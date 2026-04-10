import { useState, useEffect } from 'react'
import { api } from '../hooks/useApi'
import { useApp } from '../context'

interface PaperInfo { arxiv_id: string; title: string; categories: string[]; primary_category: string; abstract?: string; chars: number; source_method: string }
interface PhaseResult { phase: string; input_count: number; output_count: number; rejected_count: number; keep_rate: number; reject_reasons: Record<string, number>; duration_seconds: number }
interface ParamDef { type: string; label: string; default?: any; required?: boolean }
interface SourceDef { name: string; domain: string; priority: number; params: Record<string, ParamDef> }

const phaseOrder = ['ingestion', 'extraction', 'curation', 'packaging']
const LABELS: Record<string, string> = { ingestion: '1. Ingest', extraction: '2. Extract', curation: '3. Curate', packaging: '4. Package' }

const DOMAIN_COLORS: Record<string, string> = {
  arxiv: 'bg-blue-100 text-blue-700',
  local: 'bg-amber-100 text-amber-700',
  web: 'bg-green-100 text-green-700',
  code: 'bg-purple-100 text-purple-700',
}

function getDefaults(params: Record<string, ParamDef>): Record<string, any> {
  const vals: Record<string, any> = {}
  for (const [k, v] of Object.entries(params)) {
    vals[k] = v.default ?? (v.type === 'list' ? '' : v.type === 'number' || v.type === 'float' ? 0 : '')
  }
  return vals
}

export default function PipelineControl() {
  const { outputDir, setOutputDir, refresh } = useApp()

  // ── Source discovery ──
  const [sourcesByDomain, setSourcesByDomain] = useState<Record<string, SourceDef[]>>({})
  const [activeDomain, setActiveDomain] = useState('')
  const [activeSource, setActiveSource] = useState('')
  const [paramValues, setParamValues] = useState<Record<string, any>>({})

  // ── Ingest state ──
  const [outputPath, setOutputPath] = useState('/tmp/arxiv_pipeline/input.jsonl')
  const [ingestLimit, setIngestLimit] = useState(0)
  const [ingestStatus, setIngestStatus] = useState('idle')
  const [papers, setPapers] = useState<PaperInfo[]>([])
  const [ingestError, setIngestError] = useState<string | null>(null)

  // ── Pipeline state ──
  const [inputPath, setInputPath] = useState('/tmp/arxiv_pipeline/input.jsonl')
  const [configPath, setConfigPath] = useState((window as any).__DQ_CONFIG || 'configs/arxiv.yaml')
  const [workers, setWorkers] = useState(1)
  const [resume, setResume] = useState(false)
  const [pipeStatus, setPipeStatus] = useState('idle')
  const [phaseResults, setPhaseResults] = useState<Record<string, PhaseResult>>({})
  const [pipeError, setPipeError] = useState<string | null>(null)

  // Load sources on mount
  useEffect(() => {
    api<Record<string, any[]>>('/api/sources').then(data => {
      setSourcesByDomain(data)
      const doms = Object.keys(data)
      if (doms.length > 0) {
        setActiveDomain(doms[0])
        const first = data[doms[0]][0]
        if (first) { setActiveSource(first.name); setParamValues(getDefaults(first.params)) }
      }
    }).catch(() => {})
  }, [])

  useEffect(() => { loadPhaseResults() }, [outputDir])

  const domains = Object.keys(sourcesByDomain)
  const sourcesInDomain = sourcesByDomain[activeDomain] || []
  const currentSource = sourcesInDomain.find(s => s.name === activeSource)

  function selectDomain(d: string) {
    setActiveDomain(d)
    const srcs = sourcesByDomain[d] || []
    if (srcs.length) { setActiveSource(srcs[0].name); setParamValues(getDefaults(srcs[0].params)) }
  }

  function selectSource(name: string) {
    setActiveSource(name)
    const src = sourcesInDomain.find(s => s.name === name)
    if (src) setParamValues(getDefaults(src.params))
  }

  // ── Ingest ──
  const startDownload = async () => {
    setPapers([]); setIngestError(null); setIngestStatus('downloading')
    try {
      const params: Record<string, any> = {}
      if (currentSource) {
        for (const [k, def] of Object.entries(currentSource.params)) {
          let val = paramValues[k]
          if (def.type === 'list' && typeof val === 'string') val = val.split(/[\n,\s]+/).map((s: string) => s.trim()).filter(Boolean)
          if ((def.type === 'number' || def.type === 'float') && typeof val === 'string') val = Number(val)
          if (val !== undefined && val !== '' && !(Array.isArray(val) && val.length === 0)) params[k] = val
        }
      }
      await api('/api/ingest', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source: activeSource, params, output_path: outputPath, limit: ingestLimit || 0 })
      })
    } catch (e: any) { setIngestError(e.message); setIngestStatus('error') }
  }

  // poll ingest
  useEffect(() => {
    if (ingestStatus !== 'downloading') return
    let seenDownloading = false
    const timer = setInterval(async () => {
      try {
        const data = await api<any>('/api/ingest/status')
        if (data.status === 'downloading') seenDownloading = true
        setPapers(data.papers || [])
        if (seenDownloading && data.status === 'done') { setIngestStatus('done'); setInputPath(outputPath); clearInterval(timer) }
        else if (seenDownloading && data.status === 'error') { setIngestStatus('error'); setIngestError(data.error); clearInterval(timer) }
      } catch {}
    }, 1000)
    return () => clearInterval(timer)
  }, [ingestStatus])

  // ── Pipeline ──
  useEffect(() => {
    if (pipeStatus !== 'running') return
    const timer = setInterval(async () => {
      try {
        const data = await api<any>('/api/status')
        if (data.progress) {
          const r: Record<string, PhaseResult> = {}
          for (const p of data.progress) r[p.phase] = p
          setPhaseResults(r)
        }
        if (data.status === 'finished') { setPipeStatus('finished'); refresh(); clearInterval(timer) }
        else if (data.status === 'error') { setPipeStatus('error'); setPipeError(data.error); clearInterval(timer) }
      } catch {}
    }, 1000)
    return () => clearInterval(timer)
  }, [pipeStatus])

  const loadPhaseResults = async () => {
    try {
      const phases = await api<any[]>(`/api/phases?output_dir=${encodeURIComponent(outputDir)}`)
      const r: Record<string, PhaseResult> = {}
      for (const p of phases) if (p.done) try { r[p.name] = await api<any>(`/api/phase-stats/${p.name}?output_dir=${encodeURIComponent(outputDir)}`) } catch {}
      setPhaseResults(r)
    } catch {}
  }

  const startPipeline = async () => {
    setPipeError(null); setPhaseResults({}); setPipeStatus('running')
    try {
      await api('/api/run', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ input_path: inputPath, output_dir: outputDir, config_path: configPath, workers, num_samples: 0, resume }) })
    } catch (e: any) { setPipeError(e.message); setPipeStatus('error') }
  }

  const runPhase = async (n: number) => {
    try {
      setPipeStatus('running')
      await api('/api/run-phase', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ input_path: inputPath, output_dir: outputDir, config_path: configPath, phase: n, workers, num_samples: 0 }) })
    } catch (e: any) { setPipeError(e.message); setPipeStatus('error') }
  }

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">Pipeline Control</h2>

      {/* ═══ Step 1: Ingest ═══ */}
      <div className="bg-white rounded-lg shadow p-5">
        <h3 className="font-semibold text-lg mb-3">Step 1: Ingest Data</h3>

        {/* Domain tabs */}
        <div className="flex gap-2 mb-3">
          {domains.map(d => (
            <button key={d} onClick={() => {
              selectDomain(d)
            }} className={`px-3 py-1.5 text-sm rounded font-medium capitalize ${activeDomain === d ? (DOMAIN_COLORS[d]?.split(' ').join(' ') || 'bg-gray-200 text-gray-800') : 'bg-gray-50 text-gray-400'}`}
            >{d}</button>
          ))}
        </div>

        {/* Source selector */}
        {sourcesInDomain.length > 1 && (
          <div className="flex gap-2 mb-3">
            {sourcesInDomain.map(s => (
              <button key={s.name} onClick={() => selectSource(s.name)}
                className={`px-3 py-1 text-xs rounded border ${activeSource === s.name ? 'border-indigo-400 bg-indigo-50 text-indigo-700 font-medium' : 'border-gray-200 text-gray-500 hover:bg-gray-50'}`}
              >{s.name.replace(/^[^_]+_/, '')}</button>
            ))}
          </div>
        )}

        {/* Dynamic param form */}
        {currentSource && (
          <div className="grid grid-cols-2 gap-3 mb-3">
            {Object.entries(currentSource.params).map(([key, def]) => (
              <label key={key} className="block text-sm text-gray-600">
                {def.label}{def.required && <span className="text-red-400 ml-0.5">*</span>}
                {def.type === 'list' ? (
                  <textarea value={paramValues[key] ?? ''} onChange={e => setParamValues(v => ({...v, [key]: e.target.value}))}
                    rows={3} className="mt-1 block w-full rounded border border-gray-300 px-3 py-2 text-sm font-mono"
                    placeholder="one per line or comma-separated" />
                ) : (
                  <input type={def.type === 'number' || def.type === 'float' ? 'number' : 'text'}
                    value={paramValues[key] ?? ''} onChange={e => setParamValues(v => ({...v, [key]: e.target.value}))}
                    className="mt-1 block w-full rounded border border-gray-300 px-3 py-1.5 text-sm font-mono" />
                )}
              </label>
            ))}
          </div>
        )}

        <div className="flex items-center gap-3">
          <label className="block text-sm text-gray-600 flex-1">
            Output path
            <input value={outputPath} onChange={e => setOutputPath(e.target.value)} className="mt-1 block w-full rounded border border-gray-300 px-3 py-1.5 text-sm font-mono" />
          </label>
          <label className="block text-sm text-gray-600 w-24">
            Limit
            <input type="number" value={ingestLimit} onChange={e => setIngestLimit(Number(e.target.value))} className="mt-1 block w-full rounded border border-gray-300 px-2 py-1.5 text-sm" placeholder="0=all" />
          </label>
          <button onClick={startDownload} disabled={ingestStatus === 'downloading' || !activeSource}
            className={`mt-5 px-4 py-2 rounded text-white text-sm font-medium ${ingestStatus === 'downloading' ? 'bg-gray-400' : 'bg-indigo-600 hover:bg-indigo-700'}`}>
            {ingestStatus === 'downloading' ? `Downloading (${papers.length})...` : 'Download'}
          </button>
        </div>

        {ingestError && <div className="mt-2 text-sm text-red-600">{ingestError}</div>}
        {ingestStatus === 'done' && <div className="mt-2 text-sm text-green-600">Done! {papers.length} documents.</div>}

        {papers.length > 0 && (
          <div className="mt-4">
            <div className="text-sm font-medium text-gray-700 mb-2">Downloaded ({papers.length})</div>
            <div className="overflow-auto max-h-64 border rounded">
              <table className="w-full text-xs">
                <thead className="bg-gray-50 sticky top-0">
                  <tr>
                    <th className="px-3 py-2 text-left">ID</th>
                    <th className="px-3 py-2 text-left">Title</th>
                    <th className="px-3 py-2 text-right">Size</th>
                    <th className="px-3 py-2 text-left">Source</th>
                  </tr>
                </thead>
                <tbody className="divide-y">
                  {papers.map((p, i) => (
                    <tr key={i} className="hover:bg-gray-50">
                      <td className="px-3 py-1.5 font-mono">{p.arxiv_id}</td>
                      <td className="px-3 py-1.5 truncate max-w-xs">{p.title}</td>
                      <td className="px-3 py-1.5 text-right">{(p.chars / 1000).toFixed(1)}k</td>
                      <td className="px-3 py-1.5"><span className="bg-gray-100 text-gray-600 px-1.5 rounded text-[11px]">{p.source_method}</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>

      {/* ═══ Step 2: Pipeline ═══ */}
      <div className="bg-white rounded-lg shadow p-5">
        <h3 className="font-semibold text-lg mb-3">Step 2: Run Pipeline</h3>
        <div className="grid grid-cols-2 gap-4 mb-4">
          <label className="block text-sm text-gray-600">Input<input value={inputPath} onChange={e => setInputPath(e.target.value)} className="mt-1 block w-full rounded border border-gray-300 px-3 py-2 text-sm font-mono" /></label>
          <label className="block text-sm text-gray-600">Output directory<input value={outputDir} onChange={e => setOutputDir(e.target.value)} className="mt-1 block w-full rounded border border-gray-300 px-3 py-2 text-sm font-mono" /></label>
          <label className="block text-sm text-gray-600">Config YAML<input value={configPath} onChange={e => setConfigPath(e.target.value)} className="mt-1 block w-full rounded border border-gray-300 px-3 py-2 text-sm font-mono" /></label>
          <div className="flex gap-3 items-end">
            <label className="block text-sm text-gray-600">Workers<input type="number" value={workers} onChange={e => setWorkers(Number(e.target.value))} className="mt-1 block w-20 rounded border border-gray-300 px-2 py-2 text-sm" /></label>
            <label className="flex items-center gap-2 pb-2 text-sm"><input type="checkbox" checked={resume} onChange={e => setResume(e.target.checked)} />Resume</label>
          </div>
        </div>
        <div className="flex gap-3">
          <button onClick={startPipeline} disabled={pipeStatus === 'running'} className={`px-4 py-2 rounded font-medium text-white ${pipeStatus === 'running' ? 'bg-gray-400' : 'bg-blue-600 hover:bg-blue-700'}`}>
            {pipeStatus === 'running' ? 'Running...' : 'Run All Phases'}
          </button>
          {[1,2,3,4].map(n => (
            <button key={n} disabled={pipeStatus === 'running'} onClick={() => runPhase(n)} className="px-3 py-2 text-xs rounded border border-gray-300 hover:bg-gray-50 disabled:opacity-50">S{n}</button>
          ))}
        </div>
      </div>

      {pipeError && <div className="bg-red-50 border border-red-200 rounded-lg p-3 text-sm text-red-700">{pipeError}</div>}

      {Object.keys(phaseResults).length > 0 && (
        <div className="bg-white rounded-lg shadow p-5">
          <h3 className="font-semibold mb-3">Phase Progress</h3>
          <div className="space-y-2">
            {phaseOrder.map((name, i) => {
              const r = phaseResults[name]
              return (
                <div key={name} className={`flex items-center gap-3 p-3 rounded ${r ? 'bg-green-50' : 'bg-gray-50'}`}>
                  <div className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold ${r ? 'bg-green-500 text-white' : 'bg-gray-300'}`}>{i+1}</div>
                  <div className="flex-1">
                    <div className="font-medium text-sm">{LABELS[name]}</div>
                    {r && <div className="text-xs text-gray-500">{r.input_count} in → {r.output_count} kept, {r.rejected_count} rejected ({(r.keep_rate*100).toFixed(1)}%, {r.duration_seconds.toFixed(1)}s)</div>}
                    {r && Object.entries(r.reject_reasons).filter(([,v]) => v > 0).length > 0 && (
                      <div className="flex flex-wrap gap-1 mt-1">{Object.entries(r.reject_reasons).filter(([,v]) => v > 0).map(([k,v]) => <span key={k} className="bg-red-50 text-red-600 text-[10px] px-1.5 rounded">{k}: {v}</span>)}</div>
                    )}
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}
