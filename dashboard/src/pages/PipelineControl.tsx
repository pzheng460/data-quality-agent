import { useState, useEffect } from 'react'
import { api } from '../hooks/useApi'
import { useApp } from '../context'

interface PaperInfo { arxiv_id: string; title: string; chars: number; source_method: string }
interface StageResult { phase: string; input_count: number; output_count: number; rejected_count: number; keep_rate: number; reject_reasons: Record<string, number>; duration_seconds: number }
interface ParamDef { type: string; label: string; default?: any; required?: boolean }
interface SourceDef { name: string; domain: string; priority: number; params: Record<string, ParamDef> }

const STAGES = [
  { key: 'ingestion', num: 1, label: 'Ingestion', desc: 'Fetch raw data from source' },
  { key: 'extraction', num: 2, label: 'Extraction', desc: 'Convert to clean text' },
  { key: 'curation', num: 3, label: 'Curation', desc: 'Filter + dedup + contamination' },
  { key: 'packaging', num: 4, label: 'Packaging', desc: 'Sort, shard, manifest' },
]

const DOMAIN_COLORS: Record<string, string> = {
  arxiv: 'bg-blue-100 text-blue-700',
  local: 'bg-amber-100 text-amber-700',
  web: 'bg-green-100 text-green-700',
  code: 'bg-purple-100 text-purple-700',
}

function getDefaults(params: Record<string, ParamDef>): Record<string, any> {
  const v: Record<string, any> = {}
  for (const [k, d] of Object.entries(params))
    v[k] = d.default ?? (d.type === 'list' ? '' : d.type === 'number' || d.type === 'float' ? 0 : '')
  return v
}

export default function PipelineControl() {
  const { outputDir, setOutputDir, refresh } = useApp()

  // ── Global config ──
  const [configPath, setConfigPath] = useState('configs/arxiv.yaml')
  const [workers, setWorkers] = useState(4)
  const [resume, setResume] = useState(true)

  // ── Source discovery ──
  const [sourcesByDomain, setSourcesByDomain] = useState<Record<string, SourceDef[]>>({})
  const [activeDomain, setActiveDomain] = useState('')
  const [activeSource, setActiveSource] = useState('')
  const [paramValues, setParamValues] = useState<Record<string, any>>({})
  const [ingestLimit, setIngestLimit] = useState(0)

  // ── Ingest state ──
  const [ingestStatus, setIngestStatus] = useState('idle')
  const [papers, setPapers] = useState<PaperInfo[]>([])
  const [ingestError, setIngestError] = useState<string | null>(null)

  // ── Pipeline state ──
  const [pipeStatus, setPipeStatus] = useState('idle')
  const [stageResults, setStageResults] = useState<Record<string, StageResult>>({})
  const [pipeError, setPipeError] = useState<string | null>(null)

  // Derived
  const inputPath = `${outputDir}/stage1_ingested/kept`
  const domains = Object.keys(sourcesByDomain)
  const sourcesInDomain = sourcesByDomain[activeDomain] || []
  const currentSource = sourcesInDomain.find(s => s.name === activeSource)

  // Load sources
  useEffect(() => {
    api<Record<string, SourceDef[]>>('/api/sources').then(data => {
      setSourcesByDomain(data)
      const d = Object.keys(data)
      if (d.length) { setActiveDomain(d[0]); const f = data[d[0]][0]; if (f) { setActiveSource(f.name); setParamValues(getDefaults(f.params)) } }
    }).catch(() => {})
  }, [])

  useEffect(() => { loadStageResults() }, [outputDir])

  function selectDomain(d: string) { setActiveDomain(d); const s = sourcesByDomain[d] || []; if (s.length) { setActiveSource(s[0].name); setParamValues(getDefaults(s[0].params)) } }
  function selectSource(n: string) { setActiveSource(n); const s = sourcesInDomain.find(x => x.name === n); if (s) setParamValues(getDefaults(s.params)) }

  // ── Ingest ──
  const startIngest = async () => {
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
        body: JSON.stringify({ source: activeSource, params, output_path: `${outputDir}/stage1_ingested/kept/input.jsonl`, limit: ingestLimit || 0 })
      })
    } catch (e: any) { setIngestError(e.message); setIngestStatus('error') }
  }

  useEffect(() => {
    if (ingestStatus !== 'downloading') return
    let seen = false
    const timer = setInterval(async () => {
      try {
        const d = await api<any>('/api/ingest/status')
        if (d.status === 'downloading') seen = true
        setPapers(d.papers || [])
        if (seen && d.status === 'done') { setIngestStatus('done'); clearInterval(timer) }
        else if (seen && d.status === 'error') { setIngestStatus('error'); setIngestError(d.error); clearInterval(timer) }
      } catch {}
    }, 1000)
    return () => clearInterval(timer)
  }, [ingestStatus])

  // ── Pipeline ──
  const startPipeline = async () => {
    setPipeError(null); setStageResults({}); setPipeStatus('running')
    try {
      await api('/api/run', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ input_path: inputPath, output_dir: outputDir, config_path: configPath, workers, num_samples: 0, resume }) })
    } catch (e: any) { setPipeError(e.message); setPipeStatus('error') }
  }

  const runStage = async (n: number) => {
    setPipeError(null); setPipeStatus('running')
    try {
      await api('/api/run-phase', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ input_path: inputPath, output_dir: outputDir, config_path: configPath, phase: n, workers, num_samples: 0 }) })
    } catch (e: any) { setPipeError(e.message); setPipeStatus('error') }
  }

  useEffect(() => {
    if (pipeStatus !== 'running') return
    const timer = setInterval(async () => {
      try {
        const d = await api<any>('/api/status')
        if (d.progress) { const r: Record<string, StageResult> = {}; for (const p of d.progress) r[p.phase] = p; setStageResults(r) }
        if (d.status === 'finished') { setPipeStatus('finished'); refresh(); clearInterval(timer) }
        else if (d.status === 'error') { setPipeStatus('error'); setPipeError(d.error); clearInterval(timer) }
      } catch {}
    }, 1000)
    return () => clearInterval(timer)
  }, [pipeStatus])

  const loadStageResults = async () => {
    try {
      const stages = await api<any[]>(`/api/phases?output_dir=${encodeURIComponent(outputDir)}`)
      const r: Record<string, StageResult> = {}
      for (const s of stages) if (s.done) try { r[s.name] = await api<any>(`/api/phase-stats/${s.name}?output_dir=${encodeURIComponent(outputDir)}`) } catch {}
      setStageResults(r)
    } catch {}
  }

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">Pipeline Control</h2>

      {/* Global config */}
      <div className="bg-white rounded-lg shadow p-4">
        <div className="grid grid-cols-4 gap-4">
          <label className="block text-sm text-gray-600">Output directory
            <input value={outputDir} onChange={e => setOutputDir(e.target.value)} className="mt-1 block w-full rounded border border-gray-300 px-3 py-1.5 text-sm font-mono" />
          </label>
          <label className="block text-sm text-gray-600">Config YAML
            <input value={configPath} onChange={e => setConfigPath(e.target.value)} className="mt-1 block w-full rounded border border-gray-300 px-3 py-1.5 text-sm font-mono" />
          </label>
          <label className="block text-sm text-gray-600">Workers
            <input type="number" value={workers} onChange={e => setWorkers(Number(e.target.value))} className="mt-1 block w-full rounded border border-gray-300 px-3 py-1.5 text-sm" />
          </label>
          <label className="flex items-center gap-2 pt-6 text-sm">
            <input type="checkbox" checked={resume} onChange={e => setResume(e.target.checked)} /> Resume
          </label>
        </div>
      </div>

      {/* 4-Stage cards */}
      <div className="space-y-4">
        {STAGES.map(stage => {
          const result = stageResults[stage.key]
          const isSkipped = result && (result as any).skipped
          const isDone = !!result && !isSkipped
          const isRunning = pipeStatus === 'running'

          return (
            <div key={stage.key} className={`bg-white rounded-lg shadow overflow-hidden ${isDone ? 'ring-1 ring-green-200' : isSkipped ? 'ring-1 ring-yellow-200' : ''}`}>
              {/* Stage header */}
              <div className="flex items-center gap-3 px-5 py-3 border-b border-gray-100">
                <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold ${isDone ? 'bg-green-500 text-white' : isSkipped ? 'bg-yellow-400 text-white' : 'bg-gray-200 text-gray-600'}`}>
                  {stage.num}
                </div>
                <div className="flex-1">
                  <div className="font-semibold text-sm">{stage.label}</div>
                  <div className="text-xs text-gray-400">{stage.desc}</div>
                </div>
                {isSkipped && <span className="text-xs text-yellow-600 bg-yellow-50 px-2 py-0.5 rounded">skipped (resume)</span>}
                {isDone && result && (
                  <div className="text-xs text-gray-500">
                    {result.input_count} in → {result.output_count} kept ({(result.keep_rate * 100).toFixed(1)}%, {result.duration_seconds?.toFixed(1)}s)
                  </div>
                )}
                <button disabled={pipeStatus === 'running'} onClick={() => runStage(stage.num)}
                  className="px-3 py-1 text-xs rounded border border-gray-300 hover:bg-gray-50 disabled:opacity-40">
                  Run
                </button>
              </div>

              {/* Stage-specific config (expanded for ingestion) */}
              {stage.key === 'ingestion' && (
                <div className="px-5 py-3 bg-gray-50 space-y-3">
                  {/* Domain tabs */}
                  <div className="flex gap-2">
                    {domains.map(d => (
                      <button key={d} onClick={() => selectDomain(d)}
                        className={`px-2.5 py-1 text-xs rounded font-medium capitalize ${activeDomain === d ? (DOMAIN_COLORS[d] || 'bg-gray-200 text-gray-700') : 'bg-white text-gray-400 border border-gray-200'}`}
                      >{d}</button>
                    ))}
                  </div>
                  {/* Source picker */}
                  {sourcesInDomain.length > 1 && (
                    <div className="flex gap-2">
                      {sourcesInDomain.map(s => (
                        <button key={s.name} onClick={() => selectSource(s.name)}
                          className={`px-2.5 py-1 text-xs rounded border ${activeSource === s.name ? 'border-indigo-400 bg-indigo-50 text-indigo-700 font-medium' : 'border-gray-200 text-gray-500'}`}
                        >{s.name.replace(/^[^_]+_/, '')}</button>
                      ))}
                    </div>
                  )}
                  {/* Source params */}
                  {currentSource && (
                    <div className="grid grid-cols-3 gap-3">
                      {Object.entries(currentSource.params).map(([key, def]) => (
                        <label key={key} className="block text-xs text-gray-600">
                          {def.label}{def.required && <span className="text-red-400">*</span>}
                          {def.type === 'list' ? (
                            <textarea value={paramValues[key] ?? ''} onChange={e => setParamValues(v => ({...v, [key]: e.target.value}))}
                              rows={2} className="mt-1 block w-full rounded border border-gray-300 px-2 py-1 text-xs font-mono" placeholder="one per line" />
                          ) : (
                            <input type={def.type === 'number' || def.type === 'float' ? 'number' : 'text'}
                              value={paramValues[key] ?? ''} onChange={e => setParamValues(v => ({...v, [key]: e.target.value}))}
                              className="mt-1 block w-full rounded border border-gray-300 px-2 py-1 text-xs font-mono" />
                          )}
                        </label>
                      ))}
                      <label className="block text-xs text-gray-600">Limit (0=all)
                        <input type="number" value={ingestLimit} onChange={e => setIngestLimit(Number(e.target.value))} className="mt-1 block w-full rounded border border-gray-300 px-2 py-1 text-xs" />
                      </label>
                    </div>
                  )}
                  <button onClick={startIngest} disabled={ingestStatus === 'downloading' || !activeSource}
                    className={`px-3 py-1.5 rounded text-white text-xs font-medium ${ingestStatus === 'downloading' ? 'bg-gray-400' : 'bg-indigo-600 hover:bg-indigo-700'}`}>
                    {ingestStatus === 'downloading' ? `Ingesting (${papers.length})...` : 'Ingest Data'}
                  </button>
                  {ingestError && <span className="ml-2 text-xs text-red-600">{ingestError}</span>}
                  {ingestStatus === 'done' && <span className="ml-2 text-xs text-green-600">Done! {papers.length} docs</span>}

                  {papers.length > 0 && (
                    <div className="overflow-auto max-h-40 border rounded text-xs">
                      <table className="w-full">
                        <thead className="bg-gray-50 sticky top-0"><tr><th className="px-2 py-1 text-left">ID</th><th className="px-2 py-1 text-left">Title</th><th className="px-2 py-1 text-right">Size</th><th className="px-2 py-1">Source</th></tr></thead>
                        <tbody className="divide-y">{papers.map((p, i) => (
                          <tr key={i}><td className="px-2 py-1 font-mono">{p.arxiv_id}</td><td className="px-2 py-1 truncate max-w-[200px]">{p.title}</td><td className="px-2 py-1 text-right">{(p.chars/1000).toFixed(1)}k</td><td className="px-2 py-1"><span className="bg-gray-100 px-1 rounded text-[10px]">{p.source_method}</span></td></tr>
                        ))}</tbody>
                      </table>
                    </div>
                  )}
                </div>
              )}

              {/* Reject reasons for completed stages */}
              {result && Object.entries(result.reject_reasons).filter(([,v]) => v > 0).length > 0 && (
                <div className="px-5 py-2 bg-red-50/50 flex flex-wrap gap-1">
                  {Object.entries(result.reject_reasons).filter(([,v]) => v > 0).map(([k,v]) => (
                    <span key={k} className="bg-red-100 text-red-600 text-[10px] px-1.5 rounded">{k}: {v}</span>
                  ))}
                </div>
              )}
            </div>
          )
        })}
      </div>

      {/* Run controls */}
      <div className="bg-white rounded-lg shadow p-4 flex items-center gap-4">
        <button onClick={startPipeline} disabled={pipeStatus === 'running'}
          className={`px-5 py-2 rounded font-medium text-white ${pipeStatus === 'running' ? 'bg-gray-400' : 'bg-blue-600 hover:bg-blue-700'}`}>
          {pipeStatus === 'running' ? 'Running...' : 'Run All Stages'}
        </button>
        <label className="flex items-center gap-2 text-sm text-gray-600">
          <input type="checkbox" checked={resume} onChange={e => setResume(e.target.checked)} /> Resume from checkpoint
        </label>
        {pipeError && <span className="text-sm text-red-600">{pipeError}</span>}
        {pipeStatus === 'finished' && <span className="text-sm text-green-600">Pipeline complete!</span>}
      </div>
    </div>
  )
}
