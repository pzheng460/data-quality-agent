import { useState, useEffect } from 'react'
import { api } from '../hooks/useApi'
import { useApp } from '../context'

interface PaperInfo { arxiv_id: string; title: string; chars: number; source_method: string }
interface StageResult { phase: string; input_count: number; output_count: number; rejected_count: number; keep_rate: number; reject_reasons: Record<string, number>; duration_seconds: number }
interface ParamDef { type: string; label: string; default?: any; required?: boolean }
interface SourceDef { name: string; domain: string; priority: number; params: Record<string, ParamDef> }

const PIPELINE_STAGES = [
  { key: 'ingestion', num: 1, label: 'Ingestion', desc: 'Copy input into pipeline' },
  { key: 'extraction', num: 2, label: 'Extraction', desc: 'Convert raw format to text' },
  { key: 'curation', num: 3, label: 'Curation', desc: 'Filter + dedup + contamination' },
  { key: 'packaging', num: 4, label: 'Packaging', desc: 'Sort, shard, manifest' },
]

const DOMAIN_COLORS: Record<string, string> = {
  arxiv: 'bg-blue-100 text-blue-700', local: 'bg-amber-100 text-amber-700',
  web: 'bg-green-100 text-green-700', code: 'bg-purple-100 text-purple-700',
}

function getDefaults(params: Record<string, ParamDef>): Record<string, any> {
  const v: Record<string, any> = {}
  for (const [k, d] of Object.entries(params))
    v[k] = d.default ?? (d.type === 'list' ? '' : d.type === 'number' || d.type === 'float' ? 0 : '')
  return v
}

function PathLabel({ label, path }: { label: string; path: string }) {
  return (
    <span className="inline-flex items-center gap-1 text-[11px] text-gray-400">
      {label}: <code className="bg-gray-100 text-gray-600 px-1 rounded font-mono truncate max-w-[250px]">{path}</code>
    </span>
  )
}

export default function PipelineControl() {
  const {
    outputDir, setOutputDir, refresh,
    ingestOutput, setIngestOutput,
    ingestStatus, setIngestStatus,
    papers, setPapers,
    activeDomain, setActiveDomain,
    activeSource, setActiveSource,
    paramValues, setParamValues,
    pipeInput, setPipeInput,
    configPath, setConfigPath,
    workers, setWorkers,
    resume, setResume,
    pipeStatus, setPipeStatus,
    stageResults, setStageResults,
    pipeError, setPipeError,
  } = useApp()

  // Local-only state (transient, OK to lose)
  const [sourcesByDomain, setSourcesByDomain] = useState<Record<string, SourceDef[]>>({})
  const [ingestLimit, setIngestLimit] = useState(0)
  const [ingestError, setIngestError] = useState<string | null>(null)

  // Derived
  const domains = Object.keys(sourcesByDomain)
  const sourcesInDomain = sourcesByDomain[activeDomain] || []
  const currentSource = sourcesInDomain.find(s => s.name === activeSource)

  // Stage I/O paths
  const stageIO: Record<string, { input: string; output: string }> = {
    ingestion: { input: pipeInput, output: `${outputDir}/stage1_ingested/kept` },
    extraction: { input: `${outputDir}/stage1_ingested/kept`, output: `${outputDir}/stage2_extracted/kept` },
    curation: { input: `${outputDir}/stage2_extracted/kept`, output: `${outputDir}/stage3_curated/kept` },
    packaging: { input: `${outputDir}/stage3_curated/kept`, output: `${outputDir}/stage4_final` },
  }

  // Load sources
  useEffect(() => {
    api<Record<string, SourceDef[]>>('/api/sources').then(data => {
      setSourcesByDomain(data)
      const d = Object.keys(data)
      // Only set defaults if not already selected (preserve state across tab switches)
      if (d.length && !activeDomain) { setActiveDomain(d[0]); const f = data[d[0]][0]; if (f) { setActiveSource(f.name); setParamValues(getDefaults(f.params)) } }
    }).catch(() => {})
  }, [])

  useEffect(() => { loadStageResults() }, [outputDir])

  function selectDomain(d: string) { setActiveDomain(d); const s = sourcesByDomain[d] || []; if (s.length) { setActiveSource(s[0].name); setParamValues(getDefaults(s[0].params)) } }
  function selectSource(n: string) { setActiveSource(n); const s = sourcesInDomain.find(x => x.name === n); if (s) setParamValues(getDefaults(s.params)) }

  // ── Ingest action ──
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
        body: JSON.stringify({ source: activeSource, params, output_path: ingestOutput, limit: ingestLimit || 0 })
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
        if (seen && d.status === 'done') { setIngestStatus('done'); setPipelineInput(ingestOutput); clearInterval(timer) }
        else if (seen && d.status === 'error') { setIngestStatus('error'); setIngestError(d.error); clearInterval(timer) }
      } catch {}
    }, 1000)
    return () => clearInterval(timer)
  }, [ingestStatus])

  // When ingest finishes, auto-set pipeline input
  function setPipelineInput(path: string) { setPipeInput(path) }

  // ── Pipeline polling ──
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
      const data = await api<any[]>(`/api/stages/all?output_dir=${encodeURIComponent(outputDir)}`)
      const r: Record<string, StageResult> = {}
      for (const s of data) { if (s.stats) r[s.name] = s.stats }
      setStageResults(r)
    } catch {}
  }

  const runStage = async (n: number) => {
    setPipeError(null); setPipeStatus('running')
    try {
      await api('/api/run-phase', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ input_path: pipeInput, output_dir: outputDir, config_path: configPath, phase: n, workers, num_samples: 0 }) })
    } catch (e: any) { setPipeError(e.message); setPipeStatus('error') }
  }

  const startFullPipeline = async () => {
    setPipeError(null); setStageResults({}); setPipeStatus('running')
    try {
      await api('/api/run', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ input_path: pipeInput, output_dir: outputDir, config_path: configPath, workers, num_samples: 0, resume }) })
    } catch (e: any) { setPipeError(e.message); setPipeStatus('error') }
  }

  return (
    <div className="space-y-6">

      {/* ═══════════ Section 1: Ingest Data ═══════════ */}
      <div className="bg-white rounded-lg shadow">
        <div className="px-5 py-3 border-b border-gray-100">
          <h2 className="text-lg font-bold">Ingest Data</h2>
          <p className="text-xs text-gray-400">Download or load raw data from a source</p>
        </div>
        <div className="px-5 py-4 space-y-3">
          {/* Domain tabs */}
          <div className="flex gap-2 items-center">
            {domains.map(d => (
              <button key={d} onClick={() => selectDomain(d)}
                className={`px-2.5 py-1 text-xs rounded font-medium capitalize ${activeDomain === d ? (DOMAIN_COLORS[d] || 'bg-gray-200 text-gray-700') : 'bg-gray-50 text-gray-400'}`}
              >{d}</button>
            ))}
          </div>

          {/* Arxiv domain: unified inputs + source picker */}
          {activeDomain === 'arxiv' && (
            <div className="space-y-3">
              <div className="grid grid-cols-2 gap-3">
                <label className="block text-xs text-gray-600">
                  Arxiv IDs <span className="text-gray-400">(or use date range below)</span>
                  <textarea value={paramValues['ids'] ?? ''} onChange={e => setParamValues(v => ({...v, ids: e.target.value}))}
                    rows={2} className="mt-1 block w-full rounded border border-gray-300 px-2 py-1 text-xs font-mono" placeholder="2310.06825, 1706.03762" />
                </label>
                <div className="grid grid-cols-3 gap-2">
                  <label className="block text-xs text-gray-600">From date
                    <input type="date" value={paramValues['from_date'] ?? ''} onChange={e => setParamValues(v => ({...v, from_date: e.target.value}))}
                      className="mt-1 block w-full rounded border border-gray-300 px-1.5 py-1 text-xs" />
                  </label>
                  <label className="block text-xs text-gray-600">To date
                    <input type="date" value={paramValues['to_date'] ?? ''} onChange={e => setParamValues(v => ({...v, to_date: e.target.value}))}
                      className="mt-1 block w-full rounded border border-gray-300 px-1.5 py-1 text-xs" />
                  </label>
                  <label className="block text-xs text-gray-600">Categories
                    <input value={paramValues['categories'] ?? ''} onChange={e => setParamValues(v => ({...v, categories: e.target.value}))}
                      className="mt-1 block w-full rounded border border-gray-300 px-1.5 py-1 text-xs font-mono" placeholder="cs.CL" />
                  </label>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <span className="text-xs text-gray-500">Source:</span>
                {sourcesInDomain.map(s => (
                  <button key={s.name} onClick={() => selectSource(s.name)}
                    className={`px-2.5 py-1 text-xs rounded border ${activeSource === s.name ? 'border-indigo-400 bg-indigo-50 text-indigo-700 font-medium' : 'border-gray-200 text-gray-500 hover:bg-gray-50'}`}
                  >{s.name.replace(/^[^_]+_/, '')}</button>
                ))}
                <span className="text-gray-200">|</span>
                <label className="flex items-center gap-1 text-xs text-gray-600">
                  Limit <input type="number" value={ingestLimit} onChange={e => setIngestLimit(Number(e.target.value))} className="w-16 rounded border border-gray-300 px-2 py-0.5 text-xs" placeholder="0" />
                </label>
              </div>
            </div>
          )}

          {/* Non-arxiv domains: dynamic params from schema */}
          {activeDomain !== 'arxiv' && currentSource && (
            <div className="grid grid-cols-3 gap-3">
              {Object.entries(currentSource.params).map(([key, def]) => (
                <label key={key} className="block text-xs text-gray-600">
                  {def.label}{def.required && <span className="text-red-400">*</span>}
                  {def.type === 'list' ? (
                    <textarea value={paramValues[key] ?? ''} onChange={e => setParamValues(v => ({...v, [key]: e.target.value}))}
                      rows={2} className="mt-1 block w-full rounded border border-gray-300 px-2 py-1 text-xs font-mono" />
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

          {/* Output path + action */}
          <div className="flex items-end gap-3">
            <label className="block text-xs text-gray-600 flex-1">
              Output path
              <input value={ingestOutput} onChange={e => { setIngestOutput(e.target.value); setPipeInput(e.target.value) }}
                className="mt-1 block w-full rounded border border-gray-300 px-2 py-1.5 text-xs font-mono" />
            </label>
            <button onClick={startIngest} disabled={ingestStatus === 'downloading' || !activeSource}
              className={`px-4 py-1.5 rounded text-white text-xs font-medium ${ingestStatus === 'downloading' ? 'bg-gray-400' : 'bg-indigo-600 hover:bg-indigo-700'}`}>
              {ingestStatus === 'downloading' ? `Ingesting (${papers.length})...` : 'Ingest'}
            </button>
          </div>

          {ingestError && <div className="text-xs text-red-600 mt-1">{ingestError}</div>}
          {ingestStatus === 'done' && <div className="text-xs text-green-600 mt-1">Done! {papers.length} docs → {ingestOutput}</div>}

          {papers.length > 0 && (
            <div className="overflow-auto max-h-36 border rounded text-xs">
              <table className="w-full">
                <thead className="bg-gray-50 sticky top-0"><tr><th className="px-2 py-1 text-left">ID</th><th className="px-2 py-1 text-left">Title</th><th className="px-2 py-1 text-right">Size</th><th className="px-2 py-1">Source</th></tr></thead>
                <tbody className="divide-y">{papers.map((p, i) => (
                  <tr key={i}><td className="px-2 py-1 font-mono">{p.arxiv_id}</td><td className="px-2 py-1 truncate max-w-[200px]">{p.title}</td><td className="px-2 py-1 text-right">{(p.chars/1000).toFixed(1)}k</td><td className="px-2 py-1"><span className="bg-gray-100 text-gray-600 px-1 rounded text-[10px]">{p.source_method}</span></td></tr>
                ))}</tbody>
              </table>
            </div>
          )}
        </div>
      </div>

      {/* ═══════════ Section 2: Pipeline ═══════════ */}
      <div className="bg-white rounded-lg shadow">
        <div className="px-5 py-3 border-b border-gray-100">
          <h3 className="font-bold text-lg">Pipeline</h3>
          <p className="text-xs text-gray-400">4-stage processing: ingest → extract → curate → package</p>
        </div>

        {/* Pipeline config */}
        <div className="px-5 py-3 bg-gray-50 border-b border-gray-100">
          <div className="grid grid-cols-4 gap-3">
            <label className="block text-xs text-gray-600">Input path
              <input value={pipeInput} onChange={e => setPipeInput(e.target.value)} className="mt-1 block w-full rounded border border-gray-300 px-2 py-1 text-xs font-mono" />
            </label>
            <label className="block text-xs text-gray-600">Output directory
              <input value={outputDir} onChange={e => setOutputDir(e.target.value)} className="mt-1 block w-full rounded border border-gray-300 px-2 py-1 text-xs font-mono" />
            </label>
            <label className="block text-xs text-gray-600">Config YAML
              <input value={configPath} onChange={e => setConfigPath(e.target.value)} className="mt-1 block w-full rounded border border-gray-300 px-2 py-1 text-xs font-mono" />
            </label>
            <div className="flex gap-3 items-end">
              <label className="block text-xs text-gray-600 flex-1">Workers
                <input type="number" value={workers} onChange={e => setWorkers(Number(e.target.value))} className="mt-1 block w-full rounded border border-gray-300 px-2 py-1 text-xs" />
              </label>
              <label className="flex items-center gap-1.5 pb-1 text-xs text-gray-600">
                <input type="checkbox" checked={resume} onChange={e => setResume(e.target.checked)} /> Resume
              </label>
            </div>
          </div>
        </div>

        {/* Stage cards */}
        <div className="divide-y divide-gray-100">
          {PIPELINE_STAGES.map(stage => {
            const result = stageResults[stage.key]
            const isSkipped = result && (result as any).skipped
            const isDone = !!result && !isSkipped
            const io = stageIO[stage.key]

            return (
              <div key={stage.key} className={`px-5 py-3 ${isDone ? 'bg-green-50/30' : isSkipped ? 'bg-yellow-50/30' : ''}`}>
                <div className="flex items-center gap-3">
                  <div className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold shrink-0 ${isDone ? 'bg-green-500 text-white' : isSkipped ? 'bg-yellow-400 text-white' : 'bg-gray-200 text-gray-500'}`}>
                    {stage.num}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="font-semibold text-sm">{stage.label}</span>
                      <span className="text-xs text-gray-400">{stage.desc}</span>
                    </div>
                    <div className="flex gap-4 mt-0.5">
                      <PathLabel label="in" path={io.input} />
                      <PathLabel label="out" path={io.output} />
                    </div>
                  </div>
                  {isSkipped && <span className="text-xs text-yellow-600 bg-yellow-50 px-2 py-0.5 rounded shrink-0">skipped</span>}
                  {isDone && result && (
                    <span className="text-xs text-gray-500 shrink-0">
                      {result.input_count}→{result.output_count} ({(result.keep_rate * 100).toFixed(1)}%, {result.duration_seconds?.toFixed(1)}s)
                    </span>
                  )}
                  <button disabled={pipeStatus === 'running'} onClick={() => runStage(stage.num)}
                    className="px-3 py-1 text-xs rounded border border-gray-300 hover:bg-gray-50 disabled:opacity-40 shrink-0">Run</button>
                </div>
                {isDone && result && Object.entries(result.reject_reasons || {}).filter(([,v]) => v > 0).length > 0 && (
                  <div className="ml-10 mt-1 flex flex-wrap gap-1">
                    {Object.entries(result.reject_reasons).filter(([,v]) => v > 0).map(([k,v]) => (
                      <span key={k} className="bg-red-100 text-red-600 text-[10px] px-1.5 rounded">{k}: {v}</span>
                    ))}
                  </div>
                )}
              </div>
            )
          })}
        </div>

        {/* Run all */}
        <div className="px-5 py-3 border-t border-gray-100 flex items-center gap-4">
          <button onClick={startFullPipeline} disabled={pipeStatus === 'running'}
            className={`px-5 py-2 rounded font-medium text-white text-sm ${pipeStatus === 'running' ? 'bg-gray-400' : 'bg-blue-600 hover:bg-blue-700'}`}>
            {pipeStatus === 'running' ? 'Running...' : 'Run All Stages'}
          </button>
          {pipeError && <span className="text-sm text-red-600">{pipeError}</span>}
          {pipeStatus === 'finished' && <span className="text-sm text-green-600">Pipeline complete!</span>}
        </div>
      </div>
    </div>
  )
}
