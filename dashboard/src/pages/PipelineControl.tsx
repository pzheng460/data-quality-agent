import { useState, useEffect, useRef } from 'react'
import { api } from '../hooks/useApi'
import { useApp } from '../context'

interface PaperInfo { arxiv_id: string; title: string; categories: string[]; primary_category: string; abstract?: string; chars: number; source_method: string }
interface PhaseResult { phase: string; input_count: number; output_count: number; rejected_count: number; keep_rate: number; reject_reasons: Record<string, number>; duration_seconds: number }

const phaseOrder = ['phase1_parse', 'phase2_filter', 'phase3_dedup', 'phase4_contamination', 'phase5_package']
const LABELS: Record<string, string> = { phase1_parse: '1. Parse', phase2_filter: '2. Filter', phase3_dedup: '3. Dedup', phase4_contamination: '4. Contam.', phase5_package: '5. Package' }

export default function PipelineControl() {
  const { outputDir, setOutputDir, refresh } = useApp()

  // ── Ingest state ──
  const [ingestMode, setIngestMode] = useState<'ids' | 'date'>('ids')
  const [arxivIds, setArxivIds] = useState('1706.03762\n2310.06825\n2303.08774\n1312.6114\n2203.15556')
  const [fromDate, setFromDate] = useState('2025-04-01')
  const [toDate, setToDate] = useState('2025-04-07')
  const [catFilter, setCatFilter] = useState('cs.CL')
  const [maxPapers, setMaxPapers] = useState(20)
  const [outputPath, setOutputPath] = useState('/tmp/arxiv_pipeline/input.jsonl')
  const [ingestStatus, setIngestStatus] = useState('idle')
  const [papers, setPapers] = useState<PaperInfo[]>([])
  const [ingestError, setIngestError] = useState<string | null>(null)

  // ── Pipeline state ──
  const [inputPath, setInputPath] = useState('/tmp/arxiv_pipeline/input.jsonl')
  const [configPath, setConfigPath] = useState((window as any).__DQ_CONFIG || '/tmp/arxiv_test/arxiv_test.yaml')
  const [workers, setWorkers] = useState(1)
  const [resume, setResume] = useState(false)
  const [pipeStatus, setPipeStatus] = useState('idle')
  const [phaseResults, setPhaseResults] = useState<Record<string, PhaseResult>>({})
  const [events, _setEvents] = useState<any[]>([])
  const [pipeError, setPipeError] = useState<string | null>(null)
  const logRef = useRef<HTMLDivElement>(null)

  // auto-scroll log
  useEffect(() => { logRef.current && (logRef.current.scrollTop = logRef.current.scrollHeight) }, [events])
  // Load existing results
  useEffect(() => { loadPhaseResults() }, [outputDir])

  // ── Ingest ──
  const startDownload = async () => {
    setPapers([])
    setIngestError(null)
    setIngestStatus('downloading')
    try {
      if (ingestMode === 'ids') {
        const ids = arxivIds.split(/[\n,\s]+/).map(s => s.trim()).filter(Boolean)
        await api('/api/ingest/by-ids', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ids, output_path: outputPath, delay: 3 }) })
      } else {
        await api('/api/ingest/by-date', { method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ from_date: fromDate, to_date: toDate, categories: catFilter.split(/[,\s]+/).filter(Boolean), max_papers: maxPapers, output_path: outputPath, delay: 3 }) })
      }
    } catch (e: any) { setIngestError(e.message); setIngestStatus('error') }
  }

  // poll ingest
  // Simpler: just use state vars directly

  // Actually let me simplify — use a single poll approach
  useEffect(() => {
    if (ingestStatus !== 'downloading') return
    const timer = setInterval(async () => {
      try {
        const data = await api<any>('/api/ingest/status')
        setPapers(data.papers || [])
        if (data.status === 'done') { setIngestStatus('done'); setInputPath(outputPath); clearInterval(timer) }
        else if (data.status === 'error') { setIngestStatus('error'); setIngestError(data.error); clearInterval(timer) }
      } catch {}
    }, 2000)
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

      {/* ═══ Step 1: Download ═══ */}
      <div className="bg-white rounded-lg shadow p-5">
        <h3 className="font-semibold text-lg mb-3">Step 1: Download Papers</h3>
        <div className="flex gap-2 mb-4">
          <button onClick={() => setIngestMode('ids')} className={`px-3 py-1 text-sm rounded ${ingestMode === 'ids' ? 'bg-blue-100 text-blue-700 font-medium' : 'bg-gray-100'}`}>By Arxiv IDs</button>
          <button onClick={() => setIngestMode('date')} className={`px-3 py-1 text-sm rounded ${ingestMode === 'date' ? 'bg-blue-100 text-blue-700 font-medium' : 'bg-gray-100'}`}>By Date Range</button>
        </div>

        {ingestMode === 'ids' ? (
          <div className="space-y-3">
            <label className="block text-sm text-gray-600">Arxiv IDs (one per line or comma-separated)
              <textarea value={arxivIds} onChange={e => setArxivIds(e.target.value)} rows={4} className="mt-1 block w-full rounded border border-gray-300 px-3 py-2 text-sm font-mono" placeholder="2310.06825&#10;2307.09288" />
            </label>
          </div>
        ) : (
          <div className="grid grid-cols-4 gap-3">
            <label className="block text-sm text-gray-600">From<input type="date" value={fromDate} onChange={e => setFromDate(e.target.value)} className="mt-1 block w-full rounded border border-gray-300 px-2 py-1.5 text-sm" /></label>
            <label className="block text-sm text-gray-600">To<input type="date" value={toDate} onChange={e => setToDate(e.target.value)} className="mt-1 block w-full rounded border border-gray-300 px-2 py-1.5 text-sm" /></label>
            <label className="block text-sm text-gray-600">Categories<input value={catFilter} onChange={e => setCatFilter(e.target.value)} className="mt-1 block w-full rounded border border-gray-300 px-2 py-1.5 text-sm" placeholder="cs.CL, cs.LG" /></label>
            <label className="block text-sm text-gray-600">Max papers<input type="number" value={maxPapers} onChange={e => setMaxPapers(Number(e.target.value))} className="mt-1 block w-full rounded border border-gray-300 px-2 py-1.5 text-sm" /></label>
          </div>
        )}

        <div className="mt-3 flex items-center gap-3">
          <label className="block text-sm text-gray-600 flex-1">
            Output path
            <input value={outputPath} onChange={e => setOutputPath(e.target.value)} className="mt-1 block w-full rounded border border-gray-300 px-3 py-1.5 text-sm font-mono" />
          </label>
          <button onClick={startDownload} disabled={ingestStatus === 'downloading'} className={`mt-5 px-4 py-2 rounded text-white text-sm font-medium ${ingestStatus === 'downloading' ? 'bg-gray-400' : 'bg-indigo-600 hover:bg-indigo-700'}`}>
            {ingestStatus === 'downloading' ? `Downloading (${papers.length})...` : 'Download'}
          </button>
        </div>

        {ingestError && <div className="mt-2 text-sm text-red-600">{ingestError}</div>}

        {/* Downloaded papers table */}
        {papers.length > 0 && (
          <div className="mt-4">
            <div className="text-sm font-medium text-gray-700 mb-2">Downloaded Papers ({papers.length})</div>
            <div className="overflow-auto max-h-64 border rounded">
              <table className="w-full text-xs">
                <thead className="bg-gray-50 sticky top-0">
                  <tr>
                    <th className="px-3 py-2 text-left">arxiv ID</th>
                    <th className="px-3 py-2 text-left">Title</th>
                    <th className="px-3 py-2 text-left">Category</th>
                    <th className="px-3 py-2 text-right">Size</th>
                    <th className="px-3 py-2 text-left">Source</th>
                  </tr>
                </thead>
                <tbody className="divide-y">
                  {papers.map((p, i) => (
                    <tr key={i} className="hover:bg-gray-50">
                      <td className="px-3 py-1.5 font-mono">{p.arxiv_id}</td>
                      <td className="px-3 py-1.5 truncate max-w-xs">{p.title}</td>
                      <td className="px-3 py-1.5"><span className="bg-blue-50 text-blue-700 px-1.5 rounded text-[11px]">{p.primary_category}</span></td>
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
          <label className="block text-sm text-gray-600">Input (from download)<input value={inputPath} onChange={e => setInputPath(e.target.value)} className="mt-1 block w-full rounded border border-gray-300 px-3 py-2 text-sm font-mono" /></label>
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
          {[1,2,3,4,5].map(n => (
            <button key={n} disabled={pipeStatus === 'running'} onClick={() => runPhase(n)} className="px-3 py-2 text-xs rounded border border-gray-300 hover:bg-gray-50 disabled:opacity-50">P{n}</button>
          ))}
        </div>
      </div>

      {/* Pipeline errors */}
      {pipeError && <div className="bg-red-50 border border-red-200 rounded-lg p-3 text-sm text-red-700">{pipeError}</div>}

      {/* Phase progress */}
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

      {pipeError && <div className="bg-red-50 border border-red-200 rounded-lg p-3 text-sm text-red-700">{pipeError}</div>}
    </div>
  )
}
