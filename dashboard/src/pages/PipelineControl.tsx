import { useState, useEffect, useCallback, useRef } from 'react'
import { api } from '../hooks/useApi'
import { useApp } from '../context'

interface PhaseResult {
  phase: string
  input_count: number
  output_count: number
  rejected_count: number
  keep_rate: number
  reject_reasons: Record<string, number>
  duration_seconds: number
}

const PHASE_LABELS: Record<string, string> = {
  phase1_parse: '1. Parse & Validate',
  phase2_filter: '2. Quality Filter',
  phase3_dedup: '3. Dedup',
  phase4_contamination: '4. Contamination',
  phase5_package: '5. Package',
}

export default function PipelineControl() {
  const { outputDir, setOutputDir, refresh } = useApp()
  const [inputPath, setInputPath] = useState('/tmp/arxiv_test/real_papers.jsonl')
  const [configPath, setConfigPath] = useState('/tmp/arxiv_test/arxiv_test.yaml')
  const [workers, setWorkers] = useState(1)
  const [numSamples, setNumSamples] = useState(0)
  const [resume, setResume] = useState(false)

  const [status, setStatus] = useState<string>('idle')
  const [events, setEvents] = useState<Array<{ type: string; phase?: string; stats?: PhaseStats; message?: string }>>([])
  const [phaseResults, setPhaseResults] = useState<Record<string, PhaseResult>>({})
  const [error, setError] = useState<string | null>(null)
  const logRef = useRef<HTMLDivElement>(null)

  // Poll status
  const pollStatus = async () => {
    try {
      const data = await api<any>('/api/status')
      setStatus(data.status)
      if (data.progress) {
        const results: Record<string, PhaseResult> = {}
        for (const p of data.progress) results[p.phase] = p
        setPhaseResults(results)
      }
      if (data.error) setError(data.error)
      return data.status
    } catch { return 'idle' }
  }

  // Listen to SSE events
  const listenEvents = useCallback(() => {
    const es = new EventSource('http://localhost:8001/api/events')
    es.onmessage = (e) => {
      const ev = JSON.parse(e.data)
      if (ev.type === 'error') {
        setError(ev.message)
        setStatus('error')
        es.close()
      } else if (ev.type === 'end' || ev.type === 'pipeline_done') {
        setStatus('finished')
        refresh()  // Trigger other pages to reload
        es.close()
        loadPhaseResults()
      } else if (ev.type === 'phase_done') {
        setPhaseResults(prev => ({ ...prev, [ev.phase]: ev.stats }))
      }
      setEvents(prev => [...prev, ev])
    }
    es.onerror = () => {
      es.close()
      pollStatus()
    }
    return () => es.close()
  }, [outputDir])

  const loadPhaseResults = async () => {
    try {
      const phases = await api<any[]>(`/api/phases?output_dir=${encodeURIComponent(outputDir)}`)
      const results: Record<string, any> = {}
      for (const p of phases) {
        if (p.done) {
          try {
            const stats = await api<any>(`/api/phase-stats/${p.name}?output_dir=${encodeURIComponent(outputDir)}`)
            results[p.name] = stats
          } catch { /* skip */ }
        }
      }
      setPhaseResults(results)
    } catch { /* skip */ }
  }

  useEffect(() => { loadPhaseResults() }, [outputDir])
  useEffect(() => { if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight }, [events])

  const runPipeline = async () => {
    setError(null)
    setEvents([])
    setPhaseResults({})
    setStatus('running')
    try {
      await api('/api/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          input_path: inputPath,
          output_dir: outputDir,
          config_path: configPath,
          workers,
          num_samples: numSamples,
          resume,
        }),
      })
      listenEvents()
    } catch (e: any) {
      setError(e.message)
      setStatus('error')
    }
  }

  const runSinglePhase = async (phase: number) => {
    setError(null)
    setEvents([])
    setStatus('running')
    try {
      await api('/api/run-phase', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          input_path: inputPath,
          output_dir: outputDir,
          config_path: configPath,
          phase,
          workers,
          num_samples: numSamples,
        }),
      })
      listenEvents()
    } catch (e: any) {
      setError(e.message)
      setStatus('error')
    }
  }

  const phaseOrder = ['phase1_parse', 'phase2_filter', 'phase3_dedup', 'phase4_contamination', 'phase5_package']

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-gray-900">Pipeline Control</h2>

      {/* Config form */}
      <div className="bg-white rounded-lg shadow p-5 space-y-4">
        <h3 className="font-semibold">Configuration</h3>
        <div className="grid grid-cols-2 gap-4">
          <label className="block">
            <span className="text-sm text-gray-600">Input Path</span>
            <input value={inputPath} onChange={e => setInputPath(e.target.value)}
              className="mt-1 block w-full rounded border-gray-300 border px-3 py-2 text-sm" />
          </label>
          <label className="block">
            <span className="text-sm text-gray-600">Output Directory</span>
            <input value={outputDir} onChange={e => setOutputDir(e.target.value)}
              className="mt-1 block w-full rounded border-gray-300 border px-3 py-2 text-sm" />
          </label>
          <label className="block">
            <span className="text-sm text-gray-600">Config YAML</span>
            <input value={configPath} onChange={e => setConfigPath(e.target.value)}
              className="mt-1 block w-full rounded border-gray-300 border px-3 py-2 text-sm" />
          </label>
          <div className="flex gap-4">
            <label className="block">
              <span className="text-sm text-gray-600">Workers</span>
              <input type="number" value={workers} onChange={e => setWorkers(Number(e.target.value))}
                className="mt-1 block w-24 rounded border-gray-300 border px-3 py-2 text-sm" />
            </label>
            <label className="block">
              <span className="text-sm text-gray-600">Sample limit (0=all)</span>
              <input type="number" value={numSamples} onChange={e => setNumSamples(Number(e.target.value))}
                className="mt-1 block w-24 rounded border-gray-300 border px-3 py-2 text-sm" />
            </label>
            <label className="flex items-end gap-2 pb-2">
              <input type="checkbox" checked={resume} onChange={e => setResume(e.target.checked)} />
              <span className="text-sm text-gray-600">Resume</span>
            </label>
          </div>
        </div>

        <div className="flex gap-3 pt-2">
          <button onClick={runPipeline} disabled={status === 'running'}
            className={`px-4 py-2 rounded font-medium text-white ${status === 'running' ? 'bg-gray-400' : 'bg-blue-600 hover:bg-blue-700'}`}>
            {status === 'running' ? 'Running...' : 'Run Full Pipeline'}
          </button>
          {[1,2,3,4,5].map(n => (
            <button key={n} disabled={status === 'running'}
              onClick={() => runSinglePhase(n)}
              className="px-3 py-2 text-xs rounded border border-gray-300 hover:bg-gray-50 disabled:opacity-50">
              Phase {n}
            </button>
          ))}
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700 text-sm">{error}</div>
      )}

      {/* Phase progress */}
      <div className="bg-white rounded-lg shadow p-5">
        <h3 className="font-semibold mb-4">Phase Progress</h3>
        <div className="space-y-3">
          {phaseOrder.map((name, i) => {
            const result = phaseResults[name]
            const isRunning = status === 'running' && events.some(e => e.type === 'phase_start' && e.phase === name) &&
              !events.some(e => e.type === 'phase_done' && e.phase === name)
            return (
              <div key={name} className={`flex items-center gap-4 p-3 rounded-lg ${
                result ? 'bg-green-50' : isRunning ? 'bg-blue-50' : 'bg-gray-50'
              }`}>
                <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold ${
                  result ? 'bg-green-500 text-white' : isRunning ? 'bg-blue-500 text-white animate-pulse' : 'bg-gray-300 text-gray-600'
                }`}>{i + 1}</div>
                <div className="flex-1">
                  <div className="font-medium text-sm">{PHASE_LABELS[name] || name}</div>
                  {result && (
                    <div className="text-xs text-gray-600 mt-0.5">
                      {result.input_count} in &rarr; {result.output_count} kept, {result.rejected_count} rejected
                      ({(result.keep_rate * 100).toFixed(1)}%)
                      <span className="ml-2 text-gray-400">{result.duration_seconds.toFixed(1)}s</span>
                    </div>
                  )}
                  {result && Object.keys(result.reject_reasons || {}).filter(k => (result.reject_reasons as Record<string, number>)[k] > 0).length > 0 && (
                    <div className="flex flex-wrap gap-1 mt-1">
                      {Object.entries(result.reject_reasons).filter(([, v]) => v > 0).map(([rule, count]) => (
                        <span key={rule} className="bg-red-100 text-red-700 text-[10px] px-1.5 py-0.5 rounded">{rule}: {count}</span>
                      ))}
                    </div>
                  )}
                  {isRunning && <div className="text-xs text-blue-600 mt-0.5 animate-pulse">Running...</div>}
                </div>
                {!result && !isRunning && (
                  <button onClick={() => runSinglePhase(i + 1)}
                    disabled={status === 'running'}
                    className="text-xs px-2 py-1 rounded border hover:bg-gray-100 disabled:opacity-50">
                    Run
                  </button>
                )}
              </div>
            )
          })}
        </div>
      </div>

      {/* Event log */}
      <div className="bg-white rounded-lg shadow p-5">
        <h3 className="font-semibold mb-2">Event Log</h3>
        <div ref={logRef} className="bg-gray-900 text-gray-300 rounded p-3 font-mono text-xs max-h-48 overflow-y-auto">
          {events.length === 0 && <div className="text-gray-500">No events yet. Start a pipeline run.</div>}
          {events.map((ev, i) => (
            <div key={i} className={ev.type === 'error' ? 'text-red-400' : ev.type.includes('done') ? 'text-green-400' : ''}>
              [{ev.type}] {ev.phase || ''} {ev.stats ? `(${ev.stats.input_count}→${ev.stats.output_count})` : ''} {ev.message || ''}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

interface PhaseStats {
  input_count: number
  output_count: number
  rejected_count: number
  keep_rate: number
  reject_reasons: Record<string, number>
  duration_seconds: number
}
