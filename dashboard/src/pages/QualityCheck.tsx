import { useState, useEffect } from 'react'
import { useApp } from '../context'
import { api } from '../hooks/useApi'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Cell } from 'recharts'

interface FilterData {
  total: number; passed: number; failed: number; pass_rate: number
  sample_failed?: { text_preview: string; reason: { filter: string; reason: string; value?: any } }[]
  rules?: Record<string, { total: number; passed: number; failed: number; pass_rate: number }>
}

interface DatasetResult {
  num_docs: number; overall_pass_rate: number; data_type: string
  per_filter: Record<string, FilterData>
  dataset_stats?: {
    avg_word_count: number; min_word_count: number; max_word_count: number
    avg_word_length: number; exact_duplicates?: number; duplicate_rate?: number
  }
}

interface BenchResult {
  datasets: Record<string, DatasetResult>
}

function passColor(rate: number) {
  if (rate >= 0.95) return '#22c55e'
  if (rate >= 0.8) return '#eab308'
  return '#ef4444'
}

export default function QualityCheck() {
  const { outputDir } = useApp()

  const [inputPath, setInputPath] = useState('/tmp/arxiv_pipeline/input.jsonl')
  const [configPath, setConfigPath] = useState('configs/arxiv.yaml')
  const [numSamples, setNumSamples] = useState(100)
  const [workers, setWorkers] = useState(4)
  const [dataType, setDataType] = useState('auto')
  const [status, setStatus] = useState('idle')
  const [result, setResult] = useState<BenchResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [expandedFilter, setExpandedFilter] = useState<string | null>(null)

  const startBench = async () => {
    setStatus('running'); setError(null); setResult(null)
    try {
      await api('/api/bench', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ input_path: inputPath, config_path: configPath, num_samples: numSamples, data_type: dataType, workers })
      })
    } catch (e: any) { setStatus('error'); setError(e.message) }
  }

  // Poll
  useEffect(() => {
    if (status !== 'running') return
    const timer = setInterval(async () => {
      try {
        const data = await api<any>('/api/bench/status')
        if (data.status === 'done') { setStatus('done'); setResult(data.result); clearInterval(timer) }
        else if (data.status === 'error') { setStatus('error'); setError(data.error); clearInterval(timer) }
      } catch {}
    }, 1000)
    return () => clearInterval(timer)
  }, [status])

  const ds = result?.datasets ? Object.values(result.datasets)[0] : null
  const stats = ds?.dataset_stats
  const filters = ds?.per_filter || {}
  const filterNames = Object.keys(filters)

  const chartData = filterNames.map(name => ({
    name: name.length > 18 ? name.slice(0, 17) + '...' : name,
    fullName: name,
    pass_rate: Math.round((filters[name].pass_rate || 0) * 100),
  }))

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">Quality Benchmark</h2>

      {/* Config */}
      <div className="bg-white rounded-lg shadow p-5">
        <h3 className="font-semibold text-lg mb-3">Configuration</h3>
        <div className="grid grid-cols-2 gap-4 mb-4">
          <label className="block text-sm text-gray-600">Input path
            <input value={inputPath} onChange={e => setInputPath(e.target.value)} className="mt-1 block w-full rounded border border-gray-300 px-3 py-2 text-sm font-mono" />
          </label>
          <label className="block text-sm text-gray-600">Config YAML
            <input value={configPath} onChange={e => setConfigPath(e.target.value)} className="mt-1 block w-full rounded border border-gray-300 px-3 py-2 text-sm font-mono" />
          </label>
          <div className="flex gap-3">
            <label className="block text-sm text-gray-600 flex-1">Samples
              <input type="number" value={numSamples} onChange={e => setNumSamples(Number(e.target.value))} className="mt-1 block w-full rounded border border-gray-300 px-3 py-1.5 text-sm" />
            </label>
            <label className="block text-sm text-gray-600 flex-1">Workers
              <input type="number" value={workers} onChange={e => setWorkers(Number(e.target.value))} className="mt-1 block w-full rounded border border-gray-300 px-3 py-1.5 text-sm" />
            </label>
            <label className="block text-sm text-gray-600 flex-1">Data type
              <select value={dataType} onChange={e => setDataType(e.target.value)} className="mt-1 block w-full rounded border border-gray-300 px-2 py-1.5 text-sm">
                <option value="auto">auto</option>
                <option value="pretrain">pretrain</option>
                <option value="sft">sft</option>
              </select>
            </label>
          </div>
        </div>
        <button onClick={startBench} disabled={status === 'running'}
          className={`px-4 py-2 rounded font-medium text-white text-sm ${status === 'running' ? 'bg-gray-400' : 'bg-indigo-600 hover:bg-indigo-700'}`}>
          {status === 'running' ? 'Running...' : 'Run Benchmark'}
        </button>
        {error && <div className="mt-2 text-sm text-red-600">{error}</div>}
      </div>

      {ds && (
        <>
          {/* KPI cards */}
          <div className="grid grid-cols-4 gap-4">
            {[
              { label: 'Documents', value: ds.num_docs },
              { label: 'Overall Pass Rate', value: `${(ds.overall_pass_rate * 100).toFixed(1)}%`, color: passColor(ds.overall_pass_rate) },
              { label: 'Avg Word Count', value: stats?.avg_word_count?.toFixed(0) ?? '—' },
              { label: 'Data Type', value: ds.data_type, className: 'capitalize' },
            ].map((kpi, i) => (
              <div key={i} className="bg-white rounded-lg shadow p-4">
                <div className="text-xs text-gray-500 uppercase tracking-wide">{kpi.label}</div>
                <div className={`text-2xl font-bold mt-1 ${kpi.className || ''}`} style={{ color: kpi.color }}>{kpi.value}</div>
              </div>
            ))}
          </div>

          {/* Stats detail */}
          {stats && (
            <div className="bg-white rounded-lg shadow p-5">
              <h3 className="font-semibold mb-3">Dataset Statistics</h3>
              <div className="grid grid-cols-4 gap-4 text-sm">
                <div><span className="text-gray-500">Min words:</span> {stats.min_word_count}</div>
                <div><span className="text-gray-500">Max words:</span> {stats.max_word_count}</div>
                <div><span className="text-gray-500">Avg word len:</span> {stats.avg_word_length?.toFixed(1)}</div>
                {stats.exact_duplicates !== undefined && (
                  <div>
                    <span className="text-gray-500">Exact dupes:</span> {stats.exact_duplicates}
                    {stats.duplicate_rate !== undefined && <span className="text-gray-400 ml-1">({(stats.duplicate_rate * 100).toFixed(1)}%)</span>}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Bar chart */}
          {chartData.length > 0 && (
            <div className="bg-white rounded-lg shadow p-5">
              <h3 className="font-semibold mb-3">Filter Pass Rates</h3>
              <ResponsiveContainer width="100%" height={Math.max(200, chartData.length * 40)}>
                <BarChart data={chartData} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                  <XAxis type="number" domain={[0, 100]} tickFormatter={v => `${v}%`} />
                  <YAxis type="category" dataKey="name" tick={{ fontSize: 12 }} width={140} />
                  <Tooltip formatter={(v: number) => [`${v}%`, 'Pass Rate']} />
                  <Bar dataKey="pass_rate" radius={[0, 4, 4, 0]}>
                    {chartData.map((entry, i) => (
                      <Cell key={i} fill={passColor(entry.pass_rate / 100)} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Filter detail with rule breakdown */}
          <div className="bg-white rounded-lg shadow p-5">
            <h3 className="font-semibold mb-3">Filter Details</h3>
            <div className="space-y-1">
              {filterNames.map(fname => {
                const f = filters[fname]
                const rules = f.rules || {}
                const ruleNames = Object.keys(rules).sort((a, b) => (rules[b]?.failed || 0) - (rules[a]?.failed || 0))
                const isOpen = expandedFilter === fname

                return (
                  <div key={fname}>
                    <button onClick={() => setExpandedFilter(isOpen ? null : fname)}
                      className="flex items-center gap-3 w-full px-3 py-2 rounded hover:bg-gray-50 text-left">
                      <span className="text-xs text-gray-400 w-4">{isOpen ? '▼' : '▶'}</span>
                      <span className="font-medium text-sm flex-1">{fname}</span>
                      <span className="text-xs text-gray-500">{f.failed} failed / {f.total}</span>
                      <span className={`px-2 py-0.5 rounded text-xs font-medium ${f.pass_rate >= 0.95 ? 'bg-green-100 text-green-700' : f.pass_rate >= 0.8 ? 'bg-yellow-100 text-yellow-700' : 'bg-red-100 text-red-700'}`}>
                        {(f.pass_rate * 100).toFixed(1)}%
                      </span>
                    </button>

                    {isOpen && (
                      <div className="ml-8 mb-3">
                        {ruleNames.length > 0 && (
                          <table className="w-full text-xs mt-1">
                            <thead><tr className="text-gray-500 border-b"><th className="text-left py-1 pr-4">Rule</th><th className="text-right pr-4">Failed</th><th className="text-right">Pass Rate</th></tr></thead>
                            <tbody>
                              {ruleNames.map(rule => (
                                <tr key={rule} className="border-b border-gray-50">
                                  <td className="py-1 pr-4 font-mono">{rule}</td>
                                  <td className="text-right pr-4 text-red-600">{rules[rule].failed}</td>
                                  <td className="text-right">{(rules[rule].pass_rate * 100).toFixed(1)}%</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        )}

                        {f.sample_failed && f.sample_failed.length > 0 && (
                          <div className="mt-2 space-y-1">
                            <div className="text-xs font-medium text-gray-500">Sample failures:</div>
                            {f.sample_failed.map((s, i) => (
                              <div key={i} className="bg-red-50 rounded p-2 text-xs">
                                <span className="text-red-700 font-medium">{s.reason?.reason || 'unknown'}</span>
                                {s.reason?.value && <span className="text-gray-500 ml-2">({String(s.reason.value).slice(0, 50)})</span>}
                                <div className="text-gray-500 mt-0.5 font-mono text-[11px] truncate">{s.text_preview?.slice(0, 150)}</div>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          </div>
        </>
      )}
    </div>
  )
}
