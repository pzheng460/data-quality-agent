import { useState } from 'react'
import { useApp } from '../context'
import { api } from '../hooks/useApi'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'

interface BenchResult {
  total_docs: number
  sampled: number
  per_filter: Record<string, { total: number; passed: number; failed: number; pass_rate: number }>
  per_rule: Record<string, Record<string, number>>
  dataset_stats: { avg_word_count: number; avg_word_length: number }
}

export default function QualityCheck() {
  const { outputDir } = useApp()
  const [numSamples, _setNumSamples] = useState(50)
  const [configPath, _setConfigPath] = useState('/tmp/arxiv_test/arxiv_test.yaml')
  const [status, setStatus] = useState<'idle' | 'running' | 'done' | 'error'>('idle')
  const [result, setResult] = useState<BenchResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  const runBench = async () => {
    setStatus('running'); setError(null); setResult(null)
    try {
      const data = await api<BenchResult>('/api/quality-check', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ output_dir: outputDir, num_samples: numSamples, config_path: configPath }),
      })
      setResult(data)
      setStatus('done')
    } catch (e: any) {
      setError(e.message)
      setStatus('error')
    }
  }

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">Quality Check</h2>
      <p className="text-sm text-gray-500">
        Run <code className="bg-gray-100 px-1 rounded">dq bench</code> on the cleaned output to verify data quality.
        This randomly samples documents from the final stage and checks them against all filters.
      </p>

      {/* Config */}
      <div className="bg-white rounded-lg shadow p-5">
        <div className="flex items-end gap-4">
          <label className="block text-sm text-gray-600 flex-1">
            Output directory (stage5_final)
            <input value={outputDir} disabled className="mt-1 block w-full rounded border-gray-300 border px-3 py-2 text-sm bg-gray-50 font-mono" />
          </label>
          <label className="block text-sm text-gray-600 w-32">
            Samples
            <input type="number" value={100} className="mt-1 block w-full rounded border-gray-300 border px-3 py-2 text-sm" disabled />
          </label>
          <button onClick={runBench} disabled={status === 'running'}
            className={`px-4 py-2 rounded text-white text-sm font-medium ${status === 'running' ? 'bg-gray-400' : 'bg-green-600 hover:bg-green-700'}`}>
            {status === 'running' ? 'Running...' : 'Run Quality Check'}
          </button>
        </div>
        <div className="mt-2 text-xs text-gray-400">
          Runs dq bench on a random sample from the final cleaned output to check pass rates against all quality filters.
        </div>
      </div>

      {error && <div className="bg-red-50 border border-red-200 rounded-lg p-3 text-sm text-red-700">{error}</div>}

      {/* Placeholder — will show results when backend endpoint is ready */}
      {status === 'idle' && (
        <div className="bg-gray-50 rounded-lg p-8 text-center text-gray-400">
          Click "Run Quality Check" to sample and analyze the cleaned output.
        </div>
      )}

      {status === 'done' && result && (
        <div className="space-y-4">
          <div className="grid grid-cols-3 gap-4">
            <div className="bg-white rounded-lg shadow p-4">
              <div className="text-sm text-gray-500">Docs Sampled</div>
              <div className="text-2xl font-bold">{result.sampled}</div>
            </div>
            <div className="bg-white rounded-lg shadow p-4">
              <div className="text-sm text-gray-500">Avg Word Count</div>
              <div className="text-2xl font-bold">{result.dataset_stats?.avg_word_count?.toFixed(0) || '—'}</div>
            </div>
            <div className="bg-white rounded-lg shadow p-4">
              <div className="text-sm text-gray-500">Overall Pass Rate</div>
              <div className="text-2xl font-bold">{((result as any).overall_pass_rate * 100)?.toFixed(1) || '—'}%</div>
            </div>
          </div>

          {/* Per-filter pass rates */}
          <div className="bg-white rounded-lg shadow p-5">
            <h3 className="font-semibold mb-3">Filter Pass Rates</h3>
            {result.per_filter && (
              <ResponsiveContainer width="100%" height={Math.max(150, Object.keys(result.per_filter).length * 35)}>
                <BarChart data={Object.entries(result.per_filter).map(([name, f]) => ({ name, rate: (f as any).pass_rate * 100 }))} layout="vertical" margin={{ left: 150 }}>
                  <XAxis type="number" domain={[0, 100]} unit="%" />
                  <YAxis type="category" dataKey="name" tick={{ fontSize: 12 }} width={140} />
                  <Tooltip formatter={(v: any) => `${Number(v).toFixed(1)}%`} />
                  <Bar dataKey="rate" fill="#10b981" />
                </BarChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

