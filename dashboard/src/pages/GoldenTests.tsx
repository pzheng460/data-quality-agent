import { useData } from '../hooks/useData'
import type { GoldenTestResult } from '../types'

interface GoldenData {
  tests: GoldenTestResult[]
  summary: { passed: number; failed: number; total: number }
}

export default function GoldenTests() {
  const { data, loading } = useData<GoldenData>('golden_results.json')

  if (loading) return <p className="text-gray-500">Loading...</p>
  if (!data) return <p className="text-gray-400">No golden test results. Run golden tests first.</p>

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-gray-900">Golden Tests</h2>
      <div className="flex gap-4">
        <div className="bg-green-50 text-green-700 rounded-lg px-4 py-2 font-semibold">
          Pass: {data.tests.filter((t) => t.status === 'pass').length}
        </div>
        <div className="bg-red-50 text-red-700 rounded-lg px-4 py-2 font-semibold">
          Fail: {data.tests.filter((t) => t.status === 'fail').length}
        </div>
      </div>

      <div className="space-y-3">
        {data.tests.map((t) => (
          <div
            key={t.id}
            className={`bg-white rounded-lg shadow p-4 border-l-4 ${
              t.status === 'pass' ? 'border-green-500' : 'border-red-500'
            }`}
          >
            <div className="flex justify-between">
              <span className="font-mono text-sm">{t.id}</span>
              <span className={`text-sm font-semibold ${
                t.status === 'pass' ? 'text-green-600' : 'text-red-600'
              }`}>
                {t.status.toUpperCase()}
              </span>
            </div>
            {t.diff && (
              <pre className="mt-2 text-xs bg-gray-50 p-2 rounded overflow-x-auto max-h-40 overflow-y-auto">
                {t.diff}
              </pre>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
