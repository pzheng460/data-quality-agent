import { useData } from '../hooks/useData'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts'

interface SignalsData {
  signals: Array<{
    name: string
    bins: Array<{ edge: number; count: number }>
    threshold?: number
  }>
}

export default function QualitySignals() {
  const { data, loading } = useData<SignalsData>('signals_histograms.json')

  if (loading) return <p className="text-gray-500">Loading...</p>
  if (!data?.signals) return <p className="text-gray-400">No signal data available. Run pipeline to generate histograms.</p>

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-gray-900">Quality Signals</h2>
      <div className="grid grid-cols-2 gap-6">
        {data.signals.map((sig) => (
          <div key={sig.name} className="bg-white rounded-lg shadow p-4">
            <h3 className="text-sm font-semibold mb-2">{sig.name}</h3>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={sig.bins}>
                <XAxis dataKey="edge" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip />
                <Bar dataKey="count" fill="#3b82f6" />
                {sig.threshold && (
                  <ReferenceLine x={sig.threshold} stroke="red" strokeDasharray="3 3" />
                )}
              </BarChart>
            </ResponsiveContainer>
          </div>
        ))}
      </div>
    </div>
  )
}
