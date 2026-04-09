import { useData } from '../hooks/useData'

interface ContaminationData {
  total_docs: number
  contaminated_docs: number
  per_benchmark: Record<string, {
    contaminated_docs: number
    contamination_rate: number
    avg_overlap: number
    samples: Array<{ id: string; overlap: number; benchmark: string }>
  }>
}

export default function Contamination() {
  const { data, loading } = useData<ContaminationData>('contamination.json')

  if (loading) return <p className="text-gray-500">Loading...</p>
  if (!data) return <p className="text-gray-400">No contamination data available.</p>

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-gray-900">Contamination Detection</h2>
      <div className="grid grid-cols-3 gap-4">
        <div className="bg-white rounded-lg shadow p-4">
          <div className="text-sm text-gray-500">Total Docs Checked</div>
          <div className="text-xl font-bold">{data.total_docs.toLocaleString()}</div>
        </div>
        <div className="bg-white rounded-lg shadow p-4">
          <div className="text-sm text-gray-500">Contaminated</div>
          <div className="text-xl font-bold text-red-600">{data.contaminated_docs}</div>
        </div>
      </div>

      {data && Object.entries(data.per_benchmark ?? {}).map(([bm, info]) => (
        <div key={bm} className="bg-white rounded-lg shadow p-4">
          <h3 className="font-semibold">{bm}</h3>
          <div className="text-sm text-gray-600 mt-1">
            {info.contaminated_docs} contaminated ({(info.contamination_rate * 100).toFixed(2)}%)
          </div>
        </div>
      ))}
    </div>
  )
}
