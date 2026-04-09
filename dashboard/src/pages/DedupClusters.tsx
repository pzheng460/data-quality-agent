import { useData } from '../hooks/useData'

interface ClustersData {
  clusters: Array<{
    cluster_id: number
    size: number
    members: Array<{ id: string; text: string }>
  }>
}

export default function DedupClusters() {
  const { data, loading } = useData<ClustersData>('clusters.json')

  if (loading) return <p className="text-gray-500">Loading...</p>
  if (!data?.clusters) return <p className="text-gray-400">No dedup cluster data available.</p>

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-gray-900">Dedup Clusters</h2>
      <div className="space-y-4">
        {data.clusters.map((cluster) => (
          <div key={cluster.cluster_id} className="bg-white rounded-lg shadow p-4">
            <h3 className="font-semibold">
              Cluster #{cluster.cluster_id}{' '}
              <span className="text-gray-500 font-normal text-sm">({cluster.size} docs)</span>
            </h3>
            <div className="grid grid-cols-2 gap-4 mt-3">
              {cluster.members.slice(0, 2).map((member, i) => (
                <pre key={i} className="text-xs bg-gray-50 p-3 rounded overflow-x-auto max-h-48 overflow-y-auto">
                  {member.text?.slice(0, 500) || member.id}
                </pre>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
