import { useEffect, useState } from 'react'
import { useApp } from '../context'
import { api } from '../hooks/useApi'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'

interface PhaseInfo { input: number; output: number; keep_rate: number; reject_reasons?: Record<string, number> }
interface OverviewData { version: string; phases: Record<string, PhaseInfo> }

function KPI({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="bg-white rounded-lg shadow p-5">
      <div className="text-sm text-gray-500">{label}</div>
      <div className="text-3xl font-bold text-gray-900 mt-1">{value}</div>
      {sub && <div className="text-xs text-gray-400 mt-1">{sub}</div>}
    </div>
  )
}

export default function Overview() {
  const { outputDir, refreshKey } = useApp()
  const [data, setData] = useState<OverviewData | null>(null)
  const [error, setError] = useState(false)

  useEffect(() => {
    api<OverviewData>(`/api/overview?output_dir=${encodeURIComponent(outputDir)}`)
      .then(setData)
      .catch(() => setError(true))
  }, [outputDir, refreshKey])

  if (error || !data) return <p className="text-gray-400 p-8">No overview data. Run the pipeline first.</p>

  const phases = Object.entries(data.phases)
  const raw = phases[0]?.[1]?.input ?? 0
  const final_ = phases[phases.length - 1]?.[1]?.output ?? 0
  const rate = raw > 0 ? ((final_ / raw) * 100).toFixed(1) : '0'

  const funnelData = phases.map(([name, p]) => ({
    name: name.replace('phase', 'P').replace(/_/g, ' '),
    Input: p.input,
    Output: p.output,
  }))

  return (
    <div className="space-y-8">
      <h2 className="text-2xl font-bold">Pipeline Overview</h2>
      <div className="grid grid-cols-4 gap-4">
        <KPI label="Raw Documents" value={raw.toLocaleString()} />
        <KPI label="Final Output" value={final_.toLocaleString()} />
        <KPI label="Retention" value={`${rate}%`} sub={`${raw - final_} rejected`} />
        <KPI label="Version" value={data.version.replace('arxiv-', '')} />
      </div>
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4">Pipeline Funnel</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={funnelData} barGap={-20}>
            <XAxis dataKey="name" tick={{ fontSize: 12 }} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="Input" fill="#bfdbfe" radius={[4, 4, 0, 0]} />
            <Bar dataKey="Output" fill="#3b82f6" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="bg-white rounded-lg shadow overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-4 py-3 text-left font-medium text-gray-500">Phase</th>
              <th className="px-4 py-3 text-right font-medium text-gray-500">Input</th>
              <th className="px-4 py-3 text-right font-medium text-gray-500">Output</th>
              <th className="px-4 py-3 text-right font-medium text-gray-500">Rejected</th>
              <th className="px-4 py-3 text-right font-medium text-gray-500">Keep Rate</th>
            </tr>
          </thead>
          <tbody className="divide-y">
            {phases.map(([name, p]) => (
              <tr key={name} className="hover:bg-gray-50">
                <td className="px-4 py-3 font-medium">{name}</td>
                <td className="px-4 py-3 text-right">{p.input.toLocaleString()}</td>
                <td className="px-4 py-3 text-right">{p.output.toLocaleString()}</td>
                <td className="px-4 py-3 text-right text-red-600">{(p.input - p.output).toLocaleString()}</td>
                <td className="px-4 py-3 text-right">
                  <span className={`px-2 py-0.5 rounded text-xs font-medium ${p.keep_rate >= 0.9 ? 'bg-green-100 text-green-700' : p.keep_rate >= 0.7 ? 'bg-yellow-100 text-yellow-700' : 'bg-red-100 text-red-700'}`}>
                    {(p.keep_rate * 100).toFixed(1)}%
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
