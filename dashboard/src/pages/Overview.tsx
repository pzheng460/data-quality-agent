import { useEffect, useState } from 'react'
import { useApp } from '@/context'
import { api } from '@/hooks/useApi'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts'

interface StageInfo {
  phase: string; input_count: number; output_count: number
  rejected_count: number; keep_rate: number
  reject_reasons: Record<string, number>; duration_seconds: number
}

interface OverviewData { version: string; phases: Record<string, { input: number; output: number; keep_rate: number }> }

const STAGE_LABELS: Record<string, string> = {
  ingestion: 'Ingestion', extraction: 'Extraction', curation: 'Curation', packaging: 'Packaging',
}

function KPI({ label, value, color, sub }: { label: string; value: string; color?: string; sub?: string }) {
  return (
    <div className="bg-white rounded-lg shadow p-4">
      <div className="text-xs text-gray-500 uppercase tracking-wide">{label}</div>
      <div className="text-3xl font-bold mt-1" style={color ? { color } : undefined}>{value}</div>
      {sub && <div className="text-xs text-gray-400 mt-0.5">{sub}</div>}
    </div>
  )
}

export default function Overview() {
  const { outputDir, refreshKey } = useApp()
  const [overview, setOverview] = useState<OverviewData | null>(null)
  const [stages, setStages] = useState<StageInfo[]>([])
  const [error, setError] = useState(false)

  useEffect(() => {
    // Single request for all stages + stats
    api<any[]>(`/api/stages/all?output_dir=${encodeURIComponent(outputDir)}`).then(data => {
      setStages(data.filter((s: any) => s.stats).map((s: any) => s.stats))
    }).catch(() => setError(true))

    api<OverviewData>(`/api/overview?output_dir=${encodeURIComponent(outputDir)}`)
      .then(setOverview).catch(() => {})
  }, [outputDir, refreshKey])

  if (error && !overview && !stages.length) return <p className="text-gray-400 p-8">No pipeline data yet. Run the pipeline first.</p>

  const raw = stages[0]?.input_count ?? 0
  const final_ = stages.length ? stages[stages.length - 1]?.output_count ?? 0 : 0
  const retention = raw > 0 ? ((final_ / raw) * 100).toFixed(1) : '0'
  const totalTime = stages.reduce((sum, s) => sum + (s.duration_seconds || 0), 0)

  const funnelData = stages.map(s => ({
    name: STAGE_LABELS[s.phase] || s.phase,
    Input: s.input_count,
    Output: s.output_count,
  }))

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">Overview</h2>

      {/* KPI cards */}
      <div className="grid grid-cols-5 gap-4">
        <KPI label="Input Docs" value={raw.toLocaleString()} />
        <KPI label="Final Output" value={final_.toLocaleString()} />
        <KPI label="Retention" value={`${retention}%`} sub={`${(raw - final_).toLocaleString()} rejected`}
          color={Number(retention) >= 80 ? '#22c55e' : Number(retention) >= 50 ? '#eab308' : '#ef4444'} />
        <KPI label="Total Time" value={`${totalTime.toFixed(1)}s`} />
        <KPI label="Version" value={overview?.version || '—'} />
      </div>

      {/* Funnel chart */}
      {funnelData.length > 0 && (
        <div className="bg-white rounded-lg shadow p-5">
          <h3 className="font-semibold mb-3">Pipeline Funnel</h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={funnelData} barGap={-20}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="name" tick={{ fontSize: 12 }} />
              <YAxis />
              <Tooltip />
              <Bar dataKey="Input" fill="#bfdbfe" radius={[4, 4, 0, 0]} />
              <Bar dataKey="Output" fill="#3b82f6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Per-stage details with reject reasons */}
      {stages.map(s => {
        const reasons = Object.entries(s.reject_reasons || {}).filter(([, v]) => v > 0).sort((a, b) => b[1] - a[1])
        const label = STAGE_LABELS[s.phase] || s.phase

        return (
          <div key={s.phase} className="bg-white rounded-lg shadow overflow-hidden">
            <div className="flex items-center gap-3 px-5 py-4 border-b border-gray-100">
              <div className="flex-1">
                <h3 className="font-bold text-lg">{label}</h3>
                <p className="text-sm text-gray-500 mt-0.5">
                  {s.input_count.toLocaleString()} in &rarr; {s.output_count.toLocaleString()} kept, {s.rejected_count.toLocaleString()} rejected
                  <span className="text-gray-400 ml-2">({s.duration_seconds?.toFixed(1)}s)</span>
                </p>
              </div>
              <span className={`px-3 py-1 rounded text-sm font-medium ${s.keep_rate >= 0.9 ? 'bg-green-100 text-green-700' : s.keep_rate >= 0.7 ? 'bg-yellow-100 text-yellow-700' : 'bg-red-100 text-red-700'}`}>
                {(s.keep_rate * 100).toFixed(1)}% kept
              </span>
            </div>
            {reasons.length > 0 && (
              <div className="p-5">
                <div className="text-sm font-medium text-gray-500 mb-2">Rejection Reasons</div>
                <ResponsiveContainer width="100%" height={Math.max(80, reasons.length * 32)}>
                  <BarChart data={reasons.map(([rule, count]) => ({ rule, count }))} layout="vertical" margin={{ left: 180 }}>
                    <XAxis type="number" />
                    <YAxis type="category" dataKey="rule" tick={{ fontSize: 11 }} width={170} />
                    <Tooltip />
                    <Bar dataKey="count" fill="#ef4444" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
            {reasons.length === 0 && (
              <div className="px-5 py-3 text-sm text-green-600">No rejections</div>
            )}
          </div>
        )
      })}
    </div>
  )
}
