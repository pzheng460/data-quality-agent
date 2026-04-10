import { useEffect, useState } from 'react'
import { useApp } from '@/context'
import { api } from '@/hooks/useApi'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
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

export default function Overview() {
  const { outputDir, refreshKey } = useApp()
  const [overview, setOverview] = useState<OverviewData | null>(null)
  const [stages, setStages] = useState<StageInfo[]>([])
  const [error, setError] = useState(false)

  useEffect(() => {
    api<any[]>(`/api/stages/all?output_dir=${encodeURIComponent(outputDir)}`).then(data => {
      setStages(data.filter((s: any) => s.stats).map((s: any) => s.stats))
    }).catch(() => setError(true))

    api<OverviewData>(`/api/overview?output_dir=${encodeURIComponent(outputDir)}`)
      .then(setOverview).catch(() => {})
  }, [outputDir])

  if (error && !stages.length) return <p className="text-muted-foreground p-8">No pipeline data yet. Run the pipeline first.</p>

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
        {[
          { label: 'Input Docs', value: raw.toLocaleString() },
          { label: 'Final Output', value: final_.toLocaleString() },
          { label: 'Retention', value: `${retention}%`, color: Number(retention) >= 80 ? 'text-green-600' : Number(retention) >= 50 ? 'text-yellow-600' : 'text-red-600' },
          { label: 'Total Time', value: `${totalTime.toFixed(1)}s` },
          { label: 'Version', value: overview?.version || '—' },
        ].map((kpi, i) => (
          <Card key={i}>
            <CardContent className="pt-4 pb-3">
              <p className="text-xs text-muted-foreground uppercase tracking-wide">{kpi.label}</p>
              <p className={`text-2xl font-bold mt-1 ${kpi.color || ''}`}>{kpi.value}</p>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Funnel chart */}
      {funnelData.length > 0 && (
        <Card>
          <CardHeader><CardTitle>Pipeline Funnel</CardTitle></CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={funnelData} barGap={-20}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="Input" fill="hsl(var(--muted))" radius={[4, 4, 0, 0]} />
                <Bar dataKey="Output" fill="hsl(var(--primary))" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}

      {/* Per-stage details */}
      {stages.map(s => {
        const reasons = Object.entries(s.reject_reasons || {}).filter(([, v]) => v > 0).sort((a, b) => b[1] - a[1])
        return (
          <Card key={s.phase}>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <div>
                <CardTitle>{STAGE_LABELS[s.phase] || s.phase}</CardTitle>
                <p className="text-sm text-muted-foreground">
                  {s.input_count.toLocaleString()} in &rarr; {s.output_count.toLocaleString()} kept, {s.rejected_count.toLocaleString()} rejected ({s.duration_seconds?.toFixed(1)}s)
                </p>
              </div>
              <Badge variant={s.keep_rate >= 0.9 ? 'default' : s.keep_rate >= 0.7 ? 'secondary' : 'destructive'}>
                {(s.keep_rate * 100).toFixed(1)}% kept
              </Badge>
            </CardHeader>
            {reasons.length > 0 && (
              <CardContent>
                <p className="text-sm font-medium text-muted-foreground mb-2">Rejection Reasons</p>
                <ResponsiveContainer width="100%" height={Math.max(80, reasons.length * 32)}>
                  <BarChart data={reasons.map(([rule, count]) => ({ rule, count }))} layout="vertical" margin={{ left: 180 }}>
                    <XAxis type="number" />
                    <YAxis type="category" dataKey="rule" tick={{ fontSize: 11 }} width={170} />
                    <Tooltip />
                    <Bar dataKey="count" fill="hsl(var(--destructive))" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            )}
            {reasons.length === 0 && (
              <CardContent><p className="text-sm text-green-600">No rejections</p></CardContent>
            )}
          </Card>
        )
      })}
    </div>
  )
}
