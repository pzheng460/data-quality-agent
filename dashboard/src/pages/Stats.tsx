import { useEffect, useState } from 'react'
import { useApp } from '@/context'
import { api } from '@/hooks/useApi'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { ChartContainer, ChartTooltip, ChartTooltipContent, type ChartConfig } from '@/components/ui/chart'
import { Bar, BarChart, CartesianGrid, XAxis, YAxis } from 'recharts'

interface StageInfo {
  phase: string; input_count: number; output_count: number
  rejected_count: number; keep_rate: number
  reject_reasons: Record<string, number>; duration_seconds: number
}

interface OverviewData { version: string; phases: Record<string, { input: number; output: number; keep_rate: number }> }

const STAGE_LABELS: Record<string, string> = {
  ingestion: 'Ingest', extraction: 'Extract', curation: 'Curate', packaging: 'Package',
}

const funnelConfig = {
  Input: { label: 'Input', color: '#93c5fd' },
  Output: { label: 'Output', color: '#3b82f6' },
} as ChartConfig

const rejectConfig = {
  count: { label: 'Count', color: '#ef4444' },
} as ChartConfig

export default function Stats() {
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
  }, [outputDir, refreshKey])

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
      <h2 className="text-2xl font-bold">Stats</h2>

      <div className="grid grid-cols-5 gap-4">
        {[
          { label: 'Input Docs', value: raw.toLocaleString() },
          { label: 'Final Output', value: final_.toLocaleString() },
          { label: 'Retention', value: `${retention}%`, color: Number(retention) >= 80 ? 'text-green-600' : 'text-yellow-600' },
          { label: 'Total Time', value: `${totalTime.toFixed(1)}s` },
          { label: 'Version', value: overview?.version || '—' },
        ].map((kpi, i) => (
          <Card key={i}>
            <CardContent className="pt-4">
              <p className="text-xs text-muted-foreground uppercase tracking-wide">{kpi.label}</p>
              <p className={`text-2xl font-bold mt-1 ${kpi.color || ''}`}>{kpi.value}</p>
            </CardContent>
          </Card>
        ))}
      </div>

      {funnelData.length > 0 && (
        <Card>
          <CardHeader><CardTitle>Pipeline Funnel</CardTitle></CardHeader>
          <CardContent>
            <ChartContainer config={funnelConfig} className="h-[250px] w-full">
              <BarChart data={funnelData} barGap={4} barCategoryGap="20%">
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="name" tickLine={false} axisLine={false} />
                <YAxis />
                <ChartTooltip content={<ChartTooltipContent />} />
                <Bar dataKey="Input" fill="var(--color-Input)" radius={[4, 4, 0, 0]} />
                <Bar dataKey="Output" fill="var(--color-Output)" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ChartContainer>
          </CardContent>
        </Card>
      )}

      {stages.map(s => {
        const reasons = Object.entries(s.reject_reasons || {}).filter(([, v]) => v > 0).sort((a, b) => b[1] - a[1])
        const rejectData = reasons.map(([rule, count]) => ({ rule, count }))

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
            {rejectData.length > 0 ? (
              <CardContent>
                <p className="text-sm font-medium text-muted-foreground mb-2">Rejection Reasons</p>
                <ChartContainer config={rejectConfig} className="w-full" style={{ height: Math.max(80, rejectData.length * 32) }}>
                  <BarChart data={rejectData} layout="vertical" margin={{ left: 120 }}>
                    <XAxis type="number" />
                    <YAxis type="category" dataKey="rule" tick={{ fontSize: 11 }} width={110} />
                    <ChartTooltip content={<ChartTooltipContent />} />
                    <Bar dataKey="count" fill="var(--color-count)" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ChartContainer>
              </CardContent>
            ) : (
              <CardContent><p className="text-sm text-green-600">No rejections.</p></CardContent>
            )}
          </Card>
        )
      })}
    </div>
  )
}
