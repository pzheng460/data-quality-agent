import { useEffect, useState } from 'react'
import { useApp } from '@/context'
import { api } from '@/hooks/useApi'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { KpiCard, KpiGrid } from '@/components/kpi-card'
import { StatusMessage } from '@/components/status-message'
import { ChartContainer, ChartTooltip, ChartTooltipContent, type ChartConfig } from '@/components/ui/chart'
import { Bar, BarChart, CartesianGrid, XAxis, YAxis } from 'recharts'

interface StageInfo {
  phase: string; input_count: number; output_count: number
  rejected_count: number; keep_rate: number
  reject_reasons: Record<string, number>; duration_seconds: number
}

const STAGE_LABELS: Record<string, string> = {
  ingestion: 'Ingest', extraction: 'Extract', curation: 'Curate', packaging: 'Package',
}

const funnelConfig: ChartConfig = {
  Input: { label: 'Input', color: '#93c5fd' },
  Output: { label: 'Output', color: '#3b82f6' },
}

const rejectConfig: ChartConfig = {
  count: { label: 'Count', color: '#ef4444' },
}

export default function Stats() {
  const { outputDir, refreshKey } = useApp()
  const [overview, setOverview] = useState<any>(null)
  const [stages, setStages] = useState<StageInfo[]>([])
  const [error, setError] = useState(false)

  useEffect(() => {
    api<any[]>(`/api/stages/all?output_dir=${encodeURIComponent(outputDir)}`).then(data => {
      setStages(data.filter((s: any) => s.stats).map((s: any) => s.stats))
    }).catch(() => setError(true))

    api<any>(`/api/overview?output_dir=${encodeURIComponent(outputDir)}`)
      .then(setOverview).catch(() => {})
  }, [outputDir, refreshKey])

  if (error && !stages.length) return <StatusMessage status="info" message="No pipeline data yet. Run the pipeline first." />

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

      <KpiGrid cols={5}>
        <KpiCard label="Input Docs" value={raw.toLocaleString()} />
        <KpiCard label="Final Output" value={(stages[stages.length - 1]?.output_count ?? 0).toLocaleString()} />
        <KpiCard label="Retention" value={`${retention}%`} color={Number(retention) >= 80 ? 'text-green-600' : 'text-yellow-600'} />
        <KpiCard label="Total Time" value={`${totalTime.toFixed(1)}s`} />
        <KpiCard label="Version" value={overview?.version || '—'} />
      </KpiGrid>

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
        return (
          <Card key={s.phase}>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <div>
                <CardTitle>{STAGE_LABELS[s.phase] || s.phase}</CardTitle>
                <p className="text-sm text-muted-foreground">
                  {s.input_count.toLocaleString()} in → {s.output_count.toLocaleString()} kept, {s.rejected_count.toLocaleString()} rejected ({s.duration_seconds?.toFixed(1)}s)
                </p>
              </div>
              <Badge variant={s.keep_rate >= 0.9 ? 'default' : s.keep_rate >= 0.7 ? 'secondary' : 'destructive'}>
                {(s.keep_rate * 100).toFixed(1)}% kept
              </Badge>
            </CardHeader>
            {reasons.length > 0 ? (
              <CardContent>
                <p className="text-sm font-medium text-muted-foreground mb-2">Rejection Reasons</p>
                <ChartContainer config={rejectConfig} className="w-full" style={{ height: Math.max(80, reasons.length * 32) }}>
                  <BarChart data={reasons.map(([rule, count]) => ({ rule, count }))} layout="vertical" margin={{ left: 120 }}>
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
