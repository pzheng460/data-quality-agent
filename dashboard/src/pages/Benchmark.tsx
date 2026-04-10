import { useState, useEffect } from 'react'
import { useApp } from '@/context'
import { api } from '@/hooks/useApi'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Badge } from '@/components/ui/badge'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { ChartContainer, ChartTooltip, ChartTooltipContent, type ChartConfig } from '@/components/ui/chart'
import { Bar, BarChart, CartesianGrid, Cell, XAxis, YAxis } from 'recharts'

interface FilterData {
  total: number; passed: number; failed: number; pass_rate: number
  sample_failed?: { text_preview: string; reason: { filter: string; reason: string; value?: any } }[]
  rules?: Record<string, { total: number; passed: number; failed: number; pass_rate: number }>
}

interface DatasetResult {
  num_docs: number; overall_pass_rate: number; data_type: string
  per_filter: Record<string, FilterData>
  dataset_stats?: {
    avg_word_count: number; min_word_count: number; max_word_count: number
    avg_word_length: number; exact_duplicates?: number; duplicate_rate?: number
  }
}

interface BenchResult { datasets: Record<string, DatasetResult> }

const passRateConfig: ChartConfig = {
  pass_rate: { label: 'Pass Rate', color: '#3b82f6' },
}

function passColor(rate: number) {
  if (rate >= 0.95) return '#22c55e'
  if (rate >= 0.8) return '#eab308'
  return '#ef4444'
}

export default function Benchmark() {
  const { outputDir } = useApp()

  const [inputPath, setInputPath] = useState('/tmp/arxiv_pipeline/input.jsonl')
  const [configPath, setConfigPath] = useState('configs/arxiv.yaml')
  const [numSamples, setNumSamples] = useState(100)
  const [workers, setWorkers] = useState(4)
  const [dataType, setDataType] = useState('auto')
  const [status, setStatus] = useState<'idle' | 'running' | 'done' | 'error'>('idle')
  const [result, setResult] = useState<BenchResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [expandedFilter, setExpandedFilter] = useState<string | null>(null)

  const startBench = async () => {
    setStatus('running'); setError(null); setResult(null)
    try {
      await api('/api/bench', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ input_path: inputPath, config_path: configPath, num_samples: numSamples, data_type: dataType, workers })
      })
    } catch (e: any) { setStatus('error'); setError(e.message) }
  }

  useEffect(() => {
    if (status !== 'running') return
    const timer = setInterval(async () => {
      try {
        const data = await api<any>('/api/bench/status')
        if (data.status === 'done') { setStatus('done'); setResult(data.result); clearInterval(timer) }
        else if (data.status === 'error') { setStatus('error'); setError(data.error); clearInterval(timer) }
      } catch {}
    }, 1000)
    return () => clearInterval(timer)
  }, [status])

  const ds = result?.datasets ? Object.values(result.datasets)[0] : null
  const stats = ds?.dataset_stats
  const filters = ds?.per_filter || {}
  const filterNames = Object.keys(filters)

  const chartData = filterNames.map(name => ({
    name: name.length > 15 ? name.slice(0, 14) + '…' : name,
    pass_rate: Math.round((filters[name].pass_rate || 0) * 100),
  }))

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">Benchmark</h2>

      {/* Config */}
      <Card>
        <CardHeader>
          <CardTitle>Configuration</CardTitle>
          <CardDescription>Run quality analysis on any JSONL dataset</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label>Input path</Label>
              <Input value={inputPath} onChange={e => setInputPath(e.target.value)} className="font-mono text-sm" />
            </div>
            <div className="space-y-2">
              <Label>Config YAML</Label>
              <Input value={configPath} onChange={e => setConfigPath(e.target.value)} className="font-mono text-sm" />
            </div>
          </div>
          <div className="grid grid-cols-4 gap-4">
            <div className="space-y-2">
              <Label>Samples</Label>
              <Input type="number" value={numSamples} onChange={e => setNumSamples(Number(e.target.value))} />
            </div>
            <div className="space-y-2">
              <Label>Workers</Label>
              <Input type="number" value={workers} onChange={e => setWorkers(Number(e.target.value))} />
            </div>
            <div className="space-y-2">
              <Label>Data type</Label>
              <Select value={dataType} onValueChange={setDataType}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="auto">auto</SelectItem>
                  <SelectItem value="pretrain">pretrain</SelectItem>
                  <SelectItem value="sft">sft</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="flex items-end">
              <Button onClick={startBench} disabled={status === 'running'} className="w-full">
                {status === 'running' ? 'Running…' : 'Run Benchmark'}
              </Button>
            </div>
          </div>
          {error && <p className="text-sm text-destructive">{error}</p>}
        </CardContent>
      </Card>

      {ds && (
        <>
          {/* KPIs */}
          <div className="grid grid-cols-4 gap-4">
            {[
              { label: 'Documents', value: String(ds.num_docs) },
              { label: 'Overall Pass Rate', value: `${(ds.overall_pass_rate * 100).toFixed(1)}%`, color: ds.overall_pass_rate >= 0.9 ? 'text-green-600' : 'text-yellow-600' },
              { label: 'Avg Word Count', value: stats?.avg_word_count?.toFixed(0) ?? '—' },
              { label: 'Data Type', value: ds.data_type, className: 'capitalize' },
            ].map((kpi, i) => (
              <Card key={i}>
                <CardContent className="pt-4">
                  <p className="text-xs text-muted-foreground uppercase tracking-wide">{kpi.label}</p>
                  <p className={`text-2xl font-bold mt-1 ${kpi.color || ''} ${kpi.className || ''}`}>{kpi.value}</p>
                </CardContent>
              </Card>
            ))}
          </div>

          {/* Dataset Stats */}
          {stats && (
            <Card>
              <CardHeader><CardTitle className="text-base">Dataset Statistics</CardTitle></CardHeader>
              <CardContent>
                <div className="grid grid-cols-4 gap-3 text-sm">
                  <div><span className="text-muted-foreground">Min words:</span> {stats.min_word_count}</div>
                  <div><span className="text-muted-foreground">Max words:</span> {stats.max_word_count}</div>
                  <div><span className="text-muted-foreground">Avg word len:</span> {stats.avg_word_length?.toFixed(1)}</div>
                  {stats.exact_duplicates !== undefined && (
                    <div><span className="text-muted-foreground">Exact dupes:</span> {stats.exact_duplicates} ({((stats.duplicate_rate ?? 0) * 100).toFixed(1)}%)</div>
                  )}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Chart */}
          {chartData.length > 0 && (
            <Card>
              <CardHeader><CardTitle className="text-base">Filter Pass Rates</CardTitle></CardHeader>
              <CardContent>
                <ChartContainer config={passRateConfig} className="w-full" style={{ height: Math.max(200, chartData.length * 40) }}>
                  <BarChart data={chartData} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" domain={[0, 100]} tickFormatter={(v: number) => `${v}%`} />
                    <YAxis type="category" dataKey="name" tick={{ fontSize: 12 }} width={120} />
                    <ChartTooltip content={<ChartTooltipContent />} />
                    <Bar dataKey="pass_rate" radius={[0, 4, 4, 0]}>
                      {chartData.map((entry, i) => (
                        <Cell key={i} fill={passColor(entry.pass_rate / 100)} />
                      ))}
                    </Bar>
                  </BarChart>
                </ChartContainer>
              </CardContent>
            </Card>
          )}

          {/* Filter Details */}
          <Card>
            <CardHeader><CardTitle className="text-base">Filter Details</CardTitle></CardHeader>
            <CardContent className="space-y-1">
              {filterNames.map(fname => {
                const f = filters[fname]
                const rules = f.rules || {}
                const ruleNames = Object.keys(rules).sort((a, b) => (rules[b]?.failed || 0) - (rules[a]?.failed || 0))
                const isOpen = expandedFilter === fname

                return (
                  <Collapsible key={fname} open={isOpen} onOpenChange={(open) => setExpandedFilter(open ? fname : null)}>
                    <CollapsibleTrigger asChild>
                      <Button variant="ghost" className="w-full justify-start gap-3 px-3 py-2 h-auto">
                        <span className="text-xs text-muted-foreground w-4">{isOpen ? '▼' : '▶'}</span>
                        <span className="font-medium text-sm flex-1 text-left">{fname}</span>
                        <span className="text-xs text-muted-foreground">{f.failed} failed / {f.total}</span>
                        <Badge variant={f.pass_rate >= 0.95 ? 'default' : f.pass_rate >= 0.8 ? 'secondary' : 'destructive'}>
                          {(f.pass_rate * 100).toFixed(1)}%
                        </Badge>
                      </Button>
                    </CollapsibleTrigger>

                    <CollapsibleContent className="ml-8 mb-3 space-y-2">
                      {ruleNames.length > 0 && (
                        <Table>
                          <TableHeader>
                            <TableRow>
                              <TableHead>Rule</TableHead>
                              <TableHead className="text-right">Failed</TableHead>
                              <TableHead className="text-right">Pass Rate</TableHead>
                            </TableRow>
                          </TableHeader>
                          <TableBody>
                            {ruleNames.map(rule => (
                              <TableRow key={rule}>
                                <TableCell className="font-mono text-xs">{rule}</TableCell>
                                <TableCell className="text-right text-destructive text-xs">{rules[rule].failed}</TableCell>
                                <TableCell className="text-right text-xs">{(rules[rule].pass_rate * 100).toFixed(1)}%</TableCell>
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      )}
                      {f.sample_failed && f.sample_failed.length > 0 && (
                        <div>
                          <p className="text-xs font-medium text-muted-foreground mb-1">Sample failures:</p>
                          {f.sample_failed.map((s, i) => (
                            <Card key={i} className="bg-destructive/5 mb-1">
                              <CardContent className="py-2 px-3 text-xs">
                                <p className="text-destructive font-medium">{s.reason?.reason || 'unknown'}</p>
                                <p className="text-muted-foreground mt-0.5 font-mono text-[11px] truncate">{s.text_preview?.slice(0, 150)}</p>
                              </CardContent>
                            </Card>
                          ))}
                        </div>
                      )}
                    </CollapsibleContent>
                  </Collapsible>
                )
              })}
            </CardContent>
          </Card>
        </>
      )}
    </div>
  )
}
