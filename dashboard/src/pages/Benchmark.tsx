import { useState, useEffect } from 'react'
import { useApp } from '@/context'
import { api } from '@/hooks/useApi'
import { usePersistedState } from '@/hooks/usePersistedState'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { Checkbox } from '@/components/ui/checkbox'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { ChartContainer, ChartTooltip, ChartTooltipContent, type ChartConfig } from '@/components/ui/chart'
import { Bar, BarChart, CartesianGrid, Cell, XAxis, YAxis } from 'recharts'
import { KpiCard, KpiGrid } from '@/components/kpi-card'
import { FormField } from '@/components/form-field'
import { StatusMessage } from '@/components/status-message'

interface FilterData {
  total: number; passed: number; failed: number; pass_rate: number
  sample_failed?: { text_preview: string; reason: { filter: string; reason: string; value?: any } }[]
  rules?: Record<string, { total: number; passed: number; failed: number; pass_rate: number }>
}

interface LLMScores {
  type: 'sft' | 'pretrain'
  high_count: number
  low_count: number
  high_rate: number
  num_scored: number
  scoring_errors: number
  rule_fail_counts: Record<string, number>
  rule_modes?: Record<string, 'binary' | 'score'>
  rule_score_avg?: Record<string, number>
  rule_thresholds?: Record<string, number>
  rule_max_scores?: Record<string, number>
}

interface DatasetResult {
  num_docs: number; overall_pass_rate: number; data_type: string
  per_filter: Record<string, FilterData>
  dataset_stats?: {
    avg_word_count: number; min_word_count: number; max_word_count: number
    avg_word_length: number; exact_duplicates?: number; duplicate_rate?: number
  }
  llm_scores?: LLMScores
}

interface BenchResult { datasets: Record<string, DatasetResult> }

const chartConfig: ChartConfig = { pass_rate: { label: 'Pass Rate', color: '#3b82f6' } }

function passColor(rate: number) {
  if (rate >= 0.95) return '#22c55e'
  if (rate >= 0.8) return '#eab308'
  return '#ef4444'
}

export default function Benchmark() {
  const { outputDir } = useApp()
  // Default to pipeline's final output directory so benchmark runs on cleaned data.
  const [inputPath, setInputPath] = usePersistedState('bench.inputPath', `${outputDir}/stage4_final`)
  const [inputPathDirty, setInputPathDirty] = usePersistedState('bench.inputPathDirty', false)
  // Follow outputDir changes unless the user has manually edited the field.
  useEffect(() => {
    if (!inputPathDirty) setInputPath(`${outputDir}/stage4_final`)
  }, [outputDir, inputPathDirty])
  const [configPath, setConfigPath] = usePersistedState('bench.configPath', 'configs/arxiv.yaml')
  const [numSamples, setNumSamples] = usePersistedState('bench.numSamples', 100)
  const [workers, setWorkers] = usePersistedState('bench.workers', 4)
  const [dataType, setDataType] = usePersistedState('bench.dataType', 'auto')
  const [withLLM, setWithLLM] = usePersistedState('bench.withLLM', false)
  const [llmSamples, setLlmSamples] = usePersistedState('bench.llmSamples', 50)
  // HuggingFace dataset (alternative to input_path)
  const [useHF, setUseHF] = usePersistedState<boolean>('bench.useHF', false)
  const [hfDataset, setHfDataset] = usePersistedState('bench.hfDataset', 'HuggingFaceFW/fineweb')
  const [hfSubset, setHfSubset] = usePersistedState('bench.hfSubset', 'sample-10BT')
  const [hfSplit, setHfSplit] = usePersistedState('bench.hfSplit', 'train')
  const [hfTextField, setHfTextField] = usePersistedState('bench.hfTextField', 'text')
  const [status, setStatus] = useState<'idle' | 'running' | 'done' | 'error'>('idle')
  const [result, setResult] = useState<BenchResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [expandedFilter, setExpandedFilter] = useState<string | null>(null)
  const [configFiles, setConfigFiles] = useState<{ name: string; path: string }[]>([])

  useEffect(() => {
    api<{ name: string; path: string }[]>('/api/configs/list')
      .then(setConfigFiles)
      .catch(() => {})
  }, [])

  // Check for existing benchmark result on mount (from auto-bench after pipeline)
  useEffect(() => {
    api<Record<string, unknown>>('/api/bench/status').then(data => {
      if (data.status === 'done' && data.result) {
        setStatus('done')
        setResult(data.result as BenchResult)
      }
    }).catch(() => {})
  }, [])

  const resetBench = async () => {
    try {
      await api('/api/bench/reset', { method: 'POST' })
      setStatus('idle'); setError(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  const startBench = async () => {
    setStatus('running'); setError(null); setResult(null)
    try {
      await api('/api/bench', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          input_path: useHF ? '' : inputPath,
          config_path: configPath,
          num_samples: numSamples,
          data_type: dataType,
          workers,
          with_llm_scoring: withLLM,
          llm_samples: llmSamples,
          hf_dataset: useHF ? hfDataset : '',
          hf_subset: useHF ? hfSubset : '',
          hf_split: useHF ? hfSplit : 'train',
          hf_text_field: useHF ? hfTextField : 'text',
        })
      })
    } catch (e: unknown) { setStatus('error'); setError((e instanceof Error ? e.message : String(e))) }
  }

  useEffect(() => {
    if (status !== 'running') return
    const timer = setInterval(async () => {
      try {
        const data = await api<Record<string, unknown>>('/api/bench/status')
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

      <Card>
        <CardHeader>
          <CardTitle>Configuration</CardTitle>
          <CardDescription>Run quality analysis on any JSONL dataset</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center gap-3">
            <Checkbox id="use-hf" checked={useHF} onCheckedChange={v => setUseHF(!!v)} />
            <Label htmlFor="use-hf" className="text-sm cursor-pointer">Load from HuggingFace dataset instead of local path</Label>
          </div>

          {useHF ? (
            <div className="grid grid-cols-4 gap-3">
              <FormField label="Dataset">
                <Input value={hfDataset} onChange={e => setHfDataset(e.target.value)} className="font-mono text-xs" placeholder="HuggingFaceFW/fineweb" />
              </FormField>
              <FormField label="Subset/config">
                <Input value={hfSubset} onChange={e => setHfSubset(e.target.value)} className="font-mono text-xs" placeholder="sample-10BT" />
              </FormField>
              <FormField label="Split">
                <Input value={hfSplit} onChange={e => setHfSplit(e.target.value)} className="font-mono text-xs" placeholder="train" />
              </FormField>
              <FormField label="Text column">
                <Input value={hfTextField} onChange={e => setHfTextField(e.target.value)} className="font-mono text-xs" placeholder="text" />
              </FormField>
            </div>
          ) : (
            <div className="grid grid-cols-2 gap-4">
              <FormField label="Input path">
                <Input value={inputPath} onChange={e => { setInputPath(e.target.value); setInputPathDirty(true) }} className="font-mono text-sm" />
              </FormField>
              <FormField label="Config YAML">
                <Select value={configPath} onValueChange={setConfigPath}>
                  <SelectTrigger><SelectValue placeholder="Pick a config..." /></SelectTrigger>
                  <SelectContent>
                    {configFiles.length === 0 && <SelectItem value={configPath}>{configPath}</SelectItem>}
                    {configFiles.map(f => <SelectItem key={f.path} value={f.path}>{f.name}</SelectItem>)}
                  </SelectContent>
                </Select>
              </FormField>
            </div>
          )}

          {useHF && (
            <FormField label="Config YAML">
              <Select value={configPath} onValueChange={setConfigPath}>
                <SelectTrigger><SelectValue placeholder="Pick a config..." /></SelectTrigger>
                <SelectContent>
                  {configFiles.length === 0 && <SelectItem value={configPath}>{configPath}</SelectItem>}
                  {configFiles.map(f => <SelectItem key={f.path} value={f.path}>{f.name}</SelectItem>)}
                </SelectContent>
              </Select>
            </FormField>
          )}
          <div className="grid grid-cols-3 gap-4">
            <FormField label="Samples">
              <Input type="number" value={numSamples} onChange={e => setNumSamples(Number(e.target.value))} />
            </FormField>
            <FormField label="Workers">
              <Input type="number" value={workers} onChange={e => setWorkers(Number(e.target.value))} />
            </FormField>
            <FormField label="Data type">
              <Select value={dataType} onValueChange={setDataType}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="auto">auto</SelectItem>
                  <SelectItem value="pretrain">pretrain</SelectItem>
                  <SelectItem value="sft">sft</SelectItem>
                </SelectContent>
              </Select>
            </FormField>
          </div>
          <div className="rounded border p-3 bg-muted/30 space-y-2">
            <div className="flex items-center gap-3">
              <Checkbox id="bench_llm" checked={withLLM} onCheckedChange={(v) => setWithLLM(!!v)} />
              <Label htmlFor="bench_llm" className="text-sm font-medium cursor-pointer">
                Enable LLM quality judge
              </Label>
              {withLLM && (
                <div className="flex items-center gap-2 ml-2">
                  <Label htmlFor="bench_llm_samples" className="text-xs text-muted-foreground">samples</Label>
                  <Input id="bench_llm_samples" type="number" value={llmSamples}
                    onChange={e => setLlmSamples(Number(e.target.value))} className="w-20 h-7" />
                </div>
              )}
            </div>
            <p className="text-xs text-muted-foreground">
              Configure API at <a href="#/config" className="underline">Config → LLM API</a>.
              Falls back to <code>DQ_API_KEY</code> / <code>DQ_API_BASE_URL</code> env vars.
            </p>
          </div>

          <div className="flex items-center gap-4">
            <Button onClick={startBench} disabled={status === 'running'} size="lg">
              {status === 'running' ? 'Running…' : 'Run Benchmark'}
            </Button>
            {(status === 'running' || (error && /already running/i.test(error))) && (
              <Button onClick={resetBench} variant="outline" size="sm">
                Reset / cancel
              </Button>
            )}
            <StatusMessage status={status === 'error' ? 'error' : 'idle'} message={error} />
          </div>
        </CardContent>
      </Card>

      {ds && (
        <>
          <KpiGrid cols={4}>
            <KpiCard label="Documents" value={String(ds.num_docs)} />
            <KpiCard label="Overall Pass Rate" value={`${(ds.overall_pass_rate * 100).toFixed(1)}%`}
              color={ds.overall_pass_rate >= 0.9 ? 'text-green-600' : 'text-yellow-600'} />
            <KpiCard label="Avg Word Count" value={stats?.avg_word_count?.toFixed(0) ?? '—'} />
            <KpiCard label="Data Type" value={ds.data_type} className="capitalize" />
          </KpiGrid>

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

          {ds.llm_scores && (
            <Card>
              <CardHeader>
                <CardTitle className="text-base">LLM Judge Results</CardTitle>
                <CardDescription>
                  Binary classification by the LLM judge ({ds.llm_scores.type}). {ds.llm_scores.num_scored ?? (ds.llm_scores.high_count + ds.llm_scores.low_count + ds.llm_scores.scoring_errors)} docs scored.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <KpiGrid cols={4}>
                  <KpiCard label="High quality" value={String(ds.llm_scores.high_count)} color="text-green-600" />
                  <KpiCard label="Low quality" value={String(ds.llm_scores.low_count)} color="text-amber-600" />
                  <KpiCard label="Errors" value={String(ds.llm_scores.scoring_errors)}
                    color={ds.llm_scores.scoring_errors > 0 ? 'text-destructive' : undefined} />
                  <KpiCard label="High rate" value={`${(ds.llm_scores.high_rate * 100).toFixed(1)}%`}
                    color={ds.llm_scores.high_rate >= 0.8 ? 'text-green-600' : 'text-amber-600'} />
                </KpiGrid>

                {(() => {
                  const s = ds.llm_scores!
                  const ruleNames = Array.from(new Set([
                    ...Object.keys(s.rule_fail_counts || {}),
                    ...Object.keys(s.rule_score_avg || {}),
                    ...Object.keys(s.rule_modes || {}),
                  ]))
                  if (ruleNames.length === 0) return null
                  const denom = Math.max(1, s.num_scored - (s.scoring_errors || 0))
                  return (
                    <div>
                      <div className="text-sm font-medium mb-2">Per-rule breakdown</div>
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>Rule</TableHead>
                            <TableHead>Mode</TableHead>
                            <TableHead className="text-right">Avg score</TableHead>
                            <TableHead className="text-right">Threshold</TableHead>
                            <TableHead className="text-right">Failed</TableHead>
                            <TableHead className="text-right">Fail rate</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {ruleNames.map(name => {
                            const mode = s.rule_modes?.[name] ?? 'binary'
                            const avg = s.rule_score_avg?.[name]
                            const thr = s.rule_thresholds?.[name]
                            const maxS = s.rule_max_scores?.[name]
                            const fails = s.rule_fail_counts?.[name] ?? 0
                            return (
                              <TableRow key={name}>
                                <TableCell className="font-mono text-xs">{name}</TableCell>
                                <TableCell className="text-xs"><Badge variant={mode === 'score' ? 'secondary' : 'outline'}>{mode}</Badge></TableCell>
                                <TableCell className="text-right text-xs">
                                  {avg === undefined ? '—'
                                    : mode === 'score' ? `${avg.toFixed(2)}${maxS ? ` / ${maxS}` : ''}`
                                    : `${(avg * 100).toFixed(0)}%`}
                                </TableCell>
                                <TableCell className="text-right text-xs text-muted-foreground">
                                  {mode === 'score' && thr !== undefined ? thr : '—'}
                                </TableCell>
                                <TableCell className="text-right text-xs text-destructive">{fails}</TableCell>
                                <TableCell className="text-right text-xs text-muted-foreground">
                                  {`${((fails / Math.max(1, ds.llm_scores!.num_scored - (ds.llm_scores!.scoring_errors || 0))) * 100).toFixed(0)}%`}
                                </TableCell>
                              </TableRow>
                            )
                          })}
                        </TableBody>
                      </Table>
                    </div>
                  )
                })()}
              </CardContent>
            </Card>
          )}

          {chartData.length > 0 && (
            <Card>
              <CardHeader><CardTitle className="text-base">Filter Pass Rates</CardTitle></CardHeader>
              <CardContent>
                <ChartContainer config={chartConfig} className="w-full" style={{ height: Math.max(200, chartData.length * 40) }}>
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

          <Card>
            <CardHeader><CardTitle className="text-base">Filter Details</CardTitle></CardHeader>
            <CardContent className="space-y-1">
              {filterNames.map(fname => {
                const f = filters[fname]
                const rules = f.rules || {}
                const ruleNames = Object.keys(rules).sort((a, b) => (rules[b]?.failed || 0) - (rules[a]?.failed || 0))
                const isOpen = expandedFilter === fname

                return (
                  <Collapsible key={fname} open={isOpen} onOpenChange={open => setExpandedFilter(open ? fname : null)}>
                    <CollapsibleTrigger asChild>
                      <Button variant="ghost" className="w-full justify-start gap-3 px-3 py-2">
                        <span className="text-xs text-muted-foreground w-4">{isOpen ? '▼' : '▶'}</span>
                        <span className="font-medium text-sm flex-1 text-left">{fname}</span>
                        <span className="text-xs text-muted-foreground">{f.failed} / {f.total}</span>
                        <Badge variant={f.pass_rate >= 0.95 ? 'default' : f.pass_rate >= 0.8 ? 'secondary' : 'destructive'}>
                          {(f.pass_rate * 100).toFixed(1)}%
                        </Badge>
                      </Button>
                    </CollapsibleTrigger>
                    <CollapsibleContent className="ml-8 mb-3 space-y-3">
                      {fname === 'pii' && (
                        <div className="text-xs text-muted-foreground italic">
                          Note: the PII filter never rejects docs — it only detects/redacts.
                          Counts below are pattern hits, not rejections.
                        </div>
                      )}
                      {ruleNames.length > 0 && (
                        <Table>
                          <TableHeader>
                            <TableRow>
                              <TableHead>Rule</TableHead>
                              <TableHead className="text-right">{fname === 'pii' ? 'Detected' : 'Failed'}</TableHead>
                              <TableHead className="text-right">{fname === 'pii' ? 'Hit rate' : 'Pass rate'}</TableHead>
                            </TableRow>
                          </TableHeader>
                          <TableBody>
                            {ruleNames.map(rule => (
                              <TableRow key={rule}>
                                <TableCell className="font-mono text-xs">{rule}</TableCell>
                                <TableCell className={`text-right text-xs ${fname === 'pii' ? '' : 'text-destructive'}`}>{rules[rule].failed}</TableCell>
                                <TableCell className="text-right text-xs">
                                  {fname === 'pii'
                                    ? `${((1 - rules[rule].pass_rate) * 100).toFixed(1)}%`
                                    : `${(rules[rule].pass_rate * 100).toFixed(1)}%`}
                                </TableCell>
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      )}
                      {f.sample_failed && f.sample_failed.length > 0 && (
                        <div className="space-y-2">
                          <div className="text-xs font-medium text-muted-foreground">
                            Rejected examples ({f.sample_failed.length})
                          </div>
                          {f.sample_failed.map((s, i) => (
                            <div key={i} className="rounded border bg-muted/40 p-2 space-y-1">
                              <div className="flex items-center gap-2 text-xs">
                                <Badge variant="destructive" className="font-mono">{s.reason?.reason || s.reason?.filter || 'unknown'}</Badge>
                                {s.reason?.value !== undefined && s.reason?.value !== '' && (
                                  <span className="text-muted-foreground">
                                    value: <code className="font-mono">{typeof s.reason.value === 'object' ? JSON.stringify(s.reason.value) : String(s.reason.value)}</code>
                                  </span>
                                )}
                              </div>
                              <pre className="text-xs font-mono whitespace-pre-wrap break-words text-muted-foreground max-h-32 overflow-auto">
                                {s.text_preview || '(empty)'}
                              </pre>
                            </div>
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
