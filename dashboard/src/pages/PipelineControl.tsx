import { useState, useEffect } from 'react'
import { api } from '@/hooks/useApi'
import { useApp } from '@/context'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Badge } from '@/components/ui/badge'
import { Checkbox } from '@/components/ui/checkbox'
import { Separator } from '@/components/ui/separator'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'

interface ParamDef { type: string; label: string; default?: any; required?: boolean }
interface SourceDef { name: string; domain: string; priority: number; params: Record<string, ParamDef> }

const STAGES = [
  { key: 'ingestion', num: 1, label: 'Ingestion', desc: 'Copy input into pipeline' },
  { key: 'extraction', num: 2, label: 'Extraction', desc: 'Convert raw format to text' },
  { key: 'curation', num: 3, label: 'Curation', desc: 'Filter + dedup + contamination' },
  { key: 'packaging', num: 4, label: 'Packaging', desc: 'Sort, shard, manifest' },
]

function getDefaults(params: Record<string, ParamDef>): Record<string, any> {
  const v: Record<string, any> = {}
  for (const [k, d] of Object.entries(params))
    v[k] = d.default ?? (d.type === 'list' ? '' : d.type === 'number' || d.type === 'float' ? 0 : '')
  return v
}

export default function PipelineControl() {
  const {
    outputDir, setOutputDir, refresh,
    ingestOutput, setIngestOutput, ingestStatus, setIngestStatus,
    papers, setPapers, activeDomain, setActiveDomain,
    activeSource, setActiveSource, paramValues, setParamValues,
    pipeInput, setPipeInput, configPath, setConfigPath,
    workers, setWorkers, resume, setResume,
    pipeStatus, setPipeStatus, stageResults, setStageResults, pipeError, setPipeError,
  } = useApp()

  const [sourcesByDomain, setSourcesByDomain] = useState<Record<string, SourceDef[]>>({})
  const [ingestLimit, setIngestLimit] = useState(0)
  const [ingestError, setIngestError] = useState<string | null>(null)

  const domains = Object.keys(sourcesByDomain)
  const sourcesInDomain = sourcesByDomain[activeDomain] || []
  const currentSource = sourcesInDomain.find(s => s.name === activeSource)

  const stageIO: Record<string, { input: string; output: string }> = {
    ingestion: { input: pipeInput, output: `${outputDir}/stage1_ingested/kept` },
    extraction: { input: `${outputDir}/stage1_ingested/kept`, output: `${outputDir}/stage2_extracted/kept` },
    curation: { input: `${outputDir}/stage2_extracted/kept`, output: `${outputDir}/stage3_curated/kept` },
    packaging: { input: `${outputDir}/stage3_curated/kept`, output: `${outputDir}/stage4_final` },
  }

  // ── Load sources ──
  useEffect(() => {
    api<Record<string, SourceDef[]>>('/api/sources').then(data => {
      setSourcesByDomain(data)
      const d = Object.keys(data)
      if (d.length && !activeDomain) { setActiveDomain(d[0]); const f = data[d[0]][0]; if (f) { setActiveSource(f.name); setParamValues(getDefaults(f.params)) } }
    }).catch(() => {})
  }, [])

  useEffect(() => { loadStageResults() }, [outputDir])

  function selectDomain(d: string) { setActiveDomain(d); const s = sourcesByDomain[d] || []; if (s.length) { setActiveSource(s[0].name); setParamValues(getDefaults(s[0].params)) } }
  function selectSource(n: string) { setActiveSource(n); const s = sourcesInDomain.find(x => x.name === n); if (s) setParamValues(getDefaults(s.params)) }

  // ── Ingest ──
  const startIngest = async () => {
    setPapers([]); setIngestError(null); setIngestStatus('downloading')
    try {
      const params: Record<string, any> = {}
      if (currentSource) {
        for (const [k, def] of Object.entries(currentSource.params)) {
          let val = paramValues[k]
          if (def.type === 'list' && typeof val === 'string') val = val.split(/[\n,\s]+/).map((s: string) => s.trim()).filter(Boolean)
          if ((def.type === 'number' || def.type === 'float') && typeof val === 'string') val = Number(val)
          if (val !== undefined && val !== '' && !(Array.isArray(val) && val.length === 0)) params[k] = val
        }
      }
      await api('/api/ingest', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source: activeSource, params, output_path: ingestOutput, limit: ingestLimit || 0 }) })
    } catch (e: any) { setIngestError(e.message); setIngestStatus('error') }
  }

  useEffect(() => {
    if (ingestStatus !== 'downloading') return
    let seen = false
    const timer = setInterval(async () => {
      try {
        const d = await api<any>('/api/ingest/status')
        if (d.status === 'downloading') seen = true
        setPapers(d.papers || [])
        if (seen && d.status === 'done') { setIngestStatus('done'); setPipeInput(ingestOutput); clearInterval(timer) }
        else if (seen && d.status === 'error') { setIngestStatus('error'); setIngestError(d.error); clearInterval(timer) }
      } catch {}
    }, 1000)
    return () => clearInterval(timer)
  }, [ingestStatus])

  // ── Pipeline ──
  useEffect(() => {
    if (pipeStatus !== 'running') return
    const timer = setInterval(async () => {
      try {
        const d = await api<any>('/api/status')
        if (d.progress) { const r: Record<string, any> = {}; for (const p of d.progress) r[p.phase] = p; setStageResults(r) }
        if (d.status === 'finished') { setPipeStatus('finished'); refresh(); clearInterval(timer) }
        else if (d.status === 'error') { setPipeStatus('error'); setPipeError(d.error); clearInterval(timer) }
      } catch {}
    }, 1000)
    return () => clearInterval(timer)
  }, [pipeStatus])

  const loadStageResults = async () => {
    try {
      const data = await api<any[]>(`/api/stages/all?output_dir=${encodeURIComponent(outputDir)}`)
      const r: Record<string, any> = {}
      for (const s of data) { if (s.stats) r[s.name] = s.stats }
      setStageResults(r)
    } catch {}
  }

  const runStage = async (n: number) => {
    setPipeError(null); setPipeStatus('running')
    try {
      await api('/api/run-phase', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ input_path: pipeInput, output_dir: outputDir, config_path: configPath, phase: n, workers, num_samples: 0 }) })
    } catch (e: any) { setPipeError(e.message); setPipeStatus('error') }
  }

  const startFullPipeline = async () => {
    setPipeError(null); setStageResults({}); setPipeStatus('running')
    try {
      await api('/api/run', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ input_path: pipeInput, output_dir: outputDir, config_path: configPath, workers, num_samples: 0, resume }) })
    } catch (e: any) { setPipeError(e.message); setPipeStatus('error') }
  }

  return (
    <div className="space-y-6">

      {/* ═══ Ingest ═══ */}
      <Card>
        <CardHeader>
          <CardTitle>Ingest Data</CardTitle>
          <CardDescription>Download or load raw data from a source</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Domain tabs */}
          <div className="flex gap-2 items-center">
            {domains.map(d => (
              <Button key={d} variant={activeDomain === d ? 'default' : 'outline'} size="sm"
                onClick={() => selectDomain(d)} className="capitalize">{d}</Button>
            ))}
          </div>

          {/* Arxiv unified inputs */}
          {activeDomain === 'arxiv' && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>Arxiv IDs <span className="text-muted-foreground">(or use date range)</span></Label>
                  <Textarea value={paramValues['ids'] ?? ''} onChange={e => setParamValues((v: any) => ({...v, ids: e.target.value}))}
                    rows={2} className="font-mono text-xs" placeholder="2310.06825, 1706.03762" />
                </div>
                <div className="grid grid-cols-3 gap-2">
                  <div className="space-y-2">
                    <Label htmlFor="from_date">From date</Label>
                    <Input id="from_date" type="date" value={paramValues['from_date'] ?? ''} onChange={e => setParamValues((v: any) => ({...v, from_date: e.target.value}))} />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="to_date">To date</Label>
                    <Input id="to_date" type="date" value={paramValues['to_date'] ?? ''} onChange={e => setParamValues((v: any) => ({...v, to_date: e.target.value}))} />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="categories">Categories</Label>
                    <Input id="categories" value={paramValues['categories'] ?? ''} onChange={e => setParamValues((v: any) => ({...v, categories: e.target.value}))} placeholder="cs.CL" className="font-mono" />
                  </div>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <span className="text-sm text-muted-foreground">Source:</span>
                {sourcesInDomain.map(s => (
                  <Button key={s.name} variant={activeSource === s.name ? 'default' : 'outline'} size="sm"
                    onClick={() => selectSource(s.name)}>{s.name.replace(/^arxiv_/, '')}</Button>
                ))}
                <Separator orientation="vertical" className="h-6" />
                <div className="flex items-center gap-1.5">
                  <Label>Limit</Label>
                  <Input type="number" value={ingestLimit} onChange={e => setIngestLimit(Number(e.target.value))}
                    className="w-20" placeholder="0" />
                </div>
              </div>
            </div>
          )}

          {/* Non-arxiv: dynamic params */}
          {activeDomain !== 'arxiv' && currentSource && (
            <div className="grid grid-cols-3 gap-4">
              {Object.entries(currentSource.params).map(([key, def]) => (
                <div key={key} className="space-y-2">
                  <Label>{def.label}{def.required && <span className="text-destructive">*</span>}</Label>
                  {def.type === 'list' ? (
                    <Textarea value={paramValues[key] ?? ''} onChange={e => setParamValues((v: any) => ({...v, [key]: e.target.value}))}
                      rows={2} className="font-mono text-xs" />
                  ) : (
                    <Input type={def.type === 'number' || def.type === 'float' ? 'number' : 'text'}
                      value={paramValues[key] ?? ''} onChange={e => setParamValues((v: any) => ({...v, [key]: e.target.value}))} className="font-mono" />
                  )}
                </div>
              ))}
            </div>
          )}

          {/* Output + action */}
          <div className="flex items-end gap-3">
            <div className="flex-1 space-y-2">
              <Label>Output path</Label>
              <Input value={ingestOutput} onChange={e => { setIngestOutput(e.target.value); setPipeInput(e.target.value) }} className="font-mono text-xs" />
            </div>
            <Button onClick={startIngest} disabled={ingestStatus === 'downloading' || !activeSource}>
              {ingestStatus === 'downloading' ? `Ingesting (${papers.length})...` : 'Ingest'}
            </Button>
          </div>

          {ingestError && <p className="text-sm text-destructive">{ingestError}</p>}
          {ingestStatus === 'done' && <p className="text-sm text-green-600">Done! {papers.length} docs</p>}

          {papers.length > 0 && (
            <div className="border rounded-md overflow-auto max-h-36">
              <Table>
                <thead><TableRow><TableHead>ID</TableHead><TableHead>Title</TableHead><TableHead className="text-right">Size</TableHead><TableHead>Source</TableHead></TableRow></thead>
                <TableBody>{papers.map((p, i) => (
                  <tr key={i} className="text-xs"><TableCell className="font-mono">{p.arxiv_id}</TableCell><TableCell className="truncate max-w-[200px]">{p.title}</TableCell><TableCell className="text-right">{(p.chars/1000).toFixed(1)}k</TableCell><TableCell><Badge variant="secondary">{p.source_method}</Badge></TableCell></tr>
                ))}</TableBody>
              </Table>
            </div>
          )}
        </CardContent>
      </Card>

      {/* ═══ Pipeline ═══ */}
      <Card>
        <CardHeader>
          <CardTitle>Pipeline</CardTitle>
          <CardDescription>4-stage processing: ingest → extract → curate → package</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Config */}
          <div className="grid grid-cols-4 gap-3">
            <div className="space-y-1.5">
              <Label className="text-xs">Input path</Label>
              <Input value={pipeInput} onChange={e => setPipeInput(e.target.value)} className="font-mono text-xs" />
            </div>
            <div className="space-y-1.5">
              <Label className="text-xs">Output directory</Label>
              <Input value={outputDir} onChange={e => setOutputDir(e.target.value)} className="font-mono text-xs" />
            </div>
            <div className="space-y-1.5">
              <Label className="text-xs">Config YAML</Label>
              <Input value={configPath} onChange={e => setConfigPath(e.target.value)} className="font-mono text-xs" />
            </div>
            <div className="flex gap-3 items-end">
              <div className="space-y-1.5 flex-1">
                <Label className="text-xs">Workers</Label>
                <Input type="number" value={workers} onChange={e => setWorkers(Number(e.target.value))} />
              </div>
              <div className="flex items-center gap-2 pb-2">
                <Checkbox id="resume" checked={resume} onCheckedChange={(v) => setResume(!!v)} />
                <Label htmlFor="resume" className="text-xs">Resume</Label>
              </div>
            </div>
          </div>

          <Separator />

          {/* Stages */}
          <div className="space-y-1">
            {STAGES.map(stage => {
              const result = stageResults[stage.key]
              const isSkipped = result && (result as any).skipped
              const isDone = !!result && !isSkipped
              const io = stageIO[stage.key]

              return (
                <div key={stage.key} className={`flex items-center gap-3 px-4 py-3 rounded-lg ${isDone ? 'bg-green-50' : isSkipped ? 'bg-yellow-50' : 'bg-muted/50'}`}>
                  <div className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold shrink-0 ${isDone ? 'bg-green-500 text-white' : isSkipped ? 'bg-yellow-400 text-white' : 'bg-muted-foreground/20 text-muted-foreground'}`}>
                    {stage.num}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="font-semibold text-sm">{stage.label}</span>
                      <span className="text-xs text-muted-foreground">{stage.desc}</span>
                    </div>
                    <div className="flex gap-4 mt-0.5 text-[11px] text-muted-foreground">
                      <span>in: <code className="bg-muted px-1 rounded">{stageIO[stage.key].input}</code></span>
                      <span>out: <code className="bg-muted px-1 rounded">{stageIO[stage.key].output}</code></span>
                    </div>
                  </div>
                  {result && !isSkipped && (
                    <span className="text-xs text-muted-foreground shrink-0">
                      {result.input_count}→{result.output_count} ({(result.keep_rate * 100).toFixed(1)}%, {result.duration_seconds?.toFixed(1)}s)
                    </span>
                  )}
                  {isSkipped && <Badge variant="outline" className="text-yellow-600 border-yellow-300">skipped</Badge>}
                  <Button variant="outline" size="sm" disabled={pipeStatus === 'running'} onClick={() => runStage(stage.num)}>Run</Button>
                </div>
              )
            })}
          </div>

          <Separator />

          <div className="flex items-center gap-4">
            <Button onClick={startFullPipeline} disabled={pipeStatus === 'running'}>
              {pipeStatus === 'running' ? 'Running...' : 'Run All Stages'}
            </Button>
            <div className="flex items-center gap-2">
              <Checkbox id="resume-bottom" checked={resume} onCheckedChange={(v) => setResume(!!v)} />
              <Label htmlFor="resume-bottom" className="text-sm">Resume from checkpoint</Label>
            </div>
            {pipeError && <p className="text-sm text-destructive">{pipeError}</p>}
            {pipeStatus === 'finished' && <p className="text-sm text-green-600">Pipeline complete!</p>}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
