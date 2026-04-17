import { useState, useEffect } from 'react'
import { usePersistedState } from '@/hooks/usePersistedState'
import { createPortal } from 'react-dom'
import { useApp } from '@/context'
import { api, apiUrl } from '@/hooks/useApi'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Separator } from '@/components/ui/separator'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import 'katex/dist/katex.min.css'

interface Doc { id: string; text: string; text_preview?: string; metadata?: any; structural_checks?: Record<string, unknown>; trace?: Record<string, unknown>; __dq_rejections?: Array<{ filter: string; rule: string; value?: unknown; threshold?: unknown }>; [k: string]: unknown }

const STAGES = [
  { stage: 'stage1_ingested', sub: 'kept', label: 'S1 Ingested', color: 'amber' },
  { stage: 'stage2_extracted', sub: 'kept', label: 'S2 Extracted', color: 'green' },
  { stage: 'stage2_extracted', sub: 'rejected', label: 'S2 Rejected', color: 'red' },
  { stage: 'stage3_curated', sub: 'kept', label: 'S3 Curated', color: 'green' },
  { stage: 'stage3_curated', sub: 'rejected', label: 'S3 Rejected', color: 'red' },
  { stage: 'stage4_final', sub: '', label: 'S4 Final', color: 'blue' },
]

// Rewrites backend-absolute image paths to the /api/image proxy so the browser can fetch them.
// Extension-less stems (from \includegraphics{fig}) are left alone — the user sees the missing
// image placeholder, which is actually useful feedback.
function Md({ children }: { children: string }) {
  const imgComponent = ({ src, alt, ...rest }: any) => {
    let url: string | undefined
    if (typeof src === 'string' && src.startsWith('/')) {
      url = apiUrl(`/api/image?path=${encodeURIComponent(src)}`)
    }
    if (!url) {
      // Relative / unknown ref (common for data ingested before save_figures existed).
      // Render an inline placeholder so the doc doesn't look broken.
      const label = alt || src || 'figure'
      return (
        <span className="inline-block my-2 px-3 py-2 text-xs font-mono
                         border border-dashed border-muted-foreground/40 text-muted-foreground
                         rounded bg-muted/30" title={String(src)}>
          📎 {String(label).slice(0, 120)}
        </span>
      )
    }
    return (
      <img src={url} alt={alt}
           style={{ maxWidth: '100%', height: 'auto' }}
           onError={(e) => { (e.target as HTMLImageElement).style.display = 'none' }}
           {...rest} />
    )
  }
  return (
    <article className="prose-article">
      <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}
                     components={{ img: imgComponent }}>
        {children}
      </ReactMarkdown>
    </article>
  )
}

/* ── DocDetail ── */
function DocDetail({ doc, compareDoc, isRawInput = false }: { doc: Doc; compareDoc: Doc | null; isRawInput?: boolean }) {
  const [compareView, setCompareView] = useState<'split' | 'original' | 'cleaned'>('split')
  const meta = doc.metadata as any
  const sc = doc.structural_checks as Record<string, unknown> | undefined
  const trace = doc.trace as Record<string, unknown> | undefined
  const arxivId = meta?.arxiv_id as string | undefined
  const arxivHtml = arxivId ? `https://arxiv.org/html/${arxivId}` : null
  const arxivPdf = arxivId ? `https://arxiv.org/pdf/${arxivId}` : null

  const OriginalPanel = ({ className = '' }: { className?: string }) => (
    arxivPdf ? (
      <iframe src={arxivPdf} className={`w-full rounded-lg border bg-background ${className}`} title="Original PDF" />
    ) : compareDoc ? (
      <ScrollArea className={`rounded-lg border bg-muted/50 p-4 ${className}`}>
        <pre className="text-xs leading-relaxed font-mono whitespace-pre-wrap">{compareDoc.text}</pre>
      </ScrollArea>
    ) : (
      <div className={`flex items-center justify-center text-muted-foreground border rounded-lg border-dashed text-sm ${className}`}>No original. Select a Stage 2+ document.</div>
    )
  )

  const CleanedPanel = ({ className = '' }: { className?: string }) => (
    <ScrollArea className={`rounded-lg border p-4 ${className}`}>
      <Md>{doc.text}</Md>
    </ScrollArea>
  )

  return (
    <div className="p-5 space-y-4">
      {/* Header */}
      <div>
        <h3 className="text-xl font-bold">{doc.metadata?.title || doc.id}</h3>
        <div className="flex flex-wrap items-center gap-1.5 mt-2">
          {doc.metadata?.arxiv_id && <Badge variant="outline">arxiv:{doc.metadata.arxiv_id}</Badge>}
          {meta?.version && <Badge variant="outline">{meta.version}</Badge>}
          {meta?.primary_category && <Badge>{meta.primary_category}</Badge>}
          <Badge variant="secondary">{doc.text?.length ?? 0} chars</Badge>
          {arxivHtml && <a href={arxivHtml} target="_blank" rel="noreferrer" className="text-xs text-primary hover:underline ml-2">HTML ↗</a>}
          {arxivPdf && <a href={arxivPdf} target="_blank" rel="noreferrer" className="text-xs text-primary hover:underline">PDF ↗</a>}
        </div>
      </div>

      {/* Rejection */}
      {doc.__dq_rejections && doc.__dq_rejections.length > 0 && (
        <Card className="border-destructive/50 bg-destructive/5">
          <CardContent className="py-3">
            <p className="font-semibold text-destructive text-sm">Rejected</p>
            {doc.__dq_rejections.map((r, i) => (
              <div key={i} className="text-sm mt-1">
                <Badge variant="destructive" className="font-mono">{r.filter}.{r.rule}</Badge>
                {r.value !== undefined && <span className="text-muted-foreground ml-2 text-xs">val={String(r.value)}</span>}
                {r.threshold !== undefined && <span className="text-muted-foreground ml-1 text-xs">thr={String(r.threshold)}</span>}
              </div>
            ))}
          </CardContent>
        </Card>
      )}

      {/* Structural checks */}
      {sc && (
        <div className="flex flex-wrap gap-1.5">
          {Object.entries(sc).map(([k, v]) => (
            <Badge key={k} variant={v === true ? 'default' : v === false ? 'destructive' : 'secondary'} className="text-[11px]">
              {k}: {String(v)}
            </Badge>
          ))}
        </div>
      )}

      {/* Content tabs */}
      <Tabs defaultValue={isRawInput ? 'text' : 'compare'}>
        <TabsList>
          {isRawInput ? (
            <>
              <TabsTrigger value="text">Text</TabsTrigger>
              <TabsTrigger value="pdf">PDF</TabsTrigger>
              <TabsTrigger value="json">JSON</TabsTrigger>
            </>
          ) : (
            <>
              <TabsTrigger value="compare">Compare</TabsTrigger>
              <TabsTrigger value="rendered">Rendered</TabsTrigger>
              <TabsTrigger value="raw">Source</TabsTrigger>
              <TabsTrigger value="json">JSON</TabsTrigger>
              <TabsTrigger value="trace">Trace</TabsTrigger>
            </>
          )}
        </TabsList>

        <TabsContent value="text">
          <ScrollArea className="h-[75vh] rounded-lg border bg-muted/50 p-5">
            <pre className="text-sm leading-relaxed font-mono whitespace-pre-wrap">{doc.text}</pre>
          </ScrollArea>
        </TabsContent>

        <TabsContent value="pdf">
          {arxivPdf ? (
            <iframe src={arxivPdf} className="w-full h-[75vh] rounded-lg border" title="PDF" />
          ) : (
            <div className="flex items-center justify-center h-48 text-muted-foreground border rounded-lg border-dashed">No arxiv ID — PDF unavailable.</div>
          )}
        </TabsContent>

        <TabsContent value="compare">
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <div className="inline-flex rounded-md border p-0.5">
                {(['split', 'original', 'cleaned'] as const).map(v => (
                  <Button key={v} variant={compareView === v ? 'default' : 'ghost'} size="sm"
                    onClick={() => setCompareView(v)}>
                    {v === 'split' ? 'Side by Side' : v === 'original' ? 'Original' : 'Cleaned'}
                  </Button>
                ))}
              </div>
            </div>

            {compareView === 'split' && (
              <div className="grid grid-cols-2 gap-3 h-[70vh]">
                <div className="flex flex-col min-h-0">
                  <p className="text-xs font-medium text-muted-foreground mb-1 shrink-0">Original</p>
                  <div className="flex-1 overflow-auto rounded-lg border bg-muted/50 p-4">
                    {arxivPdf ? (
                      <iframe src={arxivPdf} className="w-full h-full rounded border-0" title="Original PDF" />
                    ) : compareDoc ? (
                      <pre className="text-xs leading-relaxed font-mono whitespace-pre-wrap">{compareDoc.text}</pre>
                    ) : (
                      <p className="text-muted-foreground text-sm text-center pt-8">No original. Select a Stage 2+ document.</p>
                    )}
                  </div>
                </div>
                <div className="flex flex-col min-h-0">
                  <p className="text-xs font-medium text-muted-foreground mb-1 shrink-0">Cleaned</p>
                  <div className="flex-1 overflow-auto rounded-lg border p-4">
                    <Md>{doc.text}</Md>
                  </div>
                </div>
              </div>
            )}
            {compareView === 'original' && <OriginalPanel className="h-[75vh]" />}
            {compareView === 'cleaned' && <CleanedPanel className="max-h-[75vh]" />}
          </div>
        </TabsContent>

        <TabsContent value="rendered">
          <div className="max-w-3xl"><Md>{doc.text}</Md></div>
        </TabsContent>
        <TabsContent value="raw">
          <ScrollArea className="h-[70vh] rounded-lg border bg-muted/50 p-4">
            <pre className="text-sm font-mono whitespace-pre-wrap">{doc.text}</pre>
          </ScrollArea>
        </TabsContent>
        <TabsContent value="json">
          <ScrollArea className="h-[70vh] rounded-lg border bg-muted/50 p-4">
            <pre className="text-xs font-mono">{JSON.stringify(doc, null, 2)}</pre>
          </ScrollArea>
        </TabsContent>
        <TabsContent value="trace">
          {trace ? (
            <div className="space-y-2">
              {Object.entries(trace).map(([phase, info]) => (
                <div key={phase} className="flex items-center gap-3 p-3 rounded-lg bg-muted/50">
                  <span className={`w-3 h-3 rounded-full shrink-0 ${(info as any)?.status === 'ok' ? 'bg-green-500' : 'bg-red-500'}`} />
                  <span className="font-mono text-sm w-48">{phase}</span>
                  <span className="text-muted-foreground text-xs">{JSON.stringify(info)}</span>
                </div>
              ))}
            </div>
          ) : <p className="text-muted-foreground">No trace data.</p>}
        </TabsContent>
      </Tabs>
    </div>
  )
}

/* ── Main page ── */
export default function Samples() {
  const { outputDir } = useApp()
  const [curStage, setCurStage] = useState<typeof STAGES[0] | null>(null)
  const [docs, setDocs] = useState<Doc[]>([])
  const [curDoc, setCurDoc] = useState<Doc | null>(null)
  const [compareDoc, setCompareDoc] = useState<Doc | null>(null)
  const [loading, setLoading] = useState(false)
  const [sidebarOpen, setSidebarOpen] = usePersistedState('samples.sidebarOpen', true)

  const loadDocs = async (s: typeof STAGES[0]) => {
    setCurStage(s); setCurDoc(null); setCompareDoc(null); setLoading(true)
    try {
      const path = s.sub ? `/api/docs/${s.stage}/${s.sub}` : `/api/docs/${s.stage}`
      const data = await api<Record<string, unknown>>(`${path}?output_dir=${encodeURIComponent(outputDir)}&limit=100`)
      setDocs(data.docs || [])
    } catch { setDocs([]) }
    setLoading(false)
  }

  const selectDoc = async (doc: Doc) => {
    if (!curStage) return
    try {
      const sub = curStage.sub ? `&sub=${curStage.sub}` : ''
      const full = await api<Doc>(`/api/doc?output_dir=${encodeURIComponent(outputDir)}&stage=${curStage.stage}${sub}&doc_id=${encodeURIComponent(doc.id)}`)
      setCurDoc(full)
      if (curStage.stage !== 'stage1_ingested') {
        try {
          const before = await api<Doc>(`/api/doc?output_dir=${encodeURIComponent(outputDir)}&stage=stage1_ingested&sub=kept&doc_id=${encodeURIComponent(doc.id)}`)
          setCompareDoc(before)
        } catch { setCompareDoc(null) }
      } else { setCompareDoc(null) }
    } catch { setCurDoc(doc); setCompareDoc(null) }
  }

  const headerEl = typeof document !== 'undefined' ? document.getElementById('header-actions') : null

  return (
    <>
      {headerEl && createPortal(
        <Button variant="ghost" size="sm" onClick={() => setSidebarOpen(!sidebarOpen)}>
          {sidebarOpen ? 'Hide sidebar' : 'Show sidebar'}
        </Button>,
        headerEl
      )}
    <div className="flex h-[calc(100vh-6rem)] gap-2">
      {sidebarOpen && (
        <aside className="w-64 shrink-0 border-r bg-sidebar flex flex-col">
          <div className="px-3 py-2">
            <p className="text-xs font-semibold text-sidebar-foreground/70 uppercase tracking-wider">Stages</p>
          </div>
          <nav className="px-2 space-y-0.5">
            {STAGES.map(s => {
              const isActive = curStage?.stage === s.stage && curStage?.sub === s.sub
              return (
                <button key={`${s.stage}/${s.sub}`} onClick={() => loadDocs(s)}
                  className={`flex items-center gap-2 w-full rounded-md px-2 py-1.5 text-sm transition-colors ${isActive ? 'bg-sidebar-accent text-sidebar-accent-foreground font-medium' : 'text-sidebar-foreground hover:bg-sidebar-accent/50'}`}>
                  <span className={`w-2 h-2 rounded-full shrink-0 ${s.color === 'green' ? 'bg-green-500' : s.color === 'red' ? 'bg-red-500' : s.color === 'blue' ? 'bg-blue-500' : 'bg-amber-500'}`} />
                  {s.label}
                </button>
              )
            })}
          </nav>

          <Separator className="my-2" />

          {/* Docs */}
          <div className="px-3 py-1 flex items-center justify-between">
            <p className="text-xs font-semibold text-sidebar-foreground/70 uppercase tracking-wider">Documents</p>
            <span className="text-xs text-muted-foreground">{docs.length}</span>
          </div>
          <ScrollArea className="flex-1 px-1">
            {loading && <div className="p-2 space-y-1"><Skeleton className="h-10 w-full rounded-md" /><Skeleton className="h-10 w-full rounded-md" /><Skeleton className="h-10 w-3/4 rounded-md" /></div>}
            {docs.map((d, i) => {
              const isActive = curDoc?.id === d.id
              return (
                <button key={d.id || i} onClick={() => selectDoc(d)}
                  className={`w-full text-left rounded-md px-2 py-1.5 mb-0.5 transition-colors ${isActive ? 'bg-sidebar-accent text-sidebar-accent-foreground' : 'hover:bg-sidebar-accent/50'}`}>
                  <p className="text-xs font-mono truncate">{d.id}</p>
                  <p className="text-[11px] text-muted-foreground truncate">{d.metadata?.title || d.text_preview?.slice(0, 60)}</p>
                  {(d.__dq_rejections?.length ?? 0) > 0 && (
                    <div className="flex flex-wrap gap-0.5 mt-0.5">
                      {d.__dq_rejections!.map((r, j) => <Badge key={j} variant="destructive" className="text-[9px] px-1 py-0">{r.rule}</Badge>)}
                    </div>
                  )}
                </button>
              )
            })}
          </ScrollArea>
        </aside>
      )}

      {/* Detail */}
      <Card className="flex-1 overflow-hidden flex flex-col min-w-0">
        <div className="flex-1 overflow-auto">
          {curDoc ? (
            <DocDetail doc={curDoc} compareDoc={compareDoc} isRawInput={curStage?.stage === 'stage1_ingested'} />
          ) : (
            <div className="flex items-center justify-center h-full text-muted-foreground text-sm">
              Select a stage → document
            </div>
          )}
        </div>
      </Card>
    </div>
    </>
  )
}
