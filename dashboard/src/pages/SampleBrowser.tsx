import { useState } from 'react'
import { useApp } from '../context'
import { api } from '../hooks/useApi'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import 'katex/dist/katex.min.css'

interface Doc {
  id: string; text: string; text_preview?: string; metadata?: any
  structural_checks?: Record<string, unknown>; trace?: Record<string, unknown>
  __dq_rejections?: Array<{ filter: string; rule: string; value?: unknown; threshold?: unknown }>
  [k: string]: unknown
}

const STAGES = [
  { stage: 'stage1_parsed', sub: 'kept', label: 'Phase 1 — Kept', color: 'green' },
  { stage: 'stage1_parsed', sub: 'rejected', label: 'Phase 1 — Rejected', color: 'red' },
  { stage: 'stage2_filtered', sub: 'kept', label: 'Phase 2 — Kept', color: 'green' },
  { stage: 'stage2_filtered', sub: 'rejected', label: 'Phase 2 — Rejected', color: 'red' },
  { stage: 'stage3_dedup', sub: 'kept', label: 'Phase 3 — Kept', color: 'green' },
  { stage: 'stage5_final', sub: '', label: 'Final Output', color: 'blue' },
]

function Tag({ label, color = 'gray' }: { label: string; color?: string }) {
  const c: Record<string, string> = { gray: 'bg-gray-100 text-gray-700', blue: 'bg-blue-50 text-blue-700', red: 'bg-red-50 text-red-600' }
  return <span className={`text-xs px-2 py-0.5 rounded ${c[color] || c.gray}`}>{label}</span>
}

/* ── Document Detail with Original vs Cleaned comparison ── */
function DocDetail({ doc, compareDoc }: { doc: Doc; compareDoc: Doc | null }) {
  const [tab, setTab] = useState<string>('compare')
  const meta = doc.metadata as any
  const rej = doc.__dq_rejections
  const sc = doc.structural_checks as Record<string, unknown> | undefined
  const trace = doc.trace as Record<string, unknown> | undefined

  const arxivId = meta?.arxiv_id as string | undefined
  const arxivHtml = arxivId ? `https://arxiv.org/html/${arxivId}` : null
  const arxivPdf = arxivId ? `https://arxiv.org/pdf/${arxivId}` : null
  const arxivAbs = arxivId ? `https://arxiv.org/abs/${arxivId}` : null

  return (
    <div className="p-5 space-y-4">
      {/* Header */}
      <div>
        <h3 className="text-xl font-bold text-gray-900">{meta?.title || doc.id}</h3>
        <div className="flex flex-wrap items-center gap-1.5 mt-2">
          {arxivId && <Tag label={`arxiv:${arxivId}`} />}
          {meta?.version && <Tag label={meta.version} />}
          {meta?.primary_category && <Tag label={meta.primary_category} color="blue" />}
          <Tag label={`${doc.text?.length ?? 0} chars`} />
          {arxivAbs && <a href={arxivAbs} target="_blank" rel="noreferrer" className="text-xs text-blue-600 hover:underline ml-2">arxiv page ↗</a>}
          {arxivPdf && <a href={arxivPdf} target="_blank" rel="noreferrer" className="text-xs text-blue-600 hover:underline">PDF ↗</a>}
        </div>
      </div>

      {/* Rejection banner */}
      {rej && rej.length > 0 && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-3">
          <div className="font-semibold text-red-700 text-sm">Rejected</div>
          {rej.map((r, i) => (
            <div key={i} className="text-sm mt-1">
              <code className="text-red-600 bg-red-100 px-1.5 py-0.5 rounded">{r.filter}.{r.rule}</code>
              {r.value !== undefined && <span className="text-gray-500 ml-2">value={String(r.value)}</span>}
              {r.threshold !== undefined && <span className="text-gray-500 ml-1">thr={String(r.threshold)}</span>}
            </div>
          ))}
        </div>
      )}

      {/* Structural checks */}
      {sc && (
        <div className="flex flex-wrap gap-2">
          {Object.entries(sc).map(([k, v]) => (
            <span key={k} className={`text-[11px] px-2 py-0.5 rounded ${v === true ? 'bg-green-50 text-green-700' : v === false ? 'bg-red-50 text-red-600' : 'bg-gray-100 text-gray-600'}`}>
              {k}: {String(v)}
            </span>
          ))}
        </div>
      )}

      {/* Tabs */}
      <div className="flex gap-0.5 border-b border-gray-200">
        {['compare', 'rendered', 'raw', 'json', 'trace'].map(t => (
          <button key={t} onClick={() => setTab(t as any)}
            className={`px-4 py-2 text-sm border-b-2 -mb-px transition-colors ${tab === t ? 'border-blue-500 text-blue-700 font-medium' : 'border-transparent text-gray-500 hover:text-gray-700'}`}>
            {{ compare: 'Original vs Cleaned', rendered: 'Cleaned', raw: 'Raw Text', json: 'JSON', trace: 'Trace' }[t]}
          </button>
        ))}
      </div>

      {/* ── Original vs Cleaned (side by side) ── */}
      {tab === 'compare' && (
        <div className="grid grid-cols-2 gap-3 h-[70vh]">
          {/* Left: Original — iframe from arxiv, or rendered pre-clean text */}
          <div className="flex flex-col min-h-0">
            <div className="text-sm font-medium text-gray-500 mb-2 flex items-center gap-2 shrink-0">
              <span className="w-2.5 h-2.5 rounded-full bg-amber-400" />
              Original
              {arxivHtml && <a href={arxivHtml} target="_blank" rel="noreferrer" className="text-xs text-blue-500 hover:underline ml-auto">Open HTML ↗</a>}
              {arxivPdf && <a href={arxivPdf} target="_blank" rel="noreferrer" className="text-xs text-blue-500 hover:underline ml-2">Open PDF ↗</a>}
            </div>
            {arxivHtml ? (
              <iframe src={arxivHtml} className="flex-1 rounded-lg border border-amber-200 bg-white" title="Original paper" sandbox="allow-same-origin allow-scripts" />
            ) : compareDoc ? (
              <div className="flex-1 overflow-auto rounded-lg border border-amber-200 bg-amber-50/30 p-4">
                <article className="prose-article">
                  <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>
                    {compareDoc.text}
                  </ReactMarkdown>
                </article>
              </div>
            ) : (
              <div className="flex-1 flex items-center justify-center text-gray-400 rounded-lg border border-dashed border-gray-300">
                No original version available.<br/>Original is only shown for Phase 2+ stages.
              </div>
            )}
          </div>

          {/* Right: Cleaned output */}
          <div className="flex flex-col min-h-0">
            <div className="text-sm font-medium text-gray-500 mb-2 flex items-center gap-2 shrink-0">
              <span className="w-2.5 h-2.5 rounded-full bg-green-400" />
              Cleaned Output
            </div>
            <div className="flex-1 overflow-auto rounded-lg border border-green-200 bg-green-50/20 p-4">
              <article className="prose-article">
                <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>
                  {doc.text}
                </ReactMarkdown>
              </article>
            </div>
          </div>
        </div>
      )}

      {/* ── Cleaned (full width, comfortable reading) ── */}
      {tab === 'rendered' && (
        <div className="max-w-3xl mx-auto">
          <article className="prose-article">
            <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>
              {doc.text}
            </ReactMarkdown>
          </article>
        </div>
      )}

      {tab === 'raw' && (
        <pre className="text-[13px] leading-relaxed bg-gray-50 border border-gray-200 p-4 rounded-lg overflow-auto max-h-[70vh] whitespace-pre-wrap font-mono text-gray-800">{doc.text}</pre>
      )}

      {tab === 'json' && (
        <pre className="text-xs bg-gray-50 border border-gray-200 p-4 rounded-lg overflow-auto max-h-[70vh] font-mono">{JSON.stringify(doc, null, 2)}</pre>
      )}

      {tab === 'trace' && (trace ? (
        <div className="space-y-2">
          {Object.entries(trace).map(([phase, info]) => (
            <div key={phase} className="flex items-center gap-3 p-3 rounded-lg bg-gray-50">
              <span className={`w-3 h-3 rounded-full shrink-0 ${(info as any)?.status === 'ok' ? 'bg-green-500' : 'bg-red-500'}`} />
              <span className="font-mono font-medium text-sm w-48">{phase}</span>
              <span className="text-gray-500 text-xs">{JSON.stringify(info)}</span>
            </div>
          ))}
        </div>
      ) : <p className="text-gray-400">No trace data.</p>)}
    </div>
  )
}

/* ── Main page component ── */

export default function SampleBrowser() {
  const { outputDir } = useApp()
  const [curStage, setCurStage] = useState<typeof STAGES[0] | null>(null)
  const [docs, setDocs] = useState<Doc[]>([])
  const [curDoc, setCurDoc] = useState<Doc | null>(null)
  const [compareDoc, setCompareDoc] = useState<Doc | null>(null)
  const [loading, setLoading] = useState(false)

  const loadDocs = async (s: typeof STAGES[0]) => {
    setCurStage(s); setCurDoc(null); setCompareDoc(null); setLoading(true)
    try {
      const path = s.sub ? `/api/docs/${s.stage}/${s.sub}` : `/api/docs/${s.stage}`
      const data = await api<any>(`${path}?output_dir=${encodeURIComponent(outputDir)}&limit=100`)
      setDocs(data.docs || [])
    } catch { setDocs([]) }
    setLoading(false)
  }

  const selectDoc = async (doc: Doc) => {
    if (!curStage) return
    try {
      const subParam = curStage.sub ? `&sub=${curStage.sub}` : ''
      const full = await api<Doc>(`/api/doc?output_dir=${encodeURIComponent(outputDir)}&stage=${curStage.stage}${subParam}&doc_id=${encodeURIComponent(doc.id)}`)
      setCurDoc(full)
      // Load the pre-cleaning version for comparison
      if (curStage.stage !== 'stage1_parsed') {
        try {
          const before = await api<Doc>(`/api/doc?output_dir=${encodeURIComponent(outputDir)}&stage=stage1_parsed&sub=kept&doc_id=${encodeURIComponent(doc.id)}`)
          setCompareDoc(before)
        } catch { setCompareDoc(null) }
      } else {
        setCompareDoc(null)
      }
    } catch { setCurDoc(doc); setCompareDoc(null) }
  }

  return (
    <div className="flex h-[calc(100vh-3rem)] gap-3">
      {/* Stage list */}
      <div className="w-48 bg-white rounded-lg shadow overflow-auto shrink-0">
        <div className="px-3 py-2 border-b font-semibold text-sm sticky top-0 bg-white z-10">Stages</div>
        {STAGES.map(s => (
          <button key={`${s.stage}/${s.sub}`} onClick={() => loadDocs(s)}
            className={`block w-full text-left px-3 py-2.5 text-sm border-b border-gray-100 hover:bg-blue-50 ${curStage?.stage === s.stage && curStage?.sub === s.sub ? 'bg-blue-100 font-medium' : ''}`}>
            <span className={`inline-block w-2 h-2 rounded-full mr-1.5 align-middle ${s.color === 'red' ? 'bg-red-400' : s.color === 'green' ? 'bg-green-500' : 'bg-blue-500'}`} />
            {s.label}
          </button>
        ))}
      </div>

      {/* Doc list */}
      <div className="w-64 bg-white rounded-lg shadow overflow-auto shrink-0">
        <div className="px-3 py-2 border-b font-semibold text-sm sticky top-0 bg-white z-10">
          Documents <span className="font-normal text-gray-400">({docs.length})</span>
        </div>
        {loading && <p className="p-3 text-xs text-gray-400">Loading...</p>}
        {docs.map((d, i) => (
          <button key={d.id || i} onClick={() => selectDoc(d)}
            className={`block w-full text-left px-3 py-2 border-b border-gray-50 hover:bg-blue-50 ${curDoc?.id === d.id ? 'bg-blue-50' : ''}`}>
            <div className="text-xs font-mono font-medium truncate">{d.id}</div>
            <div className="text-[11px] text-gray-500 truncate">{d.metadata?.title || d.text_preview?.slice(0, 60)}</div>
            {(d.__dq_rejections?.length ?? 0) > 0 && (
              <div className="flex flex-wrap gap-0.5 mt-0.5">
                {d.__dq_rejections!.map((r, j) => <span key={j} className="bg-red-100 text-red-600 text-[10px] px-1 rounded">{r.rule}</span>)}
              </div>
            )}
          </button>
        ))}
      </div>

      {/* Detail panel */}
      <div className="flex-1 bg-white rounded-lg shadow overflow-auto">
        {curDoc ? <DocDetail doc={curDoc} compareDoc={compareDoc} /> : <div className="flex items-center justify-center h-full text-gray-400">Select a stage → document</div>}
      </div>
    </div>
  )
}
