// ── API response types ──

export interface PaperInfo {
  arxiv_id: string
  title: string
  chars: number
  source_method: string
}

export interface StageResult {
  phase: string
  input_count: number
  output_count: number
  rejected_count: number
  keep_rate: number
  reject_reasons: Record<string, number>
  duration_seconds: number
  skipped?: boolean
}

export interface ParamDef {
  type: string
  label: string
  default?: string | number
  required?: boolean
}

export interface SourceDef {
  name: string
  domain: string
  priority: number
  params: Record<string, ParamDef>
}

export interface DocMetadata {
  arxiv_id?: string
  title?: string
  version?: string
  primary_category?: string
  [key: string]: unknown
}

export interface Doc {
  id: string
  text: string
  text_preview?: string
  metadata?: DocMetadata
  structural_checks?: Record<string, unknown>
  trace?: Record<string, unknown>
  __dq_rejections?: Array<{
    filter: string
    rule: string
    value?: unknown
    threshold?: unknown
  }>
  [key: string]: unknown
}

export interface StageInfo {
  phase: number
  name: string
  done: boolean
  stats: StageResult | null
}

export interface OverviewData {
  version: string
  phases: Record<string, { input: number; output: number; keep_rate: number }>
}

export interface IngestStatus {
  status: string
  total: number
  downloaded: number
  papers: PaperInfo[]
  error: string | null
  output_path: string | null
}

export interface PipelineStatus {
  status: string
  current_phase: string | null
  progress: StageResult[]
  error: string | null
  config_path: string | null
  input_path: string | null
  output_dir: string | null
}
