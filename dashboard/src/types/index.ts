export interface PhaseData {
  input: number
  output: number
  keep_rate: number
  reject_reasons: Record<string, number>
  duration_seconds: number
}

export interface OverviewData {
  version: string
  phases: Record<string, PhaseData>
  config_sha256: string
}

export interface SignalHistogram {
  signal: string
  bins: number[]
  counts: number[]
  threshold?: number
}

export interface SampleDocument {
  id: string
  text: string
  metadata?: Record<string, unknown>
  structural_checks?: Record<string, unknown>
  trace?: Record<string, unknown>
  quality_signals?: Record<string, number>
  __dq_rejections?: Array<{
    filter: string
    rule: string
    value?: unknown
    threshold?: unknown
  }>
  __raw_preview_head?: string
  __raw_preview_tail?: string
}

export interface SampleFile {
  name: string
  path: string
  count: number
}

export interface ClusterData {
  cluster_id: number
  size: number
  members: SampleDocument[]
}

export interface ContaminationEntry {
  benchmark: string
  contaminated_docs: number
  contamination_rate: number
  sample_docs: SampleDocument[]
}

export interface GoldenTestResult {
  id: string
  status: "pass" | "fail"
  expected?: Record<string, unknown>
  actual?: Record<string, unknown>
  diff?: string
}
