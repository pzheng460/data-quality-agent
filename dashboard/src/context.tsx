import { createContext, useContext, useState, type ReactNode } from 'react'
import type { PaperInfo, StageResult } from '@/types/api'
import { usePersistedState } from '@/hooks/usePersistedState'

interface AppState {
  // Global
  outputDir: string
  setOutputDir: (v: string) => void
  refreshKey: number
  refresh: () => void

  // Ingest
  ingestOutput: string
  setIngestOutput: (v: string) => void
  ingestStatus: string
  setIngestStatus: (v: string) => void
  papers: PaperInfo[]
  setPapers: (v: PaperInfo[]) => void
  activeDomain: string
  setActiveDomain: (v: string) => void
  activeSource: string
  setActiveSource: (v: string) => void
  paramValues: Record<string, string | number>
  setParamValues: (v: Record<string, string | number> | ((prev: Record<string, string | number>) => Record<string, string | number>)) => void

  // Pipeline
  pipeInput: string
  setPipeInput: (v: string) => void
  configPath: string
  setConfigPath: (v: string) => void
  workers: number
  setWorkers: (v: number) => void
  resume: boolean
  setResume: (v: boolean) => void
  enableLLMJudge: boolean
  setEnableLLMJudge: (v: boolean) => void
  pipeStatus: string
  setPipeStatus: (v: string) => void
  stageResults: Record<string, StageResult>
  setStageResults: (v: Record<string, StageResult>) => void
  pipeError: string | null
  setPipeError: (v: string | null) => void
}

const AppContext = createContext<AppState>(null!)

export function AppProvider({ children }: { children: ReactNode }) {
  const [outputDir, setOutputDir] = usePersistedState('pipeline.outputDir', '/tmp/arxiv_test/real_output')
  const [refreshKey, setRefreshKey] = useState(0)
  const refresh = () => setRefreshKey(k => k + 1)

  const [ingestOutput, setIngestOutput] = usePersistedState('ingest.outputPath', '/tmp/dq_data/raw.jsonl')
  const [ingestStatus, setIngestStatus] = useState('')
  const [papers, setPapers] = useState<PaperInfo[]>([])
  const [activeDomain, setActiveDomain] = usePersistedState('ingest.activeDomain', '')
  const [activeSource, setActiveSource] = usePersistedState('ingest.activeSource', '')
  const [paramValues, setParamValues] = usePersistedState<Record<string, string | number>>('ingest.paramValues', {})

  const [pipeInput, setPipeInput] = usePersistedState('pipeline.inputPath', '/tmp/dq_data/raw.jsonl')
  const [configPath, setConfigPath] = usePersistedState('pipeline.configPath', 'configs/arxiv.yaml')
  const [workers, setWorkers] = usePersistedState('pipeline.workers', 4)
  const [resume, setResume] = usePersistedState('pipeline.resume', false)
  const [enableLLMJudge, setEnableLLMJudge] = usePersistedState('pipeline.enableLLMJudge', false)
  const [pipeStatus, setPipeStatus] = useState('idle')
  const [stageResults, setStageResults] = useState<Record<string, StageResult>>({})
  const [pipeError, setPipeError] = useState<string | null>(null)

  return (
    <AppContext.Provider value={{
      outputDir, setOutputDir, refreshKey, refresh,
      ingestOutput, setIngestOutput,
      ingestStatus, setIngestStatus,
      papers, setPapers,
      activeDomain, setActiveDomain,
      activeSource, setActiveSource,
      paramValues, setParamValues,
      pipeInput, setPipeInput,
      configPath, setConfigPath,
      workers, setWorkers,
      resume, setResume,
      enableLLMJudge, setEnableLLMJudge,
      pipeStatus, setPipeStatus,
      stageResults, setStageResults,
      pipeError, setPipeError,
    }}>
      {children}
    </AppContext.Provider>
  )
}

export function useApp() {
  return useContext(AppContext)
}
