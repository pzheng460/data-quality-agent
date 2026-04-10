import { createContext, useContext, useState, type ReactNode } from 'react'
import type { PaperInfo, StageResult } from '@/types/api'

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
  pipeStatus: string
  setPipeStatus: (v: string) => void
  stageResults: Record<string, StageResult>
  setStageResults: (v: Record<string, StageResult>) => void
  pipeError: string | null
  setPipeError: (v: string | null) => void
}

const AppContext = createContext<AppState>(null!)

export function AppProvider({ children }: { children: ReactNode }) {
  const [outputDir, setOutputDir] = useState('/tmp/arxiv_test/real_output')
  const [refreshKey, setRefreshKey] = useState(0)
  const refresh = () => setRefreshKey(k => k + 1)

  const [ingestOutput, setIngestOutput] = useState('/tmp/dq_data/raw.jsonl')
  const [ingestStatus, setIngestStatus] = useState('idle')
  const [papers, setPapers] = useState<PaperInfo[]>([])
  const [activeDomain, setActiveDomain] = useState('')
  const [activeSource, setActiveSource] = useState('')
  const [paramValues, setParamValues] = useState<Record<string, string | number>>({})

  const [pipeInput, setPipeInput] = useState('/tmp/dq_data/raw.jsonl')
  const [configPath, setConfigPath] = useState('configs/arxiv.yaml')
  const [workers, setWorkers] = useState(4)
  const [resume, setResume] = useState(false)
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
