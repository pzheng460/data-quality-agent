import { createContext, useContext, useState, type ReactNode } from 'react'

interface AppState {
  outputDir: string
  setOutputDir: (v: string) => void
  refreshKey: number
  refresh: () => void
}

const Ctx = createContext<AppState>(null!)

export function AppProvider({ children }: { children: ReactNode }) {
  const [outputDir, setOutputDir] = useState('/tmp/arxiv_test/real_output')
  const [refreshKey, setRefreshKey] = useState(0)
  const refresh = () => setRefreshKey(k => k + 1)
  return (
    <Ctx.Provider value={{ outputDir, setOutputDir, refreshKey, refresh }}>
      {children}
    </Ctx.Provider>
  )
}

export function useApp() {
  return useContext(Ctx)
}
