import { useEffect, useState } from 'react'
import { useApp } from '../context'
import { api } from '../hooks/useApi'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'

interface PhaseInfo {
  phase: string; input_count: number; output_count: number
  rejected_count: number; keep_rate: number
  reject_reasons: Record<string, number>; duration_seconds: number
}

export default function PhaseDetails() {
  const { outputDir, refreshKey } = useApp()
  const [phases, setPhases] = useState<PhaseInfo[]>([])

  useEffect(() => {
    (async () => {
      try {
        const list = await api<any[]>(`/api/phases?output_dir=${encodeURIComponent(outputDir)}`)
        const results: PhaseInfo[] = []
        for (const p of list) {
          if (p.done && p.stats_file) {
            try {
              const s = await api<PhaseInfo>(`/api/phase-stats/${p.name}?output_dir=${encodeURIComponent(outputDir)}`)
              results.push(s)
            } catch { /* skip */ }
          }
        }
        setPhases(results)
      } catch { setPhases([]) }
    })()
  }, [outputDir, refreshKey])

  if (!phases.length) return <p className="text-gray-400 p-8">No phase data. Run the pipeline first.</p>

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">Phase Details</h2>
      {phases.map(p => {
        const reasons = Object.entries(p.reject_reasons).filter(([, v]) => v > 0)
        return (
          <div key={p.phase} className="bg-white rounded-lg shadow p-6">
            <div className="flex justify-between items-start mb-4">
              <div>
                <h3 className="font-bold text-lg">{p.phase}</h3>
                <p className="text-sm text-gray-500 mt-1">
                  {p.input_count} in &rarr; {p.output_count} kept, {p.rejected_count} rejected
                  <span className="ml-2 text-xs text-gray-400">{p.duration_seconds.toFixed(1)}s</span>
                </p>
              </div>
              <span className={`px-3 py-1 rounded text-sm font-medium ${p.keep_rate >= 0.9 ? 'bg-green-100 text-green-700' : p.keep_rate >= 0.7 ? 'bg-yellow-100 text-yellow-700' : 'bg-red-100 text-red-700'}`}>
                {(p.keep_rate * 100).toFixed(1)}% kept
              </span>
            </div>
            {reasons.length > 0 && (
              <>
                <div className="text-sm font-medium text-gray-600 mb-2">Rejection Reasons</div>
                <ResponsiveContainer width="100%" height={Math.max(100, reasons.length * 36)}>
                  <BarChart data={reasons.map(([r, c]) => ({ rule: r, count: c }))} layout="vertical" margin={{ left: 180 }}>
                    <XAxis type="number" />
                    <YAxis type="category" dataKey="rule" tick={{ fontSize: 12 }} width={170} />
                    <Tooltip />
                    <Bar dataKey="count" fill="#ef4444" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </>
            )}
            {reasons.length === 0 && p.rejected_count === 0 && (
              <div className="text-sm text-green-600">No rejections.</div>
            )}
          </div>
        )
      })}
    </div>
  )
}
