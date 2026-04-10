import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'

interface StageRowProps {
  num: number
  label: string
  desc: string
  input: string
  output: string
  result?: {
    input_count: number
    output_count: number
    keep_rate: number
    duration_seconds: number
    skipped?: boolean
  }
  disabled?: boolean
  onRun: () => void
}

export function StageRow({ num, label, desc, input, output, result, disabled, onRun }: StageRowProps) {
  const isDone = !!result && !result.skipped
  const isSkipped = result?.skipped

  return (
    <div className={`flex items-center gap-3 px-4 py-3 rounded-lg ${isDone ? 'bg-green-50' : isSkipped ? 'bg-yellow-50' : 'bg-muted/30'}`}>
      <div className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold shrink-0 ${isDone ? 'bg-green-500 text-white' : isSkipped ? 'bg-yellow-400 text-white' : 'bg-muted-foreground/20 text-muted-foreground'}`}>
        {num}
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="font-semibold text-sm">{label}</span>
          <span className="text-xs text-muted-foreground">{desc}</span>
        </div>
        <div className="flex gap-4 mt-0.5 text-[11px] text-muted-foreground">
          <span>in: <code className="bg-muted px-1 rounded truncate max-w-[200px] inline-block align-bottom">{input}</code></span>
          <span>out: <code className="bg-muted px-1 rounded truncate max-w-[200px] inline-block align-bottom">{output}</code></span>
        </div>
      </div>
      {isDone && result && (
        <span className="text-xs text-muted-foreground shrink-0">
          {result.input_count}→{result.output_count} ({(result.keep_rate * 100).toFixed(1)}%, {result.duration_seconds?.toFixed(1)}s)
        </span>
      )}
      {isSkipped && <Badge variant="outline" className="text-yellow-600">skipped</Badge>}
      <Button variant="outline" size="sm" disabled={disabled} onClick={onRun}>Run</Button>
    </div>
  )
}
