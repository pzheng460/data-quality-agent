import { ScrollArea } from '@/components/ui/scroll-area'

interface CodePanelProps {
  children: string
  height?: string
  className?: string
}

export function CodePanel({ children, height = '70vh', className }: CodePanelProps) {
  return (
    <ScrollArea className={`rounded-lg border bg-muted/50 p-4 ${className || ''}`} style={{ height }}>
      <pre className="text-sm font-mono whitespace-pre-wrap">{children}</pre>
    </ScrollArea>
  )
}
