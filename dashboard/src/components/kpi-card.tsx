import { Card, CardContent } from '@/components/ui/card'
import { cn } from '@/lib/utils'

interface KpiCardProps {
  label: string
  value: string | number
  color?: string
  className?: string
}

export function KpiCard({ label, value, color, className }: KpiCardProps) {
  return (
    <Card>
      <CardContent className="pt-4 pb-3">
        <p className="text-xs text-muted-foreground uppercase tracking-wide">{label}</p>
        <p className={cn('text-2xl font-bold mt-1', color, className)}>{value}</p>
      </CardContent>
    </Card>
  )
}

interface KpiGridProps {
  cols?: 3 | 4 | 5
  children: React.ReactNode
}

export function KpiGrid({ cols = 4, children }: KpiGridProps) {
  const gridClass = {
    3: 'grid-cols-3',
    4: 'grid-cols-4',
    5: 'grid-cols-5',
  }[cols]
  return <div className={`grid ${gridClass} gap-4`}>{children}</div>
}
