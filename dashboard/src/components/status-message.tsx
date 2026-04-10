import { Alert, AlertDescription } from '@/components/ui/alert'

type Status = 'error' | 'success' | 'info' | 'idle'

interface StatusMessageProps {
  status: Status
  message?: string | null
}

export function StatusMessage({ status, message }: StatusMessageProps) {
  if (!message || status === 'idle') return null

  const variant = status === 'error' ? 'destructive' : 'default'

  return (
    <Alert variant={variant} className={status === 'success' ? 'border-green-200 text-green-700 bg-green-50' : ''}>
      <AlertDescription>{message}</AlertDescription>
    </Alert>
  )
}
