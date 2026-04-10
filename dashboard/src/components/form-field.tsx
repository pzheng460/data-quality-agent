import { Label } from '@/components/ui/label'
import { cn } from '@/lib/utils'

interface FormFieldProps {
  label: string
  htmlFor?: string
  required?: boolean
  hint?: string
  className?: string
  children: React.ReactNode
}

export function FormField({ label, htmlFor, required, hint, className, children }: FormFieldProps) {
  return (
    <div className={`space-y-2 ${className || ''}`}>
      <label htmlFor={htmlFor} className="flex items-center gap-2 text-sm leading-none font-medium">
        {label}
        {required && <span className="text-destructive">*</span>}
        {hint && <span className="text-muted-foreground font-normal">{hint}</span>}
      </label>
      {children}
    </div>
  )
}
