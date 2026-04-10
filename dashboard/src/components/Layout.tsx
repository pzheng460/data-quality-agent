import { NavLink, Outlet } from 'react-router-dom'
import { Button } from '@/components/ui/button'
import { Separator } from '@/components/ui/separator'

const navItems = [
  { to: '/', label: 'Pipeline' },
  { to: '/overview', label: 'Overview' },
  { to: '/samples', label: 'Samples' },
  { to: '/benchmark', label: 'Benchmark' },
]

export default function Layout() {
  return (
    <div className="flex h-screen bg-muted/30">
      <aside className="w-56 bg-background border-r flex flex-col">
        <div className="px-4 py-5">
          <h1 className="text-lg font-bold">dq Dashboard</h1>
          <p className="text-xs text-muted-foreground mt-1">Data Quality Agent</p>
        </div>
        <Separator />
        <nav className="flex-1 px-2 py-4 space-y-1">
          {navItems.map(({ to, label }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                `block px-3 py-2 rounded-md text-sm transition-colors ${
                  isActive
                    ? 'bg-primary/10 text-primary font-medium'
                    : 'text-muted-foreground hover:bg-muted'
                }`
              }
            >
              {label}
            </NavLink>
          ))}
        </nav>
      </aside>
      <main className="flex-1 overflow-auto p-6">
        <Outlet />
      </main>
    </div>
  )
}
