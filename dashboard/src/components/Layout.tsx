import { NavLink, Outlet } from 'react-router-dom'

const navItems = [
  { to: '/', label: 'Pipeline Control' },
  { to: '/overview', label: 'Overview' },
  { to: '/phases', label: 'Phase Details' },
  { to: '/signals', label: 'Quality Signals' },
  { to: '/samples', label: 'Sample Browser' },
  { to: '/dedup', label: 'Dedup Clusters' },
  { to: '/contamination', label: 'Contamination' },
  { to: '/golden', label: 'Golden Tests' },
]

export default function Layout() {
  return (
    <div className="flex h-screen bg-gray-50">
      <aside className="w-56 bg-white border-r border-gray-200 flex flex-col">
        <div className="px-4 py-5 border-b border-gray-200">
          <h1 className="text-lg font-bold text-gray-900">dq Dashboard</h1>
          <p className="text-xs text-gray-500 mt-1">Arxiv Pipeline</p>
        </div>
        <nav className="flex-1 px-2 py-4 space-y-1">
          {navItems.map(({ to, label }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                `block px-3 py-2 rounded-md text-sm ${
                  isActive
                    ? 'bg-blue-50 text-blue-700 font-medium'
                    : 'text-gray-600 hover:bg-gray-100'
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
