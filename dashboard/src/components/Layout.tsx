import { NavLink, Outlet } from 'react-router-dom'
import {
  Sidebar, SidebarContent, SidebarGroup, SidebarGroupContent,
  SidebarGroupLabel, SidebarHeader, SidebarMenu, SidebarMenuButton,
  SidebarMenuItem, SidebarProvider, SidebarTrigger, SidebarInset,
} from '@/components/ui/sidebar'
import { Separator } from '@/components/ui/separator'
import { LayoutDashboard, BarChart3, FileSearch, FlaskConical } from 'lucide-react'

const navItems = [
  { to: '/', label: 'Pipeline', icon: LayoutDashboard },
  { to: '/stats', label: 'Stats', icon: BarChart3 },
  { to: '/samples', label: 'Samples', icon: FileSearch },
  { to: '/benchmark', label: 'Benchmark', icon: FlaskConical },
]

export default function Layout() {
  return (
    <SidebarProvider>
      <div className="flex min-h-screen w-full">
        <Sidebar>
          <SidebarHeader className="p-4">
            <h1 className="text-lg font-bold">DQ Dashboard</h1>
            <p className="text-xs text-muted-foreground">Data Quality Agent</p>
          </SidebarHeader>
          <SidebarContent>
            <SidebarGroup>
              <SidebarGroupLabel>Navigation</SidebarGroupLabel>
              <SidebarGroupContent>
                <SidebarMenu>
                  {navItems.map(({ to, label, icon: Icon }) => (
                    <SidebarMenuItem key={to}>
                      <NavLink to={to}>
                        {({ isActive }) => (
                          <SidebarMenuButton isActive={isActive} tooltip={label}>
                            <Icon />
                            <span>{label}</span>
                          </SidebarMenuButton>
                        )}
                      </NavLink>
                    </SidebarMenuItem>
                  ))}
                </SidebarMenu>
              </SidebarGroupContent>
            </SidebarGroup>
          </SidebarContent>
        </Sidebar>
        <SidebarInset>
          <header className="flex h-12 shrink-0 items-center gap-2 border-b px-4">
            <SidebarTrigger className="-ml-1" />
            <Separator orientation="vertical" className="mr-2 h-4" />
            <div className="flex-1" />
            <div id="header-actions" />
          </header>
          <main className="flex-1 overflow-auto p-6">
            <Outlet />
          </main>
        </SidebarInset>
      </div>
    </SidebarProvider>
  )
}

