import { useState } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import {
  LayoutDashboard,
  Files,
  BarChart3,
  TrendingUp,
  Lightbulb,
  Settings,
  Gauge,
  Menu,
  FolderOpen,
} from 'lucide-react';
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarHeader,
  useSidebar,
} from '@/components/ui/sidebar';
import { cn } from '@/lib/utils';

const navigation = [
  { title: 'Dashboard', url: '/', icon: LayoutDashboard },
  { title: 'Explorer', url: '/explorer', icon: FolderOpen },
  { title: 'Files', url: '/files', icon: Files },
  { title: 'Analysis', url: '/analysis', icon: Gauge },
  { title: 'Charts', url: '/charts', icon: BarChart3 },
  { title: 'Insights', url: '/insights', icon: Lightbulb },
  { title: 'Advanced', url: '/advanced', icon: TrendingUp },
  { title: 'Settings', url: '/settings', icon: Settings },
];

export function AppSidebar() {
  const { state } = useSidebar();
  const location = useLocation();
  const currentPath = location.pathname;
  
  const isCollapsed = state === 'collapsed';
  
  const isActive = (path: string) => {
    if (path === '/') {
      return currentPath === '/';
    }
    return currentPath.startsWith(path);
  };

  return (
    <Sidebar className={cn(
      'border-r border-border bg-card transition-all duration-300',
      isCollapsed ? 'w-16' : 'w-64'
    )}>
      <SidebarHeader className="border-b border-border px-4 py-4">
        <div className="flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-primary">
            <Gauge className="h-5 w-5 text-primary-foreground" />
          </div>
          {!isCollapsed && (
            <div>
              <h2 className="font-semibold text-foreground">ISA Analyzer</h2>
              <p className="text-xs text-muted-foreground">Alarm Rationalization</p>
            </div>
          )}
        </div>
      </SidebarHeader>
      
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>Navigation</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {navigation.map((item) => (
                <SidebarMenuItem key={item.title}>
                  <SidebarMenuButton asChild>
                    <NavLink
                      to={item.url}
                      className={cn(
                        'flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors',
                        'hover:bg-accent hover:text-accent-foreground',
                        isActive(item.url) 
                          ? 'bg-isa-primary text-primary-foreground shadow-isa-sm' 
                          : 'text-muted-foreground'
                      )}
                    >
                      <item.icon className="h-5 w-5 flex-shrink-0" />
                      {!isCollapsed && <span>{item.title}</span>}
                    </NavLink>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
    </Sidebar>
  );
}