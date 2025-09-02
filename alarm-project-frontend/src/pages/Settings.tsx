import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Button } from '@/components/ui/button';
import { Switch } from '@/components/ui/switch';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  Settings as SettingsIcon, 
  Server, 
  Palette, 
  Database,
  CheckCircle,
  AlertTriangle,
  Info,
  Save
} from 'lucide-react';
import { getHealth, getBenchmarks } from '@/api/client';
import { setApiBaseUrl } from '@/api/config';
import { useAppStore } from '@/store/app-store';
import { toast } from '@/hooks/use-toast';
import { cn } from '@/lib/utils';

export default function Settings() {
  const { 
    darkMode, 
    setDarkMode, 
    apiBaseUrl, 
    setApiBaseUrl 
  } = useAppStore();

  const [tempApiUrl, setTempApiUrl] = useState(apiBaseUrl);
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);

  // Health check query
  const healthQuery = useQuery({
    queryKey: ['health'],
    queryFn: getHealth,
    retry: false,
  });

  // Benchmarks query
  const benchmarksQuery = useQuery({
    queryKey: ['benchmarks'],
    queryFn: getBenchmarks,
    retry: false,
  });

  const handleApiUrlChange = (value: string) => {
    setTempApiUrl(value);
    setHasUnsavedChanges(value !== apiBaseUrl);
  };

  const handleSaveSettings = () => {
    setApiBaseUrl(tempApiUrl);
    setHasUnsavedChanges(false);
    
    toast({
      title: 'Settings Saved',
      description: 'Your settings have been saved successfully.',
    });

    // Trigger health check with new URL
    healthQuery.refetch();
  };

  const handleResetDefaults = () => {
    const defaultUrl = 'http://localhost:8000';
    setTempApiUrl(defaultUrl);
    setDarkMode(false);
    setHasUnsavedChanges(tempApiUrl !== defaultUrl);
  };

  const formatBenchmarkValue = (key: string, value: number) => {
    if (key.includes('time') || key.includes('duration')) {
      return `${value.toFixed(3)}s`;
    }
    if (key.includes('rate') || key.includes('per')) {
      return `${value.toFixed(1)}/hr`;
    }
    if (key.includes('percentage') || key.includes('percent')) {
      return `${value.toFixed(1)}%`;
    }
    return value.toString();
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-foreground">Settings</h1>
        <p className="text-muted-foreground">
          Configure your ISA Analyzer application preferences and API settings
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        {/* Settings Panel */}
        <div className="lg:col-span-2 space-y-6">
          {/* API Configuration */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Server className="h-5 w-5" />
                API Configuration
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="api-url">Base URL</Label>
                <Input
                  id="api-url"
                  value={tempApiUrl}
                  onChange={(e) => handleApiUrlChange(e.target.value)}
                  placeholder="http://localhost:8000"
                />
                <p className="text-xs text-muted-foreground">
                  The base URL for your ISA Analyzer API backend
                </p>
              </div>

              {/* API Status */}
              <div className="space-y-2">
                <Label>Connection Status</Label>
                <div className="flex items-center gap-2">
                  {healthQuery.isLoading ? (
                    <Badge variant="outline">Testing...</Badge>
                  ) : healthQuery.isError ? (
                    <Badge variant="destructive">Disconnected</Badge>
                  ) : (
                    <Badge variant="default" className="bg-status-good">Connected</Badge>
                  )}
                  
                  {healthQuery.data && (
                    <span className="text-sm text-muted-foreground">
                      Status: {healthQuery.data.status}
                    </span>
                  )}
                </div>
              </div>

              {hasUnsavedChanges && (
                <Alert>
                  <Info className="h-4 w-4" />
                  <AlertDescription>
                    You have unsaved changes. Click "Save Settings" to apply them.
                  </AlertDescription>
                </Alert>
              )}
            </CardContent>
          </Card>

          {/* Appearance */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Palette className="h-5 w-5" />
                Appearance
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label>Dark Mode</Label>
                  <div className="text-xs text-muted-foreground">
                    Toggle between light and dark themes
                  </div>
                </div>
                <Switch
                  checked={darkMode}
                  onCheckedChange={setDarkMode}
                />
              </div>

              <Separator />

              <div className="space-y-2">
                <Label>Color Scheme</Label>
                <div className="text-xs text-muted-foreground mb-3">
                  ISA-18.2 compliant professional color scheme
                </div>
                <div className="grid grid-cols-4 gap-2">
                  <div className="space-y-1">
                    <div className="h-8 w-full rounded bg-isa-primary"></div>
                    <div className="text-xs text-center">Primary</div>
                  </div>
                  <div className="space-y-1">
                    <div className="h-8 w-full rounded bg-status-good"></div>
                    <div className="text-xs text-center">Good</div>
                  </div>
                  <div className="space-y-1">
                    <div className="h-8 w-full rounded bg-status-acceptable"></div>
                    <div className="text-xs text-center">Warning</div>
                  </div>
                  <div className="space-y-1">
                    <div className="h-8 w-full rounded bg-status-poor"></div>
                    <div className="text-xs text-center">Critical</div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Actions */}
          <Card>
            <CardHeader>
              <CardTitle>Actions</CardTitle>
            </CardHeader>
            <CardContent className="flex gap-2">
              <Button 
                onClick={handleSaveSettings}
                disabled={!hasUnsavedChanges}
              >
                <Save className="h-4 w-4 mr-2" />
                Save Settings
              </Button>
              
              <Button 
                variant="outline"
                onClick={handleResetDefaults}
              >
                Reset to Defaults
              </Button>
            </CardContent>
          </Card>
        </div>

        {/* System Information */}
        <div className="space-y-6">
          {/* System Status */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Database className="h-5 w-5" />
                System Status
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm">API Connection</span>
                  {healthQuery.isError ? (
                    <AlertTriangle className="h-4 w-4 text-status-poor" />
                  ) : (
                    <CheckCircle className="h-4 w-4 text-status-good" />
                  )}
                </div>
                
                <div className="flex items-center justify-between">
                  <span className="text-sm">Dark Mode</span>
                  {darkMode ? (
                    <CheckCircle className="h-4 w-4 text-status-good" />
                  ) : (
                    <div className="w-4 h-4 border border-muted-foreground rounded-full" />
                  )}
                </div>
              </div>

              {healthQuery.isError && (
                <Alert variant="destructive">
                  <AlertTriangle className="h-4 w-4" />
                  <AlertDescription>
                    Cannot connect to API. Please check the URL and ensure the backend is running.
                  </AlertDescription>
                </Alert>
              )}
            </CardContent>
          </Card>

          {/* Performance Benchmarks */}
          {benchmarksQuery.data && (
            <Card>
              <CardHeader>
                <CardTitle>ISA Benchmarks</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {Object.entries(benchmarksQuery.data).map(([key, value]) => (
                    <div key={key} className="flex justify-between text-sm">
                      <span className="capitalize">
                        {key.replace(/_/g, ' ')}
                      </span>
                      <span className="font-mono">
                        {formatBenchmarkValue(key, value as number)}
                      </span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Version Information */}
          <Card>
            <CardHeader>
              <CardTitle>Version Information</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span>Frontend Version</span>
                <span className="font-mono">v1.0.0</span>
              </div>
              <div className="flex justify-between">
                <span>Build Date</span>
                <span className="font-mono">{new Date().toLocaleDateString()}</span>
              </div>
              <div className="flex justify-between">
                <span>Framework</span>
                <span className="font-mono">React 18</span>
              </div>
              <div className="flex justify-between">
                <span>UI Library</span>
                <span className="font-mono">Shadcn/UI</span>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}