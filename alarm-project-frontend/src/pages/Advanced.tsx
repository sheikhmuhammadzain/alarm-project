import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { FileSelector } from '@/components/common/FileSelector';
import { DataTable, Column } from '@/components/common/DataTable';
import { Skeleton } from '@/components/ui/skeleton';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  Zap, 
  Clock, 
  TrendingUp, 
  Play,
  AlertTriangle,
  Activity
} from 'lucide-react';
import { getChatteringAlarms, getStandingAlarms } from '@/api/client';
import { useAppStore } from '@/store/app-store';
import type { FileInfo, ChatteringAlarm, StandingAlarm } from '@/api/types';
import { cn } from '@/lib/utils';

export default function Advanced() {
  const { selectedFile, setSelectedFile } = useAppStore();
  
  // Chattering alarm parameters
  const [windowMinutes, setWindowMinutes] = useState(10);
  const [threshold, setThreshold] = useState(5);
  
  // Standing alarm parameters
  const [minDurationHours, setMinDurationHours] = useState(24);

  const handleFileSelect = (file: FileInfo | null) => {
    setSelectedFile(file);
  };

  // Chattering alarms query
  const chatteringQuery = useQuery({
    queryKey: ['chattering-alarms', selectedFile?.file_id, windowMinutes, threshold],
    queryFn: () => selectedFile ? getChatteringAlarms(selectedFile.file_id, windowMinutes, threshold) : null,
    enabled: !!selectedFile,
  });

  // Standing alarms query
  const standingQuery = useQuery({
    queryKey: ['standing-alarms', selectedFile?.file_id, minDurationHours],
    queryFn: () => selectedFile ? getStandingAlarms(selectedFile.file_id, minDurationHours) : null,
    enabled: !!selectedFile,
  });

  // Chattering alarms table columns
  const chatteringColumns: Column[] = [
    { key: 'tag', label: 'Alarm Tag' },
    { 
      key: 'count', 
      label: 'Total Count',
      render: (value) => value.toLocaleString()
    },
    { 
      key: 'rate_per_hour', 
      label: 'Rate (per hour)',
      render: (value) => value.toFixed(1)
    },
    { 
      key: 'severity', 
      label: 'Severity',
      render: (value) => (
        <Badge variant={
          value === 'High' ? 'destructive' :
          value === 'Medium' ? 'secondary' : 'outline'
        }>
          {value}
        </Badge>
      )
    },
  ];

  // Standing alarms table columns
  const standingColumns: Column[] = [
    { key: 'tag', label: 'Alarm Tag' },
    { 
      key: 'duration_hours', 
      label: 'Duration (hours)',
      render: (value) => value.toFixed(1)
    },
    { 
      key: 'start_time', 
      label: 'Start Time',
      render: (value) => new Date(value).toLocaleString()
    },
    { 
      key: 'end_time', 
      label: 'End Time',
      render: (value) => value ? new Date(value).toLocaleString() : '—'
    },
    { 
      key: 'is_active', 
      label: 'Status',
      render: (value) => (
        <Badge variant={value ? 'destructive' : 'secondary'}>
          {value ? 'Active' : 'Resolved'}
        </Badge>
      )
    },
  ];

  const getSeverityStats = (data: ChatteringAlarm[]) => {
    const stats = data.reduce((acc, alarm) => {
      acc[alarm.severity] = (acc[alarm.severity] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);
    
    return {
      high: stats.High || 0,
      medium: stats.Medium || 0,
      low: stats.Low || 0,
    };
  };

  const getActiveStandingCount = (data: StandingAlarm[]) => {
    return data.filter(alarm => alarm.is_active).length;
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-foreground">Advanced Analysis</h1>
        <p className="text-muted-foreground">
          Deep dive into chattering and standing alarms with advanced detection algorithms
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-4">
        {/* Configuration Sidebar */}
        <div className="lg:col-span-1">
          <Card className="sticky top-6">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <TrendingUp className="h-5 w-5" />
                Analysis Configuration
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <FileSelector 
                onFileSelect={handleFileSelect}
                allowEmpty={false}
              />

              {selectedFile && (
                <div className="space-y-4 pt-4 border-t border-border">
                  <div className="space-y-3">
                    <h4 className="text-sm font-semibold">Chattering Detection</h4>
                    <div className="space-y-2">
                      <Label htmlFor="window-minutes">Time Window (minutes)</Label>
                      <Input
                        id="window-minutes"
                        type="number"
                        value={windowMinutes}
                        onChange={(e) => setWindowMinutes(Number(e.target.value))}
                        min={1}
                        max={60}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="threshold">Alarm Threshold</Label>
                      <Input
                        id="threshold"
                        type="number"
                        value={threshold}
                        onChange={(e) => setThreshold(Number(e.target.value))}
                        min={2}
                        max={50}
                      />
                    </div>
                    <p className="text-xs text-muted-foreground">
                      Detect alarms that trigger ≥{threshold} times in {windowMinutes} minutes
                    </p>
                  </div>

                  <div className="space-y-3">
                    <h4 className="text-sm font-semibold">Standing Alarm Detection</h4>
                    <div className="space-y-2">
                      <Label htmlFor="min-duration">Min Duration (hours)</Label>
                      <Input
                        id="min-duration"
                        type="number"
                        value={minDurationHours}
                        onChange={(e) => setMinDurationHours(Number(e.target.value))}
                        min={1}
                        max={168}
                      />
                    </div>
                    <p className="text-xs text-muted-foreground">
                      Detect alarms active for ≥{minDurationHours} hours
                    </p>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Analysis Results */}
        <div className="lg:col-span-3">
          {!selectedFile ? (
            <Card className="h-96">
              <CardContent className="flex items-center justify-center h-full">
                <div className="text-center text-muted-foreground">
                  <TrendingUp className="mx-auto h-12 w-12 opacity-50 mb-4" />
                  <h3 className="text-lg font-semibold mb-2">No File Selected</h3>
                  <p>Select a file from the sidebar to perform advanced analysis</p>
                </div>
              </CardContent>
            </Card>
          ) : (
            <Tabs defaultValue="chattering" className="space-y-6">
              <TabsList className="grid w-full grid-cols-2">
                <TabsTrigger value="chattering" className="flex items-center gap-2">
                  <Zap className="h-4 w-4" />
                  Chattering Alarms
                </TabsTrigger>
                <TabsTrigger value="standing" className="flex items-center gap-2">
                  <Clock className="h-4 w-4" />
                  Standing Alarms
                </TabsTrigger>
              </TabsList>

              <TabsContent value="chattering" className="space-y-6">
                {/* Chattering Summary */}
                {chatteringQuery.data && (
                  <div className="grid gap-4 md:grid-cols-4">
                    <Card>
                      <CardContent className="p-4">
                        <div className="flex items-center gap-2">
                          <Zap className="h-5 w-5 text-isa-primary" />
                          <div>
                            <div className="text-2xl font-bold">{chatteringQuery.data.length}</div>
                            <div className="text-sm text-muted-foreground">Total Chattering</div>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                    
                    <Card>
                      <CardContent className="p-4">
                        <div className="flex items-center gap-2">
                          <AlertTriangle className="h-5 w-5 text-status-poor" />
                          <div>
                            <div className="text-2xl font-bold text-status-poor">
                              {getSeverityStats(chatteringQuery.data).high}
                            </div>
                            <div className="text-sm text-muted-foreground">High Severity</div>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                    
                    <Card>
                      <CardContent className="p-4">
                        <div className="flex items-center gap-2">
                          <Activity className="h-5 w-5 text-status-acceptable" />
                          <div>
                            <div className="text-2xl font-bold text-status-acceptable">
                              {getSeverityStats(chatteringQuery.data).medium}
                            </div>
                            <div className="text-sm text-muted-foreground">Medium Severity</div>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                    
                    <Card>
                      <CardContent className="p-4">
                        <div className="flex items-center gap-2">
                          <Activity className="h-5 w-5 text-status-good" />
                          <div>
                            <div className="text-2xl font-bold text-status-good">
                              {getSeverityStats(chatteringQuery.data).low}
                            </div>
                            <div className="text-sm text-muted-foreground">Low Severity</div>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                )}

                {/* Chattering Analysis Info */}
                <Alert>
                  <Zap className="h-4 w-4" />
                  <AlertDescription>
                    Chattering alarms are alarms that repeatedly activate and deactivate in rapid succession, 
                    causing operator distraction and potentially masking more critical issues. 
                    Current settings: ≥{threshold} activations within {windowMinutes} minutes.
                  </AlertDescription>
                </Alert>

                {/* Chattering Results */}
                {chatteringQuery.isLoading ? (
                  <div className="space-y-4">
                    {Array.from({ length: 5 }).map((_, i) => (
                      <Skeleton key={i} className="h-16 w-full" />
                    ))}
                  </div>
                ) : chatteringQuery.error ? (
                  <Alert variant="destructive">
                    <AlertTriangle className="h-4 w-4" />
                    <AlertDescription>
                      Failed to load chattering alarms: {chatteringQuery.error.message}
                    </AlertDescription>
                  </Alert>
                ) : chatteringQuery.data ? (
                  <Card>
                    <CardHeader>
                      <CardTitle>Chattering Alarms Detected</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <DataTable
                        data={chatteringQuery.data}
                        columns={chatteringColumns}
                        searchable={true}
                        sortable={true}
                        paginated={true}
                        pageSize={15}
                        emptyMessage="No chattering alarms detected with current settings"
                      />
                    </CardContent>
                  </Card>
                ) : null}
              </TabsContent>

              <TabsContent value="standing" className="space-y-6">
                {/* Standing Summary */}
                {standingQuery.data && (
                  <div className="grid gap-4 md:grid-cols-3">
                    <Card>
                      <CardContent className="p-4">
                        <div className="flex items-center gap-2">
                          <Clock className="h-5 w-5 text-isa-primary" />
                          <div>
                            <div className="text-2xl font-bold">{standingQuery.data.length}</div>
                            <div className="text-sm text-muted-foreground">Total Standing</div>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                    
                    <Card>
                      <CardContent className="p-4">
                        <div className="flex items-center gap-2">
                          <AlertTriangle className="h-5 w-5 text-status-poor" />
                          <div>
                            <div className="text-2xl font-bold text-status-poor">
                              {getActiveStandingCount(standingQuery.data)}
                            </div>
                            <div className="text-sm text-muted-foreground">Currently Active</div>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                    
                    <Card>
                      <CardContent className="p-4">
                        <div className="flex items-center gap-2">
                          <Activity className="h-5 w-5 text-status-good" />
                          <div>
                            <div className="text-2xl font-bold text-status-good">
                              {standingQuery.data.length - getActiveStandingCount(standingQuery.data)}
                            </div>
                            <div className="text-sm text-muted-foreground">Resolved</div>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                )}

                {/* Standing Analysis Info */}
                <Alert>
                  <Clock className="h-4 w-4" />
                  <AlertDescription>
                    Standing alarms are alarms that remain active for extended periods, often indicating 
                    underlying process issues that require attention. Current settings: active for ≥{minDurationHours} hours.
                  </AlertDescription>
                </Alert>

                {/* Standing Results */}
                {standingQuery.isLoading ? (
                  <div className="space-y-4">
                    {Array.from({ length: 5 }).map((_, i) => (
                      <Skeleton key={i} className="h-16 w-full" />
                    ))}
                  </div>
                ) : standingQuery.error ? (
                  <Alert variant="destructive">
                    <AlertTriangle className="h-4 w-4" />
                    <AlertDescription>
                      Failed to load standing alarms: {standingQuery.error.message}
                    </AlertDescription>
                  </Alert>
                ) : standingQuery.data ? (
                  <Card>
                    <CardHeader>
                      <CardTitle>Standing Alarms Detected</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <DataTable
                        data={standingQuery.data}
                        columns={standingColumns}
                        searchable={true}
                        sortable={true}
                        paginated={true}
                        pageSize={15}
                        emptyMessage="No standing alarms detected with current settings"
                      />
                    </CardContent>
                  </Card>
                ) : null}
              </TabsContent>
            </Tabs>
          )}
        </div>
      </div>
    </div>
  );
}