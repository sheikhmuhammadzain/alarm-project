import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { FileSelector } from '@/components/common/FileSelector';
import { MetricCard } from '@/components/common/MetricCard';
import { DataTable, Column } from '@/components/common/DataTable';
import { Skeleton } from '@/components/ui/skeleton';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  Gauge, 
  Activity, 
  AlertTriangle, 
  TrendingUp, 
  PlayCircle,
  FileCheck
} from 'lucide-react';
import { analyzeFile, batchAnalyze } from '@/api/client';
import { useAppStore } from '@/store/app-store';
import { toast } from '@/hooks/use-toast';
import type { FileInfo, MetricsResponse } from '@/api/types';
import { cn } from '@/lib/utils';

export default function Analysis() {
  const { selectedFile, setSelectedFile } = useAppStore();
  const [sampleSize, setSampleSize] = useState<number>(10000);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [analysisResults, setAnalysisResults] = useState<MetricsResponse | null>(null);
  
  const queryClient = useQueryClient();

  // File analysis mutation
  const analysisMutation = useMutation({
    mutationFn: ({ fileId, options }: { fileId: string; options: { sample_size?: number; show_advanced?: boolean } }) =>
      analyzeFile(fileId, options),
    onSuccess: (data) => {
      setAnalysisResults(data);
      toast({
        title: 'Analysis Complete',
        description: `Analysis completed for ${data.filename}`,
      });
    },
    onError: (error: any) => {
      toast({
        title: 'Analysis Failed',
        description: error.response?.data?.detail || 'Failed to analyze file',
        variant: 'destructive',
      });
    },
  });

  // Batch analysis mutation
  const batchMutation = useMutation({
    mutationFn: batchAnalyze,
    onSuccess: (data) => {
      toast({
        title: 'Batch Analysis Complete',
        description: `Analyzed ${data.summary.successful} of ${data.summary.total_processed} files`,
      });
      queryClient.invalidateQueries({ queryKey: ['dashboard-summary'] });
    },
    onError: (error: any) => {
      toast({
        title: 'Batch Analysis Failed',
        description: error.response?.data?.detail || 'Failed to run batch analysis',
        variant: 'destructive',
      });
    },
  });

  const handleFileSelect = (file: FileInfo | null) => {
    setSelectedFile(file);
    setAnalysisResults(null);
  };

  const handleAnalyze = () => {
    if (!selectedFile) return;
    
    analysisMutation.mutate({
      fileId: selectedFile.file_id,
      options: {
        sample_size: sampleSize,
        show_advanced: showAdvanced,
      },
    });
  };

  const getPerformanceStatus = (performance: string) => {
    switch (performance) {
      case 'Good': return 'good';
      case 'Acceptable': return 'acceptable';
      case 'Poor': return 'poor';
      default: return 'neutral';
    }
  };

  // Batch results table columns
  const batchColumns: Column[] = [
    { key: 'filename', label: 'File Name' },
    { 
      key: 'isa_performance', 
      label: 'Performance', 
      render: (value) => (
        <span className={cn(
          'px-2 py-1 rounded-full text-xs font-medium',
          value === 'Good' ? 'bg-status-good/10 text-status-good' :
          value === 'Acceptable' ? 'bg-status-acceptable/10 text-status-acceptable' :
          value === 'Poor' ? 'bg-status-poor/10 text-status-poor' :
          'bg-muted text-muted-foreground'
        )}>
          {value}
        </span>
      )
    },
    { 
      key: 'total_alarms', 
      label: 'Total Alarms',
      render: (value) => value.toLocaleString()
    },
    { 
      key: 'avg_alarms_per_hour', 
      label: 'Avg Rate (/hr)',
      render: (value) => value.toFixed(1)
    },
    { 
      key: 'flood_percentage', 
      label: 'Flood %',
      render: (value) => `${value.toFixed(1)}%`
    },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-foreground">Analysis</h1>
        <p className="text-muted-foreground">
          Analyze alarm data files for ISA-18.2 compliance and performance metrics
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        {/* File Selection & Configuration */}
        <div className="lg:col-span-1 space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <FileCheck className="h-5 w-5" />
                File Selection
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <FileSelector 
                onFileSelect={handleFileSelect}
                allowEmpty={false}
              />
            </CardContent>
          </Card>

          {/* Analysis Configuration */}
          <Card>
            <CardHeader>
              <CardTitle>Analysis Settings</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="sample-size">Sample Size</Label>
                <Input
                  id="sample-size"
                  type="number"
                  value={sampleSize}
                  onChange={(e) => setSampleSize(Number(e.target.value))}
                  placeholder="Enter sample size"
                  min={1000}
                  max={1000000}
                />
                <p className="text-xs text-muted-foreground">
                  Number of records to analyze (1,000 - 1,000,000)
                </p>
              </div>

              <div className="flex items-center space-x-2">
                <Switch
                  id="advanced-mode"
                  checked={showAdvanced}
                  onCheckedChange={setShowAdvanced}
                />
                <Label htmlFor="advanced-mode">Show Advanced Metrics</Label>
              </div>

              <Button
                onClick={handleAnalyze}
                disabled={!selectedFile || analysisMutation.isPending}
                className="w-full"
              >
                <Gauge className="h-4 w-4 mr-2" />
                {analysisMutation.isPending ? 'Analyzing...' : 'Analyze File'}
              </Button>
            </CardContent>
          </Card>

          {/* Batch Analysis */}
          <Card>
            <CardHeader>
              <CardTitle>Batch Analysis</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground mb-4">
                Analyze all available files at once for system-wide insights.
              </p>
              <Button
                onClick={() => batchMutation.mutate()}
                disabled={batchMutation.isPending}
                variant="outline"
                className="w-full"
              >
                <PlayCircle className="h-4 w-4 mr-2" />
                {batchMutation.isPending ? 'Running Batch Analysis...' : 'Run Batch Analysis'}
              </Button>
            </CardContent>
          </Card>
        </div>

        {/* Analysis Results */}
        <div className="lg:col-span-2 space-y-6">
          {analysisMutation.isPending && (
            <div className="space-y-4">
              <Skeleton className="h-32 w-full" />
              <div className="grid gap-4 md:grid-cols-2">
                <Skeleton className="h-24" />
                <Skeleton className="h-24" />
              </div>
            </div>
          )}

          {analysisMutation.isError && (
            <Alert variant="destructive">
              <AlertTriangle className="h-4 w-4" />
              <AlertDescription>
                Analysis failed: {analysisMutation.error?.message}
              </AlertDescription>
            </Alert>
          )}

          {analysisResults && (
            <>
              {/* Main Metrics */}
              <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
                <MetricCard
                  title="ISA Performance"
                  value={analysisResults.isa_performance}
                  subtitle="Overall system rating"
                  icon={Gauge}
                  status={getPerformanceStatus(analysisResults.isa_performance)}
                />
                
                <MetricCard
                  title="Total Alarms"
                  value={analysisResults.total_alarms}
                  subtitle="In analysis period"
                  icon={Activity}
                  status="neutral"
                />
                
                <MetricCard
                  title="Avg Alarm Rate"
                  value={`${analysisResults.avg_alarms_per_hour.toFixed(1)}/hr`}
                  subtitle="Per hour average"
                  icon={TrendingUp}
                  status={
                    analysisResults.avg_alarms_per_hour <= 144 ? 'good' :
                    analysisResults.avg_alarms_per_hour <= 288 ? 'acceptable' : 'poor'
                  }
                />
                
                <MetricCard
                  title="Flood Percentage"
                  value={`${analysisResults.flood_percentage.toFixed(1)}%`}
                  subtitle="Time in flood conditions"
                  icon={AlertTriangle}
                  status={
                    analysisResults.flood_percentage <= 1 ? 'good' :
                    analysisResults.flood_percentage <= 5 ? 'acceptable' : 'poor'
                  }
                />
              </div>

              {/* Analysis Details */}
              <Card>
                <CardHeader>
                  <CardTitle>Analysis Details</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid gap-4 md:grid-cols-2">
                    <div className="space-y-2">
                      <Label>File Information</Label>
                      <div className="text-sm space-y-1">
                        <p><strong>Filename:</strong> {analysisResults.filename}</p>
                        <p><strong>Analysis Date:</strong> {new Date(analysisResults.analysis_datetime).toLocaleString()}</p>
                        {analysisResults.sample_size && (
                          <p><strong>Sample Size:</strong> {analysisResults.sample_size.toLocaleString()}</p>
                        )}
                      </div>
                    </div>
                    
                    <div className="space-y-2">
                      <Label>Performance Rating</Label>
                      <div className="text-sm space-y-1">
                        <p><strong>Alarm Rate Performance:</strong> {analysisResults.alarm_rate_performance}</p>
                        <p><strong>ISA-18.2 Compliance:</strong> {analysisResults.isa_performance}</p>
                      </div>
                    </div>
                  </div>

                  {/* Advanced Metrics */}
                  {showAdvanced && analysisResults.advanced_metrics && (
                    <div className="space-y-3 pt-4 border-t border-border">
                      <Label>Advanced Metrics</Label>
                      <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-4">
                        <div className="p-3 rounded-lg border border-border bg-muted/30">
                          <div className="text-xs font-medium text-muted-foreground">Peak Alarm Rate</div>
                          <div className="text-lg font-semibold">{analysisResults.advanced_metrics.peak_alarm_rate}/hr</div>
                        </div>
                        <div className="p-3 rounded-lg border border-border bg-muted/30">
                          <div className="text-xs font-medium text-muted-foreground">Flood Episodes</div>
                          <div className="text-lg font-semibold">{analysisResults.advanced_metrics.flood_episodes}</div>
                        </div>
                        <div className="p-3 rounded-lg border border-border bg-muted/30">
                          <div className="text-xs font-medium text-muted-foreground">Chattering Alarms</div>
                          <div className="text-lg font-semibold">{analysisResults.advanced_metrics.chattering_alarms}</div>
                        </div>
                        <div className="p-3 rounded-lg border border-border bg-muted/30">
                          <div className="text-xs font-medium text-muted-foreground">Standing Alarms</div>
                          <div className="text-lg font-semibold">{analysisResults.advanced_metrics.standing_alarms}</div>
                        </div>
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            </>
          )}

          {/* Batch Results */}
          {batchMutation.data && (
            <Card>
              <CardHeader>
                <CardTitle>Batch Analysis Results</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid gap-4 md:grid-cols-3">
                  <div className="p-3 rounded-lg border border-border bg-status-good/10">
                    <div className="text-sm font-medium text-muted-foreground">Successful</div>
                    <div className="text-2xl font-bold text-status-good">
                      {batchMutation.data.summary.successful}
                    </div>
                  </div>
                  <div className="p-3 rounded-lg border border-border bg-status-poor/10">
                    <div className="text-sm font-medium text-muted-foreground">Failed</div>
                    <div className="text-2xl font-bold text-status-poor">
                      {batchMutation.data.summary.failed}
                    </div>
                  </div>
                  <div className="p-3 rounded-lg border border-border bg-muted/30">
                    <div className="text-sm font-medium text-muted-foreground">Total</div>
                    <div className="text-2xl font-bold">
                      {batchMutation.data.summary.total_processed}
                    </div>
                  </div>
                </div>

                {batchMutation.data.completed_files.length > 0 && (
                  <DataTable
                    data={batchMutation.data.completed_files.map(item => ({
                      filename: item.filename,
                      ...item.metrics
                    }))}
                    columns={batchColumns}
                    searchable={true}
                    sortable={true}
                    paginated={true}
                    pageSize={10}
                    emptyMessage="No completed analyses"
                  />
                )}
              </CardContent>
            </Card>
          )}

          {!analysisMutation.isPending && !analysisResults && !batchMutation.data && (
            <Card className="h-64">
              <CardContent className="flex items-center justify-center h-full">
                <div className="text-center text-muted-foreground">
                  <Gauge className="mx-auto h-12 w-12 opacity-50 mb-4" />
                  <p>Select a file and click "Analyze File" to view metrics</p>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}