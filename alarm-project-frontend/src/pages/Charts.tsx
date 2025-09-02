import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { FileSelector } from '@/components/common/FileSelector';
import { ChartWrapper } from '@/components/common/ChartWrapper';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  BarChart3, 
  TrendingUp, 
  PieChart, 
  Activity, 
  Zap,
  BarChart2,
  LineChart,
  Target
} from 'lucide-react';
import {
  getAlarmRateChart,
  getTopContributorsChart,
  getPriorityDistributionChart,
  getParetoAnalysisChart,
  getFloodAnalysisChart,
  getDistributionHistogramChart,
  getRollingAverageChart,
} from '@/api/client';
import { useAppStore } from '@/store/app-store';
import type { FileInfo } from '@/api/types';

export default function Charts() {
  const { selectedFile, setSelectedFile } = useAppStore();
  const [topN, setTopN] = useState(15);
  const [windowHours, setWindowHours] = useState(24);

  const handleFileSelect = (file: FileInfo | null) => {
    setSelectedFile(file);
  };

  // Chart queries - only enabled when file is selected
  const alarmRateQuery = useQuery({
    queryKey: ['alarm-rate-chart', selectedFile?.file_id],
    queryFn: () => selectedFile ? getAlarmRateChart(selectedFile.file_id) : null,
    enabled: !!selectedFile,
  });

  const topContributorsQuery = useQuery({
    queryKey: ['top-contributors-chart', selectedFile?.file_id, topN],
    queryFn: () => selectedFile ? getTopContributorsChart(selectedFile.file_id, topN) : null,
    enabled: !!selectedFile,
  });

  const priorityDistributionQuery = useQuery({
    queryKey: ['priority-distribution-chart', selectedFile?.file_id],
    queryFn: () => selectedFile ? getPriorityDistributionChart(selectedFile.file_id) : null,
    enabled: !!selectedFile,
  });

  const paretoAnalysisQuery = useQuery({
    queryKey: ['pareto-analysis-chart', selectedFile?.file_id],
    queryFn: () => selectedFile ? getParetoAnalysisChart(selectedFile.file_id) : null,
    enabled: !!selectedFile,
  });

  const floodAnalysisQuery = useQuery({
    queryKey: ['flood-analysis-chart', selectedFile?.file_id],
    queryFn: () => selectedFile ? getFloodAnalysisChart(selectedFile.file_id) : null,
    enabled: !!selectedFile,
  });

  const histogramQuery = useQuery({
    queryKey: ['distribution-histogram-chart', selectedFile?.file_id],
    queryFn: () => selectedFile ? getDistributionHistogramChart(selectedFile.file_id) : null,
    enabled: !!selectedFile,
  });

  const rollingAverageQuery = useQuery({
    queryKey: ['rolling-average-chart', selectedFile?.file_id, windowHours],
    queryFn: () => selectedFile ? getRollingAverageChart(selectedFile.file_id, windowHours) : null,
    enabled: !!selectedFile,
  });

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-foreground">Charts & Visualizations</h1>
        <p className="text-muted-foreground">
          Visualize alarm data with interactive charts and analytics
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-4">
        {/* File Selection Sidebar */}
        <div className="lg:col-span-1">
          <Card className="sticky top-6">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="h-5 w-5" />
                Chart Configuration
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <FileSelector 
                onFileSelect={handleFileSelect}
                allowEmpty={false}
              />

              {/* Chart Parameters */}
              {selectedFile && (
                <div className="space-y-4 pt-4 border-t border-border">
                  <div className="space-y-2">
                    <Label htmlFor="top-n">Top Contributors (N)</Label>
                    <Input
                      id="top-n"
                      type="number"
                      value={topN}
                      onChange={(e) => setTopN(Number(e.target.value))}
                      min={5}
                      max={50}
                      className="w-full"
                    />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="window-hours">Rolling Window (Hours)</Label>
                    <Input
                      id="window-hours"
                      type="number"
                      value={windowHours}
                      onChange={(e) => setWindowHours(Number(e.target.value))}
                      min={1}
                      max={168}
                      className="w-full"
                    />
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Charts Area */}
        <div className="lg:col-span-3">
          {!selectedFile ? (
            <Card className="h-96">
              <CardContent className="flex items-center justify-center h-full">
                <div className="text-center text-muted-foreground">
                  <BarChart3 className="mx-auto h-12 w-12 opacity-50 mb-4" />
                  <h3 className="text-lg font-semibold mb-2">No File Selected</h3>
                  <p>Select a file from the sidebar to view charts and visualizations</p>
                </div>
              </CardContent>
            </Card>
          ) : (
            <Tabs defaultValue="alarm-rate" className="space-y-6">
              <TabsList className="grid w-full grid-cols-4 lg:grid-cols-7">
                <TabsTrigger value="alarm-rate" className="text-xs">
                  <Activity className="h-3 w-3 mr-1" />
                  Rate
                </TabsTrigger>
                <TabsTrigger value="contributors" className="text-xs">
                  <BarChart2 className="h-3 w-3 mr-1" />
                  Top
                </TabsTrigger>
                <TabsTrigger value="priority" className="text-xs">
                  <PieChart className="h-3 w-3 mr-1" />
                  Priority
                </TabsTrigger>
                <TabsTrigger value="pareto" className="text-xs">
                  <Target className="h-3 w-3 mr-1" />
                  Pareto
                </TabsTrigger>
                <TabsTrigger value="flood" className="text-xs">
                  <Zap className="h-3 w-3 mr-1" />
                  Flood
                </TabsTrigger>
                <TabsTrigger value="histogram" className="text-xs">
                  <BarChart3 className="h-3 w-3 mr-1" />
                  Histogram
                </TabsTrigger>
                <TabsTrigger value="rolling" className="text-xs">
                  <LineChart className="h-3 w-3 mr-1" />
                  Rolling
                </TabsTrigger>
              </TabsList>

              <TabsContent value="alarm-rate">
                <ChartWrapper
                  data={alarmRateQuery.data}
                  title="Alarm Rate Over Time"
                  isLoading={alarmRateQuery.isLoading}
                  error={alarmRateQuery.error?.message}
                  metrics={alarmRateQuery.data?.metrics}
                  height={500}
                />
              </TabsContent>

              <TabsContent value="contributors">
                <div className="space-y-4">
                  <Alert>
                    <AlertDescription>
                      Showing top {topN} alarm contributors. Adjust the "Top Contributors (N)" 
                      setting in the sidebar to change the number of contributors displayed.
                    </AlertDescription>
                  </Alert>
                  
                  <ChartWrapper
                    data={topContributorsQuery.data}
                    title={`Top ${topN} Alarm Contributors`}
                    isLoading={topContributorsQuery.isLoading}
                    error={topContributorsQuery.error?.message}
                    metrics={topContributorsQuery.data?.metrics}
                    height={500}
                  />
                </div>
              </TabsContent>

              <TabsContent value="priority">
                <ChartWrapper
                  data={priorityDistributionQuery.data}
                  title="Alarm Priority Distribution"
                  isLoading={priorityDistributionQuery.isLoading}
                  error={priorityDistributionQuery.error?.message}
                  metrics={priorityDistributionQuery.data?.metrics}
                  height={500}
                />
              </TabsContent>

              <TabsContent value="pareto">
                <div className="space-y-4">
                  <Alert>
                    <AlertDescription>
                      Pareto analysis identifies the critical few alarm sources that contribute 
                      to the majority of alarm activity (80/20 rule).
                    </AlertDescription>
                  </Alert>
                  
                  <ChartWrapper
                    data={paretoAnalysisQuery.data}
                    title="Pareto Analysis - Alarm Contributors"
                    isLoading={paretoAnalysisQuery.isLoading}
                    error={paretoAnalysisQuery.error?.message}
                    metrics={paretoAnalysisQuery.data?.metrics}
                    height={500}
                  />
                </div>
              </TabsContent>

              <TabsContent value="flood">
                <div className="space-y-4">
                  <Alert>
                    <AlertDescription>
                      Flood analysis identifies periods when alarm rates exceed manageable levels 
                      (typically &gt;10 alarms per 10 minutes).
                    </AlertDescription>
                  </Alert>
                  
                  <ChartWrapper
                    data={floodAnalysisQuery.data}
                    title="Alarm Flood Analysis"
                    isLoading={floodAnalysisQuery.isLoading}
                    error={floodAnalysisQuery.error?.message}
                    metrics={floodAnalysisQuery.data?.metrics}
                    height={500}
                  />
                </div>
              </TabsContent>

              <TabsContent value="histogram">
                <ChartWrapper
                  data={histogramQuery.data}
                  title="Alarm Rate Distribution Histogram"
                  isLoading={histogramQuery.isLoading}
                  error={histogramQuery.error?.message}
                  metrics={histogramQuery.data?.metrics}
                  height={500}
                />
              </TabsContent>

              <TabsContent value="rolling">
                <div className="space-y-4">
                  <Alert>
                    <AlertDescription>
                      Rolling average chart shows alarm rate trends over a {windowHours}-hour 
                      moving window. Adjust the "Rolling Window" setting to change the time period.
                    </AlertDescription>
                  </Alert>
                  
                  <ChartWrapper
                    data={rollingAverageQuery.data}
                    title={`${windowHours}-Hour Rolling Average Alarm Rate`}
                    isLoading={rollingAverageQuery.isLoading}
                    error={rollingAverageQuery.error?.message}
                    metrics={rollingAverageQuery.data?.metrics}
                    height={500}
                  />
                </div>
              </TabsContent>
            </Tabs>
          )}
        </div>
      </div>
    </div>
  );
}