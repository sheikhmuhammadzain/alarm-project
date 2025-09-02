import { useQuery } from '@tanstack/react-query';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { MetricCard } from '@/components/common/MetricCard';
import { ChartWrapper } from '@/components/common/ChartWrapper';
import { Skeleton } from '@/components/ui/skeleton';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  Activity, 
  FileText, 
  TrendingUp, 
  AlertTriangle,
  CheckCircle,
  Clock
} from 'lucide-react';
import { getDashboardSummary, getPerformanceComparisonChart } from '@/api/client';
import { cn } from '@/lib/utils';

export default function Dashboard() {
  // Fetch dashboard summary
  const { data: summary, isLoading: summaryLoading, error: summaryError } = useQuery({
    queryKey: ['dashboard-summary'],
    queryFn: getDashboardSummary,
    refetchInterval: 30000, // Refresh every 30 seconds
  });

  // Fetch performance comparison chart
  const { data: chartData, isLoading: chartLoading, error: chartError } = useQuery({
    queryKey: ['performance-comparison-chart'],
    queryFn: getPerformanceComparisonChart,
  });

  if (summaryError) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold text-foreground">Dashboard</h1>
          <p className="text-muted-foreground">
            Overview of your alarm analysis system
          </p>
        </div>
        
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>
            Failed to load dashboard data. Please check your connection to the ISA Analyzer API.
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-foreground">Dashboard</h1>
          <p className="text-muted-foreground">
            Overview of your alarm analysis system performance
          </p>
        </div>
        
        {summary && (
          <div className="text-right text-sm text-muted-foreground">
            <p>Last updated: {new Date().toLocaleTimeString()}</p>
          </div>
        )}
      </div>

      {/* Key Metrics */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {summaryLoading ? (
          Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} className="h-32" />
          ))
        ) : summary ? (
          <>
            <MetricCard
              title="Total Files"
              value={summary.total_files}
              subtitle="Available for analysis"
              icon={FileText}
              status="neutral"
            />
            
            <MetricCard
              title="Total Alarms"
              value={summary.total_alarms}
              subtitle="Across all sources"
              icon={Activity}
              status="neutral"
            />
            
            <MetricCard
              title="Average Alarm Rate"
              value={`${summary.avg_alarm_rate.toFixed(1)}/hr`}
              subtitle="System-wide performance"
              icon={TrendingUp}
              status={
                summary.avg_alarm_rate <= 144 ? 'good' :
                summary.avg_alarm_rate <= 288 ? 'acceptable' : 'poor'
              }
            />
            
            <MetricCard
              title="Flood Percentage"
              value={`${summary.flood_percentage.toFixed(1)}%`}
              subtitle="Flood conditions detected"
              icon={AlertTriangle}
              status={
                summary.flood_percentage <= 1 ? 'good' :
                summary.flood_percentage <= 5 ? 'acceptable' : 'poor'
              }
            />
          </>
        ) : (
          <div className="col-span-full text-center py-8">
            <p className="text-muted-foreground">No summary data available</p>
          </div>
        )}
      </div>

      {/* Performance Breakdown */}
      {summary && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5" />
              Performance Breakdown
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4 md:grid-cols-3">
              <div className="flex items-center gap-3 p-4 rounded-lg bg-status-good/10 border border-status-good/20">
                <CheckCircle className="h-8 w-8 text-status-good" />
                <div>
                  <div className="text-2xl font-bold text-foreground">
                    {summary.performance_breakdown.good}
                  </div>
                  <div className="text-sm text-muted-foreground">Good Performance</div>
                </div>
              </div>
              
              <div className="flex items-center gap-3 p-4 rounded-lg bg-status-acceptable/10 border border-status-acceptable/20">
                <Clock className="h-8 w-8 text-status-acceptable" />
                <div>
                  <div className="text-2xl font-bold text-foreground">
                    {summary.performance_breakdown.acceptable}
                  </div>
                  <div className="text-sm text-muted-foreground">Acceptable Performance</div>
                </div>
              </div>
              
              <div className="flex items-center gap-3 p-4 rounded-lg bg-status-poor/10 border border-status-poor/20">
                <AlertTriangle className="h-8 w-8 text-status-poor" />
                <div>
                  <div className="text-2xl font-bold text-foreground">
                    {summary.performance_breakdown.poor}
                  </div>
                  <div className="text-sm text-muted-foreground">Poor Performance</div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Performance Comparison Charts */}
      <div className="grid gap-6 lg:grid-cols-2">
        <ChartWrapper
          data={chartData?.alarm_rate_chart || chartData}
          title="Alarm Rate Performance Comparison"
          isLoading={chartLoading}
          error={chartError?.message}
          height={500}
          className="col-span-1"
        />
        <ChartWrapper
          data={chartData?.flood_rate_chart || chartData}
          title="Flood Performance Comparison"
          isLoading={chartLoading}
          error={chartError?.message}
          height={500}
          className="col-span-1"
        />
      </div>

      {/* Recent Activity */}
      {summary?.recent_activity && summary.recent_activity.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Recent Activity</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {summary.recent_activity.slice(0, 5).map((activity, index) => (
                <div key={index} className="flex items-center justify-between p-3 rounded-lg border border-border bg-muted/30">
                  <div className="flex items-center gap-3">
                    <div className="w-2 h-2 rounded-full bg-isa-primary" />
                    <div>
                      <p className="font-medium">{activity.filename}</p>
                      <p className="text-sm text-muted-foreground">
                        {activity.action}
                      </p>
                    </div>
                  </div>
                  <div className="text-sm text-muted-foreground">
                    {new Date(activity.timestamp).toLocaleString()}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}