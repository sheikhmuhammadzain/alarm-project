import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { FileSelector } from '@/components/common/FileSelector';
import { Skeleton } from '@/components/ui/skeleton';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  Lightbulb, 
  CheckCircle, 
  AlertTriangle, 
  XCircle,
  Download,
  FileText,
  TrendingUp
} from 'lucide-react';
import { getInsights, exportMetrics } from '@/api/client';
import { useAppStore } from '@/store/app-store';
import { toast } from '@/hooks/use-toast';
import type { FileInfo, InsightItem } from '@/api/types';
import { cn } from '@/lib/utils';

export default function Insights() {
  const { selectedFile, setSelectedFile } = useAppStore();
  const [isExporting, setIsExporting] = useState(false);

  const handleFileSelect = (file: FileInfo | null) => {
    setSelectedFile(file);
  };

  // Fetch insights for selected file
  const insightsQuery = useQuery({
    queryKey: ['insights', selectedFile?.file_id],
    queryFn: () => selectedFile ? getInsights(selectedFile.file_id) : null,
    enabled: !!selectedFile,
  });

  const getInsightIcon = (type: string) => {
    switch (type) {
      case 'success':
        return <CheckCircle className="h-5 w-5 text-status-good" />;
      case 'warning':
        return <AlertTriangle className="h-5 w-5 text-status-acceptable" />;
      case 'error':
        return <XCircle className="h-5 w-5 text-status-poor" />;
      default:
        return <Lightbulb className="h-5 w-5 text-isa-primary" />;
    }
  };

  const getInsightBadgeVariant = (type: string) => {
    switch (type) {
      case 'success':
        return 'default';
      case 'warning':
        return 'secondary';
      case 'error':
        return 'destructive';
      default:
        return 'outline';
    }
  };

  const handleExport = async (format: 'json' | 'csv') => {
    if (!selectedFile) return;
    
    setIsExporting(true);
    try {
      const data = await exportMetrics(selectedFile.file_id, format);
      
      // Create and trigger download
      const blob = new Blob(
        [format === 'json' ? JSON.stringify(data, null, 2) : data], 
        { type: format === 'json' ? 'application/json' : 'text/csv' }
      );
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${selectedFile.filename.replace(/\.[^/.]+$/, '')}_metrics.${format}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      
      toast({
        title: 'Export Successful',
        description: `Metrics exported as ${format.toUpperCase()} file`,
      });
    } catch (error: any) {
      toast({
        title: 'Export Failed',
        description: error.response?.data?.detail || 'Failed to export metrics',
        variant: 'destructive',
      });
    } finally {
      setIsExporting(false);
    }
  };

  const groupedInsights = insightsQuery.data?.insights.reduce((acc, insight) => {
    if (!acc[insight.type]) {
      acc[insight.type] = [];
    }
    acc[insight.type].push(insight);
    return acc;
  }, {} as Record<string, InsightItem[]>);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-foreground">Insights & Recommendations</h1>
        <p className="text-muted-foreground">
          AI-powered insights and actionable recommendations for alarm system optimization
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-4">
        {/* File Selection Sidebar */}
        <div className="lg:col-span-1">
          <Card className="sticky top-6">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Lightbulb className="h-5 w-5" />
                Analysis Target
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <FileSelector 
                onFileSelect={handleFileSelect}
                allowEmpty={false}
              />

              {selectedFile && (
                <div className="space-y-3 pt-4 border-t border-border">
                  <h4 className="text-sm font-semibold">Export Options</h4>
                  <div className="space-y-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handleExport('json')}
                      disabled={isExporting}
                      className="w-full justify-start"
                    >
                      <Download className="h-4 w-4 mr-2" />
                      Export JSON
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handleExport('csv')}
                      disabled={isExporting}
                      className="w-full justify-start"
                    >
                      <FileText className="h-4 w-4 mr-2" />
                      Export CSV
                    </Button>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Insights Area */}
        <div className="lg:col-span-3 space-y-6">
          {!selectedFile ? (
            <Card className="h-96">
              <CardContent className="flex items-center justify-center h-full">
                <div className="text-center text-muted-foreground">
                  <Lightbulb className="mx-auto h-12 w-12 opacity-50 mb-4" />
                  <h3 className="text-lg font-semibold mb-2">No File Selected</h3>
                  <p>Select a file from the sidebar to view insights and recommendations</p>
                </div>
              </CardContent>
            </Card>
          ) : insightsQuery.isLoading ? (
            <div className="space-y-4">
              <Skeleton className="h-32 w-full" />
              {Array.from({ length: 6 }).map((_, i) => (
                <Skeleton key={i} className="h-24 w-full" />
              ))}
            </div>
          ) : insightsQuery.error ? (
            <Alert variant="destructive">
              <AlertTriangle className="h-4 w-4" />
              <AlertDescription>
                Failed to load insights: {insightsQuery.error.message}
              </AlertDescription>
            </Alert>
          ) : insightsQuery.data ? (
            <>
              {/* Summary */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <TrendingUp className="h-5 w-5" />
                    Analysis Summary
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm leading-relaxed">{insightsQuery.data.summary}</p>
                  
                  {/* Quick Stats */}
                  <div className="grid gap-4 md:grid-cols-3 mt-4 pt-4 border-t border-border">
                    <div className="text-center p-3 rounded-lg bg-status-good/10 border border-status-good/20">
                      <div className="text-2xl font-bold text-status-good">
                        {groupedInsights?.success?.length || 0}
                      </div>
                      <div className="text-sm text-muted-foreground">Good Practices</div>
                    </div>
                    <div className="text-center p-3 rounded-lg bg-status-acceptable/10 border border-status-acceptable/20">
                      <div className="text-2xl font-bold text-status-acceptable">
                        {groupedInsights?.warning?.length || 0}
                      </div>
                      <div className="text-sm text-muted-foreground">Warnings</div>
                    </div>
                    <div className="text-center p-3 rounded-lg bg-status-poor/10 border border-status-poor/20">
                      <div className="text-2xl font-bold text-status-poor">
                        {groupedInsights?.error?.length || 0}
                      </div>
                      <div className="text-sm text-muted-foreground">Critical Issues</div>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Insights by Category */}
              {Object.entries(groupedInsights || {}).map(([type, insights]) => (
                <Card key={type}>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2 capitalize">
                      {getInsightIcon(type)}
                      {type === 'success' ? 'Good Practices' : 
                       type === 'warning' ? 'Warnings & Recommendations' : 
                       'Critical Issues'}
                      <Badge variant={getInsightBadgeVariant(type)} className="ml-2">
                        {insights.length}
                      </Badge>
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      {insights.map((insight, index) => (
                        <div
                          key={index}
                          className={cn(
                            'p-4 rounded-lg border',
                            type === 'success' && 'border-status-good/20 bg-status-good/5',
                            type === 'warning' && 'border-status-acceptable/20 bg-status-acceptable/5',
                            type === 'error' && 'border-status-poor/20 bg-status-poor/5'
                          )}
                        >
                          <div className="flex items-start gap-3">
                            {getInsightIcon(type)}
                            <div className="flex-1 space-y-2">
                              <p className="font-medium text-foreground">{insight.message}</p>
                              
                              {insight.details && (
                                <p className="text-sm text-muted-foreground">{insight.details}</p>
                              )}
                              
                              {insight.recommendation && (
                                <div className="p-3 rounded-md bg-muted/50 border border-border">
                                  <p className="text-sm">
                                    <strong>Recommendation:</strong> {insight.recommendation}
                                  </p>
                                </div>
                              )}
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              ))}
            </>
          ) : (
            <Card className="h-96">
              <CardContent className="flex items-center justify-center h-full">
                <div className="text-center text-muted-foreground">
                  <Lightbulb className="mx-auto h-12 w-12 opacity-50 mb-4" />
                  <p>No insights available for this file</p>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}