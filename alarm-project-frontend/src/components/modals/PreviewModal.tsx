import { useQuery } from '@tanstack/react-query';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { DataTable, Column } from '@/components/common/DataTable';
import { Eye, FileText, BarChart } from 'lucide-react';
import { getFilePreview, getFileStatistics } from '@/api/client';
import type { FileInfo } from '@/api/types';
import { cn } from '@/lib/utils';

interface PreviewModalProps {
  file: FileInfo | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function PreviewModal({ file, open, onOpenChange }: PreviewModalProps) {
  // Fetch preview data
  const { data: previewData, isLoading: previewLoading, error: previewError } = useQuery({
    queryKey: ['file-preview', file?.file_id],
    queryFn: () => file ? getFilePreview(file.file_id, 10) : null,
    enabled: !!file && open,
  });

  // Fetch statistics
  const { data: statsData, isLoading: statsLoading, error: statsError } = useQuery({
    queryKey: ['file-statistics', file?.file_id],
    queryFn: () => file ? getFileStatistics(file.file_id) : null,
    enabled: !!file && open,
  });

  if (!file) return null;

  // Create columns for preview table
  const previewColumns: Column[] = (previewData?.headers || []).map(header => ({
    key: header,
    label: header,
    render: (value) => (
      <span className={cn(
        'text-sm',
        value === null || value === undefined ? 'text-muted-foreground' : ''
      )}>
        {value === null || value === undefined ? '—' : String(value)}
      </span>
    )
  }));

  // Create columns for statistics table
  const statsColumns: Column[] = [
    { key: 'name', label: 'Column Name', width: '1/3' },
    { key: 'type', label: 'Data Type', width: '1/6' },
    { 
      key: 'unique_values', 
      label: 'Unique Values', 
      width: '1/6',
      render: (value) => value.toLocaleString()
    },
    { 
      key: 'null_count', 
      label: 'Null Count', 
      width: '1/6',
      render: (value, row) => (
        <div className="space-y-1">
          <span>{value.toLocaleString()}</span>
          <div className="text-xs text-muted-foreground">
            ({((value / (statsData?.total_rows || 1)) * 100).toFixed(1)}%)
          </div>
        </div>
      )
    },
  ];

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-6xl max-h-[80vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Eye className="h-5 w-5" />
            File Preview: {file.filename}
          </DialogTitle>
          <DialogDescription>
            Explore the structure and content of your alarm data file
          </DialogDescription>
        </DialogHeader>

        {/* File Info Banner */}
        <div className="flex items-center justify-between rounded-lg border border-border bg-muted/30 p-4">
          <div className="space-y-1">
            <div className="flex items-center gap-3">
              <h3 className="font-semibold">{file.filename}</h3>
              <Badge variant={file.source_type === 'uploaded' ? 'default' : 'secondary'}>
                {file.source_type}
              </Badge>
            </div>
            <div className="flex items-center gap-4 text-sm text-muted-foreground">
              <span>Size: {file.size_mb.toFixed(1)} MB</span>
              {file.folder && <span>Folder: {file.folder}</span>}
              {statsData?.total_rows && (
                <span>Rows: {statsData.total_rows.toLocaleString()}</span>
              )}
            </div>
          </div>
          {statsData?.date_range && (
            <div className="text-right text-sm">
              <div className="font-medium">Data Range</div>
              <div className="text-muted-foreground">
                {new Date(statsData.date_range.start).toLocaleDateString()} -
                {new Date(statsData.date_range.end).toLocaleDateString()}
              </div>
            </div>
          )}
        </div>

        <Tabs defaultValue="preview" className="flex-1 overflow-hidden">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="preview" className="flex items-center gap-2">
              <FileText className="h-4 w-4" />
              Data Preview
            </TabsTrigger>
            <TabsTrigger value="statistics" className="flex items-center gap-2">
              <BarChart className="h-4 w-4" />
              Column Statistics
            </TabsTrigger>
          </TabsList>

          <TabsContent value="preview" className="flex-1 overflow-hidden mt-4">
            {previewLoading && (
              <div className="space-y-4">
                <Skeleton className="h-8 w-full" />
                {Array.from({ length: 5 }).map((_, i) => (
                  <Skeleton key={i} className="h-12 w-full" />
                ))}
              </div>
            )}

            {previewError && (
              <Alert variant="destructive">
                <AlertDescription>
                  Failed to load file preview: {previewError.message}
                </AlertDescription>
              </Alert>
            )}

            {previewData && (
              <div className="space-y-4 overflow-hidden">
                <div className="flex items-center justify-between">
                  <p className="text-sm text-muted-foreground">
                    Showing first 10 rows of {previewData.total_rows.toLocaleString()} total
                  </p>
                </div>
                
                <div className="overflow-auto max-h-96">
                  <DataTable
                    data={previewData.rows}
                    columns={previewColumns}
                    searchable={false}
                    sortable={false}
                    paginated={false}
                    emptyMessage="No preview data available"
                  />
                </div>
              </div>
            )}
          </TabsContent>

          <TabsContent value="statistics" className="flex-1 overflow-hidden mt-4">
            {statsLoading && (
              <div className="space-y-4">
                {Array.from({ length: 8 }).map((_, i) => (
                  <Skeleton key={i} className="h-16 w-full" />
                ))}
              </div>
            )}

            {statsError && (
              <Alert variant="destructive">
                <AlertDescription>
                  Failed to load file statistics: {statsError.message}
                </AlertDescription>
              </Alert>
            )}

            {statsData && (
              <div className="space-y-4 overflow-hidden">
                {/* Summary Stats */}
                <div className="grid grid-cols-3 gap-4">
                  <div className="rounded-lg border border-border bg-muted/30 p-3">
                    <div className="text-sm font-medium text-muted-foreground">Total Rows</div>
                    <div className="text-2xl font-bold">{statsData.total_rows.toLocaleString()}</div>
                  </div>
                  <div className="rounded-lg border border-border bg-muted/30 p-3">
                    <div className="text-sm font-medium text-muted-foreground">Columns</div>
                    <div className="text-2xl font-bold">{statsData.columns.length}</div>
                  </div>
                  <div className="rounded-lg border border-border bg-muted/30 p-3">
                    <div className="text-sm font-medium text-muted-foreground">Data Quality</div>
                    <div className="text-2xl font-bold text-status-good">
                      {(100 - (statsData.columns.reduce((sum, col) => sum + col.null_count, 0) / 
                        (statsData.total_rows * statsData.columns.length) * 100)).toFixed(1)}%
                    </div>
                  </div>
                </div>

                {/* Column Details */}
                <div className="overflow-auto max-h-96">
                  <DataTable
                    data={statsData.columns}
                    columns={statsColumns}
                    searchable={true}
                    sortable={true}
                    paginated={false}
                    emptyMessage="No column data available"
                  />
                </div>
              </div>
            )}
          </TabsContent>
        </Tabs>

        <div className="flex justify-end pt-4 border-t border-border">
          <Button onClick={() => onOpenChange(false)}>
            Close
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}