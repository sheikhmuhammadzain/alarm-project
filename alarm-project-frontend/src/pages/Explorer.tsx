import { useEffect, useMemo, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Skeleton } from '@/components/ui/skeleton';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { DataTable, Column } from '@/components/common/DataTable';
import { ChartWrapper } from '@/components/common/ChartWrapper';
import {
  getAutoFiles,
  getAutoFilesByFolder,
  getAllFiles,
  getFilePreview,
  getFileStatistics,
  analyzeFile,
  getAlarmRateChart,
  getTopContributorsChart,
  getPriorityDistributionChart,
  getParetoAnalysisChart,
  getFloodAnalysisChart,
  getDistributionHistogramChart,
  getRollingAverageChart,
  getInsights,
  exportMetrics,
  refreshAutoFiles,
  getChatteringAlarms,
  getStandingAlarms,
  deleteFile,
  deleteAllFiles,
} from '@/api/client';
import { FolderOpen, BarChart3, RefreshCw, Download, Lightbulb, Zap, Clock, ChevronLeft, ChevronRight, Upload, Trash2 } from 'lucide-react';
import { cn } from '@/lib/utils';
import { toast } from '@/hooks/use-toast';
import type { FileInfo, InsightItem, ChatteringAlarm, StandingAlarm } from '@/api/types';
import { UploadModal } from '@/components/modals/UploadModal';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog';

type AutoFile = {
  file_id: string;
  filename: string;
  size_mb?: number;
  modified_time?: string;
  extension?: string;
  folder?: string;
  source_type?: 'uploaded' | 'auto-discovered';
};

export default function Explorer() {
  const queryClient = useQueryClient();
  const [searchParams, setSearchParams] = useSearchParams();
  const [selectedFolder, setSelectedFolder] = useState<string>('');
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'data' | 'stats' | 'analysis' | 'insights' | 'advanced'>('data');
  const [visibleCols, setVisibleCols] = useState<string[] | null>(null);
  const [topN, setTopN] = useState<number>(15);
  const [exporting, setExporting] = useState<'csv' | 'json' | null>(null);
  const [windowMinutes, setWindowMinutes] = useState<number>(10);
  const [threshold, setThreshold] = useState<number>(5);
  const [minDurationHours, setMinDurationHours] = useState<number>(24);
  const [windowHours, setWindowHours] = useState<number>(24);
  const [filesSidebarOpen, setFilesSidebarOpen] = useState<boolean>(true);
  const [fileSearch, setFileSearch] = useState<string>('');
  const [uploadOpen, setUploadOpen] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [deleteAllDialogOpen, setDeleteAllDialogOpen] = useState(false);
  const [fileToDelete, setFileToDelete] = useState<FileInfo | null>(null);

  const { data: folders = [], error: foldersError, isLoading: foldersLoading } = useQuery({
    queryKey: ['auto-files'],
    queryFn: getAutoFiles,
  });

  const { data: folderFiles = [], isLoading: filesLoading } = useQuery<FileInfo[]>({
    queryKey: ['auto-folder', selectedFolder],
    // cast to any to align with API typing differences for auto files
    queryFn: () => (selectedFolder && selectedFolder !== '__uploaded__' ? getAutoFilesByFolder(selectedFolder) : Promise.resolve([] as FileInfo[])),
    enabled: !!selectedFolder && selectedFolder !== '__uploaded__',
  });

  // All files (for uploaded pseudo-folder and counters)
  const { data: allFiles = [], isLoading: allFilesLoading } = useQuery<FileInfo[]>({
    queryKey: ['all-files'],
    queryFn: getAllFiles,
  });

  const uploadedFiles = useMemo(() => allFiles.filter(f => f.source_type === 'uploaded'), [allFiles]);
  const uploadedFilesCount = uploadedFiles.length;

  // Determine which files to show in the sidebar based on selected folder
  const displayedFiles: FileInfo[] = useMemo(() => {
    if (selectedFolder === '__uploaded__') return uploadedFiles;
    return (folderFiles as FileInfo[]) || [];
  }, [selectedFolder, uploadedFiles, folderFiles]);

  const setActive = (id: string) => {
    setActiveId(id);
    setVisibleCols(null);
  };

  // Ensure a selected file becomes active automatically
  useEffect(() => {
    if (!activeId && selectedIds.length > 0) {
      setActiveId(selectedIds[0]);
      setVisibleCols(null);
    }
  }, [selectedIds, activeId]);

  const toggleSelect = (id: string, value: boolean) => {
    setSelectedIds((prev) => {
      const next = value ? Array.from(new Set([...prev, id])) : prev.filter((x) => x !== id);
      if (!activeId && next.length) setActiveId(next[0]);
      if (activeId && !next.includes(activeId)) setActiveId(next[0] || null);
      return next;
    });
  };

  // Minimal file columns kept only for type safety if DataTable is reused elsewhere
  const fileRowColumns: Column[] = [
    {
      key: 'select',
      label: 'Select',
      sortable: false,
      render: (_value, row: AutoFile) => (
        <Checkbox
          checked={selectedIds.includes(row.file_id)}
          onCheckedChange={(c) => toggleSelect(row.file_id, Boolean(c))}
        />
      ),
      width: '1/6',
    },
    { key: 'filename', label: 'File Name', width: '5/6' },
  ];

  const { data: preview, isLoading: previewLoading, error: previewError } = useQuery({
    queryKey: ['preview', activeId],
    queryFn: () => (activeId ? getFilePreview(activeId, 50) : null),
    enabled: !!activeId && activeTab === 'data',
  });

  const { data: stats, isLoading: statsLoading, error: statsError } = useQuery({
    queryKey: ['stats', activeId],
    queryFn: () => (activeId ? getFileStatistics(activeId) : null),
    enabled: !!activeId && activeTab === 'stats',
  });

  const { data: metrics, isLoading: metricsLoading, error: metricsError } = useQuery({
    queryKey: ['metrics', activeId],
    queryFn: () => (activeId ? analyzeFile(activeId) : null),
    enabled: !!activeId && activeTab === 'analysis',
  });

  // Charts for active file
  const SAMPLE = 15000; // cap rows for heavy charts to avoid timeouts

  const { data: rateChart, isLoading: rateLoading, error: rateError } = useQuery({
    queryKey: ['explorer-alarm-rate', activeId],
    queryFn: () => (activeId ? getAlarmRateChart(activeId, SAMPLE) : null),
    enabled: !!activeId && activeTab === 'analysis',
  });

  const { data: topChart, isLoading: topLoading, error: topError } = useQuery({
    queryKey: ['explorer-top', activeId, topN],
    queryFn: () => (activeId ? getTopContributorsChart(activeId, topN, SAMPLE) : null),
    enabled: !!activeId && activeTab === 'analysis',
  });

  // Additional charts
  const { data: priorityChart, isLoading: priorityLoading, error: priorityError } = useQuery({
    queryKey: ['explorer-priority', activeId],
    queryFn: () => (activeId ? getPriorityDistributionChart(activeId, SAMPLE) : null),
    enabled: !!activeId && activeTab === 'analysis',
  });

  const { data: paretoChart, isLoading: paretoLoading, error: paretoError } = useQuery({
    queryKey: ['explorer-pareto', activeId],
    queryFn: () => (activeId ? getParetoAnalysisChart(activeId, SAMPLE) : null),
    enabled: !!activeId && activeTab === 'analysis',
  });

  const { data: floodChart, isLoading: floodLoading, error: floodError } = useQuery({
    queryKey: ['explorer-flood', activeId],
    queryFn: () => (activeId ? getFloodAnalysisChart(activeId, SAMPLE) : null),
    enabled: !!activeId && activeTab === 'analysis',
  });

  // Extra charts
  const { data: histogramChart, isLoading: histLoading, error: histError } = useQuery({
    queryKey: ['explorer-histogram', activeId],
    queryFn: () => (activeId ? getDistributionHistogramChart(activeId) : null),
    enabled: !!activeId && activeTab === 'analysis',
  });

  const { data: rollingChart, isLoading: rollingLoading, error: rollingError } = useQuery({
    queryKey: ['explorer-rolling', activeId, windowHours],
    queryFn: () => (activeId ? getRollingAverageChart(activeId, windowHours) : null),
    enabled: !!activeId && activeTab === 'analysis',
  });

  // Insights
  const { data: insights, isLoading: insightsLoading, error: insightsError } = useQuery({
    queryKey: ['explorer-insights', activeId],
    queryFn: () => (activeId ? getInsights(activeId) : null),
    enabled: !!activeId && activeTab === 'insights',
  });

  // Advanced detections (active file)
  const { data: chatteringData, isLoading: chatteringLoading, error: chatteringError } = useQuery<ChatteringAlarm[] | null>({
    queryKey: ['explorer-chattering', activeId, windowMinutes, threshold],
    queryFn: () => (activeId ? getChatteringAlarms(activeId, windowMinutes, threshold) : null),
    enabled: !!activeId && activeTab === 'advanced',
  });

  const { data: standingData, isLoading: standingLoading, error: standingError } = useQuery<StandingAlarm[] | null>({
    queryKey: ['explorer-standing', activeId, minDurationHours],
    queryFn: () => (activeId ? getStandingAlarms(activeId, minDurationHours) : null),
    enabled: !!activeId && activeTab === 'advanced',
  });

  // Compare selected files (analyze each)
  const compareEnabled = selectedIds.length > 1;
  const { data: compareData, isLoading: compareLoading, error: compareError } = useQuery({
    queryKey: ['explorer-compare', selectedIds.join('|')],
    queryFn: async () => {
      const idToName: Record<string, string> = Object.fromEntries((displayedFiles as FileInfo[]).map(f => [f.file_id, f.filename]));
      const results = await Promise.all(selectedIds.map(async (id) => {
        const m = await analyzeFile(id);
        return { file_id: id, filename: idToName[id] || id, metrics: m };
      }));
      return results;
    },
    enabled: compareEnabled && activeTab === 'analysis',
  });

  const refreshMutation = useMutation({
    mutationFn: refreshAutoFiles,
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ['auto-files'] });
      await queryClient.invalidateQueries({ queryKey: ['auto-folder'] });
      await queryClient.invalidateQueries({ queryKey: ['all-files'] });
      toast({ title: 'Folders refreshed' });
    },
    onError: () => toast({ title: 'Failed to refresh', variant: 'destructive' }),
  });

  // Delete mutations
  const deleteMutation = useMutation({
    mutationFn: deleteFile,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['all-files'] });
      queryClient.invalidateQueries({ queryKey: ['auto-files'] });
      queryClient.invalidateQueries({ queryKey: ['auto-folder'] });
      toast({ title: 'File Deleted', description: fileToDelete?.filename });
      setFileToDelete(null);
      setDeleteDialogOpen(false);
    },
    onError: () => {
      toast({ title: 'Delete Failed', variant: 'destructive' });
    },
  });

  const deleteAllMutation = useMutation({
    mutationFn: deleteAllFiles,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['all-files'] });
      toast({ title: 'All uploaded files deleted' });
      setDeleteAllDialogOpen(false);
    },
    onError: () => toast({ title: 'Delete Failed', variant: 'destructive' }),
  });

  const downloadText = (filename: string, content: string, mime: string) => {
    const blob = new Blob([content], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  const handleExport = async (format: 'csv' | 'json') => {
    if (!activeId) return;
    try {
      setExporting(format);
      const res = await exportMetrics(activeId, format);
      if (format === 'csv') {
        const csv = String(res);
        downloadText(`metrics_${activeId}.csv`, csv, 'text/csv');
      } else {
        const jsonStr = JSON.stringify(res, null, 2);
        downloadText(`metrics_${activeId}.json`, jsonStr, 'application/json');
      }
    } catch (e) {
      toast({ title: 'Export failed', variant: 'destructive' });
    } finally {
      setExporting(null);
    }
  };

  const defaultCols = useMemo(() => preview?.headers ?? [], [preview?.headers]);
  const effectiveCols = useMemo(() => visibleCols ?? defaultCols, [visibleCols, defaultCols]);

  const previewColumns: Column[] = effectiveCols.map((h) => ({
    key: h,
    label: h,
    sortable: false,
  }));

  // Initialize state from URL
  useEffect(() => {
    const folder = searchParams.get('folder');
    const sel = searchParams.get('sel');
    const active = searchParams.get('active');
    const tab = searchParams.get('tab') as typeof activeTab | null;
    const top = searchParams.get('top');
    const rhours = searchParams.get('rhours');
    const win = searchParams.get('win');
    const thr = searchParams.get('thr');
    const minhrs = searchParams.get('minhrs');
    const q = searchParams.get('q');
    if (folder) setSelectedFolder(folder);
    if (sel) setSelectedIds(sel.split(',').filter(Boolean));
    if (active) setActiveId(active);
    if (tab && ['data','stats','analysis','insights','advanced'].includes(tab)) setActiveTab(tab as typeof activeTab);
    if (top) setTopN(Number(top));
    if (rhours) setWindowHours(Number(rhours));
    if (win) setWindowMinutes(Number(win));
    if (thr) setThreshold(Number(thr));
    if (minhrs) setMinDurationHours(Number(minhrs));
    if (q) setFileSearch(q);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Sync state to URL
  useEffect(() => {
    const next = new URLSearchParams(searchParams);
    if (selectedFolder) next.set('folder', selectedFolder); else next.delete('folder');
    if (selectedIds.length) next.set('sel', selectedIds.join(',')); else next.delete('sel');
    if (activeId) next.set('active', activeId); else next.delete('active');
    if (activeTab) next.set('tab', activeTab);
    next.set('top', String(topN));
    next.set('rhours', String(windowHours));
    next.set('win', String(windowMinutes));
    next.set('thr', String(threshold));
    next.set('minhrs', String(minDurationHours));
    if (fileSearch) next.set('q', fileSearch); else next.delete('q');
    setSearchParams(next, { replace: true });
  }, [selectedFolder, selectedIds, activeId, activeTab, topN, windowHours, windowMinutes, threshold, minDurationHours, fileSearch, setSearchParams, searchParams]);

  // Auto-select the first folder for a faster start (if none in URL)
  useEffect(() => {
    if (!selectedFolder && folders && folders.length > 0) {
      setSelectedFolder(folders[0].folder_name);
    }
  }, [folders, selectedFolder]);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-foreground">Explorer</h1>
          <p className="text-muted-foreground">Select a folder, multi-select files, and view data/metrics in one place.</p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={() => refreshMutation.mutate()} disabled={refreshMutation.isPending}>
            <RefreshCw className={cn('mr-2 h-4 w-4', refreshMutation.isPending && 'animate-spin')} /> Refresh
          </Button>
          <Button size="sm" onClick={() => setUploadOpen(true)}>
            <Upload className="mr-2 h-4 w-4" /> Upload
          </Button>
          {uploadedFilesCount > 0 && (
            <Button
              variant="outline"
              size="sm"
              onClick={() => setDeleteAllDialogOpen(true)}
              disabled={deleteAllMutation.isPending}
            >
              <Trash2 className="mr-2 h-4 w-4" /> Delete All Uploaded
            </Button>
          )}
        </div>
      </div>

      {/* Quick folder buttons */}
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2 overflow-x-auto">
          {foldersLoading ? (
            Array.from({ length: 4 }).map((_, i) => (
              <Skeleton key={i} className="h-8 w-28" />
            ))
          ) : (
            <>
          <Button
            key="__uploaded__"
            size="sm"
            variant={selectedFolder === '__uploaded__' ? 'default' : 'outline'}
            onClick={() => {
              setSelectedFolder('__uploaded__');
              setSelectedIds([]);
              setActiveId(null);
            }}
          >
            Uploaded ({uploadedFilesCount})
          </Button>
          {folders.map((f) => (
            <Button
              key={f.folder_name}
              size="sm"
              variant={selectedFolder === f.folder_name ? 'default' : 'outline'}
              onClick={() => {
                setSelectedFolder(f.folder_name);
                setSelectedIds([]);
                setActiveId(null);
              }}
            >
              {f.folder_name}
            </Button>
          ))}
          </>
          )}
        </div>
        {/* removed explicit toggle from here per request */}
      </div>

      {foldersError && (
        <Alert variant="destructive">
          <AlertDescription>
            {foldersError instanceof Error ? foldersError.message : 'Failed to load folders'}
          </AlertDescription>
        </Alert>
      )}

      <div className="flex gap-6">
        {/* Files sidebar */}
        {filesSidebarOpen && (
          <aside className="w-72 shrink-0 space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <FolderOpen className="h-5 w-5 text-isa-primary" />
                <h3 className="font-semibold">{selectedFolder || '—'}</h3>
              </div>
              <Button variant="ghost" size="icon" onClick={() => setFilesSidebarOpen(false)} title="Hide files">
                <ChevronLeft className="h-4 w-4" />
              </Button>
            </div>
            {(selectedFolder !== '__uploaded__' && filesLoading) || (selectedFolder === '__uploaded__' && allFilesLoading) ? (
              <div className="space-y-3">
                {Array.from({ length: 6 }).map((_, i) => (
                  <Skeleton key={i} className="h-10 w-full" />
                ))}
              </div>
            ) : (
              <div className="space-y-2">
                <Input
                  placeholder="Search files..."
                  value={fileSearch}
                  onChange={(e) => setFileSearch(e.target.value)}
                />
                <div className="flex items-center gap-2 text-xs text-muted-foreground">
                  <Button size="sm" variant="ghost" onClick={() => setSelectedIds(displayedFiles.map(f => f.file_id))}>Select All</Button>
                  <Button size="sm" variant="ghost" onClick={() => setSelectedIds([])}>Clear</Button>
                  <Button size="sm" variant="ghost" onClick={() => setSelectedIds((prev) => displayedFiles.map(f => f.file_id).filter(id => !prev.includes(id)).concat(prev.filter(id => !displayedFiles.map(f => f.file_id).includes(id))))}>Invert</Button>
                </div>
                <div className="max-h-[70vh] overflow-auto rounded-md border border-border">
                  <ul className="divide-y">
                    {(displayedFiles as FileInfo[])
                      .filter((f) => f.filename.toLowerCase().includes(fileSearch.toLowerCase()))
                      .map((file) => {
                        const selected = selectedIds.includes(file.file_id);
                        const active = activeId === file.file_id;
                        return (
                          <li key={file.file_id} className={cn('px-2 py-2 flex items-center gap-2 cursor-pointer hover:bg-muted/50', active && 'bg-muted')}
                              onClick={() => toggleSelect(file.file_id, !selected)}>
                            <Checkbox checked={selected} onCheckedChange={(c) => toggleSelect(file.file_id, Boolean(c))} />
                            <span className="truncate text-sm flex-1" title={file.filename}>{file.filename}</span>
                            {file.source_type === 'uploaded' && (
                              <Button
                                variant="ghost"
                                size="icon"
                                title="Delete uploaded file"
                                onClick={(e) => { e.stopPropagation(); setFileToDelete(file); setDeleteDialogOpen(true); }}
                              >
                                <Trash2 className="h-4 w-4 text-destructive" />
                              </Button>
                            )}
                          </li>
                        );
                      })}
                    {selectedFolder && (displayedFiles as FileInfo[]).length === 0 && (
                      <li className="px-3 py-4 text-sm text-muted-foreground">No files in this folder</li>
                    )}
                    {!selectedFolder && (
                      <li className="px-3 py-4 text-sm text-muted-foreground">Select a folder</li>
                    )}
                  </ul>
                </div>
              </div>
            )}
          </aside>
        )}

        {!filesSidebarOpen && (
          <div className="pt-1">
            <Button variant="ghost" size="icon" onClick={() => setFilesSidebarOpen(true)} title="Show files">
              <ChevronRight className="h-4 w-4" />
            </Button>
          </div>
        )}

        {/* Main content */}
        <div className="flex-1 space-y-4">
          <div className="rounded-lg border border-border bg-card p-4">
            <div className="flex items-center justify-between">
              <h3 className="font-semibold">Selected files ({selectedIds.length})</h3>
            </div>
            <div className="mt-3 flex flex-wrap gap-2">
              {selectedIds.length === 0 ? (
                <span className="text-sm text-muted-foreground">Select one or more files from the left.</span>
              ) : (
                selectedIds.map((id) => {
                  const file = (displayedFiles as FileInfo[]).find((f) => f.file_id === id);
                  return (
                    <Button
                      key={id}
                      size="sm"
                      variant={activeId === id ? 'default' : 'outline'}
                      onClick={() => setActive(id)}
                    >
                      {file?.filename || id}
                    </Button>
                  );
                })
              )}
            </div>
          </div>

          {activeId && (
            <div className="rounded-lg border border-border bg-card p-4 space-y-4">
              <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as typeof activeTab)}>
                <TabsList className="grid grid-cols-5 w-full">
                  <TabsTrigger value="data" className="flex items-center gap-2">
                    Data
                  </TabsTrigger>
                  <TabsTrigger value="stats">Statistics</TabsTrigger>
                  <TabsTrigger value="analysis" className="flex items-center gap-2">
                    <BarChart3 className="h-4 w-4" /> Analysis
                  </TabsTrigger>
                  <TabsTrigger value="insights" className="flex items-center gap-2">
                    <Lightbulb className="h-4 w-4" /> Insights
                  </TabsTrigger>
                  <TabsTrigger value="advanced" className="flex items-center gap-2">
                    <Zap className="h-4 w-4" /> Advanced
                  </TabsTrigger>
                </TabsList>

                <TabsContent value="data" className="mt-4 space-y-3">
                  {previewLoading && <Skeleton className="h-48 w-full" />}
                  {previewError && (
                    <Alert variant="destructive">
                      <AlertDescription>
                        {previewError instanceof Error ? previewError.message : 'Failed to load preview'}
                      </AlertDescription>
                    </Alert>
                  )}
                  {preview && (
                    <>
                      <div className="rounded-md border border-border p-3">
                        <div className="flex items-center justify-between">
                          <span className="text-sm font-medium">Columns</span>
                          <div className="flex items-center gap-2">
                            <Button variant="outline" size="sm" onClick={() => setVisibleCols(defaultCols)}>
                              Select All
                            </Button>
                            <Button variant="outline" size="sm" onClick={() => setVisibleCols([])}>
                              Clear
                            </Button>
                          </div>
                        </div>
                        <div className="mt-2 flex flex-wrap gap-3">
                          {defaultCols.map((h) => {
                            const checked = (visibleCols ?? defaultCols).includes(h);
                            return (
                              <label key={h} className={cn('flex items-center gap-2 text-sm')}>
                                <Checkbox
                                  checked={checked}
                                  onCheckedChange={(c) => {
                                    const on = Boolean(c);
                                    setVisibleCols((prev) => {
                                      const base = prev ?? defaultCols;
                                      return on ? Array.from(new Set([...base, h])) : base.filter((x) => x !== h);
                                    });
                                  }}
                                />
                                <span>{h}</span>
                              </label>
                            );
                          })}
                        </div>
                      </div>

                      <div className="overflow-auto">
                        <DataTable
                          data={preview.rows}
                          columns={previewColumns}
                          searchable={true}
                          sortable={false}
                          paginated={true}
                          pageSize={20}
                          emptyMessage="No preview data"
                        />
                      </div>
                      <div className="text-xs text-muted-foreground">
                        Showing up to 50 rows of {preview.total_rows.toLocaleString()} total.
                      </div>
                    </>
                  )}
                </TabsContent>

                <TabsContent value="stats" className="mt-4">
                  {statsLoading && <Skeleton className="h-48 w-full" />}
                  {statsError && (
                    <Alert variant="destructive">
                      <AlertDescription>
                        {statsError instanceof Error ? statsError.message : 'Failed to load statistics'}
                      </AlertDescription>
                    </Alert>
                  )}
                  {stats && (
                    <div className="space-y-4">
                      <div className="grid grid-cols-3 gap-3">
                        <div className="rounded-lg border border-border bg-muted/30 p-3">
                          <div className="text-sm text-muted-foreground">Total Rows</div>
                          <div className="text-xl font-bold">
                            {stats.total_rows?.toLocaleString?.() ?? '—'}
                          </div>
                        </div>
                        <div className="rounded-lg border border-border bg-muted/30 p-3">
                          <div className="text-sm text-muted-foreground">Columns</div>
                          <div className="text-xl font-bold">{stats.columns?.length ?? '—'}</div>
                        </div>
                      </div>
                      <div className="overflow-auto max-h-96">
                        <DataTable
                          data={stats.columns || []}
                          columns={[
                            { key: 'name', label: 'Column' },
                            { key: 'type', label: 'Type' },
                            { key: 'unique_values', label: 'Unique' },
                            { key: 'null_count', label: 'Nulls' },
                          ]}
                          paginated={false}
                          searchable={true}
                          sortable={true}
                          emptyMessage="No stats"
                        />
                      </div>
                    </div>
                  )}
                </TabsContent>

                <TabsContent value="analysis" className="mt-4 space-y-4">
                  {metricsLoading && <Skeleton className="h-40 w-full" />}
                  {metricsError && (
                    <Alert variant="destructive">
                      <AlertDescription>
                        {metricsError instanceof Error ? metricsError.message : 'Failed to load analysis'}
                      </AlertDescription>
                    </Alert>
                  )}
                  {metrics && (
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                      <div className="rounded-lg border border-border bg-muted/30 p-3">
                        <div className="text-sm text-muted-foreground">Total Alarms</div>
                        <div className="text-2xl font-bold">
                          {metrics.total_alarms?.toLocaleString?.() ?? 0}
                        </div>
                      </div>
                      <div className="rounded-lg border border-border bg-muted/30 p-3">
                        <div className="text-sm text-muted-foreground">Avg/hr</div>
                        <div className="text-2xl font-bold">{metrics.avg_alarms_per_hour?.toFixed?.(1) ?? '0.0'}</div>
                      </div>
                      <div className="rounded-lg border border-border bg-muted/30 p-3">
                        <div className="text-sm text-muted-foreground">Flood %</div>
                        <div className="text-2xl font-bold">{metrics.flood_percentage?.toFixed?.(1) ?? '0.0'}%</div>
                      </div>
                    </div>
                  )}

                  {/* Small controls for charts */}
                  <div className="grid gap-3 md:grid-cols-2">
                    <div className="space-y-2">
                      <Label htmlFor="top-n">Top Contributors (N)</Label>
                      <Input id="top-n" type="number" min={5} max={50} value={topN} onChange={(e) => setTopN(Number(e.target.value))} />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="rolling-hours">Rolling Average Window (hours)</Label>
                      <Input id="rolling-hours" type="number" min={1} max={168} value={windowHours} onChange={(e) => setWindowHours(Number(e.target.value))} />
                    </div>
                  </div>

                  {/* Inline charts: auto-fit to 1 or 2 per row based on available width */}
                  <div className="grid gap-4 grid-cols-[repeat(auto-fit,minmax(540px,1fr))]">
                    <ChartWrapper
                      data={rateChart || undefined}
                      title="Alarm Rate Over Time"
                      isLoading={rateLoading}
                      error={rateError instanceof Error ? rateError.message : null}
                      className="min-w-0"
                      height={360}
                    />
                    <ChartWrapper
                      data={topChart || undefined}
                      title={`Top ${topN} Alarm Contributors`}
                      isLoading={topLoading}
                      error={topError instanceof Error ? topError.message : null}
                      className="min-w-0"
                      height={360}
                    />
                    <ChartWrapper
                      data={priorityChart || undefined}
                      title="Priority Distribution"
                      isLoading={priorityLoading}
                      error={priorityError instanceof Error ? priorityError.message : null}
                      className="min-w-0"
                      height={360}
                    />
                    <ChartWrapper
                      data={paretoChart || undefined}
                      title="Pareto Analysis"
                      isLoading={paretoLoading}
                      error={paretoError instanceof Error ? paretoError.message : null}
                      className="min-w-0"
                      height={360}
                    />
                    <ChartWrapper
                      data={floodChart || undefined}
                      title="Flood Analysis (10‑min windows)"
                      isLoading={floodLoading}
                      error={floodError instanceof Error ? floodError.message : null}
                      className="min-w-0"
                      height={360}
                    />
                    <ChartWrapper
                      data={histogramChart || undefined}
                      title="Alarm Rate Distribution Histogram"
                      isLoading={histLoading}
                      error={histError instanceof Error ? histError.message : null}
                      className="min-w-0"
                      height={360}
                    />
                    <ChartWrapper
                      data={rollingChart || undefined}
                      title={`${windowHours}-Hour Rolling Average Alarm Rate`}
                      isLoading={rollingLoading}
                      error={rollingError instanceof Error ? rollingError.message : null}
                      className="min-w-0"
                      height={360}
                    />
                  </div>

                  <div className="flex gap-2 pt-2">
                    <Button size="sm" variant="outline" disabled={!activeId || exporting === 'json'} onClick={() => handleExport('json')}>
                      <Download className="mr-2 h-4 w-4" /> Export JSON
                    </Button>
                    <Button size="sm" variant="outline" disabled={!activeId || exporting === 'csv'} onClick={() => handleExport('csv')}>
                      <Download className="mr-2 h-4 w-4" /> Export CSV
                    </Button>
                  </div>

                  {/* Comparison of selected files */}
                  {compareEnabled && (
                    <div className="pt-4 space-y-3">
                      <h4 className="font-semibold">Comparison of selected files</h4>
                      {compareLoading ? (
                        <Skeleton className="h-32 w-full" />
                      ) : compareError ? (
                        <Alert variant="destructive">
                          <AlertDescription>
                            {compareError instanceof Error ? compareError.message : 'Failed to compare files'}
                          </AlertDescription>
                        </Alert>
                      ) : compareData ? (
                        <DataTable
                          data={compareData.map(r => ({
                            filename: r.filename,
                            total_alarms: r.metrics.total_alarms,
                            avg_alarms_per_hour: r.metrics.avg_alarms_per_hour,
                            flood_percentage: r.metrics.flood_percentage,
                            isa_performance: r.metrics.isa_performance,
                          }))}
                          columns={[
                            { key: 'filename', label: 'File' },
                            { key: 'total_alarms', label: 'Total' },
                            { key: 'avg_alarms_per_hour', label: 'Avg/hr', render: (v) => (typeof v === 'number' ? v.toFixed(1) : v) },
                            { key: 'flood_percentage', label: 'Flood %', render: (v) => (typeof v === 'number' ? `${v.toFixed(1)}%` : v) },
                            { key: 'isa_performance', label: 'ISA' },
                          ]}
                          paginated={false}
                          searchable={true}
                          sortable={true}
                          emptyMessage="No comparison data"
                        />
                      ) : null}
                    </div>
                  )}
                </TabsContent>

                <TabsContent value="insights" className="mt-4 space-y-3">
                  {insightsLoading && <Skeleton className="h-40 w-full" />}
                  {insightsError && (
                    <Alert variant="destructive">
                      <AlertDescription>
                        {insightsError instanceof Error ? insightsError.message : 'Failed to load insights'}
                      </AlertDescription>
                    </Alert>
                  )}
                  {insights && (
                    <div className="space-y-4">
                      {/* Summary & Export */}
                      <div className="flex items-start justify-between gap-3">
                        <div className="rounded-lg border border-border bg-muted/30 p-3 text-sm flex-1">
                          <span className="font-semibold">Summary:</span> {insights.summary}
                        </div>
                        <div className="shrink-0 flex gap-2">
                          <Button size="sm" variant="outline" disabled={!activeId || exporting === 'json'} onClick={() => handleExport('json')}>
                            <Download className="mr-2 h-4 w-4" /> JSON
                          </Button>
                          <Button size="sm" variant="outline" disabled={!activeId || exporting === 'csv'} onClick={() => handleExport('csv')}>
                            <Download className="mr-2 h-4 w-4" /> CSV
                          </Button>
                        </div>
                      </div>

                      {/* Grouped insights */}
                      {(() => {
                        const grouped = (insights.insights || []).reduce((acc: Record<string, InsightItem[]>, it: InsightItem) => {
                          (acc[it.type] = acc[it.type] || []).push(it);
                          return acc;
                        }, {} as Record<string, InsightItem[]>);
                        const order: Array<'success' | 'warning' | 'error'> = ['success', 'warning', 'error'];
                        const titleFor: Record<string, string> = { success: 'Good Practices', warning: 'Warnings & Recommendations', error: 'Critical Issues' };
                        const variantFor = (t: string) => t === 'error' ? 'destructive' : t === 'warning' ? 'secondary' : 'default';
                        return (
                          <div className="space-y-4">
                            {order.filter(t => grouped[t]?.length).map((t) => (
                              <div key={t} className="rounded-lg border border-border">
                                <div className="flex items-center justify-between p-3 border-b border-border">
                                  <div className="font-semibold">{titleFor[t]}</div>
                                  <Badge variant={variantFor(t)}>{grouped[t].length}</Badge>
                                </div>
                                <div className="p-3 space-y-2">
                                  {grouped[t].map((it, idx) => (
                                    <div key={idx} className={cn('rounded-md border p-3 text-sm',
                                      t === 'error' ? 'border-red-300/50 bg-red-500/5' : t === 'warning' ? 'border-yellow-300/50 bg-yellow-500/5' : 'border-emerald-300/50 bg-emerald-500/5')}
                                    >
                                      {it.message}
                                    </div>
                                  ))}
                                </div>
                              </div>
                            ))}
                          </div>
                        );
                      })()}
                    </div>
                  )}
                </TabsContent>

                <TabsContent value="advanced" className="mt-4 space-y-6">
                  {/* Configuration */}
                  <div className="grid gap-4 md:grid-cols-3">
                    <div className="space-y-2">
                      <Label htmlFor="win-min">Chattering Window (minutes)</Label>
                      <Input id="win-min" type="number" min={1} max={60} value={windowMinutes} onChange={(e) => setWindowMinutes(Number(e.target.value))} />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="thr">Chattering Threshold</Label>
                      <Input id="thr" type="number" min={2} max={50} value={threshold} onChange={(e) => setThreshold(Number(e.target.value))} />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="min-hrs">Standing Min Duration (hours)</Label>
                      <Input id="min-hrs" type="number" min={1} max={168} value={minDurationHours} onChange={(e) => setMinDurationHours(Number(e.target.value))} />
                    </div>
                  </div>

                  {/* Summary Cards */}
                  <div className="grid gap-4 md:grid-cols-4">
                    <div className="rounded-lg border border-border bg-muted/30 p-3">
                      <div className="text-sm text-muted-foreground">Chattering (High)</div>
                      <div className="text-2xl font-bold">{Array.isArray(chatteringData) ? chatteringData.filter((x: ChatteringAlarm) => x.severity === 'High').length : 0}</div>
                    </div>
                    <div className="rounded-lg border border-border bg-muted/30 p-3">
                      <div className="text-sm text-muted-foreground">Chattering (Medium)</div>
                      <div className="text-2xl font-bold">{Array.isArray(chatteringData) ? chatteringData.filter((x: ChatteringAlarm) => x.severity === 'Medium').length : 0}</div>
                    </div>
                    <div className="rounded-lg border border-border bg-muted/30 p-3">
                      <div className="text-sm text-muted-foreground">Chattering (Low)</div>
                      <div className="text-2xl font-bold">{Array.isArray(chatteringData) ? chatteringData.filter((x: ChatteringAlarm) => x.severity === 'Low').length : 0}</div>
                    </div>
                    <div className="rounded-lg border border-border bg-muted/30 p-3">
                      <div className="text-sm text-muted-foreground">Standing (Active)</div>
                      <div className="text-2xl font-bold">{Array.isArray(standingData) ? standingData.filter((x: StandingAlarm) => x.is_active).length : 0}</div>
                    </div>
                  </div>

                  {/* Chattering */}
                  <div className="space-y-3">
                    <h4 className="flex items-center gap-2 font-semibold"><Zap className="h-4 w-4" /> Chattering Alarms</h4>
                    {chatteringLoading ? (
                      <Skeleton className="h-28 w-full" />
                    ) : chatteringError ? (
                      <Alert variant="destructive"><AlertDescription>{chatteringError instanceof Error ? chatteringError.message : 'Failed to load chattering alarms'}</AlertDescription></Alert>
                    ) : chatteringData ? (
                      <DataTable
                        data={chatteringData}
                        columns={[
                          { key: 'tag', label: 'Tag' },
                          { key: 'count', label: 'Total', render: (v) => (typeof v === 'number' ? v.toLocaleString() : v) },
                          { key: 'rate_per_hour', label: 'Rate/hr', render: (v) => (typeof v === 'number' ? v.toFixed(1) : v) },
                          { key: 'severity', label: 'Severity', render: (v) => (<Badge variant={v === 'High' ? 'destructive' : v === 'Medium' ? 'secondary' : 'outline'}>{v}</Badge>) },
                        ]}
                        searchable={true}
                        sortable={true}
                        paginated={true}
                        pageSize={12}
                        emptyMessage="No chattering alarms"
                      />
                    ) : null}
                  </div>

                  {/* Standing */}
                  <div className="space-y-3">
                    <h4 className="flex items-center gap-2 font-semibold"><Clock className="h-4 w-4" /> Standing Alarms</h4>
                    {standingLoading ? (
                      <Skeleton className="h-28 w-full" />
                    ) : standingError ? (
                      <Alert variant="destructive"><AlertDescription>{standingError instanceof Error ? standingError.message : 'Failed to load standing alarms'}</AlertDescription></Alert>
                    ) : standingData ? (
                      <DataTable
                        data={standingData}
                        columns={[
                          { key: 'tag', label: 'Tag' },
                          { key: 'duration_hours', label: 'Duration (h)', render: (v) => (typeof v === 'number' ? v.toFixed(1) : v) },
                          { key: 'start_time', label: 'Start', render: (v) => new Date(v).toLocaleString() },
                          { key: 'end_time', label: 'End', render: (v) => (v ? new Date(v).toLocaleString() : '—') },
                          { key: 'is_active', label: 'Status', render: (v) => (<Badge variant={v ? 'destructive' : 'secondary'}>{v ? 'Active' : 'Resolved'}</Badge>) },
                        ]}
                        searchable={true}
                        sortable={true}
                        paginated={true}
                        pageSize={12}
                        emptyMessage="No standing alarms"
                      />
                    ) : null}
                  </div>
                </TabsContent>
              </Tabs>
            </div>
          )}
        </div>
      </div>
      {/* Upload Modal */}
      <UploadModal open={uploadOpen} onOpenChange={setUploadOpen} />

      {/* Delete confirmation for single file */}
      <AlertDialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete File</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete "{fileToDelete?.filename}"? This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => fileToDelete && deleteMutation.mutate(fileToDelete.file_id)}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* Delete All confirmation */}
      <AlertDialog open={deleteAllDialogOpen} onOpenChange={setDeleteAllDialogOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete All Uploaded Files</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete all {uploadedFilesCount} uploaded files? This cannot be undone and will not affect auto-discovered files.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => deleteAllMutation.mutate()}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              Delete All
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}


