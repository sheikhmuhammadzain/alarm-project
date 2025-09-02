import { apiClient } from './config';
import type {
  FileInfo,
  FolderGroup,
  HealthResponse,
  MetricsResponse,
  ChartResponse,
  InsightsResponse,
  DashboardSummary,
  FilePreview,
  FileStatistics,
  BatchAnalysisResult,
  ChatteringAlarm,
  StandingAlarm,
} from './types';

// Health & Info
export const getHealth = (): Promise<HealthResponse> =>
  apiClient.get('/health').then(res => res.data);

export const getApiInfo = () =>
  apiClient.get('/').then(res => res.data);

// File Management
export const getFiles = (): Promise<FileInfo[]> =>
  apiClient.get('/files').then(res => res.data);

export const getAutoFiles = (signal?: AbortSignal): Promise<FolderGroup[]> =>
  apiClient.get('/auto-files', { signal }).then(res => {
    const data = res.data as { folders?: Record<string, FileInfo[]> };
    const folders = data?.folders || {};
    const groups: FolderGroup[] = Object.entries(folders).map(([folder_name, files]) => ({
      folder_name,
      files: files as FileInfo[],
      file_count: files.length,
    }));
    return groups;
  });

export const refreshAutoFiles = (signal?: AbortSignal) =>
  apiClient.post('/auto-files/refresh', undefined, { signal }).then(res => res.data);

export const getAutoFilesByFolder = (folderName: string, signal?: AbortSignal): Promise<FileInfo[]> =>
  apiClient.get(`/auto-files/${folderName}`, { signal }).then(res => {
    const data = res.data as { files?: FileInfo[] };
    return data?.files || [];
  });

export const getAllFiles = (signal?: AbortSignal): Promise<FileInfo[]> =>
  apiClient.get('/all-files', { signal }).then(res => {
    const data = res.data as { files?: FileInfo[] };
    return data?.files || [];
  });

export const uploadFile = (file: File): Promise<{ file_id: string }> => {
  const formData = new FormData();
  formData.append('file', file);
  
  return apiClient.post('/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  }).then(res => ({ file_id: res.data.file_id }));
};

export const deleteFile = (fileId: string) =>
  apiClient.delete(`/file/${fileId}`).then(res => res.data);

export const deleteAllFiles = () =>
  apiClient.delete('/files/all').then(res => res.data);

// File Details
export const getFilePreview = (fileId: string, rows: number = 10, signal?: AbortSignal): Promise<FilePreview> =>
  apiClient.get(`/file/${fileId}/preview?rows=${rows}`, { signal }).then(res => {
    const d = res.data;
    return {
      headers: d.columns,
      rows: d.data,
      total_rows: d.total_rows,
    } as FilePreview;
  });

export const getFileStatistics = (fileId: string, signal?: AbortSignal): Promise<FileStatistics> =>
  apiClient.get(`/file/${fileId}/statistics`, { signal }).then(res => {
    const d: { basic_statistics: { filename: string; total_rows: number }; column_info: Array<{ column: string; data_type: string; unique_values: number; non_null_count: number }>; } = res.data;
    const basic = d.basic_statistics;
    const humanizeType = (t?: string): string => {
      const s = (t || '').toLowerCase();
      if (!s) return 'Unknown';
      if (s.includes('datetime')) return 'Datetime';
      if (s.startsWith('date')) return 'Date';
      if (s === 'bool' || s === 'boolean') return 'Boolean';
      if (s.startsWith('int') || s.includes('int')) return 'Integer';
      if (s.startsWith('float') || s === 'double' || s === 'number') return 'Float';
      if (s === 'category' || s === 'categorical') return 'Category';
      if (s === 'string') return 'Text';
      if (s === 'object' || s === 'mixed') return 'Text';
      return t as string;
    };
    const columns = (d.column_info || []).map((c) => ({
      name: c.column,
      type: humanizeType(c.data_type),
      unique_values: c.unique_values,
      null_count: Math.max(0, (basic?.total_rows ?? 0) - (c.non_null_count ?? 0)),
    }));
    return {
      filename: basic?.filename,
      total_rows: basic?.total_rows ?? 0,
      columns,
      date_range: undefined,
    } as FileStatistics;
  });

// Analysis
export const analyzeFile = (
  fileId: string,
  options: { sample_size?: number; show_advanced?: boolean } = {},
  signal?: AbortSignal
): Promise<MetricsResponse> =>
  apiClient.post(`/analyze/${fileId}`, options, { signal }).then(res => {
    const m: Partial<MetricsResponse> & { source_file?: string; total_alarms?: number; avg_alarms_per_hour?: number; isa_performance?: 'Good' | 'Acceptable' | 'Poor' | 'Unknown'; flood_percentage?: number } = res.data;
    return {
      file_id: fileId,
      filename: m.source_file,
      total_alarms: m.total_alarms ?? 0,
      avg_alarms_per_hour: m.avg_alarms_per_hour ?? 0,
      isa_performance: m.isa_performance ?? 'Unknown',
      flood_percentage: m.flood_percentage ?? 0,
      alarm_rate_performance: m.isa_performance ?? 'Unknown',
      analysis_datetime: new Date().toISOString(),
      sample_size: options.sample_size,
      advanced_metrics: undefined,
      // preserve other fields for charts if needed
      ...m,
    } as unknown as MetricsResponse;
  });

export const batchAnalyze = (): Promise<BatchAnalysisResult> =>
  apiClient.post('/analyze/batch').then(res => {
    const results = (res.data?.results || []) as Array<{ file_id: string; filename: string; metrics?: MetricsResponse; error?: string } >;
    const completed_files = results
      .filter((r) => r.metrics)
      .map((r) => ({ file_id: r.file_id, filename: r.filename, metrics: r.metrics! }));
    const failed_files = results
      .filter((r) => !r.metrics)
      .map((r) => ({ file_id: r.file_id, filename: r.filename, error: r.error || 'Unknown error' }));
    return {
      completed_files,
      failed_files,
      summary: {
        total_processed: results.length,
        successful: completed_files.length,
        failed: failed_files.length,
      },
    } as BatchAnalysisResult;
  });

// Charts
export const getAlarmRateChart = (fileId: string, sampleSize?: number, signal?: AbortSignal): Promise<ChartResponse> =>
  apiClient.get(`/charts/alarm-rate/${fileId}${typeof sampleSize === 'number' ? `?sample_size=${sampleSize}` : ''}`, { signal }).then(res => res.data);

export const getTopContributorsChart = (fileId: string, topN: number = 15, sampleSize?: number, signal?: AbortSignal): Promise<ChartResponse> =>
  apiClient.get(`/charts/top-contributors/${fileId}?top_n=${topN}${typeof sampleSize === 'number' ? `&sample_size=${sampleSize}` : ''}`, { signal }).then(res => res.data);

export const getPriorityDistributionChart = (fileId: string, sampleSize?: number, signal?: AbortSignal): Promise<ChartResponse> =>
  apiClient.get(`/charts/priority-distribution/${fileId}${typeof sampleSize === 'number' ? `?sample_size=${sampleSize}` : ''}`, { signal }).then(res => res.data);

export const getParetoAnalysisChart = (fileId: string, sampleSize?: number, signal?: AbortSignal): Promise<ChartResponse> =>
  apiClient.get(`/charts/pareto-analysis/${fileId}${typeof sampleSize === 'number' ? `?sample_size=${sampleSize}` : ''}`, { signal }).then(res => res.data);

export const getFloodAnalysisChart = (fileId: string, sampleSize?: number, signal?: AbortSignal): Promise<ChartResponse> =>
  apiClient.get(`/charts/flood-analysis/${fileId}${typeof sampleSize === 'number' ? `?sample_size=${sampleSize}` : ''}`, { signal }).then(res => res.data);

export const getPerformanceComparisonChart = (): Promise<{ alarm_rate_chart?: unknown; flood_rate_chart?: unknown }> =>
  apiClient.get('/charts/performance-comparison').then(res => res.data as { alarm_rate_chart?: unknown; flood_rate_chart?: unknown });

export const getDistributionHistogramChart = (fileId: string, signal?: AbortSignal): Promise<ChartResponse> =>
  apiClient.get(`/charts/distribution-histogram/${fileId}`, { signal }).then(res => res.data);

export const getRollingAverageChart = (fileId: string, windowHours: number = 24, signal?: AbortSignal): Promise<ChartResponse> =>
  apiClient.get(`/charts/rolling-average/${fileId}?window_hours=${windowHours}`, { signal }).then(res => res.data);

// Dashboard
export const getDashboardSummary = (): Promise<DashboardSummary> =>
  apiClient.get('/dashboard/summary').then(res => {
    const d: { overall_kpis?: { total_sources?: number; good_sources?: number; poor_sources?: number; total_alarms?: number; avg_alarms_per_hour?: number; avg_flood_percentage?: number } } = res.data || {};
    const kpis = d.overall_kpis || {};
    const total_files = kpis.total_sources ?? 0;
    const good = kpis.good_sources ?? 0;
    const poor = kpis.poor_sources ?? 0;
    const acceptable = Math.max(0, total_files - good - poor);
    return {
      total_files,
      total_alarms: kpis.total_alarms ?? 0,
      avg_alarm_rate: kpis.avg_alarms_per_hour ?? 0,
      flood_percentage: kpis.avg_flood_percentage ?? 0,
      performance_breakdown: { good, acceptable, poor },
      recent_activity: [],
    } as DashboardSummary;
  });

// Insights
export const getInsights = (fileId: string, signal?: AbortSignal): Promise<InsightsResponse> =>
  apiClient.get(`/insights/${fileId}`, { signal }).then(res => {
    const d: { insights?: string[]; metrics_summary?: { alarm_rate?: number; flood_percentage?: number } } = res.data;
    const categorize = (s: string) =>
      s?.startsWith('✅') ? 'success' :
      (s?.startsWith('⚠️') || s?.startsWith('🌊')) ? 'warning' :
      s?.startsWith('❌') ? 'error' : 'success';
    const insights = (d.insights || []).map((m) => ({ type: categorize(m), message: m }));
    const ms = d.metrics_summary || { alarm_rate: undefined, flood_percentage: undefined };
    const summary = `Rate: ${typeof ms.alarm_rate === 'number' ? ms.alarm_rate.toFixed(1) : '—'}/hr, Flood: ${typeof ms.flood_percentage === 'number' ? ms.flood_percentage.toFixed(1) : '—'}%`;
    return { file_id: fileId, insights, summary } as InsightsResponse;
  });

// Export
export const exportMetrics = (fileId: string, format: 'json' | 'csv' = 'json'): Promise<string | Record<string, unknown>> =>
  apiClient.get(`/export/metrics/${fileId}?format=${format}`).then(res => {
    if (format === 'csv') {
      const { csv_data } = (res.data || {}) as { csv_data?: string };
      return csv_data as string;
    }
    return res.data as Record<string, unknown>;
  });

// Advanced Analysis
export const getChatteringAlarms = (
  fileId: string,
  windowMinutes: number = 10,
  threshold: number = 5,
  signal?: AbortSignal
): Promise<ChatteringAlarm[]> =>
  apiClient.get(`/advanced/chattering/${fileId}?window_minutes=${windowMinutes}&threshold=${threshold}`,
    { signal })
    .then(res => {
      const items: Array<{ tag: string; max_repeats_in_window?: number; total_occurrences?: number }> = (res.data?.chattering_alarms || []);
      return items.map((it) => {
        const rate_per_hour = (it.max_repeats_in_window || 0) * (60 / Math.max(1, windowMinutes));
        const severity = it.max_repeats_in_window >= threshold * 2 ? 'High' : it.max_repeats_in_window >= threshold ? 'Medium' : 'Low';
        return {
          tag: it.tag,
          count: it.total_occurrences ?? it.max_repeats_in_window ?? 0,
          rate_per_hour,
          severity,
        } as ChatteringAlarm;
      });
    });

export const getStandingAlarms = (
  fileId: string,
  minDurationHours: number = 24,
  signal?: AbortSignal
): Promise<StandingAlarm[]> =>
  apiClient.get(`/advanced/standing-alarms/${fileId}?min_duration_hours=${minDurationHours}`,
    { signal })
    .then(res => {
      const items: Array<{ tag: string; span_hours?: number; first_occurrence: string; last_occurrence?: string }> = (res.data?.standing_alarms || []);
      const now = Date.now();
      return items.map((it) => {
        const end_time = it.last_occurrence;
        const is_active = end_time ? (now - new Date(end_time).getTime()) < minDurationHours * 3600 * 1000 : true;
        return {
          tag: it.tag,
          duration_hours: it.span_hours ?? 0,
          start_time: it.first_occurrence,
          end_time,
          is_active,
        } as StandingAlarm;
      });
    });

// Benchmarks
export const getBenchmarks = () =>
  apiClient.get('/benchmarks').then(res => res.data?.benchmarks ?? res.data);