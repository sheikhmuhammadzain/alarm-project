// API Response Types for ISA Alarm Analyzer

export interface FileInfo {
  file_id: string;
  filename: string;
  source_type: 'uploaded' | 'auto-discovered';
  folder?: string;
  size_mb: number;
  created_at: string;
  row_count?: number;
}

export interface FolderGroup {
  folder_name: string;
  files: FileInfo[];
  file_count: number;
}

export interface HealthResponse {
  status: string;
  benchmarks: Record<string, number>;
}

export interface MetricsResponse {
  file_id: string;
  filename: string;
  total_alarms: number;
  avg_alarms_per_hour: number;
  isa_performance: 'Good' | 'Acceptable' | 'Poor';
  flood_percentage: number;
  alarm_rate_performance: string;
  analysis_datetime: string;
  sample_size?: number;
  advanced_metrics?: AdvancedMetrics;
}

export interface AdvancedMetrics {
  peak_alarm_rate: number;
  flood_episodes: number;
  chattering_alarms: number;
  standing_alarms: number;
  priority_distribution: Record<string, number>;
}

export interface ChartResponse {
  chart_data?: any; // Plotly JSON
  chart?: any; // Backend returns { chart }
  metrics?: Record<string, any>;
  title?: string;
}

export interface InsightItem {
  type: 'success' | 'warning' | 'error';
  message: string;
  details?: string;
  recommendation?: string;
}

export interface InsightsResponse {
  file_id: string;
  insights: InsightItem[];
  summary: string;
}

export interface ChatteringAlarm {
  tag: string;
  count: number;
  rate_per_hour: number;
  severity: 'Low' | 'Medium' | 'High';
}

export interface StandingAlarm {
  tag: string;
  duration_hours: number;
  start_time: string;
  end_time?: string;
  is_active: boolean;
}

export interface DashboardSummary {
  total_files: number;
  total_alarms: number;
  avg_alarm_rate: number;
  flood_percentage: number;
  performance_breakdown: {
    good: number;
    acceptable: number;
    poor: number;
  };
  recent_activity: Array<{
    file_id: string;
    filename: string;
    action: string;
    timestamp: string;
  }>;
}

export interface FilePreview {
  headers: string[];
  rows: Array<Record<string, any>>;
  total_rows: number;
}

export interface FileStatistics {
  filename: string;
  total_rows: number;
  columns: Array<{
    name: string;
    type: string;
    unique_values: number;
    null_count: number;
  }>;
  date_range: {
    start: string;
    end: string;
  };
}

export interface BatchAnalysisResult {
  completed_files: Array<{
    file_id: string;
    filename: string;
    metrics: MetricsResponse;
  }>;
  failed_files: Array<{
    file_id: string;
    filename: string;
    error: string;
  }>;
  summary: {
    total_processed: number;
    successful: number;
    failed: number;
  };
}

// API Error Response
export interface ApiError {
  detail: string;
  status_code?: number;
}

// Chart Types for Plotly
export interface PlotlyChart {
  data: any[];
  layout: any;
  config?: any;
}