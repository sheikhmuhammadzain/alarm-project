import { useState, useEffect, useRef } from 'react';
import Plot from 'react-plotly.js';
import type { Data as PlotData, Layout, Config } from 'plotly.js';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { AlertTriangle, TrendingUp } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useAppStore } from '@/store/app-store';

interface ChartWrapperProps {
  data?: unknown; // Plotly JSON data (various backend formats supported)
  title?: string;
  isLoading?: boolean;
  error?: string | null;
  metrics?: Record<string, unknown>;
  className?: string;
  height?: number;
  showMetrics?: boolean;
}

export function ChartWrapper({ 
  data, 
  title = 'Chart', 
  isLoading = false,
  error = null,
  metrics,
  className,
  height = 400,
  showMetrics = true
}: ChartWrapperProps) {
  const [plotData, setPlotData] = useState<PlotData[] | null>(null);
  const [plotLayout, setPlotLayout] = useState<Partial<Layout> | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [computedHeight, setComputedHeight] = useState<number>(height);
  const isDarkMode = useAppStore((s) => s.darkMode);

  // Process Plotly data when it changes
  useEffect(() => {
    if (!data) return;

    try {
      // Handle different response formats
      let chartData: PlotData[] | undefined;
      let chartLayout: Partial<Layout> | undefined;

      const asRecord = (v: unknown): Record<string, unknown> | null =>
        typeof v === 'object' && v !== null ? (v as Record<string, unknown>) : null;
      
      // Use loose checks because backend may return slightly different shapes
      const root = asRecord(data);
      if (root && 'chart_data' in root) {
        // Response format: { chart_data: { data: [], layout: {} } }
        const cd = asRecord(root['chart_data']);
        const cdData = cd?.['data'];
        const cdLayout = cd?.['layout'];
        if (Array.isArray(cdData)) chartData = cdData as PlotData[];
        if (typeof cdLayout === 'object' && cdLayout !== null) chartLayout = cdLayout as Partial<Layout>;
      } else if (root && 'chart' in root) {
        // Backend format: { chart: { data: [], layout: {} } }
        const ch = asRecord(root['chart']);
        const chData = ch?.['data'];
        const chLayout = ch?.['layout'];
        if (Array.isArray(chData)) chartData = chData as PlotData[];
        if (typeof chLayout === 'object' && chLayout !== null) chartLayout = chLayout as Partial<Layout>;
      } else if (root && 'data' in root && 'layout' in root) {
        // Direct Plotly format: { data: [], layout: {} }
        const d = root['data'];
        const l = root['layout'];
        if (Array.isArray(d)) chartData = d as PlotData[];
        if (typeof l === 'object' && l !== null) chartLayout = l as Partial<Layout>;
      } else if (Array.isArray(data)) {
        // Just data array
        chartData = data as unknown as PlotData[];
        chartLayout = {} as Partial<Layout>;
      } else {
        console.warn('Unexpected chart data format:', data);
        return;
      }

      // Enhance layout for responsiveness and theme
      const isDark = isDarkMode || document.documentElement.classList.contains('dark');
      const enhancedLayout = {
        ...chartLayout,
        autosize: true,
        height: computedHeight,
        // Slightly larger bottom margin to accommodate rotated ticks; smaller top (no internal title)
        margin: { l: 50, r: 30, t: 40, b: 70 },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: {
          family: 'inherit',
          size: 12,
          color: isDark ? '#e5e5e5' : '#111827',
        },
        // Remove internal title to avoid duplication with Card header
        title: undefined,
        // Prevent axis labels from being clipped/overlapping in tight spaces
        xaxis: {
          ...chartLayout?.xaxis,
          automargin: true,
          tickangle: chartLayout?.xaxis?.tickangle ?? -45,
          title: { ...chartLayout?.xaxis?.title, standoff: 10 },
          color: isDark ? '#e5e5e5' : '#374151',
          gridcolor: isDark ? '#374151' : '#e5e7eb',
        },
        yaxis: {
          ...chartLayout?.yaxis,
          automargin: true,
          title: { ...chartLayout?.yaxis?.title, standoff: 10 },
          color: isDark ? '#e5e5e5' : '#374151',
          gridcolor: isDark ? '#374151' : '#e5e7eb',
        },
      };

      setPlotData(chartData);
      setPlotLayout(enhancedLayout);
    } catch (err) {
      console.error('Error processing chart data:', err);
    }
  }, [data, computedHeight, isDarkMode]);

  // Compute a responsive height based on container width (maintain aspect ratio)
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const compute = () => {
      const width = el.clientWidth || el.getBoundingClientRect().width || 0;
      // Prefer a 16:9-ish ratio on wide screens, but never below provided height
      const target = Math.max(height, Math.round(Math.min(560, Math.max(300, width * 0.56))));
      setComputedHeight(target);
    };
    compute();
    const ro = new ResizeObserver(() => compute());
    ro.observe(el);
    window.addEventListener('orientationchange', compute);
    return () => {
      ro.disconnect();
      window.removeEventListener('orientationchange', compute);
    };
  }, [height]);

  // Plotly config for interactivity
  const plotConfig: Partial<Config> = {
    responsive: true,
    displayModeBar: true,
    modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
    displaylogo: false,
    toImageButtonOptions: {
      format: 'png',
      filename: title.toLowerCase().replace(/\s+/g, '-'),
      height: 600,
      width: 800,
      scale: 2,
    },
  };

  return (
    <Card className={cn('shadow-isa-md', className)}>
      <CardHeader className="pb-4">
        <CardTitle className="flex items-center gap-2">
          <TrendingUp className="h-5 w-5 text-isa-primary" />
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading && (
          <div className="space-y-4">
            <Skeleton className="w-full" style={{ height: `${computedHeight}px` }} />
            {showMetrics && (
              <div className="grid grid-cols-2 gap-4">
                <Skeleton className="h-16" />
                <Skeleton className="h-16" />
              </div>
            )}
          </div>
        )}

        {error && (
          <Alert variant="destructive">
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription>
              Failed to load chart: {error}
            </AlertDescription>
          </Alert>
        )}

        {!isLoading && !error && plotData && plotLayout && (
          <div className="space-y-6">
            <div ref={containerRef} className="w-full overflow-hidden rounded-lg border border-border">
              <Plot
                data={plotData}
                layout={plotLayout}
                config={plotConfig}
                style={{ width: '100%', height: `${computedHeight}px` }}
                useResizeHandler
              />
            </div>

            {/* Display metrics if available */}
            {showMetrics && metrics && Object.keys(metrics).length > 0 && (
              <div className="space-y-3">
                <h4 className="text-sm font-semibold text-foreground">Chart Metrics</h4>
                <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                  {Object.entries(metrics).map(([key, value]) => (
                    <div
                      key={key}
                      className="rounded-lg border border-border bg-muted/30 px-3 py-2"
                    >
                      <div className="text-xs font-medium text-muted-foreground">
                        {key.replace(/_/g, ' ').toUpperCase()}
                      </div>
                      <div className="mt-1 text-sm font-semibold text-foreground">
                        {typeof value === 'number' ? value.toLocaleString() : String(value)}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {!isLoading && !error && !plotData && (
          <div className="flex items-center justify-center text-muted-foreground" style={{ height: `${computedHeight}px` }}>
            <div className="text-center">
              <TrendingUp className="mx-auto h-12 w-12 opacity-50" />
              <p className="mt-4">No chart data available</p>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}