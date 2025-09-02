import { ReactNode } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';
import { TrendingUp, TrendingDown, Minus, LucideIcon } from 'lucide-react';

interface MetricCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon?: LucideIcon;
  status?: 'good' | 'acceptable' | 'poor' | 'neutral';
  trend?: 'up' | 'down' | 'neutral';
  trendValue?: string;
  className?: string;
  children?: ReactNode;
}

export function MetricCard({
  title,
  value,
  subtitle,
  icon: Icon,
  status = 'neutral',
  trend = 'neutral',
  trendValue,
  className,
  children,
}: MetricCardProps) {
  // Status colors based on ISA performance
  const getStatusColors = (status: string) => {
    switch (status) {
      case 'good':
        return {
          bg: 'bg-status-good/10',
          border: 'border-status-good/20',
          text: 'text-status-good',
          badge: 'bg-status-good text-white',
        };
      case 'acceptable':
        return {
          bg: 'bg-status-acceptable/10',
          border: 'border-status-acceptable/20',
          text: 'text-status-acceptable',
          badge: 'bg-status-acceptable text-white',
        };
      case 'poor':
        return {
          bg: 'bg-status-poor/10',
          border: 'border-status-poor/20',
          text: 'text-status-poor',
          badge: 'bg-status-poor text-white',
        };
      default:
        return {
          bg: 'bg-muted/30',
          border: 'border-border',
          text: 'text-isa-primary',
          badge: 'bg-isa-neutral-200 text-isa-neutral-600',
        };
    }
  };

  const statusColors = getStatusColors(status);
  
  const getTrendIcon = () => {
    switch (trend) {
      case 'up':
        return <TrendingUp className="h-4 w-4 text-status-good" />;
      case 'down':
        return <TrendingDown className="h-4 w-4 text-status-poor" />;
      default:
        return <Minus className="h-4 w-4 text-muted-foreground" />;
    }
  };

  const formatValue = (val: string | number) => {
    if (typeof val === 'number') {
      // Format large numbers with commas
      if (val >= 1000) {
        return val.toLocaleString();
      }
      // Format decimals nicely
      return val % 1 === 0 ? val.toString() : val.toFixed(2);
    }
    return val;
  };

  return (
    <Card className={cn(
      'transition-all duration-200 hover:shadow-isa-md',
      statusColors.bg,
      statusColors.border,
      className
    )}>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium text-muted-foreground">
          {title}
        </CardTitle>
        {Icon && (
          <Icon className={cn('h-4 w-4', statusColors.text)} />
        )}
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          {/* Main value */}
          <div className="flex items-baseline justify-between">
            <div className="text-2xl font-bold text-foreground">
              {formatValue(value)}
            </div>
            
            {/* Status badge */}
            {status !== 'neutral' && (
              <Badge className={cn('text-xs', statusColors.badge)}>
                {status.toUpperCase()}
              </Badge>
            )}
          </div>

          {/* Subtitle and trend */}
          {(subtitle || trendValue) && (
            <div className="flex items-center justify-between text-xs">
              {subtitle && (
                <span className="text-muted-foreground">{subtitle}</span>
              )}
              
              {trendValue && (
                <div className="flex items-center gap-1">
                  {getTrendIcon()}
                  <span className={cn(
                    'font-medium',
                    trend === 'up' ? 'text-status-good' :
                    trend === 'down' ? 'text-status-poor' :
                    'text-muted-foreground'
                  )}>
                    {trendValue}
                  </span>
                </div>
              )}
            </div>
          )}

          {/* Additional content */}
          {children}
        </div>
      </CardContent>
    </Card>
  );
}