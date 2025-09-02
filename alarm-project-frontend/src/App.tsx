import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Toaster } from '@/components/ui/toaster';
import { Toaster as Sonner } from '@/components/ui/sonner';
import { TooltipProvider } from '@/components/ui/tooltip';
import { Layout } from '@/components/Layout';
import Dashboard from '@/pages/Dashboard';
import Files from '@/pages/Files';
import Analysis from '@/pages/Analysis';
import Charts from '@/pages/Charts';
import Insights from '@/pages/Insights';
import Advanced from '@/pages/Advanced';
import Settings from '@/pages/Settings';
import NotFound from '@/pages/NotFound';
import Explorer from '@/pages/Explorer';

const App = () => (
  <TooltipProvider>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="explorer" element={<Explorer />} />
          <Route path="files" element={<Files />} />
          <Route path="analysis" element={<Analysis />} />
          <Route path="charts" element={<Charts />} />
          <Route path="insights" element={<Insights />} />
          <Route path="advanced" element={<Advanced />} />
          <Route path="settings" element={<Settings />} />
          <Route path="*" element={<NotFound />} />
        </Route>
      </Routes>
      <Toaster />
      <Sonner />
    </BrowserRouter>
  </TooltipProvider>
);

export default App;