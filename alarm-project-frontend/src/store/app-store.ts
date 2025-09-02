import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { FileInfo } from '@/api/types';

export interface AppState {
  // Selected file context
  selectedFile: FileInfo | null;
  selectedFileId: string | null;
  
  // UI State
  sidebarCollapsed: boolean;
  darkMode: boolean;
  
  // API Configuration
  apiBaseUrl: string;
  
  // Actions
  setSelectedFile: (file: FileInfo | null) => void;
  setSelectedFileId: (fileId: string | null) => void;
  setSidebarCollapsed: (collapsed: boolean) => void;
  setDarkMode: (dark: boolean) => void;
  setApiBaseUrl: (url: string) => void;
  
  // Derived getters
  hasSelectedFile: () => boolean;
  getSelectedFileId: () => string | null;
}

export const useAppStore = create<AppState>()(
  persist(
    (set, get) => ({
      // Initial state
      selectedFile: null,
      selectedFileId: null,
      sidebarCollapsed: false,
      darkMode: false,
      apiBaseUrl: 'http://localhost:8000',
      
      // Actions
      setSelectedFile: (file) => set({ 
        selectedFile: file, 
        selectedFileId: file?.file_id || null 
      }),
      
      setSelectedFileId: (fileId) => set({ selectedFileId: fileId }),
      
      setSidebarCollapsed: (collapsed) => set({ sidebarCollapsed: collapsed }),
      
      setDarkMode: (dark) => {
        set({ darkMode: dark });
        // Apply dark mode to document
        if (dark) {
          document.documentElement.classList.add('dark');
        } else {
          document.documentElement.classList.remove('dark');
        }
      },
      
      setApiBaseUrl: (url) => set({ apiBaseUrl: url }),
      
      // Derived getters
      hasSelectedFile: () => !!get().selectedFile,
      getSelectedFileId: () => get().selectedFileId,
    }),
    {
      name: 'isa-analyzer-storage', // Storage key
      partialize: (state) => ({ 
        darkMode: state.darkMode,
        apiBaseUrl: state.apiBaseUrl,
        sidebarCollapsed: state.sidebarCollapsed
      }), // Only persist these values
    }
  )
);

// Initialize dark mode on load
const { darkMode } = useAppStore.getState();
if (darkMode) {
  document.documentElement.classList.add('dark');
}