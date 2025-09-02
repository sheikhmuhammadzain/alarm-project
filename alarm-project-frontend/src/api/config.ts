import axios from 'axios';

// API Configuration - make base URL configurable via environment
export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

// Create axios instance with default config
export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000, // 120 seconds to accommodate heavy analysis/charts
  headers: {
    'Content-Type': 'application/json',
  },
});

// Allow runtime override of baseURL (used by Settings page)
export const setApiBaseUrl = (url: string) => {
  apiClient.defaults.baseURL = url;
};

// Request interceptor for logging
apiClient.interceptors.request.use(
  (config) => {
    console.log(`[API] ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    const message = error.response?.data?.detail || error.message;
    console.error('[API Error]', message);
    return Promise.reject(error);
  }
);