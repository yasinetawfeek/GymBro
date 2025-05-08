// Configuration variables for the application

// First check if window._env_ exists (runtime environment variables)
// Then check Vite environment variables (build-time)
// Finally fall back to localhost
export const API_URL = 
  (window._env_ && window._env_.VITE_API_URL) || 
  import.meta.env.VITE_API_URL || 
  'http://localhost:8000';

// Other configuration variables can be added here 