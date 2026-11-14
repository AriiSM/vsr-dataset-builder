import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Vite configuration:
//  - in development (`npm run dev`), all requests to /api are proxied
//    to the FastAPI backend (python backend/run_api.py) on port 8000;
//  - at build time (`npm run build`), static files are generated in `dist/`,
//    from where FastAPI serves them directly in production.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
  },
});
