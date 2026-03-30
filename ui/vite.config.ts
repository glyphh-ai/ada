import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "path";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: {
    outDir: process.env.VITE_OUT_DIR || path.resolve(__dirname, "../glyphh/public/dist"),
    emptyOutDir: true,
  },
  server: {
    proxy: {
      // Proxy all API paths to the running runtime server
      "/ui": "http://localhost:8002",
      "/health": "http://localhost:8002",
      "/docs": "http://localhost:8002",
      "/openapi.json": "http://localhost:8002",
      // Catch-all for /{org_id}/... API routes
      "^/[0-9a-f]{8}-": {
        target: "http://localhost:8002",
        changeOrigin: true,
      },
    },
  },
});
