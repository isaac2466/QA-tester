import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Las peticiones a /api se reenvían al backend FastAPI (evita problemas de CORS).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": "http://127.0.0.1:8000" },
  },
});
