import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Las peticiones a /api se reenvían al backend FastAPI (evita problemas de CORS).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Otro puerto para el backend:  BACKEND_URL=http://127.0.0.1:8001 npm run dev
    proxy: { "/api": process.env.BACKEND_URL ?? "http://127.0.0.1:8000" },
  },
});
