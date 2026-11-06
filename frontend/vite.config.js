import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Las peticiones a /api se reenvían al backend FastAPI (evita problemas de CORS).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // timeout amplio: la IA local en CPU puede tardar varios minutos por imagen
    proxy: { "/api": { target: "http://127.0.0.1:8000", timeout: 600000, proxyTimeout: 600000 } },
  },
});
