import { defineConfig } from "vite";

export default defineConfig({
  server: {
    proxy: {
      "/ws": { target: "ws://127.0.0.1:8000", ws: true },
      "/health": "http://127.0.0.1:8000",
      "/api": "http://127.0.0.1:8000",
    },
  },
  build: {
    rollupOptions: { output: { manualChunks: { three: ["three"] } } },
    // three.js is already split into its own chunk above and legitimately exceeds the
    // default 500kB warning threshold; raise it rather than chase a warning that isn't
    // pointing at an actual problem (there's no lazy-load win here — the 3D scene is
    // needed immediately on load, so deferring the import would just delay first render).
    chunkSizeWarningLimit: 650,
  },
});
