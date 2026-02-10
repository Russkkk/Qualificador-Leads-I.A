import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";
import { resolve } from "node:path";

export default defineConfig({
  plugins: [vue()],
  build: {
    rollupOptions: {
      input: {
        index: resolve(__dirname, "index.html"),
        acao: resolve(__dirname, "acao.html"),
        onboarding: resolve(__dirname, "onboarding.html")
      }
    }
  }
});
