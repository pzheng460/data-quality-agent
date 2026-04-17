import path from 'path'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  base: './',
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  build: {
    // Split heavy libraries into their own chunks. First paint pulls only
    // shared UI + router + the current route's chunk; charts / KaTeX / markdown
    // load on demand when the relevant page mounts.
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (!id.includes('node_modules')) return
          if (id.includes('recharts') || id.includes('d3-')) return 'charts'
          if (id.includes('katex') || id.includes('rehype-katex') || id.includes('remark-math')) return 'katex'
          if (id.includes('react-markdown') || id.includes('remark-') || id.includes('rehype-') ||
              id.includes('micromark') || id.includes('mdast') || id.includes('unified')) return 'markdown'
          if (id.includes('react-diff-viewer')) return 'diff'
          if (id.includes('@base-ui') || id.includes('@radix-ui')) return 'ui-primitives'
          if (id.includes('react-router')) return 'router'
        },
      },
    },
    chunkSizeWarningLimit: 800,
  },
})
