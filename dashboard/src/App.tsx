import { HashRouter, Routes, Route } from 'react-router-dom'
import { Suspense, lazy } from 'react'
import Layout from './components/Layout'

// Lazy-load page components so each route is its own chunk.
// First paint only needs Layout + current route, not the whole app.
const Pipeline = lazy(() => import('./pages/Pipeline'))
const Stats = lazy(() => import('./pages/Stats'))
const Samples = lazy(() => import('./pages/Samples'))
const Benchmark = lazy(() => import('./pages/Benchmark'))
const Config = lazy(() => import('./pages/Config'))
const NotFound = lazy(() => import('./pages/NotFound'))

function PageFallback() {
  return (
    <div className="flex items-center justify-center h-64 text-sm text-muted-foreground">
      Loading…
    </div>
  )
}

export default function App() {
  return (
    <HashRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<Suspense fallback={<PageFallback />}><Pipeline /></Suspense>} />
          <Route path="/stats" element={<Suspense fallback={<PageFallback />}><Stats /></Suspense>} />
          <Route path="/samples" element={<Suspense fallback={<PageFallback />}><Samples /></Suspense>} />
          <Route path="/benchmark" element={<Suspense fallback={<PageFallback />}><Benchmark /></Suspense>} />
          <Route path="/config" element={<Suspense fallback={null}><Config /></Suspense>} />
          <Route path="*" element={<Suspense fallback={null}><NotFound /></Suspense>} />
        </Route>
      </Routes>
    </HashRouter>
  )
}
