import { HashRouter, Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Pipeline from './pages/Pipeline'
import Overview from './pages/Overview'
import Samples from './pages/Samples'
import Benchmark from './pages/Benchmark'

export default function App() {
  return (
    <HashRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<Pipeline />} />
          <Route path="/overview" element={<Overview />} />
          <Route path="/samples" element={<Samples />} />
          <Route path="/benchmark" element={<Benchmark />} />
        </Route>
      </Routes>
    </HashRouter>
  )
}
