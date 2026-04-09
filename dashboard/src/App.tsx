import { HashRouter, Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import PipelineControl from './pages/PipelineControl'
import Overview from './pages/Overview'
import PhaseDetails from './pages/PhaseDetails'
import QualitySignals from './pages/QualitySignals'
import SampleBrowser from './pages/SampleBrowser'
import DedupClusters from './pages/DedupClusters'
import Contamination from './pages/Contamination'
import GoldenTests from './pages/GoldenTests'

export default function App() {
  return (
    <HashRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<PipelineControl />} />
          <Route path="/overview" element={<Overview />} />
          <Route path="/phases" element={<PhaseDetails />} />
          <Route path="/signals" element={<QualitySignals />} />
          <Route path="/samples" element={<SampleBrowser />} />
          <Route path="/dedup" element={<DedupClusters />} />
          <Route path="/contamination" element={<Contamination />} />
          <Route path="/golden" element={<GoldenTests />} />
        </Route>
      </Routes>
    </HashRouter>
  )
}
