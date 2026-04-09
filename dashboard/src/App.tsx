import { HashRouter, Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import PipelineControl from './pages/PipelineControl'
import Overview from './pages/Overview'
import PhaseDetails from './pages/PhaseDetails'
import SampleBrowser from './pages/SampleBrowser'
import QualityCheck from './pages/QualityCheck'

export default function App() {
  return (
    <HashRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<PipelineControl />} />
          <Route path="/overview" element={<Overview />} />
          <Route path="/phases" element={<PhaseDetails />} />
          <Route path="/samples" element={<SampleBrowser />} />
          <Route path="/signals" element={<QualityCheck />} />
        </Route>
      </Routes>
    </HashRouter>
  )
}
