import { useState } from 'react'
import { BookOpen, FileOutput, Github, HelpCircle, X } from 'lucide-react'
import './AppHeader.css'

const githubRepository = 'https://github.com/JustinCarter-429/dataset-forge'

export function AppHeader() {
  const [showHelp, setShowHelp] = useState(false)
  return <header className="app-header">
    <a className="brand" href="#workflow" aria-label="Dataset Forge home"><div className="brand-mark"><FileOutput size={18} /></div><span>Dataset Forge</span></a>
    <nav aria-label="Application links">
      <button className="header-link" onClick={() => setShowHelp(true)}><HelpCircle size={15} /> How it works</button>
      <a href="http://127.0.0.1:8000/docs" target="_blank" rel="noreferrer"><BookOpen size={15} /> Docs</a>
      <a className="github-link" href={githubRepository} target="_blank" rel="noreferrer"><Github size={15} /> Star on GitHub</a>
    </nav>
    {showHelp && <div className="help-backdrop" role="presentation" onMouseDown={() => setShowHelp(false)}><section className="help-dialog" role="dialog" aria-modal="true" aria-labelledby="how-it-works-title" onMouseDown={event => event.stopPropagation()}><div className="help-heading"><div><p className="eyebrow">Carter 1.0 guide</p><h2 id="how-it-works-title">How it works</h2></div><button className="icon-button" aria-label="Close instructions" onClick={() => setShowHelp(false)}><X size={18} /></button></div><ol><li>Upload up to three PDF, DOCX, or TXT documents.</li><li>Ask Carter 1.0 a document-grounded question or describe the dataset you need.</li><li>Select JSON or CSV and generate a source-grounded package.</li><li>Review validation, evidence, and quality details before downloading.</li></ol><p>Carter 1.0 uses only the selected document knowledge. Local mode uses LM Studio; Cloud mode uses RunPod.</p><button className="generate-button" onClick={() => setShowHelp(false)}>Got it</button></section></div>}
  </header>
}
