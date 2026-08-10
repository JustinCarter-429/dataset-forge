import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import './index.css'
import './a11y.css'
import './dropzone.css'
import './phase4.css'
createRoot(document.getElementById('root')!).render(<StrictMode><App /></StrictMode>)
