import { useEffect, useState } from 'react'
import { Download, Sparkles } from 'lucide-react'
import { cancelGeneration, createGeneration, generationDownloadUrl, getGeneration, uploadFile } from '../api/client'
import type { GenerationJob, OutputFormat, UploadedFile } from '../api/types'
import { AppHeader } from '../components/AppHeader'
import { DatasetPreview } from '../components/DatasetPreview'
import { FileDropzone } from '../components/FileDropzone'
import { FormatSelector } from '../components/FormatSelector'
import { GenerationStatus } from '../components/GenerationStatus'
import { WorkflowStepper } from '../components/WorkflowStepper'

export function HomePage() {
  const [file, setFile] = useState<File | null>(null)
  const [uploaded, setUploaded] = useState<UploadedFile | null>(null)
  const [fileError, setFileError] = useState('')
  const [uploading, setUploading] = useState(false)
  const [prompt, setPrompt] = useState('')
  const [format, setFormat] = useState<OutputFormat>('json')
  const [job, setJob] = useState<GenerationJob | null>(null)
  const [generationId, setGenerationId] = useState('')
  const [error, setError] = useState('')

  useEffect(() => {
    if (!generationId || ['completed', 'failed', 'cancelled'].includes(job?.status || '')) return
    let stopped = false
    const poll = async () => { try { const next = await getGeneration(generationId); if (!stopped) setJob(next) } catch { if (!stopped) setError('Unable to refresh generation status. Please try again.') } }
    poll(); const timer = window.setInterval(poll, 1000)
    return () => { stopped = true; window.clearInterval(timer) }
  }, [generationId, job?.status])

  const chooseFile = async (next: File | null, nextError?: string) => {
    setFileError(nextError || '')
    if (!next) { setFile(null); setUploaded(null); return }
    setFile(next); setUploading(true); setError('')
    try { setUploaded(await uploadFile(next)) } catch (e) { setUploaded(null); setFileError(e instanceof Error ? e.message : 'Upload failed.') } finally { setUploading(false) }
  }
  const submit = async () => {
    if (!uploaded || !prompt.trim() || uploading) return
    setError('')
    try { const created = await createGeneration(uploaded.id, prompt.trim(), format); setGenerationId(created.generationId); setJob(null) } catch (e) { setError(e instanceof Error ? e.message : 'Generation could not start.') }
  }
  const cancel = async () => {
    if (!job || !running) return
    try { setJob(await cancelGeneration(job.id)); setError('') } catch (e) { setError(e instanceof Error ? e.message : 'Cancellation could not be completed.') }
  }
  const reset = () => { setFile(null); setUploaded(null); setFileError(''); setUploading(false); setPrompt(''); setFormat('json'); setJob(null); setGenerationId(''); setError('') }
  const running = !!job && !['completed', 'failed', 'cancelled'].includes(job.status)
  const step = job?.status === 'completed' ? 3 : running ? 2 : uploaded && prompt.trim() ? 2 : uploaded ? 1 : 0

  return <><AppHeader /><main id="workflow" className="app-shell"><section className="hero"><div className="hero-kicker"><Sparkles size={14} /> Simple, structured, source-aware</div><h1>Create a Dataset From a Document</h1><p>Upload a file, describe the dataset you want, and generate a downloadable JSON or CSV package.</p><WorkflowStepper step={step} /></section><div className="workspace"><div className="left-column"><section className="panel"><div className="panel-title"><span className="step-badge">1</span><div><h2>Upload your source</h2><p>One document to start your dataset.</p></div></div><FileDropzone file={file} onChange={chooseFile} error={fileError} busy={uploading} />{uploaded && <div className="accepted-file"><div className="accepted-check">✓</div><div><strong>{uploaded.name}</strong><span>{uploaded.sizeBytes.toLocaleString()} bytes · {uploaded.status === 'ready' ? 'Extraction-ready' : 'Uploaded; extraction begins when you generate'}</span></div></div>}</section><section className="panel"><div className="panel-title"><span className="step-badge">2</span><div><h2>Describe your dataset</h2><p>Be specific about the task, structure, and audience.</p></div></div><label htmlFor="prompt">Dataset prompt</label><textarea id="prompt" value={prompt} onChange={e => setPrompt(e.target.value)} placeholder="Create a QA dataset for software testing based on the uploaded document. Each record should include an instruction, context, expected output, category, and difficulty." /><p className="helper">Your prompt is sent with bounded extracted source context to the configured RunPod gpt-oss-20b provider.</p><label className="format-label">Output format</label><FormatSelector value={format} onChange={setFormat} /></section><section className="generate-panel"><div><div className="panel-title compact"><span className="step-badge">3</span><div><h2>Generate your dataset</h2><p>{running ? 'Your request is moving through the backend pipeline.' : 'Create a validated package when your inputs are ready.'}</p></div></div>{error && <p className="error-banner">{error}</p>}</div><div className="generation-actions"><button className="generate-button" disabled={!uploaded || !prompt.trim() || uploading || running} onClick={submit}>{running ? 'Generating Dataset…' : 'Generate Dataset'}<Sparkles size={16} /></button>{running && <button className="cancel-button" onClick={cancel}>Cancel Generation</button>}</div></section></div><div className="right-column"><GenerationStatus job={job} /><DatasetPreview job={job} /><div className="download-card"><div><strong>{job?.packageReady ? 'Your package is ready' : 'Download package'}</strong><span>{job?.packageReady ? 'Validated outputs are available.' : 'Download unlocks after packaging completes.'}</span></div><button className="download-button" disabled={!job?.packageReady} onClick={() => job && window.open(generationDownloadUrl(job.id), '_blank')}><Download size={16} /> Download ZIP</button>{(job?.status === 'completed' || job?.status === 'cancelled') && <button className="reset-button" onClick={reset}>Start over</button>}</div></div></div></main></>
}
