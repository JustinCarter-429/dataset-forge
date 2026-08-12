import { Circle, CircleCheck, CirclePause, CircleX, LoaderCircle } from 'lucide-react'
import { useEffect, useState } from 'react'
import type { GenerationJob } from '../api/types'
import { getGeneration, resumeGeneration } from '../api/client'
import './GenerationStatus.css'

const stages = [
  { key: 'extracting', title: 'Document extracted', copy: 'Text and tables extracted successfully' },
  { key: 'analyzing', title: 'Content analyzed', copy: 'Structure and key themes identified' },
  { key: 'planning', title: 'Dataset planned', copy: 'DatasetSpec and output schema prepared' },
  { key: 'generating', title: 'Generating training examples', copy: 'Creating dataset records with Carter 1.0' },
  { key: 'reviewing', title: 'Reviewing generated dataset', copy: 'Checking the completed candidate dataset before validation' },
  { key: 'validating', title: 'Validating dataset', copy: 'Checking records, evidence, and quality' },
  { key: 'packaging', title: 'Preparing ZIP download', copy: 'Packaging validated outputs' },
]

export function GenerationStatus({ job }: { job: GenerationJob | null }) {
  const [resuming, setResuming] = useState(false)
  const [resumedJob, setResumedJob] = useState<GenerationJob | null>(null)
  const visibleJob = resumedJob || job
  useEffect(() => { setResumedJob(null) }, [job])
  useEffect(() => { if (!resumedJob || ['completed', 'failed', 'cancelled'].includes(resumedJob.status)) return; const timer = window.setInterval(() => { void getGeneration(resumedJob.id).then(setResumedJob).catch(() => undefined) }, 1000); return () => window.clearInterval(timer) }, [resumedJob])
  const index = stages.findIndex(item => item.key === visibleJob?.stage)
  const completed = visibleJob?.status === 'completed'; const paused = !!visibleJob?.recoverable && !!visibleJob?.resumeAvailable
  const failed = visibleJob?.status === 'failed' && !paused; const cancelled = visibleJob?.status === 'cancelled'; const batch = visibleJob?.batch
  const extractionCopy = visibleJob?.analysis ? `${visibleJob.analysis.availablePageCount ? `${visibleJob.analysis.availablePageCount} pages · ` : ''}${visibleJob.extraction?.statistics.wordCount.toLocaleString()} words extracted` : 'Docling/Python extraction in progress'
  const analysisCopy = visibleJob?.analysis ? `${visibleJob.analysis.sectionCount} sections · ${visibleJob.analysis.tableCount} tables identified` : 'Deterministic structure analysis'
  const generationCopy = batch ? `Batch ${batch.currentBatch} of ${batch.totalBatches} · ${batch.recordsGenerated} / ${batch.recordsRequested} records generated` : visibleJob?.provider?.state === 'queued' ? 'Waiting for AI worker...' : stages[3].copy
  const reviewActive = ['revising', 'revalidating'].includes(visibleJob?.progress.currentStage || '')
  const resume = async () => { if (!visibleJob || resuming) return; setResuming(true); try { setResumedJob(await resumeGeneration(visibleJob.id)) } finally { setResuming(false) } }
  return <section className="side-card" aria-live="polite"><div className="side-card-heading"><div><p className="eyebrow">Live pipeline</p><h2>Generation Status</h2></div><span className={`status-chip ${completed ? 'success' : paused ? 'warning' : failed || cancelled ? 'danger' : ''}`}>{completed ? 'Ready' : paused ? 'Paused' : failed ? 'Failed' : cancelled ? 'Cancelled' : visibleJob ? 'Generating' : 'Pending'}</span></div><div className="status-list">{stages.map((stage, i) => { const done = completed || index > i || (reviewActive && i < 4); const current = !completed && !failed && !paused && !cancelled && (index === i || (reviewActive && i === 4)); const stagePaused = paused && index === i; return <div className="status-item" key={stage.key}><div className={`status-icon ${done ? 'done' : current ? 'active' : stagePaused ? 'paused' : failed && index === i ? 'failed' : ''}`}>{done ? <CircleCheck size={17} /> : stagePaused ? <CirclePause size={17} /> : failed && index === i ? <CircleX size={17} /> : current ? <LoaderCircle size={17} className="spin" /> : <Circle size={17} />}</div><div><strong>{stage.title}</strong><span>{stage.key === 'extracting' ? extractionCopy : stage.key === 'analyzing' ? analysisCopy : stage.key === 'generating' ? generationCopy : stage.copy}</span>{stage.key === 'generating' && batch && <div className="batch-progress" role="progressbar" aria-label={`${batch.recordsGenerated} of ${batch.recordsRequested} records generated`} aria-valuemin={0} aria-valuenow={batch.recordsGenerated} aria-valuemax={batch.recordsRequested}><i style={{ width: `${Math.round(100 * batch.recordsGenerated / batch.recordsRequested)}%` }} /></div>}</div></div> })}</div>{cancelled && <p className="error-banner">Generation cancelled. No package was created.</p>}{paused && <><p className="pause-banner">{visibleJob?.error?.message}</p><button className="generate-button" disabled={resuming} onClick={resume}>{resuming ? 'Resuming...' : 'Resume Generation'}</button></>}{failed && <p className="error-banner">{visibleJob?.error?.message || 'Generation failed. Your inputs have been preserved.'}</p>}</section>
}
