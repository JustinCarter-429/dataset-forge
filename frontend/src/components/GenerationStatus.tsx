import { Circle, CircleCheck, CircleX, LoaderCircle } from 'lucide-react'
import type { GenerationJob } from '../api/types'

const stages = [
  { key: 'extracting', title: 'Document extracted', copy: 'Text and tables extracted successfully' },
  { key: 'analyzing', title: 'Content analyzed', copy: 'Structure and key themes identified' },
  { key: 'generating', title: 'Generating training examples', copy: 'Creating dataset records with gpt-oss-20b' },
  { key: 'validating', title: 'Validating dataset', copy: 'Checking generated records and schema' },
  { key: 'packaging', title: 'Preparing ZIP download', copy: 'Packaging validated outputs' },
]

export function GenerationStatus({ job }: { job: GenerationJob | null }) {
  const index = stages.findIndex(item => item.key === job?.stage)
  const completed = job?.status === 'completed'
  const failed = job?.status === 'failed'
  const cancelled = job?.status === 'cancelled'
  const extractionCopy = job?.analysis ? `${job.analysis.availablePageCount ? `${job.analysis.availablePageCount} pages · ` : ''}${job.extraction?.statistics.wordCount.toLocaleString()} words extracted` : 'Docling/Python extraction in progress'
  const analysisCopy = job?.analysis ? `${job.analysis.sectionCount} sections · ${job.analysis.tableCount} tables identified` : 'Deterministic structure analysis'
  const generationCopy = job?.batch?.total ? `${job.provider?.state === 'queued' ? 'Waiting for AI worker · ' : ''}Batch ${job.batch.completed || 0} of ${job.batch.total}` : job?.provider?.state === 'queued' ? 'Waiting for AI worker...' : stages[2].copy
  const reviewActive = ['reviewing', 'revising', 'revalidating'].includes(job?.progress.currentStage || '')
  const reviewCopy = job?.progress.currentStage === 'revising' ? 'Improving flagged examples with one bounded revision' : job?.progress.currentStage === 'revalidating' ? 'Revalidating revised records against source evidence' : 'Reviewing dataset quality with the configured AI reviewer'
  return <section className="side-card" aria-live="polite"><div className="side-card-heading"><div><p className="eyebrow">Live pipeline</p><h2>Generation Status</h2></div><span className={`status-chip ${completed ? 'success' : failed || cancelled ? 'danger' : ''}`}>{completed ? 'Ready' : failed ? 'Failed' : cancelled ? 'Cancelled' : job ? `${job.progress.percent}%` : 'Pending'}</span></div><div className="status-list">{stages.map((stage, i) => { const done = completed || index > i || (reviewActive && i < 3); const current = !completed && !failed && !cancelled && (index === i || (reviewActive && i === 3)); const copy = stage.key === 'extracting' ? extractionCopy : stage.key === 'analyzing' ? analysisCopy : stage.key === 'generating' ? generationCopy : stage.key === 'validating' && reviewActive ? reviewCopy : stage.copy; return <div className="status-item" key={stage.key}><div className={`status-icon ${done ? 'done' : current ? 'active' : failed && index === i ? 'failed' : ''}`}>{done ? <CircleCheck size={17} /> : failed && index === i ? <CircleX size={17} /> : current ? <LoaderCircle size={17} className="spin" /> : <Circle size={17} />}</div><div><strong>{stage.title}</strong><span>{copy}</span></div></div> })}</div>{cancelled && <p className="error-banner">Generation cancelled. No package was created.</p>}{failed && <p className="error-banner">{job.error?.message || 'Generation failed. Your inputs have been preserved.'}</p>}</section>
}
