import { Circle, CircleCheck, CircleX, LoaderCircle } from 'lucide-react'
import type { GenerationJob } from '../api/types'

const stages = [
  { key: 'extracting', title: 'Document extracted', copy: 'Text and tables extracted successfully' },
  { key: 'analyzing', title: 'Content analyzed', copy: 'Structure and key themes identified' },
  { key: 'planning', title: 'Dataset planned', copy: 'DatasetSpec and output schema prepared' },
  { key: 'generating', title: 'Generating training examples', copy: 'Creating dataset records with Carter 1.0' },
  { key: 'validating', title: 'Validating dataset', copy: 'Checking records, evidence, and quality' },
  { key: 'packaging', title: 'Preparing ZIP download', copy: 'Packaging validated outputs' },
]

export function GenerationStatus({ job }: { job: GenerationJob | null }) {
  const index = stages.findIndex(item => item.key === job?.stage); const completed = job?.status === 'completed'; const failed = job?.status === 'failed'; const cancelled = job?.status === 'cancelled'; const batch = job?.batch
  const extractionCopy = job?.analysis ? `${job.analysis.availablePageCount ? `${job.analysis.availablePageCount} pages · ` : ''}${job.extraction?.statistics.wordCount.toLocaleString()} words extracted` : 'Docling/Python extraction in progress'
  const analysisCopy = job?.analysis ? `${job.analysis.sectionCount} sections · ${job.analysis.tableCount} tables identified` : 'Deterministic structure analysis'
  const generationCopy = batch ? `Batch ${batch.currentBatch} of ${batch.totalBatches} · ${batch.recordsGenerated} / ${batch.recordsRequested} records generated` : job?.provider?.state === 'queued' ? 'Waiting for AI worker...' : stages[3].copy
  const reviewActive = ['reviewing', 'revising', 'revalidating'].includes(job?.progress.currentStage || '')
  return <section className="side-card" aria-live="polite"><div className="side-card-heading"><div><p className="eyebrow">Live pipeline</p><h2>Generation Status</h2></div><span className={`status-chip ${completed ? 'success' : failed || cancelled ? 'danger' : ''}`}>{completed ? 'Ready' : failed ? 'Failed' : cancelled ? 'Cancelled' : job ? `${job.progress.percent}%` : 'Pending'}</span></div><div className="status-list">{stages.map((stage, i) => { const done = completed || index > i || (reviewActive && i < 4); const current = !completed && !failed && !cancelled && (index === i || (reviewActive && i === 4)); return <div className="status-item" key={stage.key}><div className={`status-icon ${done ? 'done' : current ? 'active' : failed && index === i ? 'failed' : ''}`}>{done ? <CircleCheck size={17} /> : failed && index === i ? <CircleX size={17} /> : current ? <LoaderCircle size={17} className="spin" /> : <Circle size={17} />}</div><div><strong>{stage.title}</strong><span>{stage.key === 'extracting' ? extractionCopy : stage.key === 'analyzing' ? analysisCopy : stage.key === 'generating' ? generationCopy : stage.copy}</span>{stage.key === 'generating' && batch && <div className="batch-progress" role="progressbar" aria-label={`${batch.recordsGenerated} of ${batch.recordsRequested} records generated`} aria-valuemin={0} aria-valuenow={batch.recordsGenerated} aria-valuemax={batch.recordsRequested}><i style={{ width: `${Math.round(100 * batch.recordsGenerated / batch.recordsRequested)}%` }} /></div>}</div></div> })}</div>{cancelled && <p className="error-banner">Generation cancelled. No package was created.</p>}{failed && <p className="error-banner">{job.error?.message || 'Generation failed. Your inputs have been preserved.'}</p>}</section>
}
