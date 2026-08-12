import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { vi } from 'vitest'
import { GenerationStatus } from './GenerationStatus'
import type { GenerationJob } from '../api/types'

const api = vi.hoisted(() => ({ resumeGeneration: vi.fn(), getGeneration: vi.fn() }))
vi.mock('../api/client', () => api)

const job = (changes: Partial<GenerationJob> = {}): GenerationJob => ({
  id: 'resume-job', status: 'failed', stage: 'generating', progress: { percent: 70, currentStage: 'generating' }, file: { id: 'f', name: 'source.txt' },
  output: { requestedFormat: 'jsonl', recordCount: null, finalRecordCount: null, sizeBytes: null }, validation: null, packageReady: false,
  capabilities: { extraction: 'docling', generation: 'carter', groundingValidation: 'grounding' },
  batch: { currentBatch: 4, totalBatches: 4, recordsGenerated: 15, recordsRequested: 20, currentBatchTarget: 5 },
  error: { code: 'PROVIDER_NO_FINAL_CONTENT', message: "Generation paused after 15 records. Carter couldn't complete the next batch after three attempts, but your completed work has been preserved. Resume to continue from where it stopped." },
  recoverable: true, resumeAvailable: true, ...changes,
})

beforeEach(() => vi.clearAllMocks())

test('renders a recoverable provider failure as paused with a resume CTA', () => {
  render(<GenerationStatus job={job()} />)
  expect(screen.getByText('Paused')).toBeInTheDocument()
  expect(screen.getByText('Resume Generation')).toBeInTheDocument()
  expect(screen.queryByText('Failed')).not.toBeInTheDocument()
  expect(screen.getByText(/15 records/)).toBeInTheDocument()
  expect(screen.queryByText('PROVIDER_NO_FINAL_CONTENT')).not.toBeInTheDocument()
})

test('resumes exactly once using the same job and preserves progress', async () => {
  let resolve: (value: GenerationJob) => void = () => undefined
  api.resumeGeneration.mockReturnValue(new Promise<GenerationJob>(done => { resolve = done }))
  render(<GenerationStatus job={job()} />)
  const button = screen.getByRole('button', { name: 'Resume Generation' })
  fireEvent.click(button); fireEvent.click(button)
  expect(api.resumeGeneration).toHaveBeenCalledTimes(1)
  expect(api.resumeGeneration).toHaveBeenCalledWith('resume-job')
  expect(screen.getByRole('button', { name: 'Resuming...' })).toBeDisabled()
  resolve(job({ status: 'generating', recoverable: false, resumeAvailable: false, error: undefined }))
  await waitFor(() => expect(screen.getByText('Generating')).toBeInTheDocument())
  expect(screen.getByText(/15 \/ 20 records generated/)).toBeInTheDocument()
  expect(screen.queryByText('Resume Generation')).not.toBeInTheDocument()
})

test('keeps a non-recoverable failure as failed without a resume CTA', () => {
  render(<GenerationStatus job={job({ recoverable: false, resumeAvailable: false })} />)
  expect(screen.getByText('Failed')).toBeInTheDocument()
  expect(screen.queryByText('Resume Generation')).not.toBeInTheDocument()
})

test('keeps a completed job ready without a resume CTA', () => {
  render(<GenerationStatus job={job({ status: 'completed', stage: 'completed', recoverable: false, resumeAvailable: false, packageReady: true })} />)
  expect(screen.getByText('Ready')).toBeInTheDocument()
  expect(screen.queryByText('Resume Generation')).not.toBeInTheDocument()
})
