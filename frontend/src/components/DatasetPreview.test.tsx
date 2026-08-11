import { render, screen } from '@testing-library/react'
import { DatasetPreview } from './DatasetPreview'
import type { GenerationJob } from '../api/types'

const job = (status: 'passed' | 'failed', warnings = 0): GenerationJob => ({
  id: 'g', status: 'completed', stage: 'completed', progress: { percent: 100, currentStage: 'completed' }, file: { id: 'f', name: 'source.txt' },
  output: { requestedFormat: 'json', recordCount: 2, finalRecordCount: 2, sizeBytes: 1024 }, packageReady: status === 'passed',
  validation: { schemaValid: status === 'passed', totalRecords: 2, validRecords: status === 'passed' ? 2 : 0, invalidRecords: status === 'passed' ? 0 : 2, groundingStatus: status, groundedRecords: status === 'passed' ? 2 : 0, totalEvidenceItems: 2, verifiedEvidenceItems: status === 'passed' ? 2 : 0, qualityStatus: warnings ? 'passed_with_warnings' : status, exactDuplicatesRemoved: 0, nearDuplicatePairs: warnings },
  validationReport: { schema_version: '2.0', status, schema_valid: status === 'passed', records: { generated: 2, final: 2, valid: status === 'passed' ? 2 : 0, invalid: status === 'passed' ? 0 : 2 }, grounding: { status, total_records: 2, grounded_records: status === 'passed' ? 2 : 0, ungrounded_records: status === 'passed' ? 0 : 2, total_evidence_items: 2, verified_evidence_items: status === 'passed' ? 2 : 0, failed_evidence_items: status === 'passed' ? 0 : 2, grounding_percent: status === 'passed' ? 100 : 0 }, duplicates: {}, quality: { status: warnings ? 'passed_with_warnings' : status, category_count: 1, difficulty_distribution: { easy: 2 }, exact_duplicates_removed: 0, near_duplicate_pairs: warnings, warnings: warnings ? [{ code: 'NEAR_DUPLICATE', severity: 'warning', message: 'Potential near-duplicate examples detected.' }] : [] }, issues: [] }, capabilities: { extraction: 'txt', generation: 'runpod', groundingValidation: 'phase4' }, provider: { state: 'completed' }, analysis: { sectionCount: 0, tableCount: 0, contentVolume: 100, availablePageCount: null }, batch: { currentBatch: 1, totalBatches: 1, recordsGenerated: 2, recordsRequested: 2, currentBatchTarget: 2 },
})

test('renders backend-derived passed grounding and evidence counts', () => {
  render(<DatasetPreview job={job('passed')} />)
  expect(screen.getByText('Source grounding checks passed')).toBeInTheDocument()
  expect(screen.getByText('2 / 2 records grounded')).toBeInTheDocument()
  expect(screen.getByText('2 / 2 evidence references verified')).toBeInTheDocument()
})

test('renders failed grounding and warning state from the backend', () => {
  render(<DatasetPreview job={job('failed', 1)} />)
  expect(screen.getByText('Source grounding checks failed')).toBeInTheDocument()
  expect(screen.getByText(/quality warning/)).toBeInTheDocument()
})

test('renders bounded AI review details without a numeric score', () => {
  const reviewed = job('passed')
  reviewed.qualityReview = { reviewVersion: '1.0', provider: 'runpod_serverless', model: 'gpt-oss-20b', promptVersion: 'dataset-quality-review-v1', reviewBatchCount: 1, completedReviewBatches: 1, issuesFound: 1, blockingIssues: 0, warnings: 1, revisionRequired: false, revisionAttempted: false, revisionSucceeded: false, summary: 'One warning.', status: 'passed_with_warnings', issues: [{ code: 'REPETITIVE_RECORDS', severity: 'warning', recordIds: ['g-1'], message: 'Examples repeat the same pattern.', suggestedAction: 'Vary the testing concept.' }] }
  render(<DatasetPreview job={reviewed} />)
  expect(screen.getByText('AI quality review completed with warnings')).toBeInTheDocument()
  expect(screen.getByText('0 blocking issues · 1 warnings')).toBeInTheDocument()
  expect(screen.queryByText(/% AI quality|confidence/)).not.toBeInTheDocument()
})
