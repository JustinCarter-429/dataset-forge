import { CheckCircle2, Database, FileArchive, TriangleAlert, XCircle } from 'lucide-react'
import type { GenerationJob } from '../api/types'

const size = (n: number | null) => n == null ? '—' : n < 1024 ? `${n} B` : `${(n / 1024).toFixed(1)} KB`

export function DatasetPreview({ job }: { job: GenerationJob | null }) {
  const output = job?.output
  const validation = job?.validation
  const report = job?.validationReport
  const quality = output?.qualitySummary
  const metrics = [
    ['Records', output?.recordCount == null ? '—' : String(output.recordCount), 'records'],
    ['Format', output?.requestedFormat?.toUpperCase() || '—', 'format'],
    ['Size', size(output?.sizeBytes ?? null), 'package'],
    ['Validation', validation?.qualityStatus === 'passed_with_warnings' ? 'Warnings' : validation?.schemaValid ? 'Passed' : validation?.schemaValid === false ? 'Failed' : '—', 'schema'],
  ]
  const analysis = job?.analysis
  const groundingPassed = report?.grounding.status === 'passed'
  return <section className="side-card" aria-live="polite">
    <div className="side-card-heading"><div><p className="eyebrow">Backend summary</p><h2>Dataset Preview</h2></div><Database size={19} color="#98a2b3" /></div>
    <div className="metric-grid">{metrics.map(([label, value, note]) => <div className="metric-card" key={label}><span>{label}</span><strong>{value}</strong><small>{note}</small></div>)}</div>
    {analysis && <div className="extraction-summary"><strong>Extraction</strong><span>{analysis.sectionCount} sections · {analysis.tableCount} tables · {analysis.contentVolume.toLocaleString()} characters</span></div>}
    {report ? <div className={`validation-summary ${groundingPassed ? 'validation-passed' : 'validation-failed'}`}>
      <div>{groundingPassed ? <CheckCircle2 size={16} /> : <XCircle size={16} />}<strong>{groundingPassed ? 'Source grounding checks passed' : 'Source grounding checks failed'}</strong></div>
      <span>{report.grounding.grounded_records} / {report.grounding.total_records} records grounded</span>
      <span>{report.grounding.verified_evidence_items} / {report.grounding.total_evidence_items} evidence references verified</span>
      {report.quality.warnings.length > 0 && <span className="validation-warning"><TriangleAlert size={14} /> {report.quality.warnings.length} quality warning{report.quality.warnings.length === 1 ? '' : 's'}</span>}
    </div> : <div className="grounding-note"><FileArchive size={16} /><div><strong>Source grounding pending</strong><span>Dataset Forge verifies evidence against extracted source units.</span></div></div>}
    {quality && <div className={`validation-summary ${quality.exportEligible ? 'validation-passed' : 'validation-failed'}`}><div>{quality.exportEligible ? <CheckCircle2 size={16} /> : <XCircle size={16} />}<strong>Quality check {quality.exportEligible ? 'passed' : 'blocked'}</strong></div><span>{quality.acceptedRecords} accepted · {quality.quarantinedRecords} quarantined · {quality.rejectedRecords} rejected</span>{quality.findings.length > 0 && <span className="validation-warning"><TriangleAlert size={14} /> {quality.findings.map(item => item.code).slice(0, 3).join(', ')}</span>}</div>}
    {job?.qualityReview && <div className="quality-review-summary">
      <div><CheckCircle2 size={16} /><strong>AI quality review {job.qualityReview.status === 'passed_with_warnings' ? 'completed with warnings' : 'completed'}</strong></div>
      <span>{job.qualityReview.blockingIssues} blocking issues · {job.qualityReview.warnings} warnings</span>
      <span>{job.qualityReview.revisionSucceeded ? '1 bounded quality revision applied' : job.qualityReview.revisionAttempted ? 'Bounded revision attempted' : 'Revision not needed'}</span>
      {job.qualityReview.issues.length > 0 && <details><summary>Review details</summary><ul>{job.qualityReview.issues.slice(0, 5).map(issue => <li key={`${issue.code}-${issue.recordIds.join('-')}`}><strong>{issue.code}</strong>: {issue.message}</li>)}</ul></details>}
    </div>}
  </section>
}
