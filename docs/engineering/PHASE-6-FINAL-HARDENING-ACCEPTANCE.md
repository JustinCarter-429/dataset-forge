# Phase 6 — Final hardening and acceptance

Date: 2026-08-10

## Release decision

Status: **COMPLETE / POC RELEASE READY**. Phase 6A deterministically reproduced the original multi-batch provenance failure, fixed it without weakening grounding, and the final two-batch RunPod/browser acceptance completed with all eight records and all eight evidence items grounded.

## Architecture inventory

React/Vite frontend → FastAPI → Docling/TXT extraction → canonical extraction model → bounded context planner → Techie custom agentic agents → Dataset Schema 2.0 → deterministic schema/grounding/duplicate validation → bounded AI review/revision → JSON/CSV packaging.

Phase 6 preserved the existing architecture and added cancellation state, a process-local one-active-generation guard, stale-upload cleanup, safe download filenames, response security headers, keyboard upload activation, same-external-job status retry, global application-owned record ID rebasing during multi-batch assembly, and immutable batch-scoped provider aliases. Provider-visible `source_1...source_N` aliases are resolved to canonical extraction IDs before assembly; unknown, canonical-ID, and cross-batch references fail closed, and evidence is verified first against the exact quoteable projection supplied to the model and then by the complete Phase 4 validator.

## Multi-batch grounding closure

The original failed real run (`[redacted provider job id]`) used 13 extracted units and two planned batches. It reached `completed: 1 / total: 2`, failed at validation, and produced neither quality review nor a package. Its process-local runtime record payload was not persisted, so the raw model rows cannot be honestly reconstructed after process exit. The preserved run state, source fixture, and failure code establish the failure class: batch-local aliases such as `source_1` were allowed to cross the batch boundary uncanonicalized and were subsequently treated as global extraction IDs. The first invariant break was **alias mapping / pre-assembly canonicalization**, not extraction, planning, model output, or the final Phase 4 validator.

The deterministic regression fixture represents the retained failure class record-for-record: batch 1 and batch 2 both return provider-visible `source_1`; pre-fix global assembly fails Phase 4 with `INVALID_SOURCE_REF` and `EVIDENCE_NOT_FOUND`; fixed assembly resolves the first to its batch-1 canonical ID and the second to its batch-2 canonical ID before any global record exists. Aliases are intentionally reused between batches; temporary provider record IDs may also be reused, and application record IDs are globally rebased. Prompt text and quote verification now share `quoteable_source_text`, including table row projection, so model-visible and validator-visible source text are identical.

### Final real-run ledger

Generation (`[redacted provider job id]`) used 13 extracted units in two batches. The provider returned local aliases, which were resolved before assembly; the packaged ledger below shows only the resulting canonical values. Every source was in the origin batch, each referenced extraction unit existed, every normalized evidence comparison matched, and no validation issue was emitted.

| Final record ID | Origin | Canonical source/evidence ref | Allowed aliases | Quote chars | Unit exists | Match | Issue |
|---|---|---|---|---:|---|---|---|
| `...-1` | batch-1 | `c7810a0e4e4f0e444dc9` | `source_1..source_8` | 59 | yes | yes | — |
| `...-2` | batch-1 | `c1b06bd247543c04ab7d` | `source_1..source_8` | 88 | yes | yes | — |
| `...-3` | batch-1 | `a2f14f365b97e420dd57` | `source_1..source_8` | 80 | yes | yes | — |
| `...-4` | batch-1 | `31b7df00fd12f3f6860a` | `source_1..source_8` | 92 | yes | yes | — |
| `...-5` | batch-2 | `6afe5b2e6ee3558a8c84` | `source_1..source_5` | 103 | yes | yes | — |
| `...-6` | batch-2 | `b515e0602cf31f634518` | `source_1..source_5` | 94 | yes | yes | — |
| `...-7` | batch-2 | `eceacdb6a6c4a5fbe43a` | `source_1..source_5` | 94 | yes | yes | — |
| `...-8` | batch-2 | `3d2141000102993f542a` | `source_1..source_5` | 98 | yes | yes | — |

The Phase 6A fix uses prompt `dataset-author-v3`: each batch exposes only immutable local aliases, admits no provider canonical IDs, resolves aliases before assembly, rejects unknown/cross-batch references, requires source/evidence pairing, checks the exact projected source text, and rebases application-owned record IDs globally. Deterministic alias, collision, unknown-reference, cross-batch, evidence-pairing, ambiguous-evidence, partial-failure, and multi-batch assembly tests pass.

The prior parse-failure run (`[redacted provider job id]`) remains retained defect history. The final successful real acceptance was generation (`[redacted provider job id]`) with 13 units and two planned batches. It returned 8 final records, grounded 8/8 records and 8/8 evidence items, found zero invalid/cross-batch/unresolved references, then ran one quality review with no revision. The JSON ZIP endpoint returned HTTP 200 and a 4,102-byte archive; the UI completed the Playwright flow at `http://127.0.0.1:5173`.

## Provider accounting policy

`providerSubmitAttempts` counts POST `/run` attempts. `/run` is deliberately single-shot because an unknown POST outcome could create a duplicate paid job. `providerJobsCreated`, `providerJobsCompleted`, and `providerJobsFailed` count external inference jobs. `providerStatusPolls`, `providerTransportRetries`, and `providerCancelCalls` count transport operations only; status polling never counts as inference. `qualityReviewJobs` and `revisionJobs` are subsets of external inference jobs. Dataset Forge does not use RunPod `/retry`.

Temporary status errors retry the same external job at most twice. Auth errors are non-retryable. 429/5xx/network errors remain bounded by the provider policy. The Phase 5 global one-revision budget remains one; structural repair consumes the same global budget.

## Acceptance matrix

| Test | Type | Result | Evidence | Notes |
|---|---|---:|---|---|
| TXT upload/validation | Deterministic + prior real | Pass | `tests/test_phase1.py`, Phase 5 real package | UTF-8/CP1252 supported |
| DOCX extraction | Deterministic | Pass | `tests/test_extraction_phase2.py` | Docling structure regression |
| PDF extraction | Deterministic | Pass | `tests/test_extraction_phase2.py` | OCR/scanned-only limitation remains |
| Corrupt/oversized/unsafe file | Deterministic | Pass | `tests/test_phase1.py` | 25 MB backend-enforced limit |
| Single-batch generation/review/package | Real prior acceptance | Pass | Phase 5 acceptance and retained package | 2 records, grounded |
| Multi-batch planner/all-unit allocation | Deterministic | Pass | `tests/test_phase6_hardening.py` | Stable ordered source allocation |
| Multi-batch provider assembly | Real | Pass | Browser run (`[redacted provider job id]`) | 2 generation batches, 8/8 source-grounded, then 1 quality-review job |
| Cross-batch references | Deterministic | Pass | Phase 4 grounding tests | Unknown refs fail closed |
| Cancellation/idempotency | Deterministic | Pass | `tests/test_phase6_hardening.py` | No package after cancel |
| Poll retry/no resubmit | Deterministic | Pass | `tests/test_phase6_hardening.py` | One `/run`, same job status polls |
| JSON/CSV/package/manifest | Deterministic + prior real | Pass | Phase 4/5 tests and ZIP inspection | CSV derives from canonical records |
| Accessibility basics | Static/browser inspection | Pass | keyboard upload, labels, focus styles | Full axe suite not installed |
| Responsive behavior | Browser inspection | Pass | prior matrix: 1440, 1280, 768, 390 | no known horizontal overflow |
| Security/privacy scan | Source/package inspection | Pass | redacted env audit and package checks | no secret or reasoning fields |
| Runtime cleanup | Deterministic | Pass | `storage/cleanup.py` | stale uploads only; completed outputs retained |

## Test counts

- Backend: **65 collected; targeted Phase 6A 13 passed and all remaining backend modules passed** using the bundled Python runtime; default tests make no real RunPod calls.
- Phase 6 targeted hardening: **13 passed**.
- Frontend: **3 passed**; typecheck passed; production build passed.
- Browser: final Playwright upload, two-batch generation, grounding display, quality review, package-ready state, and ZIP endpoint acceptance passed.
- Health: `GET /api/health` returns healthy without provider secrets.

## Runtime and security

Uploads are process-local and stale upload files older than 24 hours are removed at startup. Completed output directories remain for the process/local retention period so a user can download them. Active jobs do not survive backend restart. ZIP downloads use a sanitized source-derived filename. `X-Content-Type-Options: nosniff` and `Referrer-Policy: same-origin` are set; CORS is limited to the configured local frontend origins. Source/provider logging is bounded to IDs, stages, counts, and safe status metadata.

## Final provider accounting

Final successful run: 3 submit attempts, 3 jobs created/completed, 2 generation jobs, 1 quality-review job, 0 revision jobs, 145 status polls, 0 transport retries, 0 cancels, and 0 duplicate provider jobs. The cancelled prompt-format attempt occurred before conforming acceptance and produced no package; it is excluded from the acceptance totals.

Git is not configured for this standalone directory; no repository was initialized.
