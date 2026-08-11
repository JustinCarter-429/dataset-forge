# Carter 1.0 Part 2B.2 Production Orchestration

The production `POST /api/generations` path now uses
`app.carter.production.CarterDatasetGenerationService`, rather than invoking
`RunPodDatasetGenerator`.  The latter remains legacy-only for Dataset Forge's
pre-Carter fixed-record tests and is not a Carter application orchestrator.

At job creation the selected runtime (`runpod` or `local_lm_studio`) is stored
on the job.  The orchestrator maps that pinned value to the one Carter logical
Cloud/Local request contract; it does not read a mutable selector during a run
and does not fall back across providers.

Before model work it loads the immutable 14-file Carter package.  It then runs
the planner, validates and compiles the DatasetSpec with the Part 1 compiler,
performs the bounded three-tool generation loop, validates a
`CarterCanonicalDataset`, runs structured quality review, permits at most one
application-authorized revision, and exports the final dynamic records as JSON
or schema-derived CSV before existing ZIP packaging.

Provider HTTP remains in `RunPodCarterProvider` and `LMStudioCarterProvider`.
Only safe phase/call counts and tool names are retained in job metadata; source
text, prompts, credentials, tool arguments, and reasoning are not added to
artifacts.  A deterministic provider exercises a custom DatasetSpec with
`customer_intent`, `confidence_label`, and `reasoning_style`, proving the new
path does not project through the legacy six-field record shape.

The external RunPod health diagnostic remains a separate live-acceptance gate.
No live inference call is made by this implementation document or its tests.

## Part 2B.3 compatibility reconciliation

The Phase 1 assertions that exercised `/api/generate` and `/api/generations`
were classified as mixed-contract assertions: those routes now intentionally
start a Carter job, but the assertions still expected the former fixed-record
artifact.  Dataset Forge v1.0.0 compatibility preserves upload, request,
asynchronous job, polling, format-selection, ZIP/download, and error lifecycle
behavior.  It does not require a Carter dataset to be projected into six legacy
fields.  The reconciled checks validate DatasetSpec-driven JSON and CSV fields,
evidence serialization, and absence of fabricated `instruction` fields.

The deterministic browser acceptance runs through the production UI and API:
five Playwright checks cover upload limits, document Ask behavior, runtime
unavailability isolation, generation failure and validation failure handling,
advisory review completion, download readiness, keyboard access, and responsive
layout.  The deterministic adapter provides only test scenarios; it does not
alter the immutable Carter contracts or the live provider adapters.

Part 2B.4 adds a per-run safe invocation ledger to the production orchestrator.
It records only the Carter phase and selected provider runtime; no prompt,
source, credential, tool-argument, or reasoning data is retained.  Deterministic
coverage confirms planner, generation, tool continuation, review, and revision
remain on the selected RunPod or LM Studio adapter, and malformed selected-runtime
generation does not trigger a cross-runtime retry.

Active-run immutability is exercised with a controlled per-provider callback:
Run A changes only a future-run selector during its generation turn, then its
tool continuation, review, and revision remain on its already captured RunPod
adapter.  Run B is subsequently constructed with the new LM Studio selection.

## Part 2B.7 final backend certification and RunPod live re-entry (2026-08-10)

Certification used the accepted **exhaustive node-level machine-readable
fallback**: one pytest node per process, each with a unique JUnit XML report
and basetemp.  Temporary reports were written under `%TEMP%` and are not
repository artifacts.

### Remaining Phase 1 nodes

All seven nodes collected one case, passed one case, skipped/failed/errored
zero cases, and exited with code 0:

| Node ID | JUnit report |
| --- | --- |
| `tests/test_phase1.py::test_supported_files_complete_and_json_zip_is_safe` | `%TEMP%/p2b7-phase1-01.xml` |
| `tests/test_phase1.py::test_csv_zip_contains_manifest_and_dataset` | `%TEMP%/p2b7-phase1-02.xml` |
| `tests/test_phase1.py::test_unknown_and_traversal_downloads_are_safe` | `%TEMP%/p2b7-phase1-03.xml` |
| `tests/test_phase1.py::test_resource_api_upload_generation_status_and_download` | `%TEMP%/p2b7-phase1-04.xml` |
| `tests/test_phase1.py::test_generation_download_before_completion_is_safe` | `%TEMP%/p2b7-phase1-05.xml` |
| `tests/test_phase1.py::test_corrupt_pdf_generation_fails_truthfully` | `%TEMP%/p2b7-phase1-06.xml` |
| `tests/test_phase1.py::test_csv_neutralizes_formula_like_values` | `%TEMP%/p2b7-phase1-07.xml` |

Result: **7/7 PASS**.

### Extraction Phase 2 inventory and results

Collection discovered exactly five nodes.  Each collected/passed one case,
skipped/failed/errored zero cases, and exited with code 0.

| Node ID | JUnit report | Duration |
| --- | --- | --- |
| `tests/test_extraction_phase2.py::test_txt_preserves_order_unicode_and_legitimate_repetition` | `%TEMP%/p2b7-extraction-01.xml` | 0.690 s |
| `tests/test_extraction_phase2.py::test_txt_empty_is_rejected` | `%TEMP%/p2b7-extraction-02.xml` | 0.568 s |
| `tests/test_extraction_phase2.py::test_pdf_docx_real_docling_extraction_preserves_sentinels` | `%TEMP%/p2b7-extraction-03.xml` | 18.270 s |
| `tests/test_extraction_phase2.py::test_analysis_is_deterministic` | `%TEMP%/p2b7-extraction-04.xml` | 0.811 s |
| `tests/test_extraction_phase2.py::test_corrupt_pdf_maps_to_truthful_error` | `%TEMP%/p2b7-extraction-05.xml` | 8.711 s |

Result: **5/5 PASS**.

### Exhaustive accounting

Collection-only inventory covered all 11 backend files and found 125 distinct
node IDs: 3, 5, 24, 10, 18, 5, 12, 18, 11, 6, and 13 respectively in the
authoritative file order.  The prior certified baseline was 112 nodes.  The
remaining Phase 1 and extraction work added 12 nodes; a real production
adapter defect found during live planner re-entry required one additional
focused regression node.  The affected `test_carter_production.py` nodes were
re-executed individually after the fix.

| Metric | Result |
| --- | --- |
| Backend files represented | 11/11 |
| Missing node IDs | 0 |
| Duplicate certified node IDs | 0 |
| Previously certified | 112 |
| Newly certified | 13 |
| Final collected / passed / skipped / failed / errors | 125 / 125 / 0 / 0 / 0 |
| All exit codes zero | Yes |

### RunPod live re-entry

The real production `RunPodCarterProvider` was used with configured
environment credentials; no credential, authorization header, prompt content,
or reasoning content was persisted here.  Minimal health passed: authenticated
`/health`, then one bounded no-tool inference was accepted, traversed
`IN_QUEUE` then `IN_PROGRESS`, and completed with usable output (22.656 s).

The first production planner attempt revealed that the prompt-package request
did not expose `tool_choice` or `max_tokens`, although the RunPod adapter
requires both.  The shared request contract now supplies the existing defaults
(`tool_choice="auto"`, `max_tokens=4096`), and a focused regression test
verifies the adapter accepts a production prompt-package request.  The retry
submitted successfully but the external RunPod job reached terminal `FAILED`.
The adapter exposes no safe worker detail for that terminal state.  In
accordance with the live sequence, generation, tool use, review, revision, and
export were not attempted.

Live status: minimal health **PASS**; planner **FAIL** (provider terminal
failure / insufficient exposed detail); generation, tool flow, review, and
export **BLOCKED**.  There was no LM Studio crossover.
