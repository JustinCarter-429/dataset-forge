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
