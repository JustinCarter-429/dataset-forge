# Dataset Forge Engineering Implementation Log

## Phase 1A — UI contract integration and async workflow

Date: 2026-08-09

### Scope

Evolved the existing Phase 1 placeholder application into the Dataset Forge single-page workflow without adding authentication, persistence, model calls, Docling, RunPod, or external services.

### Meaningful changes

- Added resource-oriented file and generation APIs: `POST /api/files`, `POST /api/generations`, `GET /api/generations/{id}`, and `GET /api/generations/{id}/download`.
- Preserved `/api/generate` and `/api/download/{job_id}` as legacy compatibility routes using the same pipeline services.
- Added a thread-safe process-local `InMemoryJobStore` and backend-owned job state, stage, progress, output metrics, validation summary, package readiness, and safe failure contract.
- Kept extraction and dataset generation explicitly placeholder-only. The API exposes capability metadata so the UI does not claim source extraction, gpt-oss generation, or grounding validation.
- Added package `metadata.json`, `README.txt`, and `manifest.json` while retaining `generation_manifest.json`. CSV continues to derive deterministically from canonical JSON and now has a documented safe package path.
- Replaced the previous centered form with the Dataset Forge header, centered shell, state-driven stepper, responsive two-column workspace, generation status card, backend-derived preview metrics, and download card.
- Centralized the frontend resource API client and added one-second bounded polling that stops on completed/failed jobs or unmount.

### Verification evidence

- Backend: `11 passed` with pytest.
- Frontend: `npm run typecheck` passed; `npm run build` passed.
- Live API: `/api/health` returned `{"status":"ok","service":"document-dataset-maker"}`.
- Resource API tests covered safe upload names, missing file IDs, completed state, record count, output size, schema validation, grounding not evaluated, package readiness, and download.
- ZIP tests covered JSON/CSV package contents, manifest metadata, and safe paths.
- Browser preview opened at `http://127.0.0.1:5173`; the smoke TXT file uploaded through the real file chooser, the prompt enabled generation, backend-derived completion showed 3 records / JSON / 1.4 KB / Passed, grounding remained not evaluated, and the ZIP download action was enabled and exercised.
- The preview was checked at the available 1280px browser viewport with no page-wide horizontal overflow. The in-app browser did not advertise a viewport override capability, so 1440/768/390 overrides were not available in this environment; the responsive CSS includes the required stack breakpoint.
- Git was not initialized for this standalone directory, so no commit was created.

### Known limitations

Active uploaded files and generation jobs are process-local and do not survive a backend restart. Extraction, semantic analysis, real gpt-oss-20b generation, grounding, and quality validation remain Phase 2+ placeholders.

## Phase 2 — Docling extraction and canonical source preparation

Date: 2026-08-09

### Meaningful changes

- Added real Docling 2.118.1 extraction for PDF and DOCX, plus direct UTF-8/CP1252 TXT extraction.
- Added a canonical extraction model with stable element IDs, source locations, document statistics, validation, and deterministic analysis.
- Integrated extraction and analysis into the existing asynchronous generation state machine while preserving the Phase 1 resource and legacy routes.
- Kept OCR, Docling table-structure inference, model generation, grounding, RAG, and persistence out of scope; the placeholder generator remains the Phase 3 boundary.
- Updated the UI to report backend-derived extraction words, pages when available, sections, tables, validation, and the existing placeholder-generation limitation.

### Verification evidence

- Backend: `17 passed` with pytest, including real PDF/DOCX Docling extraction, TXT ordering/Unicode/repeated-content coverage, corrupt-document failure mapping, and Phase 1 route/ZIP regression coverage.
- Frontend: `npm run typecheck` passed; `npm run build` passed.
- Docling integration: PDF and DOCX fixtures retained ordered sentinels, headings, table cells, and page provenance; corrupt PDF returned a safe `CORRUPT_DOCUMENT` failure.

## Phase 3 - RunPod gpt-oss-20b generation

Date: 2026-08-09

### Scope

Replaced the production placeholder generation boundary with a RunPod Serverless/vLLM provider abstraction, bounded canonical context batching, versioned dataset-authoring prompt, JSON-schema request, authoritative canonical validation, one bounded repair, real provider-aware job metadata, and truthful Phase 3 UI status. Phase 2 extraction remains the source of truth.

### Meaningful changes

- Added `backend/app/providers/contracts.py`, `config.py`, and `runpod.py` for provider-neutral configuration, native `/run` + `/status` polling, `/health`, timeout/error mapping, and backend-only authorization.
- Added `backend/app/services/context_projection.py` and `backend/app/prompts/dataset_author_v1.py` for ordered source projection, conservative token estimation, batching, hard record caps, and prompt versioning.
- Replaced placeholder production generation with `RunPodDatasetGenerator`; missing configuration fails with `RUNPOD_CONFIGURATION_REQUIRED` and never falls back to fake records.
- Added provider/model/batch state to the generation API and updated the existing status/preview UI for waiting workers, batch progress, gpt-oss generation, schema validation, and Phase 4 grounding status.
- Updated package metadata, README, environment example, and Phase 3 engineering documentation.

### Verification evidence

- Backend: `23 passed` with mocked provider transport and Phase 2 regressions; no default test performs a real RunPod call.
- Frontend: typecheck passed; production build passed.
- RunPod real health/compatibility/generation: not run because endpoint ID and API key are not configured.
- Embedded browser: unavailable because the local URL is refused by the embedded browser environment; no false browser-acceptance claim is made.

### Phase 3B real acceptance audit

- Loaded safe RunPod configuration from `backend/.env`; endpoint/key/model/max length were valid without recording the secret.
- One real RunPod health call passed with HTTP 200 and approximately 388 ms latency.
- The first compatibility result exposed a wrapped vLLM choices response; the parser was corrected and regression-tested. The one corrected compatibility job still returned no parseable JSON content to the probe, so the acceptance stopped before real dataset generation and browser acceptance.
- Final deterministic backend result: `24 passed`; frontend typecheck/build passed. No real generation job or ZIP was produced.

## Phase 3C — RunPod passthrough correction and acceptance

Date: 2026-08-09

### Meaningful changes

- Replaced the structured-generation shorthand payload with a shared native RunPod `/run` job containing `input.openai_route="/v1/chat/completions"` and `input.openai_input`.
- Added configured model provenance, safe request/output shape telemetry, current vLLM `structured_outputs.json`, explicit worker-error classification, and assistant-content extraction that never treats reasoning as the dataset result.
- Added deterministic fixtures for current Chat Completions, array-wrapped output, legacy text, empty/invalid shapes, reasoning-only output, worker errors, and passthrough serialization.

### Acceptance evidence

- Official worker README checked: current worker source reports vLLM `0.20.2`; exact deployed image tag is not exposed by the endpoint.
- The final compatibility job (`[redacted provider job id]`) completed through native RunPod polling and validated the tiny `{status: "ok"}` schema.
- One real dataset job completed from the QA smoke TXT document with 2 records, canonical validation passed, and no placeholder fallback. ZIP inspection passed; CSV was derived from canonical JSON without a second generation.
- Live browser acceptance passed upload, prompt, generation, completion, preview, validation, and ZIP endpoint verification. Responsive checks at 1440x900, 1280x720, 768x1024, and 390x844 found no horizontal overflow.
- Full backend regression: `35 passed`; frontend typecheck and production build passed; default tests made zero RunPod calls.

### Known limitations

Grounding quality remains `not_evaluated` until Phase 4. The process-local job store remains non-durable, and the exact deployed worker image tag could not be established from endpoint metadata. Git is not configured for this standalone directory.

### Next step

## Phase 4 — Dataset schema, grounding, validation, and quality

Date: 2026-08-09

### Meaningful changes

- Added canonical Dataset schema `2.0` with source references, verified evidence, category, difficulty, and backend-owned metadata.
- Added deterministic normalization, contiguous evidence verification against extracted source units, strict all-record grounding gates, required-field checks, and safe validation failure handling.
- Added deterministic exact-duplicate removal, near-duplicate warnings, category/difficulty quality metrics, and `validation-report.json` packaging.
- Updated the RunPod authoring prompt and bounded source projection to request evidence without changing the Phase 3 RunPod transport.
- Updated the API and Dataset Preview to show backend-derived schema, grounding, evidence, duplicate, and quality results. Added frontend tests for passed and failed grounding states.

### Verification evidence

- Backend: `46 passed`; frontend: `2 passed`; frontend typecheck and production build passed.
- One real RunPod Phase 4 dataset job completed with 2 final records, 2/2 grounded records, 2/2 verified evidence items, zero exact duplicates removed, zero near-duplicate warnings, and quality status `passed`.
- JSON ZIP inspection passed with canonical dataset, validation report, metadata, manifests, README, and matching counts. CSV regression passed from the same canonical records without another inference call.
- Browser preview was opened at `http://127.0.0.1:5173` for local UI and responsive inspection.
- Git is not configured for this standalone directory, so no commit was created.

### Next step

Phase 5 — Agentic self-check, bounded repair, optional validator/research, and quality controls.

Phase 4 — Dataset schema, source grounding, validation, and quality UI.
## Phase 5 — Bounded agentic quality review

Date: 2026-08-09

Starting state: Phase 4 was complete with schema 2.0, deterministic grounding, validation reports, JSON/CSV packaging, and a real RunPod generation acceptance.

Objective: add one advisory quality-review operation, one global revision budget, safe targeted revision, full Phase 4 revalidation, truthful review status, and quality-review packaging without starting Phase 6.

Architecture: added `dataset-quality-review-v1` and `dataset-quality-revision-v1`; constrained review contracts and issue codes; same RunPod provider/client; deterministic review batching/token preflight; application-owned severity and revision decisions; targeted lineage metadata; `quality-review.json`; review/revision status in the API and single-page UI; disabled-by-default public research foundation.

Provider calls: one successful real Phase 5 acceptance used 2 RunPod jobs total — one dataset generation and one quality review. No compatibility call and no revision call were needed for the final acceptance. Earlier bounded acceptance attempts failed safely during validation/revision and produced no packages.

Verification: backend `52 passed`; frontend `3 passed`; typecheck/build passed; mock no-revision, warnings-only, targeted revision, failed revalidation, schema negative, unknown-ID, authority-field, and bounded-call tests passed. Final real job produced 2 records, 2/2 grounded, 2/2 evidence verified, 1 warning, 0 blocking issues, revision not required. ZIP and CSV regression passed; browser review state, download, responsive checks, and security/privacy checks passed.

Files: `backend/app/services/quality_review.py`, `backend/app/prompts/dataset_quality_review_v1.py`, `docs/engineering/PHASE-5-AGENTIC-QUALITY-REVIEW.md`, plus updates to the domain models, pipeline, RunPod route integration, packaging, frontend types/components/tests, README, and environment example.

Known limitations: process-local jobs remain non-durable; AI review is advisory and does not establish correctness; public research remains disabled by default. Next phase is Phase 6 final hardening and release readiness.

## Phase 6 — Final hardening and acceptance

Date: 2026-08-09

Implemented cancellation endpoint/state with idempotent provider cancellation, one-active-generation protection, stale upload cleanup, safe source-derived download filenames, response security headers, keyboard upload activation, same-external-job status retry, bounded provider accounting, globally unique application-owned multi-batch record IDs, a representative synthetic multi-batch fixture, and Phase 6 deterministic regression coverage.

Verification: backend `56 passed, 0 failed, 4 warnings`; Phase 6 targeted hardening `28 passed`; frontend `3 passed`; typecheck/build passed. Prior real TXT and PDF/Docling acceptance remains valid. A new real browser multi-batch run reached `Batch 1 of 2`, then failed deterministic source/evidence validation and produced no partial package. Therefore Phase 6 is intentionally recorded as **INCOMPLETE / NOT READY** until the configured provider output/grounding interaction is corrected and reaccepted.

Detailed matrix, accounting terminology, security/runtime review, and blocker evidence: `docs/engineering/PHASE-6-FINAL-HARDENING-ACCEPTANCE.md`.

## Phase 6A — Multi-batch grounding closure audit

Date: 2026-08-10

Forensic investigation established that the original `a0f53a96f9b1475dab2b997d2a546d27` failure was a batch-local alias leak: `source_1` was reused by different provider batches and reached global validation without origin-batch resolution. The raw process-local provider payload was not retained, but the preserved run state and deterministic fixture reproduce the exact failure class. The smallest general fix admits only immutable provider-visible aliases for the origin batch, resolves them to canonical extraction IDs before assembly, verifies quotes against the shared quoteable projection, and rebases application-owned record IDs globally. Unknown aliases, canonical IDs supplied by the model, cross-batch refs, wrong-source/ambiguous evidence, duplicate temporary IDs, partial multi-batch failures, and premature review are all explicit regressions.

Final deterministic verification: 65 backend tests collected (Phase 6A targeted: 13 passed), frontend tests: 3 passed, typecheck/build passed. Final real browser acceptance generation (`[redacted provider job id]`) used 13 extracted units and two generation batches. It completed 8 records, 8/8 grounded records, 8/8 verified evidence items, no invalid/cross-batch/unresolved source refs, then one quality-review job with no revision. Provider accounting: 3 submit attempts/jobs created/completed (2 generation + 1 review), 145 status polls, 0 retries, 0 cancels, 0 duplicate jobs. JSON ZIP endpoint passed (4,102 bytes); CSV remains deterministically derived from the same canonical records and is covered by the backend regression suite. Phase 6 is **complete / POC release ready**; the completed application remains open at `http://127.0.0.1:5173`.

## V1 public GitHub release preparation

Date: 2026-08-09

Prepared the initial public source tree without changing application behavior: added a portfolio README and owned logo asset, a source-available community license with commercial-license guidance, third-party notices, security/contribution documents, changelog, version file, comprehensive ignores, and publication record. The real local backend `.env` was preserved unread and excluded. Historical runtime artifacts and provider job IDs were excluded or redacted from the public tree. Final deterministic verification: backend 65 passed (4 Docling deprecation warnings); frontend 3 passed, typecheck/build passed. Git was initialized on `main`; initial release-preparation commit: `191fe63ca89d70c59e4ba3b18c0094b8c330b5dd`. The public repository is https://github.com/JustinCarter-429/dataset-forge; the final publication documentation commit is pushed before `v1.0.0` is tagged and released.
