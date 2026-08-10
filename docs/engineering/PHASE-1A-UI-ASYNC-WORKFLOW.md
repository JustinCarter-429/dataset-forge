# Phase 1A UI and Async Workflow Contract

## API

1. `POST /api/files` accepts exactly one PDF, DOCX, or TXT multipart upload and returns a server-owned `file.id`, sanitized display name, byte count, MIME type, extension, and `uploaded` status. Extraction begins when a generation is created.
2. `POST /api/generations` accepts `{ fileId, datasetPrompt, outputFormat }`, validates the uploaded file reference, returns `202` with `generationId`, and queues the in-memory background workflow.
3. `GET /api/generations/{generationId}` returns authoritative `status`, `stage`, stage-derived percent, file identity, requested format, record count, output size, schema validation, package readiness, capability flags, and safe failure data.
4. `GET /api/generations/{generationId}/download` returns the ZIP only after `packageReady` is true.

Legacy routes remain available for Phase 1 compatibility and call the same pipeline service.

## State model

`queued → extracting → analyzing → generating → validating → packaging → completed`.

Any active stage may become `failed`. The frontend never invents stage transitions or percentages; it polls the backend and renders the returned state.

## Truthful Phase 1A/Phase 2 behavior

PDF/DOCX extraction uses Docling 2.118.1 and TXT uses a direct safe reader, producing the canonical provenance-preserving extraction model and deterministic analysis. The dataset generator still returns three deterministic placeholder records. The validator performs real structural checks. `groundingStatus` is `not_evaluated`, and the UI explicitly describes model generation as a future capability.

## Package contract

The canonical dataset is exported to `dataset.json` or deterministically converted to `dataset.csv`. The ZIP also contains `metadata.json`, `README.txt`, `manifest.json`, and the retained `generation_manifest.json`. Metadata uses `generationMode: phase1_placeholder` and never claims a model/provider.

## Frontend interaction

The upload card calls `/api/files` before marking the file ready. The generate button remains disabled until the backend has accepted a file, the prompt is non-blank, and no job is running. A generation job is polled every second. Preview records, size, format, validation, and download state come only from the status response. Reset returns the page to its initial state.

## Responsive/accessibility decisions

The workspace uses a 1.65fr/1fr desktop layout and stacks below 900px. The upload surface remains keyboard-operable through its Browse Files button and hidden input. Labels, focus states, `aria-current`, and `aria-live` are present. Motion is limited to the status spinner and respects reduced-motion preferences.
