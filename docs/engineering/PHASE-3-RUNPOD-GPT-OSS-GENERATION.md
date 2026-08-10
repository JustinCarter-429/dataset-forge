# Phase 3 - RunPod gpt-oss-20b generation

Date: 2026-08-09

## Scope and architecture

Phase 3 replaces the Phase 1 placeholder generator while preserving Phase 2 extraction. The flow is:

`Docling/TXT extraction -> canonical document -> deterministic analysis -> bounded context batches -> RunPod Serverless vLLM -> canonical records -> structural validation -> JSON/CSV export -> ZIP`.

The application remains a single-page upload/describe/generate workflow. There is no authentication, database, RAG, vector store, multi-file project, model selector, or dataset editor.

## Official research checked

The implementation was based on the official RunPod and vLLM documentation checked on 2026-08-09:

- [RunPod worker-vLLM source](https://github.com/runpod-workers/worker-vllm): the current README reports worker-vLLM `0.20.2`; the exact deployed image tag was not exposed by the endpoint and remains unverified.
- [RunPod: Send requests to vLLM workers](https://docs.runpod.io/serverless/vllm/vllm-requests): native `/run` and `/status`, vLLM messages, and asynchronous lifecycle.
- [RunPod: Send API requests](https://docs.runpod.io/serverless/endpoints/send-requests): `/run`, `/status`, `/cancel`, `/health`, queue/execution timeouts, 30-minute async result retention, and 429/5xx handling.
- [vLLM: Structured Outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/): JSON-schema structured output, `response_format`, and current `structured_outputs` terminology replacing deprecated `guided_json`.
- [OpenAI: gpt-oss open-weight models](https://help.openai.com/en/articles/11870455): gpt-oss is self-hosted/open-weight, compatible with vLLM, and not served through the OpenAI API.

The request uses native RunPod `/run` for durability, with `input.openai_route="/v1/chat/completions"` and the complete request under `input.openai_input`. The inner request includes the configured model, `messages`, `max_tokens`, `temperature`, `stream=false`, and vLLM `structured_outputs: {json: schema}`. Backend Pydantic validation remains authoritative.

## Configuration and safety gate

`backend/.env.example` documents `RUNPOD_ENDPOINT_ID`, `RUNPOD_API_KEY`, `RUNPOD_MODEL`, `RUNPOD_MAX_MODEL_LEN`, polling/queue/execution timeouts, `RUNPOD_RECORDS_PER_BATCH`, and `MAX_DATASET_RECORDS`. Missing endpoint ID, key, model, or invalid numeric limits raises `RUNPOD_CONFIGURATION_REQUIRED` or `RUNPOD_CONFIGURATION_INVALID` before any external request.

The API key is backend-only. Default pytest uses fake providers or `httpx.MockTransport`; no test contacts RunPod.

## Provider boundary and lifecycle

`backend/app/providers/contracts.py` defines provider-neutral configuration, job, and error contracts. `runpod.py` owns URL construction, authorization, `/run`, `/status`, `/health`, state mapping, queue/execution timeouts, response parsing, and provider error classification. Unknown states, auth failures, 429s, 5xx responses, missing job IDs, and network errors fail safely. `/cancel` is documented for a later cancellation control but is not exposed by the current UI.

The default production path is `RunPodDatasetGenerator`; there is no placeholder fallback after provider failure. A missing configuration produces a failed generation with a safe user-facing message.

## Prompt, canonical ownership, and context

`backend/app/prompts/dataset_author_v1.py` owns prompt version `dataset-author-v1`. The model authors only `instruction`, `input`, and `output`; the application adds generation ID, file ID, source element references, provider, model, prompt version, and timestamp in record metadata.

`context_projection.py` projects ordered canonical extraction elements, tables, and stable element IDs into deterministic batches. It uses a conservative four-characters-per-token estimate, a 65% model-length input budget, a configured record-per-batch default of 4, and a hard maximum record cap defaulting to 20. Binary files, paths, Docling internals, and secrets are never sent to the provider.

## Structured output and repair

The provider requests a JSON object containing a `records` array with the existing canonical record fields. Provider outputs are normalized from common vLLM completion shapes, parsed as JSON, converted into the existing `CanonicalDataset`, and passed through structural validation. At most one repair request is issued for an invalid/empty/non-JSON batch. Repair failure terminates the generation; it does not create placeholder success.

Grounding is not evaluated in Phase 3. Source references are preserved as provenance metadata, but no claim of semantic grounding quality is shown.

## UI and job state

The existing UI now renders real provider-aware status text, waiting-for-worker state, batch progress when returned, model-generation copy, schema validation, real record count/size, and the Phase 4 grounding limitation. Inputs remain in place on failure. ZIP download remains disabled until all batches, validation, export, and packaging succeed.

## Verification

- Backend: `35 passed` after adding RunPod configuration, mocked lifecycle, passthrough serialization, current/array-wrapped completion parsing, worker-error handling, bounded batching, repair, and Phase 2 regression tests.
- Frontend: `npm run typecheck` passed; `npm run build` passed after Phase 3 UI changes.
- Real RunPod health, compatibility, and dataset generation were verified on 2026-08-09 with the local backend configuration; the API key is never recorded.
- Embedded Codex preview: unavailable because the local URL was refused by the embedded browser environment; no claim of browser acceptance is made.

## Limitations and Phase 4 handoff

The process-local job store is non-durable. Provider output quality and source-grounding correctness require Phase 4 validation and quality UI. The current UI does not expose cancellation. No Git repository exists in this standalone directory, so no commit was created.

## Phase 3C acceptance audit - 2026-08-09

- Safe configuration loaded from `backend/.env`: endpoint and API key present, model `openai/gpt-oss-20b`, `RUNPOD_MAX_MODEL_LEN=128000`, polling 1 second, queue timeout 300 seconds, execution timeout 600 seconds, 4 records per batch, and 20 maximum records. The key is not recorded here.
- Exactly one real RunPod `/health` request passed with HTTP 200 and approximately 388 ms latency. The response did not expose worker/job counters in the returned payload.
- The two previous compatibility attempts are documented as runtime-only failures: the first exposed an array-wrapped choices shape; the second exposed a local tiny-schema parser mismatch. Their external IDs/output bodies were not persisted by the application, so no fabricated IDs are recorded.
- One final authorized compatibility job passed: external ID `665fa65c-6f79-43ac-b2f8-a68433421bdc-u2`, status sequence `IN_QUEUE -> IN_PROGRESS -> COMPLETED`, assistant content present, JSON parseable, and `{status: "ok"}` valid.
- One real dataset generation job completed from `qa-dataset-forge-smoke.txt`: generation ID `8d36dfc1e9bb4614b91cd48d6992f75a`, 2 records, canonical schema validation passed, no placeholder records, and one RunPod batch.
- ZIP verification passed with HTTP 200 and non-empty `README.txt`, `dataset.json`, `generation_manifest.json`, `manifest.json`, and `metadata.json` entries. CSV was derived locally from the canonical JSON with 2 rows; no second RunPod generation was made.
- Deterministic verification after the correction: `35 passed`; frontend typecheck and production build passed. Default tests make zero RunPod calls.
- Browser acceptance passed through the live frontend: file chooser upload, prompt, generation progress, completion, 2-record preview, validation passed, and ZIP action. The download event did not expose a file handle because the UI opens the download URL; the same endpoint was verified directly.
- Responsive checks at 1440x900, 1280x720, 768x1024, and 390x844 showed no horizontal overflow. Embedded preview was unavailable earlier in this environment; the live Playwright-compatible browser was used for acceptance.
