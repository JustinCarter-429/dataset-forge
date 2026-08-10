# Carter 1.0 dual-runtime agent POC

## Scope

Carter 1.0 is a small, explicit agent/runtime extension of Dataset Forge, not a new agent platform. The user-facing identity is Carter 1.0; the configured base model remains technical provenance. Cloud reuses the existing RunPod transport. Local uses LM Studio's OpenAI-compatible `/v1/models` and `/v1/chat/completions` endpoints, with its local server and model lifecycle owned by LM Studio.

## Local knowledge and tools

Canonical PDF/DOCX/TXT extraction is reused and written into an ignored SQLite FTS5 runtime database. The cap is three documents. Retrieval is lexical and bounded to ten results. The only registered tools are `list_documents`, `search_local_knowledge`, and `get_source_units`; no shell, arbitrary filesystem, SQL, browser, web-research, MCP, or network tools are exposed. The agent loop is bounded to three rounds; every provider tool call is parsed as JSON, schema-checked, and resolved solely through application identifiers in SQLite.

## Safety boundary and limitations

Runtime selection is explicit: Local never silently falls back to RunPod, and Cloud does not contact LM Studio. No model weights, uploads, or knowledge database are tracked. This POC deliberately defers embeddings/vector databases, fine-tuning, automatic downloads, long-term chat, accounts, web research, and unlimited document libraries. Native tool-call interoperability is provider-dependent; the application retains ownership of tool validation and execution.

## Verification

Deterministic verification on 2026-08-10 covered final responses, one/two/three tool rounds, a rejected fourth round, unknown/malformed/schema-invalid tool arguments, unknown and duplicate document IDs, unknown source references, provider timeout propagation, and path-shaped hostile IDs. The full backend suite uses no real RunPod or LM Studio calls. The frontend suite covers Carter branding, Cloud/Local selection, unavailable Local state, loading/failure state, and multi-document citations.

Live Cloud acceptance used two temporary TXT sources and one RunPod generation plus one quality review. It produced four final JSON records, all grounded (4/4) with five verified evidence references, zero warnings, zero revisions, and a package-ready ZIP. The browser preview was opened at `http://127.0.0.1:5173`; the package contained canonical dataset, metadata, validation report, quality review, manifests, and README. LM Studio was unavailable at `127.0.0.1:1234`, so real Local acceptance remains setup-required; no model was downloaded automatically.

## Carter 1.0 software acceptance closure

Date: 2026-08-10

The final closure added only acceptance infrastructure and the missing 4th-document testability path. The backend and frontend retain a hard three-document limit; after three cards the upload surface remains visible in a disabled maximum state, and attempted drag/drop or selection is rejected with an accessible message while the existing documents remain intact. Direct `GenerationRequest` validation rejects four IDs before provider dispatch.

The deterministic browser provider is enabled only when both `APP_ENVIRONMENT=test` and `CARTER_TEST_PROVIDER=deterministic` are explicitly set. It is not a production default, is not exposed in the UI, and does not alter normal RunPod or LM Studio routing. The Playwright harness starts an isolated deterministic backend on port 8001 and Vite on port 5174, clears only its own SQLite fixture, and exercises real frontend/FastAPI contracts.

Playwright acceptance: **5 passed, 0 failed, 0 skipped**. Covered 3-document upload and 4th rejection, Ask success with two citations, no-result, provider failure, Local unavailable with no Cloud fallback, generation failure, grounding/validation failure, quality-warning completion with download enabled, keyboard focus, and responsive 1440x900, 1280x720, 768x1024, and 390x844 layouts. Axe reported **0 serious and 0 critical violations** on initial, three-document, completed-warning, and failure states. Backend full regression: **83 passed, 0 failed, 4 warnings**; frontend: **6 passed, 0 failed**; typecheck and production build passed.

The preserved real Cloud acceptance remains separate and valid: 2 documents, 4 final records, 4/4 grounded, 5/5 verified evidence items, 0 invalid source refs, 0 cross-document refs, 0 unresolved aliases, quality review passed, and 0 revisions. LM Studio real acceptance is **SETUP REQUIRED** only; no paid provider call was made during this closure.

## Final real LM Studio acceptance attempt

Date: 2026-08-10

The configured/default LM Studio URL `http://127.0.0.1:1234` was unreachable, and `backend/.env` contained no `LM_STUDIO_ENABLED`, `LM_STUDIO_BASE_URL`, or `LM_STUDIO_MODEL` override. The live Carter runtime endpoint consequently reported Local as `configured: false, available: false`; no model-discovery response or served model ID was available.

No Ask Carter request was sent, so real LM Studio inference count is **0**, RunPod inference count is **0**, and no source documents were uploaded for this blocked acceptance. No code or architecture was changed. Security checks found no tracked model weights, environment files, knowledge databases, or reasoning artifacts; the local app and frontend remain running on ports 8000 and 5173.
