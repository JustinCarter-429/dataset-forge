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
