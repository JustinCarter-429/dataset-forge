# Carter 1.0 Prompt Package Wiring — Part 1

## Scope and starting state

This Part 1 foundation materializes the frozen Carter 1.0 package at `backend/app/carter/contracts/1.0/`, with no real RunPod or LM Studio inference. Work started on `feat/carter-1.0-agent-poc` at `c86f2b5dd2aa2278a466fff67b91c991b2d976d1`; the `v1.0.0` tag remains `e36d68f8d1a189f8d7ca58e35ad868b58616c6a5`.

The package inventory is exactly 14 files: five prompts, three tools, five Draft 2020-12 schemas, and one manifest. The root `prompts/` directory remains the user-supplied staging source; the backend contract directory is the sole runtime source. Raw bytes are copied unchanged.

## Runtime foundation

`app.carter.runtime` loads only manifest-declared files from one explicit package directory. It validates inventory, declarations, prompt parent links, operation bindings, exact tool names, JSON parsing, local `$ref` targets, and schema compilation. Errors are normalized to `CARTER_PROMPT_PACKAGE_INVALID`; external references and undeclared files fail closed. JSON Schema resources are registered locally under Draft 2020-12, including tool-container resources for cross-file input-schema references.

Each contract SHA-256 is computed from raw bytes. The package fingerprint SHA-256 hashes canonical UTF-8 JSON of lexicographically sorted `{file, sha256}` records using sorted keys and compact separators.

The manifest resolver produces planning, generation, Ask, and quality-review operations. Rendering produces the same logical root/task messages, output schema, and tool schemas for Cloud and Local; only an adapter may later alter transport. Provider projection is an explicit copy boundary and canonical output is always revalidated.

## Validation and bounded behavior

DatasetSpec structural validation uses the canonical schema; semantic validation enforces ready-state count bounds, unique fields, coherent constraints, and classification enums. The generator compiler replaces the template with approved semantic fields (`string`, `integer`, `number`, `boolean`, `enum`, `array_string`), preserves `additionalProperties: false` and mandatory evidence, and applies exact generated-batch min/max counts only to the generated branch. An unapproved or uncompiled spec cannot produce a schema.

The registry contains exactly `list_documents`, `search_local_knowledge`, and `get_source_units`. It validates tool input before a bound existing SQLite KnowledgeStore handler and validates the safe normalized result before it returns to Carter. Agent normalization rejects multiple native calls and validates one logical `tool_call` or `final_response`. Quality review is advisory, validates record/field references, and revision authorization permits exactly one application-authorized revision.

The deterministic planner foundation uses the manifest planning operation, accepts only application-owned context, invokes an injected mock/provider-neutral callable, and structurally plus semantically validates its result.

## Tests and scope

Focused tests cover inventory/fingerprints, package corruption and remote refs, operation resolution, Cloud/Local equivalence, dynamic compilation, action normalization, tool validation, planner validation, quality review, and revision limits. Source text remains application input rather than executable contract state; no source text can register tools, change a runtime, change schemas, or increase limits.

Completion audit additions: generation template schemas are explicitly rejected at the request boundary until compiled from a ready DatasetSpec; application-owned `CarterAgentTurnState` rejects a fourth tool request after three normalized one-tool turns; planner fixtures cover all five supported dataset types and reject reserved fields and malformed enums; and the existing KnowledgeStore adapters reject path-like opaque identifiers outside the selected document/source scope.

Part 2 remains responsible for live provider activation, real bounded loops, live Ask/quality integration, and browser acceptance. No provider calls, secrets, model weights, knowledge databases, or hidden reasoning are introduced by this foundation.
