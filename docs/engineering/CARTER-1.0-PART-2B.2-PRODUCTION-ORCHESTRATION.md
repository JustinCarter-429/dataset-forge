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

## Part 2B.9 RunPod generation response-shape diagnosis (2026-08-10)

The corrected repository root was `C:\Users\Justin\Documents\dataset generate v1`;
the backend working directory was
`C:\Users\Justin\Documents\dataset generate v1\backend`; and the verified Python
executable was
`C:\Users\Justin\Documents\dataset generate v1\backend\.venv\Scripts\python.exe`.
The backend directory, virtual-environment Python, production RunPod module,
diagnostic module, and safe temporary diagnostic directory were verified before
the live request.  No recoverable earlier terminal response was present.

One production-compatible basic generation call was made through the real
`CarterPromptPackage` dynamic-schema compiler, `RunPodCarterProvider`, and
`RunPodProvider`; it did not run the tool/review/export lifecycle.  It reached
`COMPLETED` after `IN_QUEUE` and `IN_PROGRESS`.  A subsequent read of the same
terminal status record recovered only a sanitized structural snapshot (no new
generation was submitted).

The terminal status was an object with keys `delayTime`, `executionTime`, `id`,
`output`, `status`, and `workerId`.  `output` was an array of length one;
`output[0]` was an OpenAI-style completion object with keys `choices`,
`created`, `id`, `kv_transfer_params`, `model`, `object`,
`prompt_logprobs`, `prompt_token_ids`, `service_tier`, `system_fingerprint`,
and `usage`.  `output[0].choices` was an array of length one and its choice
had a `message` object with `content`, `reasoning`, `tool_calls`, `role`,
`annotations`, `audio`, `function_call`, and `refusal` fields.  The observed
finish reason was `stop`; usage was 11,260 input tokens, 255 output tokens,
and 11,515 total tokens.

`output[0].choices[0].message.content` was null, the `tool_calls` container
was present but empty, and no `text`, token-array, final-channel, top-level
`choices`, or `response` path was present.  The reasoning field was present
but is intentionally neither logged nor returned as Carter content.  Thus the
primary observed variant is **B: OpenAI ChatCompletion nested inside RunPod
output**, but it contains no usable final output.  This is not an unhandled
normalization path: it is an analysis-only provider/model completion.  No
response parser was changed, no speculative fallback was added, and tool,
review, revision, and export were correctly blocked.

The RunPod transport now has a bounded content-free terminal-shape diagnostic:
at most five nesting levels, twelve object keys, and two array entries are
inspected; strings are represented only by length.  Its regression test proves
reasoning and generated content are excluded.  The temporary snapshot is not
an artifact and is not committed.  Raw provider responses, reasoning,
credentials, prompts, and tool arguments were not persisted.

Affected backend recertification was deliberately scoped to the provider,
production orchestration, Carter closure, dynamic-schema, prompt-package,
grounding, quality-review, and hardening modules: 110 passed, 0 failed.  Phase
1 upload/download and Phase 2 extraction nodes do not import the changed
transport diagnostic or Carter adapter and were not rerun.  Frontend
recertification passed: Vitest 6/0, TypeScript typecheck, production build, and
deterministic Playwright 5/0.  The browser suite uses its explicit deterministic
provider and makes no additional live RunPod request.

## Part 2B.10 gpt-oss Harmony final-channel diagnosis (2026-08-10)

The verified failing response remains a nested OpenAI Chat Completion in the
native RunPod `/run` then `/status` lifecycle: it completed with `stop`, 11,260
input tokens, 255 completion tokens, `message.content: null`, empty
`message.tool_calls`, and a present reasoning field.  Carter neither returns,
persists, nor substitutes that reasoning as final content.

The endpoint is configured in this repository only as model
`openai/gpt-oss-20b` and `RUNPOD_MAX_MODEL_LEN=128000`.  Its health endpoint
exposes only worker/job counters.  No startup logs, deployment/image metadata,
vLLM version, served-model name, parser settings, custom chat template, or
reasoning-effort/stop settings are exposed through the repository or endpoint.
Accordingly the deployed vLLM version, worker image,
`REASONING_PARSER`, `TOOL_CALL_PARSER`, and `ENABLE_AUTO_TOOL_CHOICE` are **not
verifiable**.  The repository itself does not configure any of those variables.

All requests used the production native RunPod wrapper with
`input.openai_route=/v1/chat/completions`, `stream=false`, temperature 0.2, and
no manual Harmony control tokens.  The compatibility path represents a schema
as `structured_outputs.json`; it adds exactly one authoritative-schema system
message for a complex Carter schema.  No duplicate schema injection was found.

Safe live capability isolation (assistant text and reasoning were never
recorded) found the following:

| Request | Prompt tokens | Structured JSON | Tools | Final content |
| --- | ---: | --- | ---: | --- |
| Minimal one-message JSON object | 78 | yes | 0 | present; valid JSON |
| Small schema-instruction, three-message JSON object | 99 | yes | 0 | present; valid JSON |
| Short Carter-shaped, four-message JSON object | 222 | yes | 3 | present; valid JSON |
| Full Carter-shaped request without structured JSON | 11,015 | no | 3 | absent; reasoning present |

The final large no-tools control ended with a terminal worker failure without
a safe exposed category, so it neither establishes nor refutes a tool-parser
contribution.  It was not retried.  The completed large no-JSON control proves
that `json_object` is not the primary trigger; the reproducible boundary is the
large Carter-shaped prompt/serving interaction.  It also stopped voluntarily
after 228 completion tokens with a 4,096-token budget, so it is not a token-cap
truncation.

This is partially compatible with the documented upstream gpt-oss/vLLM class of
Harmony final-channel failures: current vLLM documentation states that gpt-oss
uses Harmony parsing, and an upstream report reproduces null Chat Completion
content for gpt-oss under multi-turn JSON conditions.  The exact deployed vLLM
version and the no-JSON large-prompt condition cannot be matched, so this is
not sufficient to identify a particular upstream release as the root cause.
The required external next step is to obtain sanitized RunPod worker startup
configuration/version evidence and, for the deployed version, ensure its
gpt-oss Harmony parsing and `openai` tool-call configuration follow the
corresponding vLLM recipe.  No Carter schema, normalizer, or fallback was
changed; basic generation, tools, review, and export remain blocked pending a
legitimate final channel.

## Part 2B.11 RunPod deployment configuration audit (2026-08-10)

The authenticated RunPod endpoint console and its worker startup logs were
inspected read-only.  The console release label is `vLLM v2.22.5`; the actual
worker startup log identifies the serving engine as **vLLM v0.20.2** (V1
engine).  Three idle workers were visible, all NVIDIA A40 workers with 50 GB
RAM and 9 vCPUs.

The startup configuration provides the following deployment baseline:

| Setting | Observed value |
| --- | --- |
| Model / served model name | `openai/gpt-oss-20b` |
| Quantization | `gpt_oss_mxfp4` |
| Dtype | `torch.bfloat16` |
| KV-cache dtype | `auto` |
| Maximum sequence length | `131072` |
| `ENFORCE_EAGER` | `true` |
| CUDA graph mode / capture size | `NONE` / `0` |
| Reasoning parser | `openai_gptoss` |
| Custom chat template | not exposed in the sanitized startup configuration |
| `TOOL_CALL_PARSER` / auto-tool choice | not exposed in the sanitized startup configuration |

This changes the immediate diagnosis materially: the proposed first isolation
setting, eager execution, is already enabled and CUDA graph capture is already
disabled.  The Part 2B.10 content-free completion therefore cannot be
attributed to the default CUDA-graph path or to a request exceeding a graph
capture threshold.  The configured context length also exceeds the 11,260
input-token failing request, so it was not rejected for exceeding the visible
engine context limit.

The endpoint configuration editor was not opened through `Update endpoint`,
because that control can alter/redeploy the live external endpoint.  No
environment variable, worker image, or endpoint configuration was changed;
therefore no new live inference was sent and no tool, review, revision, or
export flow was attempted.  The remaining unverified server-side settings
(`TOOL_CALL_PARSER`, auto-tool choice, and custom chat template) need a
sanitized editor/configuration read.  Any isolated setting change requires
explicit confirmation immediately before the external save/redeploy action.

## Part 2B.12 Direct OpenAI-surface isolation (2026-08-10)

The documented RunPod OpenAI-compatible base URL was exercised directly,
without changing the production native `/run` plus `/status` transport.  A
read-only `GET /openai/v1/models` returned HTTP 200 and exactly one safe served
model ID: `openai/gpt-oss-20b`.  A minimal non-streaming JSON chat request to
the direct surface also returned HTTP 200 with `finish_reason=stop`, reasoning
present, and non-empty content.

The historical failing request text was deliberately not retained.  To avoid
recovering or persisting it, the large direct isolation control was instead
constructed at runtime by the current frozen Carter package using the normal
dataset-generation operation, dynamic-schema compiler, compatibility system
instruction, three registered tools, and the same model, max-tokens,
temperature, and structured-output mode.  It contained four messages and
60,365 characters.  No message or model output content was printed or stored.

| Surface / variant | Input tokens | Completion tokens | Finish | Reasoning | Content | Tool calls |
| --- | ---: | ---: | --- | --- | --- | ---: |
| Historical native basic generation | 11,260 | 255 | `stop` | present | null | 0 |
| Direct OpenAI compatible, Carter-shaped control | 11,335 | 321 | `stop` | present | null | 0 |
| Direct OpenAI compatible, identical control with `reasoning_effort=low` | 11,335 | 114 | `stop` | present | non-empty | 0 |

The direct control reproduces the content-free completion at essentially the
same Carter prompt scale, eliminating the RunPod native wrapper as the primary
boundary for this symptom.  Repository inspection confirms that Carter does
not send `reasoning_effort`; the baseline is therefore **omitted**.  Adding
only `reasoning_effort=low` produced final content, making low reasoning effort
a model/serving mitigation candidate.  This single diagnostic does not prove
reliability or validate a complete canonical dataset, and it was not adopted
as a global production default because it could silently override Carter
reasoning-profile semantics.

Because direct and low-effort diagnostics isolated the behavior, the
conditional prompt-duplication, small-vs-full, hardware, and vLLM-version
branches were not run.  No frozen Carter contract, response normalizer,
production transport, endpoint setting, GPU, or worker image was changed.
Tool configuration remains unverified and tool, review, revision, and export
remain blocked until a selected mitigation is validated through the full Carter
workflow.
