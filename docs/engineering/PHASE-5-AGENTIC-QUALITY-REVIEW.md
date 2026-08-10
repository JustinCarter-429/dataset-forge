# Phase 5 — Bounded Agentic Quality Review

Date: 2026-08-09

## Starting state

Phase 4 remains authoritative: Dataset schema `2.0`, source references, contiguous evidence verification, exact duplicate handling, near-duplicate warnings, quality metrics, and `validation-report.json` all run before any AI review.

## Bounded loop and authority

The production loop is:

`generation → Phase 4 validation → one AI quality review → application policy → optional one targeted revision → complete Phase 4 revalidation → package`.

There is one global model-authored revision budget. Existing structural/grounding repair consumes that same budget, so a later quality revision cannot create a second repair cycle. `revisionAttempts <= 1` is tracked in job metadata and tested. A second AI review is not required after revision; deterministic revalidation is the final gate.

The reviewer is advisory. It cannot author `schemaValid`, grounding, evidence, package, security, or final pass/fail decisions. Severity is normalized by application policy. `SOURCE_SUPPORT_CONCERN` and `AMBIGUOUS_INSTRUCTION` block by default; repetition, weak outputs, diversity, and coverage observations are warnings unless a future explicit policy changes them.

## Contracts and prompts

- Review schema: version `1.0`, prompt `dataset-quality-review-v1`.
- Revision prompt: `dataset-quality-revision-v1`.
- Review issues use constrained codes, known record IDs, bounded messages/actions, and `warning`/`blocking` severity.
- Review output rejects unknown IDs, unknown codes, oversized issue lists, authority fields such as `passed`, and reasoning fields. No chain-of-thought is stored or displayed.

The review input contains only the dataset prompt, final validated records, record IDs, attached evidence, deterministic validation summary, distributions, duplicate observations, and source coverage. It does not include the binary source, filesystem paths, secrets, raw Docling objects, or full provider responses.

## Provider, batching, and privacy

Quality review reuses the existing provider-neutral `DatasetGenerationProvider` and the existing native RunPod `/run` + `/status` client with the configured `openai/gpt-oss-20b` model. It does not add a second HTTP client. Review batches are deterministic and token-budgeted; concurrency is one. A future independent validator can be selected through the small `QUALITY_VALIDATOR_MODE` extension point, but `same_model` is the default and no second credential is required.

Public research is foundation-only and disabled by default with `PUBLIC_RESEARCH_ENABLED=false`. Core acceptance makes zero research calls. Uploaded source evidence remains distinct from any future external provenance.

## Reports and UI

Successful packages include `quality-review.json` alongside the canonical dataset, `validation-report.json`, metadata, manifests, and README. The review artifact contains provider/model/prompt provenance, issue counts, application-computed status, revision accounting, and concise issues only. Metadata records review status and revision fields; the deterministic validation report remains authoritative.

The single-page UI keeps its existing three-step workflow. During review it shows “Reviewing dataset quality”; during revision it shows “Improving flagged examples” and “Revalidating revised dataset”. The final preview shows passed/warning state, blocking/warning counts, bounded revision status, and compact expandable issue details. It never displays an AI score or reasoning.

## Verification

- Backend: `52 passed`, including Phase 2–4 regression, review schema negatives, no-revision, warnings-only, targeted revision, failed revalidation, and bounded-call behavior.
- Frontend: `3 passed`; typecheck and production build passed.
- Real acceptance: generation `6355f7b853b64483895de19017c70f10` completed with 2/2 grounded records and 2/2 verified evidence items. One real review completed with 1 application-normalized warning, 0 blocking issues, revision not required, and 2 provider jobs total (generation + review).
- ZIP inspection passed with `quality-review.json`, `validation-report.json`, canonical JSON, both manifests, metadata, README, no reasoning fields, and matching counts. CSV was derived from the same final JSON without another model generation.
- Browser acceptance passed upload, extraction, generation, deterministic validation, AI review, warning display, package readiness, and download. Responsive checks cover 1440×900, 1280×720, 768×1024, and 390×844 with no horizontal overflow in the current build.
- Security checks found no API-key values, raw source/provider-response logging, reasoning persistence, unbounded revision, unsafe ZIP paths, or public research calls in core acceptance.

## Known limitations and next phase

The job store remains process-local. AI review is a bounded quality signal, not a correctness guarantee. Phase 6 owns final packaging, manual acceptance, large-document behavior, cancellation/retry, and release hardening.
