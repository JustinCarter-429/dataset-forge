# Phase 4 — Grounding, Validation, and Quality

Date: 2026-08-09

## Contract

Phase 4 makes Dataset schema `2.0` the canonical package contract. Each accepted record contains `instruction`, `context`, `expected_output`, `category`, `difficulty`, `source_refs`, and `evidence`. The backend owns IDs, metadata, validation, duplicate decisions, and package contents.

The package is rejected when any final record is ungrounded, references an unknown extracted source unit, contains missing or invalid evidence, or fails required-field/schema checks. Validation runs before export and packaging.

## Grounding algorithm

1. Extraction provides stable source-unit IDs and text.
2. The generation prompt receives bounded source units and requires source IDs plus short evidence quotes.
3. The validator normalizes Unicode, line endings, smart quotes, whitespace, and case, then verifies each quote as a contiguous substring of its referenced source unit.
4. A record is grounded only when all its references and evidence items verify. Every final accepted record must be grounded.

Evidence is bounded, non-empty, specific, and attached to an existing source unit. The validator does not infer support from semantic similarity.

## Quality analysis

The validation report records generated, valid, invalid, and final counts; grounding/evidence totals; category counts; difficulty distribution; exact duplicates removed; near-duplicate pairs; and issue codes. Exact duplicates are normalized and removed deterministically, retaining the first record. Near duplicates use a deterministic `SequenceMatcher` threshold and remain with a transparent warning.

`validation-report.json` is included in every successful Phase 4 package. The UI displays backend-derived schema, grounding, evidence, duplicate, and quality results without synthesizing success metrics locally.

## Verification

- Backend: `46 passed`, including invalid refs, missing/short/unmatched evidence, duplicates, quality metrics, and report packaging.
- Frontend: `2 passed`; typecheck and production build passed.
- Real acceptance: one RunPod dataset generation completed with 2 final records, 2/2 grounded records, 2/2 verified evidence items, zero duplicate removals, and quality status `passed`.
- ZIP acceptance: dataset, validation report, both manifests, metadata, and README were present with matching schema and counts.
- CSV regression: canonical records export deterministically with JSON-serialized evidence/source references and formula-safe text cells.

## Limitations

The process-local job store is non-durable. Literal evidence verification does not replace human review for factual nuance, ambiguity, or task usefulness. Phase 5 owns agentic self-check and bounded repair.
