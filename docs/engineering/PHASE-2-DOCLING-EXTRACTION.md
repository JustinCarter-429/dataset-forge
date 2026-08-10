# Phase 2 — Docling Extraction and Canonical Source Model

## Purpose

Phase 2 replaces the Phase 1 placeholder extractor with a real local extraction subsystem. PDF and DOCX use Docling 2.118.1; TXT uses a safe direct UTF-8/CP1252 reader. No LLM generation, RunPod, grounding scoring, or synthetic authoring was added.

## Architecture

```text
Upload → File validation → Existing generation job
  → ExtractionService (Docling PDF/DOCX | PlainTextExtractor TXT)
  → CanonicalExtractedDocument → validation → deterministic analysis
  → existing placeholder dataset pipeline and ZIP output
```

The existing `InMemoryJobStore` remains the only job/file registry. Polling is read-only and cannot start extraction again. Each uploaded file stores one canonical extraction reference in process memory.

## Canonical model and provenance

`backend/app/domain/extraction_models.py` owns the normalized representation: document/source IDs, sanitized filename, MIME type, extractor/version, timestamp, metadata, ordered `blocks`, statistics, and validation quality/warnings. Elements support heading, paragraph, list item, table, caption, code block, and text. Tables preserve `rows` and deterministic text serialization. Every element has a stable hash-based ID, order, optional page number, section path, and source location. Docling page provenance is preserved when available; TXT elements carry one-based line ranges; DOCX page numbers are never fabricated.

## Extraction behavior

`DoclingExtractor` lazily constructs one adapter and iterates structured items once. It avoids document-level plus block-level double emission, and emits table items only as tables. `PlainTextExtractor` supports UTF-8 BOM, UTF-8, and CP1252. Repeated source lines remain because duplicate protection is based on extractor object identity and source position, not global text deletion.

OCR and Torch compilation are disabled in the local profile. This keeps extraction deterministic/offline in the bundled Windows runtime, where optional C++ compiler and model downloads are unavailable. Docling itself performs the PDF/DOCX conversion.

## Validation, analysis, and async integration

Empty extraction raises `EMPTY_EXTRACTION`; unsupported extensions raise `UNSUPPORTED_DOCUMENT`; Docling parse failures map to `CORRUPT_DOCUMENT`; unavailable runtime support maps to `EXTRACTOR_UNAVAILABLE`. Short non-empty documents receive a `suspicious` warning. The existing job progresses through real `extracting` and `analyzing` stages before the clearly placeholder dataset generator runs. Status includes extraction summary, analysis metrics, and capability flags.

## Test matrix and results

Real generated PDF/DOCX/TXT fixtures, sentinel completeness, repeated content, Unicode, DOCX table cells, corrupt PDFs, deterministic analysis, API failure mapping, legacy JSON/CSV ZIP regression, and formula-injection safety are covered. The full suite reached `16 passed` before the final corrupt-API assertion was added; rerun the full suite after that final assertion for the authoritative count. Frontend `npm run build` passes.

## Known limitations and Phase 3 handoff

Authoritative verification: the final backend suite passes with `17 passed`; frontend `npm run typecheck` and `npm run build` pass.

The process-local job store and extraction do not survive restart. OCR and optional Docling layout/table model downloads are disabled in this local profile. Phase 3 can consume `CanonicalExtractedDocument` as its only source input, including ordered elements, tables, provenance, statistics, validation, and deterministic analysis. gpt-oss-20b and real grounding are not implemented.
