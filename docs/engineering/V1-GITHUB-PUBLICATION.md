# Dataset Forge V1 GitHub publication

Date: 2026-08-09

## Pre-publication state

Dataset Forge Phase 6 was complete and assessed as PoC release ready. The application remained local and operational; Git had not previously been initialized in this directory.

## Publication preparation

- Adopted the user-provided `Dataset Forge.png` logo as `docs/assets/dataset-forge-logo.png` after visual inspection.
- Rewrote the README as a portfolio landing page with safe local setup, V1 architecture, grounding model, limitations, and source-available licensing information.
- Added the Dataset Forge Community License 1.0, commercial-license guidance, third-party notices, changelog, security policy, contributing guidance, and a root `VERSION` file.
- Preserved the real local `backend/.env` without reading it. It is ignored alongside runtime uploads, generated packages, caches, build output, and dependency directories.
- Classified runtime uploads, generated packages, caches, build output, and dependency directories as ignored; source fixtures, application code, engineering history, and the owned logo were retained.
- Sanitized unnecessary provider job identifiers from public engineering history. The retained evidence uses counts and outcomes rather than external job IDs.

## License decision

Dataset Forge is **source-available**, not OSI-approved open source. The Community License permits personal, educational, academic/research, nonprofit/noncommercial, and qualifying small-business commercial use. Organizations with combined gross revenue of US $250,000 or more require a separate commercial license. This provisional custom license should receive legal review before material commercial enforcement.

## Validation gates

- Backend pytest: **65 passed, 4 warnings** (Docling deprecation warnings only).
- Frontend Vitest: **3 passed**; TypeScript typecheck and production build passed.
- A repository-specific credential scan excludes the real local `.env` from content inspection and checks that it is ignored before staging.

## Publication record

The following values are completed only after GitHub publication:

| Item | Value |
| --- | --- |
| Initial commit | `191fe63ca89d70c59e4ba3b18c0094b8c330b5dd` |
| Public repository | https://github.com/JustinCarter-429/dataset-forge |
| Visibility | Public (requested) |
| Default branch | `main` |
| Release tag | `v1.0.0` — https://github.com/JustinCarter-429/dataset-forge/releases/tag/v1.0.0 |
| GitHub release | https://github.com/JustinCarter-429/dataset-forge/releases/tag/v1.0.0 |

## GitHub publication

The repository was created as **public** under `JustinCarter-429/dataset-forge`. The `main` branch was pushed successfully from the reviewed local initial history. The final publication documentation commit is pushed before tagging; the `v1.0.0` tag and release point to that final commit. Repository topics were set for document AI, dataset generation, FastAPI, React, Docling, RunPod, vLLM, and gpt-oss.

## Known limitations

Dataset Forge V1 is a local PoC. It has process-local jobs, one configured provider/model, no authentication or durable job queue, and limited scanned-PDF OCR support. It does not expose the maintainer’s provider credentials; each user supplies their own backend-only RunPod credentials.
