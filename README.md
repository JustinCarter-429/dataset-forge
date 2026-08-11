<p align="center">
  <img src="docs/assets/dataset-forge-logo.png" alt="Dataset Forge" width="560" />
</p>

# Dataset Forge

> Turn source documents into validated, structured datasets.

Dataset Forge is a document-to-dataset application that turns uploaded source material into structured, downloadable datasets.

## What you can do

1. Upload supported PDF, DOCX, or TXT source documents.
2. Describe the dataset you want to create.
3. Choose JSON or CSV output.
4. Generate a source-grounded dataset through the remote Carter runtime.
5. Follow each stage through the Live Pipeline, including batch progress for larger requests.
6. Review validation and quality results.
7. Download a validated dataset package.

## Dataset generation and quality

Dataset Forge plans structured datasets from natural-language requests and supports dynamic dataset fields, including custom dataset structures. Larger requests are divided into sequential batches and merged only after validation.

Every generated record is checked against the requested structure and its selected source material. The application detects duplicates, quarantines sensitive records, and exports only accepted records. CSV output is formula-safe.

## Output package

Each completed run downloads as a ZIP containing the validated JSON or CSV dataset, a manifest, and a quality report.

## Reliability improvements

- Bounded sequential batch generation with live progress.
- Safe handling for failed batches.
- No partial datasets reported as successful.
- Deterministic validation and quality gates before export.
- Download and package reconciliation checks.
- Runtime isolation with no silent fallback.

## Run locally

Prerequisites: Python 3.11, Node.js, npm, and configured runtime access.

```powershell
cd backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

In another terminal:

```powershell
cd frontend
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

Open `http://127.0.0.1:5173`. Never commit populated environment files.

## Validation

- Backend: 133 passing
- Frontend: 6 passing
- Browser: 6 passing
- Typecheck: PASS
- Build: PASS

## Current PoC scope

- Remote AI-assisted planning is enabled.
- Local runtime selection is retained but disabled in the current PoC.
- Additional remote-generation hardening is deferred to post-PoC work.
- Production authentication, persistent workspaces, and production hosting are outside the current PoC scope.

## License

Dataset Forge is source-available under the [Dataset Forge Community License 1.0](LICENSE). See [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md) for commercial use and [SECURITY.md](SECURITY.md) for responsible disclosure.
