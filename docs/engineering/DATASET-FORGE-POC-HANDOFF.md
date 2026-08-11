# Dataset Forge Carter 1.0 PoC Handoff

## Product and flow

Dataset Forge converts uploaded PDF, DOCX, or TXT source material into grounded,
structured datasets: upload, describe the dataset, choose JSON or CSV, generate,
pass the quality gate, and download a ZIP. JSON is the canonical representation;
CSV is a deterministic conversion of validated canonical JSON.

Run locally with Python 3.11 and Node/npm:

```powershell
cd backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

In a second terminal:

```powershell
cd frontend
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

The frontend is `http://127.0.0.1:5173`; backend health is
`GET http://127.0.0.1:8000/api/health` and returns service status JSON.
`backend/.env.example` lists every supported variable name without values.

## Carter and runtime status

Carter 1.0 uses its authoritative 14-file prompt package, planner, DatasetSpec,
dynamic schema, generator, exactly three bounded tools (`list_documents`,
`search_local_knowledge`, `get_source_units`), advisory quality review, one
bounded revision, and the application-owned quality/export gate.

RunPod is the active PoC runtime. The logical model is Carter 1.0 and the
technical model is `openai/gpt-oss-20b`; configure `RUNPOD_ENDPOINT_ID`,
`RUNPOD_API_KEY`, `RUNPOD_MODEL`, capacity, polling, and timeout variables in
the backend environment. Remote minimal inference, Carter planning, and a valid
DatasetSpec are verified. Full large Carter generation through the current
GPT-OSS/vLLM deployment is deferred: some large requests return reasoning with
no usable final assistant content. LM Studio is retained architecturally,
visible but disabled in the UI, and its live acceptance is deferred.

## Quality and package contract

Every candidate record is dynamically schema-validated and source/evidence
revalidated. Exact duplicates are excluded; likely secrets and PII are
quarantined. Only accepted records export. `quality-report.json` contains safe
metadata/findings; CSV formula-leading values are neutralized. Semantic
near-duplicate embeddings are not implemented.

The ZIP contains `dataset.json` or `dataset.csv`, `quality-report.json`,
`validation-report.json`, `quality-review.json`, `generation_manifest.json`,
`manifest.json`, `metadata.json`, and `README.txt` when the corresponding
reports exist. No provider payloads, raw reasoning, secrets, or absolute paths
are intended to be packaged.

## Verification and next phase

Final deterministic evidence: backend 132 passed; focused quality checks 16
passed; frontend 6 passed; typecheck/build passed; browser clean flow 5 passed.
Browser quarantine and zero-accepted flows are backend-certified and deferred
in browser coverage because the deterministic provider has no such fixtures.

Deferred work: large RunPod generation hardening, LM Studio live runtime,
semantic near-duplicate detection, expanded human review, production hosting,
authentication, and durable projects/workspaces. The next recommended phase is
portfolio-ready deployment/demo hosting, without changing Carter contracts.
