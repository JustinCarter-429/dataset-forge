import io
import json
import re
import csv
import pytest
from pathlib import Path
from zipfile import ZipFile
from fastapi.testclient import TestClient
from app.main import app
from app.domain.models import CanonicalDataset, DatasetRecord
from app.domain.enums import OutputFormat
from app.services.export import DeterministicDatasetExporter
from app.providers.contracts import ProviderJob
from app.api.routes import generation as generation_route
from docx import Document
from reportlab.pdfgen import canvas

client = TestClient(app)


class FakeProvider:
    def generate(self, *, messages, schema, max_tokens):
        match = re.search(r"SOURCE UNIT: ([^\n]+)\nTEXT: ([^\n]+)", messages[-1]["content"])
        source_ref, quote = match.groups() if match else ("unknown", "hello")
        return ProviderJob("fake-job", "completed", {"records": [
            {"instruction": "Ask a testing question.", "context": quote, "expected_output": "Answer from the source.", "category": "testing", "difficulty": "easy", "source_refs": [source_ref], "evidence": [{"source_ref": source_ref, "quote": quote}]},
            {"instruction": "Identify a testing practice.", "context": quote, "expected_output": "Describe the practice.", "category": "testing", "difficulty": "medium", "source_refs": [source_ref], "evidence": [{"source_ref": source_ref, "quote": quote}]},
            {"instruction": "Explain a boundary case.", "context": quote, "expected_output": "Explain the boundary case.", "category": "testing", "difficulty": "hard", "source_refs": [source_ref], "evidence": [{"source_ref": source_ref, "quote": quote}]},
        ]})

    def health(self):
        return {"status": "ok"}


@pytest.fixture(autouse=True)
def fake_provider(monkeypatch):
    monkeypatch.setattr(generation_route, "_provider_factory", lambda config: FakeProvider())
    # The public route is Carter-backed.  The deterministic Carter adapter
    # exercises its DatasetSpec-defined artifact contract; the old fake fixed
    # record provider remains only for legacy service tests.
    monkeypatch.setenv("APP_ENVIRONMENT", "test")
    monkeypatch.setenv("CARTER_TEST_PROVIDER", "deterministic")


def upload(name="source.txt", content=b"hello", prompt="Create examples", output_format="json"):
    if name.endswith(".pdf"):
        stream = io.BytesIO(); pdf = canvas.Canvas(stream); pdf.drawString(72, 720, "PDF_SENTINEL"); pdf.save(); content = stream.getvalue()
    if name.endswith(".docx"):
        document = Document(); document.add_paragraph("DOCX_SENTINEL"); stream = io.BytesIO(); document.save(stream); content = stream.getvalue()
    return client.post("/api/generate", files={"file": (name, io.BytesIO(content), "application/octet-stream")}, data={"dataset_prompt": prompt, "output_format": output_format})


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_rejects_missing_prompt():
    assert upload(prompt="").status_code == 422


def test_rejects_unsupported_extension():
    assert upload(name="source.exe").status_code == 422


def test_rejects_empty_file():
    assert upload(content=b"").status_code == 422


def test_rejects_invalid_format():
    assert upload(output_format="yaml").status_code == 422


def test_supported_files_complete_and_json_zip_is_safe():
    for name in ("source.pdf", "source.docx", "source.txt"):
        response = upload(name=name)
        assert response.status_code == 200
        result = response.json()
        archive_response = client.get(result["download_url"])
        assert archive_response.status_code == 200
        with ZipFile(io.BytesIO(archive_response.content)) as archive:
            assert {"dataset.json", "generation_manifest.json", "manifest.json", "metadata.json", "README.txt"}.issubset(set(archive.namelist()))
            assert "app/" not in archive.read("generation_manifest.json").decode()
            payload = json.loads(archive.read("dataset.json"))
            assert payload["dataset_spec"]["dataset_type"] == "custom"
            assert payload["records"][0]["customer_intent"] == "support request"
            assert "instruction" not in payload["records"][0]


def test_csv_zip_contains_manifest_and_dataset():
    result = upload(output_format="csv").json()
    with ZipFile(io.BytesIO(client.get(result["download_url"]).content)) as archive:
        assert {"dataset.csv", "generation_manifest.json", "manifest.json", "metadata.json", "README.txt"}.issubset(set(archive.namelist()))
        rows = list(csv.DictReader(io.StringIO(archive.read("dataset.csv").decode())))
        assert rows and list(rows[0]) == ["customer_intent", "confidence_label", "reasoning_style", "evidence"]
        assert rows[0]["customer_intent"] == "support request" and "source_" in rows[0]["evidence"]


def test_unknown_and_traversal_downloads_are_safe():
    assert client.get("/api/download/unknown").status_code == 404
    assert client.get("/api/download/../unknown").status_code == 404


def test_resource_api_upload_generation_status_and_download():
    uploaded = client.post("/api/files", files={"file": ("../safe.txt", io.BytesIO(b"source"), "text/plain")})
    assert uploaded.status_code == 201
    payload = uploaded.json()["file"]
    assert payload["name"] == "safe.txt"
    assert payload["status"] == "uploaded"
    missing_file = client.post("/api/generations", json={"fileId": "0" * 32, "datasetPrompt": "x", "outputFormat": "json"})
    assert missing_file.status_code == 404
    created = client.post("/api/generations", json={"fileId": payload["id"], "datasetPrompt": "Create examples", "outputFormat": "json"})
    assert created.status_code == 202
    generation_id = created.json()["generationId"]
    status = client.get(f"/api/generations/{generation_id}")
    assert status.status_code == 200
    body = status.json()
    assert body["status"] == "completed"
    assert body["progress"]["percent"] == 100
    assert body["output"]["recordCount"] == 1
    assert body["output"]["sizeBytes"] > 0
    assert body["validation"]["schemaValid"] is True
    assert body["validation"]["groundingStatus"] == "passed"
    assert body["validation"]["groundedRecords"] == 1
    assert body["validation"]["verifiedEvidenceItems"] == 1
    assert body["validation"]["qualityStatus"] == "passed"
    assert body["packageReady"] is True
    downloaded = client.get(f"/api/generations/{generation_id}/download")
    assert downloaded.status_code == 200


def test_generation_download_before_completion_is_safe():
    assert client.get("/api/generations/unknown/download").status_code in (404, 409)


def test_corrupt_pdf_generation_fails_truthfully():
    uploaded = client.post("/api/files", files={"file": ("broken.pdf", io.BytesIO(b"not a real pdf"), "application/pdf")})
    assert uploaded.status_code == 201
    file_id = uploaded.json()["file"]["id"]
    created = client.post("/api/generations", json={"fileId": file_id, "datasetPrompt": "Create examples", "outputFormat": "json"})
    assert created.status_code == 202
    status = client.get(f"/api/generations/{created.json()['generationId']}").json()
    assert status["status"] == "failed"
    assert status["error"]["code"] in {"CORRUPT_DOCUMENT", "EXTRACTION_FAILED"}
    assert status["packageReady"] is False


def test_csv_neutralizes_formula_like_values(tmp_path: Path):
    dataset = CanonicalDataset(records=[DatasetRecord(instruction="=SUM(A1)", input="+unsafe", output="@mention")])
    output = DeterministicDatasetExporter().export(dataset, OutputFormat.CSV, tmp_path / "dataset.csv")
    text = output.read_text(encoding="utf-8")
    assert "'=SUM(A1)" in text and "'+unsafe" in text and "'@mention" in text
