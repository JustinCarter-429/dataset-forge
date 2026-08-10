import json
from pathlib import Path

import pytest

from app.domain.extraction_models import CanonicalExtractedDocument, ExtractionElement, ExtractionElementType, ExtractionStatistics, ExtractionValidation
from app.domain.models import CanonicalDataset, DatasetRecord, EvidenceReference
from app.services.export import DeterministicDatasetExporter
from app.services.packaging import ZipDatasetPackager
from app.services.validation import Phase4ValidationService, normalize_source_text
from app.services.generation import dataset_schema
from app.prompts.dataset_author_v2 import user_prompt
from app.domain.enums import OutputFormat
from app.domain.models import GenerationManifest


def source() -> CanonicalExtractedDocument:
    blocks = [
        ExtractionElement(elementId="s1", type=ExtractionElementType.PARAGRAPH, text="Functional testing checks feature workflows and expected functionality.", order=1),
        ExtractionElement(elementId="s2", type=ExtractionElementType.PARAGRAPH, text="Negative testing checks invalid inputs and safe failure behavior.", order=2),
    ]
    return CanonicalExtractedDocument(documentId="doc", sourceFileId="file", sourceFilename="source.txt", mimeType="text/plain", extractor="plain_text", extractorVersion="1", blocks=blocks, statistics=ExtractionStatistics(pageCount=None, characterCount=120, wordCount=15, elementCount=2, headingCount=0, paragraphCount=2, listItemCount=0, tableCount=0, tableRowCount=0, nonEmptyElementCount=2), validation=ExtractionValidation(valid=True, quality="good"))


def record(ref="s1", quote="Functional testing checks feature workflows and expected functionality.", instruction="What does functional testing check?", difficulty="easy"):
    return DatasetRecord(instruction=instruction, input=quote, output="It checks feature workflows.", context=quote, expected_output="It checks feature workflows.", category="testing", difficulty=difficulty, source_refs=[ref], evidence=[EvidenceReference(source_ref=ref, quote=quote)])


def validate(records):
    return Phase4ValidationService().validate(CanonicalDataset(records=records), source())


def test_grounding_and_normalization_pass():
    final, report = validate([record()])
    assert report.status == "passed"
    assert report.grounding.grounded_records == 1
    assert report.grounding.verified_evidence_items == 1
    assert normalize_source_text("Functional  testing\r\nchecks feature") in normalize_source_text(source().elements[0].text)


@pytest.mark.parametrize("bad,code", [
    (record(ref="missing"), "INVALID_SOURCE_REF"),
    (record(quote="This quote is not in source"), "EVIDENCE_NOT_FOUND"),
    (record(quote=""), "EVIDENCE_EMPTY"),
    (record(quote="tiny"), "EVIDENCE_TOO_SHORT"),
    (DatasetRecord(instruction="Q", input="x", output="y", context="x", expected_output="y", category="testing", difficulty="easy"), "MISSING_SOURCE_REF"),
])
def test_grounding_failures_are_hard_errors(bad, code):
    _, report = validate([bad])
    assert report.status == "failed"
    assert code in {issue.code for issue in report.issues}


def test_wrong_block_evidence_fails():
    _, report = validate([record(ref="s1", quote="Negative testing checks invalid inputs and safe failure behavior.")])
    assert report.grounding.status == "failed"
    assert "EVIDENCE_NOT_FOUND" in {issue.code for issue in report.issues}


def test_duplicate_policy_and_near_duplicate_warning():
    first = record()
    duplicate = record()
    final, report = validate([first, duplicate])
    assert len(final.records) == 1
    assert report.quality.exact_duplicates_removed == 1


def test_invalid_difficulty_and_zero_records_fail():
    _, invalid = validate([record(difficulty="advanced")])
    _, empty = validate([])
    assert "INVALID_DIFFICULTY" in {issue.code for issue in invalid.issues}
    assert "ZERO_RECORDS" in {issue.code for issue in empty.issues}


def test_phase4_package_contains_validation_report_and_metadata(tmp_path: Path):
    final, report = validate([record()])
    exported = DeterministicDatasetExporter().export(final, OutputFormat.JSON, tmp_path / "dataset.json")
    manifest = GenerationManifest(job_id="g", source_file="source.txt", requested_format=OutputFormat.JSON, record_count=1, provider="runpod_serverless", model="openai/gpt-oss-20b", validation_status=report.status)
    archive = ZipDatasetPackager().package(exported, manifest, tmp_path / "dataset.zip", validation_report=report.model_dump(mode="json"))
    from zipfile import ZipFile
    with ZipFile(archive) as zip_file:
        assert "validation-report.json" in zip_file.namelist()
        payload = json.loads(zip_file.read("validation-report.json"))
        assert payload["schema_version"] == "2.0"
        assert payload["grounding"]["grounded_records"] == 1
        assert json.loads(zip_file.read("metadata.json"))["groundingStatus"] == "passed"


def test_v2_schema_requires_evidence_without_model_validation_fields():
    item = dataset_schema()["properties"]["records"]["items"]
    assert set(["context", "expected_output", "category", "difficulty", "source_refs", "evidence"]).issubset(item["required"])
    assert "grounded" not in item["properties"]
    prompt = user_prompt("make examples", "SOURCE UNIT: s1\nTEXT: Functional testing checks workflows.", "batch-1", 1)
    assert "SOURCE UNIT: s1" in prompt
