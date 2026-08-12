import json

from app.carter.count_policy import resolve_count
from app.carter.dynamic_dataset import CarterCanonicalDataset, export_canonical_json, export_canonical_jsonl
from app.domain.extraction_models import CanonicalExtractedDocument, ExtractionElement, ExtractionStatistics, ExtractionValidation


def _document(units: int) -> CanonicalExtractedDocument:
    return CanonicalExtractedDocument(
        document_id="doc", source_file_id="file", source_filename="fixture.txt", mime_type="text/plain", extractor="test",
        statistics=ExtractionStatistics(pageCount=1, characterCount=100, wordCount=20, elementCount=units, headingCount=0, paragraphCount=units, listItemCount=0, tableCount=0, tableRowCount=0, nonEmptyElementCount=units), validation=ExtractionValidation(valid=True, quality="valid"),
        elements=[ExtractionElement(elementId=f"source_{index}", type="paragraph", text=f"Concept {index}. A second fact. A third fact. A fourth fact. A fifth fact.", order=index, pageNumber=1) for index in range(units)],
    )


def test_no_count_request_is_auto_sized_from_source_units_not_ten():
    plan = resolve_count("Create a source-grounded question-answer dataset.", _document(12), 100)
    assert plan.mode == "auto" and plan.requested is None
    assert plan.minimum > 10 and plan.maximum > 10 and plan.target > 10


def test_explicit_count_remains_authoritative():
    plan = resolve_count("Create exactly 10 source-grounded question-answer examples.", _document(30), 100)
    assert plan.mode == "explicit" and plan.requested == plan.target == 10


def test_explicit_count_never_schedules_padding_beyond_source_opportunities():
    plan = resolve_count("Create exactly 20 source-grounded records.", _document(6), 100)
    assert plan.mode == "explicit" and plan.requested == 20
    assert plan.target == 12 and plan.supported_count == 12


def test_jsonl_is_one_record_per_parseable_line_and_json_is_readable(tmp_path):
    spec = {"dataset_type": "custom", "fields": [{"name": "question"}]}
    dataset = CarterCanonicalDataset(spec, tuple({"question": f"Q{index}", "evidence": []} for index in range(25)), {})
    jsonl = export_canonical_jsonl(dataset, tmp_path / "dataset.jsonl")
    lines = jsonl.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 25 and jsonl.read_bytes().endswith(b"\n")
    assert [json.loads(line)["question"] for line in lines] == [f"Q{index}" for index in range(25)]
    pretty = export_canonical_json(dataset, tmp_path / "dataset.json")
    assert len(pretty.read_text(encoding="utf-8").splitlines()) > 25
    assert json.loads(pretty.read_text(encoding="utf-8"))["records"] == list(dataset.records)
