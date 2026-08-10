import csv
import json

import pytest

from app.carter.dynamic_dataset import export_canonical_csv, export_canonical_json, validate_canonical_dataset
from app.carter.runtime import CarterPromptPackage, CarterPromptPackageError


@pytest.fixture()
def package(): return CarterPromptPackage.load()


def spec(dataset_type="custom"):
    return {"status":"ready","dataset_type":dataset_type,"dataset_name":"custom-support-analysis","dataset_description":"Custom dynamic records.","requested_record_count":1,"effective_record_count":1,"fields":[{"name":"customer_intent","type":"string","required":True,"description":"Intent."},{"name":"confidence_label","type":"enum","required":True,"description":"Confidence.","enum_values":["high","low"]},{"name":"reasoning_style","type":"array_string","required":False,"description":"Styles."}],"source_policy":"selected_documents_only","grounding_required":True,"evidence_required":True,"generation_requirements":["source_grounded","avoid_exact_duplicates"],"user_constraints":[],"clarification":{"required":False,"reason_code":None,"question":None,"reason":None}}


def record(): return {"customer_intent":"refund request","confidence_label":"high","reasoning_style":["concise","empathetic"],"evidence":[{"source_ref":"source_1","quote":"Refund requests are reviewed promptly."}]}


def test_dynamic_custom_record_preserves_all_fields_in_json_and_csv(package, tmp_path):
    dataset = validate_canonical_dataset(package, spec(), [record()], allowed_source_refs={"source_1"})
    assert dataset.field_order == ("customer_intent", "confidence_label", "reasoning_style", "evidence")
    json_path = export_canonical_json(dataset, tmp_path / "dataset.json")
    csv_path = export_canonical_csv(dataset, tmp_path / "dataset.csv")
    assert json.loads(json_path.read_text(encoding="utf8"))["records"][0]["customer_intent"] == "refund request"
    with csv_path.open(encoding="utf8", newline="") as handle: row = next(csv.DictReader(handle))
    assert list(row) == list(dataset.field_order) and row["reasoning_style"] == '["concise","empathetic"]' and "source_1" in row["evidence"]


@pytest.mark.parametrize("change", ["missing", "unknown", "enum", "evidence"])
def test_dynamic_validation_rejects_invalid_or_fabricated_records(package, change):
    candidate = record()
    if change == "missing": candidate.pop("customer_intent")
    elif change == "unknown": candidate["legacy_context"] = "lossy"
    elif change == "enum": candidate["confidence_label"] = "certain"
    else: candidate["evidence"][0]["source_ref"] = "../../secret.txt"
    with pytest.raises(CarterPromptPackageError): validate_canonical_dataset(package, spec(), [candidate], allowed_source_refs={"source_1"})
