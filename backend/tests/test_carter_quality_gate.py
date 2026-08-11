import json
from zipfile import ZipFile

from app.carter.dynamic_dataset import export_canonical_csv, export_canonical_json, quality_gate, validate_canonical_dataset
from app.carter.runtime import CarterPromptPackage
from app.domain.enums import OutputFormat
from app.domain.models import GenerationManifest
from app.services.packaging import ZipDatasetPackager


def spec():
    return {"status":"ready","dataset_type":"custom","dataset_name":"quality-custom","dataset_description":"Quality test.","requested_record_count":1,"effective_record_count":1,"fields":[{"name":"customer_intent","type":"string","required":True,"description":"Intent."},{"name":"confidence_label","type":"enum","required":True,"description":"Confidence.","enum_values":["high","low"]},{"name":"reasoning_style","type":"array_string","required":False,"description":"Style."}],"source_policy":"selected_documents_only","grounding_required":True,"evidence_required":True,"generation_requirements":["source_grounded","avoid_exact_duplicates"],"user_constraints":[],"clarification":{"required":False,"reason_code":None,"question":None,"reason":None}}


def record(intent="support request"):
    return {"customer_intent":intent,"confidence_label":"high","reasoning_style":["concise"],"evidence":[{"source_ref":"source_1","quote":"Customers can request support."}]}


def dataset(records):
    package = CarterPromptPackage.load(); return package, validate_canonical_dataset(package, spec(), records, allowed_source_refs={"source_1"})


def test_quality_gate_preserves_dynamic_fields_and_accepts_clean_record():
    package, candidate = dataset([record()]); accepted, report = quality_gate(package, candidate, allowed_source_refs={"source_1"})
    assert report.export_eligible and accepted.records[0]["confidence_label"] == "high" and report.accepted_records == 1


def test_quality_gate_rejects_normalized_exact_duplicate():
    package, candidate = dataset([record(), record()]); accepted, report = quality_gate(package, candidate, allowed_source_refs={"source_1"})
    assert len(accepted.records) == 1 and report.duplicate_records == 1 and report.rejected_records == 1


def test_quality_gate_quarantines_synthetic_secret_and_pii_without_echoing_values():
    sensitive = record("email test@example.com api_key=sk_abcdefghijklmnopqrstuvwxyz")
    package, candidate = dataset([sensitive]); accepted, report = quality_gate(package, candidate, allowed_source_refs={"source_1"})
    assert not accepted.records and not report.export_eligible and report.quarantined_records == 1
    assert "test@example.com" not in json.dumps(report.as_dict())


def test_quality_gate_never_accepts_fabricated_evidence():
    package = CarterPromptPackage.load(); candidate = validate_canonical_dataset(package, spec(), [record()], allowed_source_refs={"source_1"})
    altered = candidate.__class__(candidate.specification, (record(),), candidate.compiled_schema)
    altered.records[0]["evidence"][0]["source_ref"] = "fabricated"
    accepted, report = quality_gate(package, altered, allowed_source_refs={"source_1"})
    assert not accepted.records and report.grounding_failures == 1 and report.rejected_records == 1


def test_accepted_only_exports_are_formula_safe_and_zip_includes_safe_quality_report(tmp_path):
    package, candidate = dataset([record("=not a spreadsheet formula"), record("email test@example.com")])
    accepted, report = quality_gate(package, candidate, allowed_source_refs={"source_1"})
    json_file = export_canonical_json(accepted, tmp_path / "dataset.json")
    csv_file = export_canonical_csv(accepted, tmp_path / "dataset.csv")
    archive = ZipDatasetPackager().package(json_file, GenerationManifest(job_id="quality-test", source_file="source.txt", requested_format=OutputFormat.JSON, record_count=1), tmp_path / "dataset.zip", report.as_dict())
    assert len(json.loads(json_file.read_text())["records"]) == 1
    assert "'=not a spreadsheet formula" in csv_file.read_text()
    with ZipFile(archive) as bundle:
        assert {"dataset.json", "quality-report.json", "generation_manifest.json"} <= set(bundle.namelist())
        assert "test@example.com" not in bundle.read("quality-report.json").decode()
