from __future__ import annotations

import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.e1_pipeline import (
    DATASET_TYPES,
    SCORE_FIELDS,
    aggregate,
    build_corpus,
    candidate_fingerprint,
    certify,
    deterministic_validator,
    inspect_checkpoint_archive,
    overlap_check,
    parse_and_normalize,
    redaction_check,
    safe_extract_checkpoint,
    stable_json,
)


def envelope(decision="ACCEPT"):
    return stable_json({
        "decision": decision,
        "preferred_candidate_index": None,
        "scores": {name: None for name in SCORE_FIELDS},
        "issue_codes": [],
        "critique": "ok",
        "revision_directive": None,
    })


def test_corpus_is_deterministic_balanced_and_complete():
    first, second = build_corpus(), build_corpus()
    assert stable_json(first) == stable_json(second)
    assert len(first) == 105
    assert {row["dataset_type"] for row in first} == set(DATASET_TYPES)
    for dataset_type in DATASET_TYPES:
        rows = [row for row in first if row["dataset_type"] == dataset_type]
        assert {decision: sum(row["expected_decision"] == decision for row in rows) for decision in ("accept", "revise", "reject")} == {"accept": 7, "revise": 7, "reject": 7}
        categories = {category for row in rows for category in row["defect_categories"]}
        assert {"valid_records", "malformed_json", "duplicates", "near_duplicates", "prompt_injection", "critic_override", "malicious_source_content"} <= categories


def test_overlap_check_detects_exact_and_reports_missing(tmp_path):
    held = build_corpus()[:1]
    candidate = held[0]["canonical_record"]["input"]["candidate_record"]
    source = tmp_path / "train.jsonl"
    source.write_text(json.dumps({"input": {"candidate_record": candidate}}) + "\n", encoding="utf-8")
    report = overlap_check(held, [source, tmp_path / "missing.jsonl"])
    assert report["exact_overlap_count"] == 1
    assert report["sources"][1]["status"] == "missing"
    assert report["test_splits_accessed"] is False


def test_normalization_rejects_malformed_and_maps_valid_contract():
    assert parse_and_normalize("```json\n{}\n```")[0] is None
    assert parse_and_normalize("{bad")[1] == "NOT_EXACT_JSON_OBJECT"
    normalized, error = parse_and_normalize(envelope("REVISE"))
    assert error is None
    assert normalized == {"decision": "revise", "confidence": 0.0, "reason_codes": [], "feedback": "ok", "model_version": "dataset-forge-critic-v1"}


def test_metric_calculation_confusion_and_missing_latency():
    rows = []
    for expected, predicted in (("accept", "accept"), ("revise", "reject"), ("reject", "accept")):
        rows.append({"case_id": expected, "dataset_type": "question_answer", "defect_categories": ["prompt_injection"] if expected == "reject" else ["valid_records"], "expected_decision": expected, "predicted_decision": predicted, "latency_seconds": None, "input_tokens": 0, "output_tokens": 0})
    metrics = aggregate(rows)
    assert metrics["accuracy"] == pytest.approx(1 / 3)
    assert metrics["confusion_matrix"]["reject"]["accept"] == 1
    assert metrics["dangerous_false_accepts"] == 1
    assert metrics["latency"]["mean_seconds"] is None


def test_deterministic_validator_does_not_read_expected_label():
    row = build_corpus()[0]
    row["expected_decision"] = "reject"
    assert deterministic_validator(row)["decision"] == "accept"


def test_safe_checkpoint_inspection_and_extraction(tmp_path):
    good = tmp_path / "good.tar.gz"
    with tarfile.open(good, "w:gz") as archive:
        info = tarfile.TarInfo("step/adapter/adapter_model.safetensors"); payload = b"safe"; info.size = len(payload); archive.addfile(info, io.BytesIO(payload))
    inspection = inspect_checkpoint_archive(good)
    assert inspection["contains_adapter"] and not inspection["contains_complete_model"]
    safe_extract_checkpoint(good, tmp_path / "out")
    assert (tmp_path / "out/step/adapter/adapter_model.safetensors").read_bytes() == b"safe"
    bad = tmp_path / "bad.tar.gz"
    with tarfile.open(bad, "w:gz") as archive:
        info = tarfile.TarInfo("../escape"); info.size = 1; archive.addfile(info, io.BytesIO(b"x"))
    assert inspect_checkpoint_archive(bad)["unsafe_members"] == ["../escape"]
    with pytest.raises(ValueError, match="UNSAFE_ARCHIVE_MEMBER"): safe_extract_checkpoint(bad, tmp_path / "bad-out")


def test_certification_thresholds_and_critical_failure():
    def metric(accuracy=0.8, f1=0.8, dangerous=0):
        return {"accuracy": accuracy, "macro_f1": f1, "contract_validity_rate": 1.0, "parse_failure_rate": 0.0, "false_accept_rate": 0.0, "dangerous_false_accepts": dangerous, "by_dataset_type": {name: {"accuracy": 0.8, "sample_count": 21} for name in DATASET_TYPES}}
    thresholds = json.loads((Path(__file__).parents[1] / "evaluation/thresholds.json").read_text())
    outcome, _ = certify({"base": metric(0.6, 0.6), "fine_tuned": metric()}, {"exact_overlap_count": 0, "near_overlap_count": 0}, thresholds, True)
    assert outcome == "PASS"
    outcome, reasons = certify({"base": metric(), "fine_tuned": metric(dangerous=1)}, {"exact_overlap_count": 0, "near_overlap_count": 0}, thresholds, True)
    assert outcome == "FAIL" and "Dangerous false accept detected" in reasons


def test_redaction_check_passes_and_fails(tmp_path):
    names = ("index.html", "evaluation-results.json", "evaluation-results.csv", "certification-report.md", "model-manifest.json")
    (tmp_path / "logs").mkdir()
    for name in names: (tmp_path / name).write_text("website safe", encoding="utf-8")
    redaction_check(tmp_path)
    assert json.loads((tmp_path / "logs/redaction-check.json").read_text())["status"] == "passed"
    (tmp_path / "index.html").write_text("C:\\Users\\Someone", encoding="utf-8")
    with pytest.raises(ValueError, match="REDACTION_FAILED"): redaction_check(tmp_path)
