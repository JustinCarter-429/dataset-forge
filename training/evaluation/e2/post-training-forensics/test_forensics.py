from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from dataset_forge_critic.e2_contract import REASON_CODE_SPECS, parse_strict_response


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("e25_forensics", ROOT / "analyze_forensics.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def response(decision: str, code: str = "REQUIRED_FIELD_MISSING") -> str:
    return json.dumps(
        {
            "decision": decision,
            "confidence": 1.0,
            "reason_codes": [code],
            "feedback": "Add the absent schema-required field.",
            "model_version": "dataset-forge-critic-v1",
        },
        separators=(",", ":"),
    )


def test_failed_case_reproduces_exact_parser_rejection() -> None:
    with pytest.raises(ValueError, match="^INVALID_CODE_COMBINATION:REQUIRED_FIELD_MISSING$"):
        parse_strict_response(response("reject"))


def test_required_field_missing_is_revise_only() -> None:
    assert REASON_CODE_SPECS["REQUIRED_FIELD_MISSING"].allowed_decisions == frozenset({"revise"})
    assert parse_strict_response(response("revise"))["decision"] == "revise"


@pytest.mark.parametrize("code", ["required_field_missing", "MISSING_FIELD", "UNKNOWN_CODE"])
def test_case_variants_aliases_and_unknown_codes_remain_invalid(code: str) -> None:
    with pytest.raises(ValueError, match="UNAPPROVED_REASON_CODE"):
        parse_strict_response(response("revise", code))


def test_dataset_type_does_not_change_code_compatibility() -> None:
    # Dataset type is deliberately outside the five-field response contract.
    parsed = parse_strict_response(response("revise"))
    assert tuple(parsed) == ("decision", "confidence", "reason_codes", "feedback", "model_version")


def test_metrics_keep_invalid_output_as_failure() -> None:
    rows = []
    for truth, guess in (("accept", "accept"), ("revise", "invalid"), ("reject", "reject")):
        rows.append({"expected_decision": truth, "predicted_decision": guess, "latency_seconds": 1.0, "input_tokens": 2, "output_tokens": 3})
    metrics = MODULE.classification(rows)
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["contract_validity_rate"] == pytest.approx(2 / 3)
    assert metrics["parse_failure_rate"] == pytest.approx(1 / 3)


def test_recomputed_evidence_is_complete_and_unchanged() -> None:
    metrics = json.loads((ROOT / "smoke-recomputed-metrics.json").read_text(encoding="utf-8"))
    result = json.loads((ROOT / "forensic-results.json").read_text(encoding="utf-8"))
    assert metrics["raw_response_count"] == metrics["unique_case_ids"] == 15
    assert metrics["reported_metric_discrepancies"] == {}
    assert metrics["evidence_overwritten"] is False
    assert result["evidence_integrity"]["evidence_archive_matches"] is True
    assert result["evidence_integrity"]["adapter_matches"] is True


def test_corpus_targets_are_consistent_and_leak_free() -> None:
    audit = json.loads((ROOT / "reason-code-corpus-audit.json").read_text(encoding="utf-8"))
    assert audit["all_targets_revise"] is True
    assert audit["all_single_code_targets"] is True
    assert audit["compatibility_table_violations_full_corpus"] == 0
    assert audit["contradictory_identical_inputs"] == 0
    assert audit["failed_case_overlap"]["leakage_detected"] is False
