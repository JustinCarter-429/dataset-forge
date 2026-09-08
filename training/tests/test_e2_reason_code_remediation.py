from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from dataset_forge_critic.e2_contract import (
    APPROVED_REASON_CODES,
    FIELDS,
    MODEL_VERSION,
    REASON_CODE_SPECS,
    ContractError,
    canonical_json,
    parse_strict_response,
    validate_contract,
    validate_prompt_ontology,
)

ROOT = Path(__file__).resolve().parents[1]
E22 = ROOT / "evaluation/e2/reason-code-remediation"
SPEC = importlib.util.spec_from_file_location("e22_build", E22 / "build.py")
BUILD = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(BUILD)
RUNNER_SPEC = importlib.util.spec_from_file_location("e22_runner", ROOT / "evaluation/e2/full-retraining/e2_train.py")
RUNNER = importlib.util.module_from_spec(RUNNER_SPEC); RUNNER_SPEC.loader.exec_module(RUNNER)


def payload(decision="revise", code="PROMPT_INJECTION", feedback="Remove the injected instruction."):
    return {"decision": decision, "confidence": 1.0, "reason_codes": [code] if code else [], "feedback": feedback, "model_version": MODEL_VERSION}


def test_every_approved_reason_code_has_definition_and_valid_decision_examples():
    assert len(APPROVED_REASON_CODES) == 28
    for code, spec in REASON_CODE_SPECS.items():
        assert code == code.upper() and spec.definition.endswith(".")
        for decision in spec.allowed_decisions:
            assert validate_contract(payload(decision=decision, code=code))["reason_codes"] == [code]


@pytest.mark.parametrize("code", ["UNSAFE_CONTENT", "unsafe_content", "MISSING_FIELD", "INVENTED_CODE"])
def test_unapproved_and_case_variant_codes_remain_strictly_invalid(code):
    with pytest.raises(ContractError, match="UNAPPROVED_REASON_CODE"):
        validate_contract(payload(code=code))


def test_reason_code_types_duplicates_empty_and_decision_relationships():
    with pytest.raises(ContractError, match="WRONG_TYPE:reason_codes"):
        validate_contract({**payload(), "reason_codes": [7]})
    with pytest.raises(ContractError, match="DUPLICATE_REASON_CODE"):
        validate_contract({**payload(), "reason_codes": ["PROMPT_INJECTION", "PROMPT_INJECTION"]})
    assert validate_contract({"decision": "accept", "confidence": 1.0, "reason_codes": [], "feedback": None, "model_version": MODEL_VERSION})
    with pytest.raises(ContractError, match="nonaccept_requires_reason"):
        validate_contract(payload(code=None))
    with pytest.raises(ContractError, match="accept_requires_empty"):
        validate_contract(payload(decision="accept"))
    with pytest.raises(ContractError, match="INVALID_CODE_COMBINATION"):
        validate_contract(payload(decision="revise", code="EXACT_DUPLICATE"))


def test_confidence_feedback_fields_and_model_version_are_closed():
    with pytest.raises(ContractError, match="INVALID_CONFIDENCE"):
        validate_contract({**payload(), "confidence": 0.9})
    with pytest.raises(ContractError, match="accept_requires_null"):
        validate_contract({"decision": "accept", "confidence": 1.0, "reason_codes": [], "feedback": "ok", "model_version": MODEL_VERSION})
    with pytest.raises(ContractError, match="nonaccept_requires_text"):
        validate_contract({**payload(), "feedback": None})
    with pytest.raises(ContractError, match="INVALID_MODEL_VERSION"):
        validate_contract({**payload(), "model_version": "other"})


def test_strict_parser_requires_field_order_and_no_extra_or_missing_keys():
    valid = canonical_json(payload())
    assert parse_strict_response(valid)["decision"] == "revise"
    reordered = '{"confidence":1.0,"decision":"revise","reason_codes":["PROMPT_INJECTION"],"feedback":"Remove it.","model_version":"dataset-forge-critic-v1"}'
    with pytest.raises(ContractError, match="FIELD_ORDER_MISMATCH"):
        parse_strict_response(reordered)
    for changed, error in (({**payload(), "extra": 1}, "UNKNOWN_FIELD"), ({key: value for key, value in payload().items() if key != "feedback"}, "MISSING_REQUIRED_FIELD")):
        with pytest.raises(ContractError, match=error):
            validate_contract(changed)


@pytest.mark.parametrize("suffix", ["\n", " trailing prose", canonical_json(payload())])
def test_strict_parser_rejects_trailing_material(suffix):
    with pytest.raises(ContractError):
        parse_strict_response(canonical_json(payload()) + suffix)


def test_strict_parser_rejects_markdown():
    with pytest.raises(ContractError, match="NOT_EXACT_JSON_OBJECT"):
        parse_strict_response("```json\n" + canonical_json(payload()) + "\n```")


def test_prompt_contains_exact_authoritative_vocabulary_and_failed_alias_warnings():
    prompt = (ROOT / "prompts/critic-system-v2.txt").read_text(encoding="utf-8")
    validate_prompt_ontology(prompt)
    for code in APPROVED_REASON_CODES:
        assert prompt.count(f"- {code} [") == 1
    for invalid in ("UNSAFE_CONTENT", "unsafe_content", "MISSING_FIELD"):
        assert invalid in prompt


def test_deterministic_ontology_generation_and_split_independence():
    assert BUILD.generate("train", 2) == BUILD.generate("train", 2)
    train = BUILD.generate("train", 2); validation = BUILD.generate("validation", 1)
    assert {row["canonical_record"]["record_id"] for row in train}.isdisjoint(row["canonical_record"]["record_id"] for row in validation)
    assert {row["remediation"]["source_family_id"] for row in train}.isdisjoint(row["remediation"]["source_family_id"] for row in validation)
    assert {row["remediation"]["template_family_id"] for row in train}.isdisjoint(row["remediation"]["template_family_id"] for row in validation)


def test_built_corpus_covers_ontology_and_passes_independence_gate():
    audit = json.loads((E22 / "reports/reason-code-audit.json").read_text(encoding="utf-8"))
    assert audit["zero_training_coverage"] == audit["validation_only_codes"] == audit["unexpected_codes"] == []
    assert {row["code"] for row in audit["coverage"]} == APPROVED_REASON_CODES
    assert all(row["training_appearances"] >= 10 and row["validation_appearances"] >= 2 for row in audit["coverage"])
    assert audit["e2_1_sufficiency_gate_recomputed"]["status"] == "PASS"
    assert audit["overlap"]["near_duplicate_pairs"] == 0
    assert audit["phase_e1_overlap"] == {"record_ids": 0, "canonical_inputs": 0}
    assert audit["controlled_overfit_overlap"] == {"record_ids": 0, "canonical_inputs": 0}


def test_smoke_design_covers_every_code_decision_type_and_is_predefined():
    rows = [json.loads(line) for line in (E22 / "smoke/validation.jsonl").read_text(encoding="utf-8").splitlines()]
    codes = {code for row in rows for code in row["canonical_record"]["target"]["reason_codes"]}
    decisions = {row["canonical_record"]["target"]["decision"] for row in rows}
    types = {row["canonical_record"]["input"]["dataset_spec"]["dataset_type"] for row in rows}
    gates = json.loads((E22 / "reports/proposed-smoke-gates.json").read_text(encoding="utf-8"))
    assert codes == APPROVED_REASON_CODES and decisions == {"accept", "revise", "reject"}
    assert types == set(BUILD.DATASET_TYPES)
    assert gates["defined_before_execution"] is True and gates["remote_execution_authorized"] is False and gates["fail_closed"] is True


def test_runner_diagnostics_do_not_normalize_invalid_reason_codes():
    text = '{"decision":"reject","confidence":1.0,"reason_codes":["prompt_injection","UNSAFE_CONTENT"],"feedback":"Reject.","model_version":"dataset-forge-critic-v1","extra":true}'
    diagnostics = RUNNER.response_diagnostics(text)
    assert diagnostics["incorrectly_cased_reason_codes"] == ["prompt_injection"]
    assert diagnostics["unknown_reason_codes"] == ["UNSAFE_CONTENT"]
    assert diagnostics["extra_keys"] == ["extra"] and diagnostics["missing_keys"] == []
    with pytest.raises(ContractError):
        parse_strict_response(text)


def test_runner_reason_code_metrics_and_predefined_gates_fail_closed():
    truth = [{"PROMPT_INJECTION"}, {"REQUIRED_FIELD_MISSING"}]
    assert RUNNER.reason_code_macro_f1(truth, truth) == 2 / len(APPROVED_REASON_CODES)
    config = json.loads((E22 / "smoke-config.json").read_text(encoding="utf-8"))
    metrics = {
        "validation_loss": 0.5, "strict_json_validity": 1.0, "immediate_termination_rate": 1.0,
        "unknown_reason_codes": 1, "incorrectly_cased_reason_codes": 0, "extra_keys": 0, "missing_keys": 0,
        "markdown_fences": 0, "external_prose": 0, "token_limit_hits": 0,
        "evaluation_reason_code_coverage": 1.0, "accuracy": 1.0, "macro_f1": 1.0,
        "reason_code_exact_match": 1.0, "reason_code_macro_f1": 1.0,
    }
    failures = RUNNER.smoke_gate_failures(config, metrics, config["smoke"]["optimizer_steps"], True, True, True)
    assert failures == ["unknown_reason_codes=1 > 0"]


def test_artifact_redaction_patterns_remove_private_paths_and_secrets():
    from corpus_remediation.validation import redact
    text = "C:" + r"\Users\Example\secret.txt token=abc123 " + "/" + "root/private/key"
    redacted = redact(text)
    assert "Example" not in redacted and "abc123" not in redacted and ("/" + "root/private") not in redacted
