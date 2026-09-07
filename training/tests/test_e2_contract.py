from __future__ import annotations

import json
from pathlib import Path

import pytest

from dataset_forge_critic.e2_contract import CONFIDENCE_SEMANTICS, ContractError, canonical_json, parse_strict_response, target_from_record
from dataset_forge_critic.evaluator import controlled_overfit_gate
from dataset_forge_critic.gemma import RENDERER
from dataset_forge_critic.tokenization import generation_terminator_ids, tokenize_record
from dataset_forge_critic.training_config import ModelSpec, load_training_configuration
from test_training_system import FakeProcessor, decision_record


def valid_payload():
    return {"decision": "accept", "confidence": 0.98, "reason_codes": [], "feedback": None, "model_version": "dataset-forge-critic-v1"}


def test_canonical_target_serialization_is_exact_and_ordered():
    assert canonical_json(valid_payload()) == '{"decision":"accept","confidence":0.98,"reason_codes":[],"feedback":null,"model_version":"dataset-forge-critic-v1"}'


@pytest.mark.parametrize(
    ("text", "error"),
    [
        ('{"decision":"accept","decision":"reject","confidence":1,"reason_codes":[],"feedback":null,"model_version":"dataset-forge-critic-v1"}', "DUPLICATE_KEY"),
        (json.dumps({**valid_payload(), "unknown": True}, separators=(",", ":")), "UNKNOWN_FIELD"),
        (json.dumps({**valid_payload(), "confidence": "0.9"}, separators=(",", ":")), "WRONG_TYPE"),
        (json.dumps({**valid_payload(), "decision": "ACCEPT"}, separators=(",", ":")), "INVALID_DECISION"),
        ("prefix " + canonical_json(valid_payload()), "NOT_EXACT_JSON_OBJECT"),
        (canonical_json(valid_payload()) + " suffix", "NOT_EXACT_JSON_OBJECT"),
        ("```json\n" + canonical_json(valid_payload()) + "\n```", "NOT_EXACT_JSON_OBJECT"),
        (canonical_json(valid_payload()) + canonical_json(valid_payload()), "INVALID_JSON_SYNTAX"),
        (canonical_json(valid_payload())[:-1], "NOT_EXACT_JSON_OBJECT"),
    ],
)
def test_strict_contract_rejections(text, error):
    with pytest.raises(ContractError, match=error):
        parse_strict_response(text)


def test_eos_mask_boundary_and_closing_brace_retention():
    processor = FakeProcessor()
    tokenized = tokenize_record(decision_record(), "native", processor, 4096)
    assert tokenized.input_ids[-1] == processor.tokenizer.eos_token_id
    assert tokenized.labels[-1] == processor.tokenizer.eos_token_id
    assert tokenized.input_ids[-2] == 106
    assert all(label == -100 for label in tokenized.labels[: tokenized.prompt_tokens])
    assert any(label != -100 for label in tokenized.labels[tokenized.prompt_tokens :])
    decoded_chars = [processor.tokenizer.decode([token]) for token in tokenized.input_ids[tokenized.prompt_tokens : -2]]
    assert "".join(decoded_chars).endswith("}")
    assert generation_terminator_ids(processor) == [1, 106]


def test_all_training_profiles_pin_the_e2_renderer():
    profiles = Path(__file__).resolve().parents[1] / "configs" / "profiles"
    for path in profiles.glob("*.yaml"):
        assert load_training_configuration(path).model.renderer_version == RENDERER


def test_stale_renderer_identity_is_rejected():
    with pytest.raises(ValueError, match="RENDERER_IDENTITY_MISMATCH"):
        ModelSpec(renderer_version="gemma4-critic-render-v1")


def test_controlled_overfit_gate_requires_every_output_condition():
    passing = {"strict_valid": True, "terminated": True, "output_tokens": 12, "predicted_decision": "accept", "expected_decision": "accept", "text": canonical_json(valid_payload())}
    assert controlled_overfit_gate([passing] * 20, 256)["status"] == "passed"
    assert controlled_overfit_gate([{**passing, "output_tokens": 256}], 256)["status"] == "failed"


def test_authoritative_native_mapping_policy_is_exact():
    item = decision_record()
    item["target"]["issue_codes"] = [" grounding contradiction ", "GROUNDING_CONTRADICTION"]
    mapped = target_from_record(item)
    assert mapped == {
        "decision": "reject",
        "confidence": 1.0,
        "reason_codes": ["GROUNDING_CONTRADICTION"],
        "feedback": "The answer conflicts with the source.",
        "model_version": "dataset-forge-critic-v1",
    }
    assert CONFIDENCE_SEMANTICS == "UNCALIBRATED_LABEL_AUTHORITY_PLACEHOLDER"


def test_native_mapping_rejects_unsupported_labels():
    item = decision_record()
    item["supervision"]["available_targets"].remove("decision")
    with pytest.raises(ContractError, match="DECISION_SUPERVISION_REQUIRED"):
        target_from_record(item)
    item = decision_record()
    item["target"]["issue_codes"] = ["invented-code"]
    with pytest.raises(ContractError, match="UNAPPROVED_REASON_CODE"):
        target_from_record(item)
