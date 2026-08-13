from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from dataset_forge_critic.provenance import stable_record_id
from dataset_forge_critic.schemas import CanonicalCriticRecord, CriticDecision

from conftest import valid_record


def test_valid_canonical_example_and_zero_score_is_present():
    record = CanonicalCriticRecord.model_validate(valid_record())
    assert record.target.scores.grounding is not None
    assert record.target.scores.grounding.value == 0.0


def test_invalid_task_family_rejected():
    payload = valid_record()
    payload["task_family"] = "whatever"
    with pytest.raises(ValidationError, match="task_family"):
        CanonicalCriticRecord.model_validate(payload)


@pytest.mark.parametrize("decision", list(CriticDecision))
def test_native_decisions_are_valid(decision: CriticDecision):
    payload = valid_record()
    payload["target"]["decision"] = decision.value
    payload["supervision"]["available_targets"].append("decision")
    assert CanonicalCriticRecord.model_validate(payload).target.decision is decision


def test_invalid_decision_rejected():
    payload = valid_record()
    payload["target"]["decision"] = "KEEP"
    with pytest.raises(ValidationError, match="decision"):
        CanonicalCriticRecord.model_validate(payload)


def test_optional_labels_can_be_absent():
    payload = valid_record()
    payload["target"] = {}
    payload["supervision"] = {"available_targets": []}
    record = CanonicalCriticRecord.model_validate(payload)
    assert record.target.decision is None
    assert record.target.scores.grounding is None


def test_source_scale_score_is_preserved():
    payload = valid_record()
    payload["target"]["scores"]["grounding"]["value"] = 1.5
    assert CanonicalCriticRecord.model_validate(payload).target.scores.grounding.value == 1.5


def test_provenance_requires_source_dataset_and_versions():
    payload = valid_record()
    payload["provenance"] = {"source_dataset": ""}
    with pytest.raises(ValidationError, match="source_dataset"):
        CanonicalCriticRecord.model_validate(payload)


def test_stable_id_is_deterministic_and_versioned():
    first = stable_record_id("source", "train", "1", "1.0.0")
    assert first == stable_record_id("source", "train", "1", "1.0.0")
    assert first != stable_record_id("source", "train", "1", "1.0.1")
