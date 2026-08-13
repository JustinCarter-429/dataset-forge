from __future__ import annotations

import pytest

from dataset_forge_critic.validation import validate_record

from conftest import valid_record


def test_listed_supervised_target_must_exist():
    payload = valid_record()
    payload["target"]["scores"]["grounding"] = None
    with pytest.raises(ValueError, match="scores.grounding"):
        validate_record(payload)


def test_unsupervised_missing_target_succeeds():
    payload = valid_record()
    payload["target"]["scores"]["grounding"] = None
    payload["supervision"] = {"available_targets": []}
    assert validate_record(payload).target.scores.grounding is None


def test_actionable_malformed_id_error():
    payload = valid_record()
    payload["record_id"] = "random"
    with pytest.raises(ValueError, match="record_id"):
        validate_record(payload)
