from __future__ import annotations

from collections import Counter

from corpus_remediation.injection import DATASET_TYPES, generate
from corpus_remediation.validation import distributions, strict_target


def generated():
    return generate("validation", {"benign": 10, "repairable": 10, "malicious": 10}, 7)


def test_prompt_injection_reject_construction():
    row = next(item for item in generated() if item["remediation"]["semantic_group"] == "malicious")
    assert row["canonical_record"]["target"]["decision"] == "reject" and "PROMPT_INJECTION" in row["canonical_record"]["target"]["reason_codes"]


def test_repairable_injection_revise_construction():
    row = next(item for item in generated() if item["remediation"]["semantic_group"] == "repairable")
    assert row["canonical_record"]["target"]["decision"] == "revise" and row["canonical_record"]["target"]["feedback"].startswith("Remove only")


def test_benign_discussion_accept_construction():
    row = next(item for item in generated() if item["remediation"]["semantic_group"] == "benign")
    assert row["canonical_record"]["target"]["decision"] == "accept" and row["canonical_record"]["target"]["feedback"] is None


def test_all_five_dataset_types():
    assert {item["remediation"]["dataset_type"] for item in generated()} == set(DATASET_TYPES)


def test_decision_balancing():
    counts = Counter(item["canonical_record"]["target"]["decision"] for item in generated())
    assert counts == {"accept": 10, "revise": 10, "reject": 10}


def test_dataset_type_balancing():
    counts = Counter(item["remediation"]["dataset_type"] for item in generated())
    assert set(counts.values()) == {6}


def test_template_family_caps():
    dist = distributions(generated())
    assert max(item["fraction"] for item in dist["template_families"].values()) <= 0.05


def test_strict_target_schema():
    assert all(strict_target(item["canonical_record"]["target"]) for item in generated())


def test_confidence_placeholder_value():
    assert all(item["canonical_record"]["target"]["confidence"] == 1.0 for item in generated())


def test_deterministic_reruns():
    assert generate("train", {"benign": 2, "repairable": 2, "malicious": 2}, 11) == generate("train", {"benign": 2, "repairable": 2, "malicious": 2}, 11)
