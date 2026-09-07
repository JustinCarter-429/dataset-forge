from __future__ import annotations

import copy

from corpus_remediation.canonicalization import canonical_bytes, fingerprint, normalize
from corpus_remediation.conflicts import resolve


def row(record_id: str, decision: str = "ACCEPT", *, reason=None, feedback=None, source_id="s1"):
    return {"canonical_record": {"record_id": record_id, "input": {"text": "Café\r\nvalue", "n": 7, "negated": False}, "target": {"decision": decision, "reason_codes": reason or [], "feedback": feedback}, "provenance": {"source_dataset": "unit", "source_split": "train", "source_record_id": source_id}}, "native": {"mutation_family": "unit"}}


def test_canonicalization_stability():
    assert canonical_bytes({"b": 2, "a": " e\u0301 "}) == canonical_bytes({"a": "é", "b": 2})


def test_meaning_preserving_canonicalization():
    value = {"negative": "not allowed!", "number": 12, "label": "A", "type": "classification", "inner": "x  y"}
    result = normalize(value)
    assert result["negative"] == "not allowed!" and result["number"] == 12 and result["label"] == "A" and result["inner"] == "x  y"


def test_fingerprint_is_key_order_independent():
    assert fingerprint({"a": 1, "b": 2}) == fingerprint({"b": 2, "a": 1})


def test_conflict_group_quarantine():
    kept, quarantine = resolve([row("a", "ACCEPT"), row("b", "REJECT")])
    assert kept == [] and quarantine[0]["quarantine_reason"] == "CONFLICTING_DECISIONS"


def test_exact_duplicate_removal():
    kept, quarantine = resolve([row("b", source_id="z"), row("a", source_id="a")])
    assert len(kept) == 1 and len(quarantine) == 1 and quarantine[0]["record_count"] == 1


def test_differing_target_quarantine():
    kept, quarantine = resolve([row("a", "REJECT", reason=["A"]), row("b", "REJECT", reason=["B"])])
    assert kept == [] and quarantine[0]["quarantine_reason"] == "SAME_DECISION_MATERIAL_TARGET_DIFFERENCE"


def test_same_target_deterministic_retention():
    first = row("z", source_id="z"); second = row("a", source_id="a")
    kept1, _ = resolve([first, second]); kept2, _ = resolve([second, first])
    assert kept1[0]["canonical_record"]["record_id"] == kept2[0]["canonical_record"]["record_id"] == "a"


def test_whitespace_and_order_equivalence_retained():
    first = row("a", "REJECT", reason=["B", "A"], feedback=" fix ")
    second = row("b", "REJECT", reason=["A", "B"], feedback="fix")
    kept, quarantine = resolve([first, second])
    assert len(kept) == 1 and quarantine[0]["quarantine_reason"] == "EXACT_DUPLICATE_TARGET"


def test_missing_provenance_quarantined():
    value = row("a"); del value["canonical_record"]["provenance"]["source_record_id"]
    kept, quarantine = resolve([value])
    assert kept == [] and quarantine[0]["quarantine_reason"] == "MISSING_PROVENANCE"


def test_empty_corpus_behavior():
    assert resolve([]) == ([], [])


def test_quarantine_report_is_non_sensitive_shape():
    _, quarantine = resolve([row("a", "ACCEPT"), row("b", "REJECT")])
    assert set(quarantine[0]["provenance"][0]) == {"record_id", "source_dataset", "source_record_id", "source_split"}
