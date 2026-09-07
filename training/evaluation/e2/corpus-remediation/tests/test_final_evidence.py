from __future__ import annotations

import json
from pathlib import Path

from corpus_remediation.validation import redact

ROOT = Path(__file__).resolve().parents[1]


def audit():
    return json.loads((ROOT / "reports/audit.json").read_text(encoding="utf-8"))


def test_closing_brace_supervision():
    value = audit(); assert value["tokenization"]["supervised_closing_brace"] == 6750


def test_turn_terminator_supervision():
    value = audit(); assert value["tokenization"]["supervised_turn_terminator"] == 6750


def test_eos_supervision():
    value = audit(); assert value["tokenization"]["supervised_eos"] == 6750


def test_phase_e1_overlap_prevention():
    assert audit()["phase_e1_overlap"] == {"canonical_inputs": 0, "record_ids": 0}


def test_controlled_overfit_overlap_prevention():
    assert audit()["controlled_overfit_overlap"] == {"canonical_inputs": 0, "record_ids": 0}


def test_final_cross_split_gate():
    value = audit(); assert value["overlap"]["near_duplicate_pairs"] == 0 and value["gate"]["status"] == "PASS"


def test_secret_and_path_redaction():
    text = "C:\\Users\\Alice\\data token=abc -----BEGIN TEST PRIVATE KEY-----oops-----END TEST PRIVATE KEY----- /root/admin/file"
    cleaned = redact(text)
    assert "Alice" not in cleaned and "abc" not in cleaned and "oops" not in cleaned and "/root" not in cleaned
