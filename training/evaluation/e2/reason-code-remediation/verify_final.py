"""Verify final Phase E2.2 corpus/configuration invariants and reproducibility."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[3]
sys.path.insert(0, str(WORKSPACE / "training/src"))
from dataset_forge_critic.e2_contract import APPROVED_REASON_CODES, CONFIDENCE_SEMANTICS, FIELDS  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> int:
    audit = json.loads((HERE / "reports/reason-code-audit.json").read_text(encoding="utf-8")); smoke_config = json.loads((HERE / "smoke-config.json").read_text(encoding="utf-8"))
    train = rows(HERE / "smoke/train.jsonl"); validation = rows(HERE / "smoke/validation.jsonl"); combined = train + validation
    repro = HERE / "repro-check-20260908"
    compare = ["corpus/train.jsonl", "corpus/validation.jsonl", "smoke/train.jsonl", "smoke/validation.jsonl", "smoke-config.json", "reports/reason-code-audit.json", "reports/reason-code-coverage.csv", "reports/proposed-smoke-gates.json", "reports/root-cause-report.md"]
    byte_identity = {name: sha256(HERE / name) == sha256(repro / name) for name in compare}
    train_ids = {row["canonical_record"]["record_id"] for row in train}; validation_ids = {row["canonical_record"]["record_id"] for row in validation}
    canon = lambda row: json.dumps(row["canonical_record"]["input"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    train_inputs = {canon(row) for row in train}; validation_inputs = {canon(row) for row in validation}
    codes = {code for row in combined for code in row["canonical_record"]["target"]["reason_codes"]}
    checks = {
        "deterministic_rebuild_byte_identical": all(byte_identity.values()), "smoke_train_records_340": len(train) == 340,
        "smoke_validation_records_97": len(validation) == 97, "record_id_overlap_zero": not train_ids & validation_ids,
        "exact_and_canonical_input_overlap_zero": not train_inputs & validation_inputs,
        "near_duplicate_overlap_zero": audit["overlap"]["near_duplicate_pairs"] == 0,
        "template_family_overlap_zero": audit["overlap"]["template_family_overlap"] == 0,
        "source_family_overlap_zero": audit["overlap"]["source_family_overlap"] == 0,
        "e1_overlap_zero": not any(audit["phase_e1_overlap"].values()), "controlled_overfit_overlap_zero": not any(audit["controlled_overfit_overlap"].values()),
        "all_dataset_types_represented": {row["canonical_record"]["input"]["dataset_spec"]["dataset_type"] for row in combined} == {"question_answer", "instruction_response", "classification", "scenario_expected_result", "custom"},
        "all_decisions_represented": {row["canonical_record"]["target"]["decision"] for row in combined} == {"accept", "revise", "reject"},
        "all_reason_codes_represented": codes == APPROVED_REASON_CODES, "aliases_absent_and_casing_exact": not ({"UNSAFE_CONTENT", "unsafe_content", "MISSING_FIELD"} & codes) and codes <= APPROVED_REASON_CODES,
        "targets_have_exactly_five_fields": all(set(row["canonical_record"]["target"]) == set(FIELDS) and len(row["canonical_record"]["target"]) == 5 for row in combined),
        "closing_brace_supervised_all": audit["tokenization"]["supervised_closing_brace"] == audit["tokenization"]["records"],
        "turn_terminator_106_supervised_all": audit["tokenization"]["supervised_turn_terminator"] == audit["tokenization"]["records"],
        "eos_1_supervised_all": audit["tokenization"]["supervised_eos"] == audit["tokenization"]["records"],
        "fully_masked_zero": audit["tokenization"]["fully_masked_completions"] == 0,
        "no_excluded_or_quarantined_reintroduced": smoke_config["data"]["quarantined_records"] == 0 and all(not row.get("excluded") and not row.get("quarantined") for row in combined),
        "confidence_semantics_explicit": smoke_config["data"]["confidence_semantics"] == CONFIDENCE_SEMANTICS == audit["confidence_semantics"],
    }
    result = {"schema_version": "phase-e2.2-final-corpus-verification-v1", "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
              "byte_identity": byte_identity, "counts": {"training": len(train), "validation": len(validation), "approved_reason_codes": len(codes)},
              "tokenization": audit["tokenization"], "optimizer_steps": smoke_config["smoke"]["optimizer_steps"]}
    (HERE / "reports/final-corpus-verification.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": result["status"], "checks": len(checks), "byte_identical_files": sum(byte_identity.values())}, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
