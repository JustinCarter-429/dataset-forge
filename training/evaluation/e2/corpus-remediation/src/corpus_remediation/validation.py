"""Corpus gates, overlap scanning, token-boundary checks, and redaction."""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

from .canonicalization import fingerprint
from .clustering import cross_split_near, input_fingerprint

FIELDS = ("decision", "confidence", "reason_codes", "feedback", "model_version")
DECISIONS = {"accept", "revise", "reject"}
TYPES = {"question_answer", "instruction_response", "classification", "scenario_expected_result", "custom"}


def metadata(row: dict[str, Any]) -> dict[str, Any]:
    if "remediation" in row:
        return row["remediation"]
    native = row.get("native", {})
    return {
        "injection_related": False,
        "semantic_group": "native",
        "dataset_type": row["canonical_record"]["input"].get("dataset_spec", {}).get("dataset_type"),
        "template_family_id": native.get("template_family_id"),
        "source_family_id": native.get("source_family_id"),
        "mutation_family": native.get("mutation_family"),
        "generator_seed": row["canonical_record"].get("provenance", {}).get("source_record_id"),
    }


def strict_target(target: Any) -> bool:
    return (
        isinstance(target, dict) and set(target) == set(FIELDS) and target["decision"] in DECISIONS
        and target["confidence"] == 1.0 and not isinstance(target["confidence"], bool)
        and isinstance(target["reason_codes"], list)
        and (target["feedback"] is None or isinstance(target["feedback"], str))
        and target["model_version"] == "dataset-forge-critic-v1"
    )


def distributions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    decisions = Counter(row["canonical_record"]["target"]["decision"] for row in rows)
    types = Counter(metadata(row)["dataset_type"] for row in rows)
    templates = Counter(metadata(row)["template_family_id"] for row in rows)
    defects = Counter(metadata(row)["mutation_family"] for row in rows)
    injections = Counter(metadata(row)["semantic_group"] for row in rows if metadata(row)["injection_related"])
    def packed(values: Counter[str]) -> dict[str, dict[str, float | int]]:
        return {str(key): {"count": value, "fraction": value / count if count else 0.0} for key, value in sorted(values.items(), key=lambda item: str(item[0]))}
    return {"records": count, "decisions": packed(decisions), "dataset_types": packed(types), "template_families": packed(templates), "defect_categories": packed(defects), "prompt_injection": packed(injections)}


def reference_fingerprints(rows: list[dict[str, Any]]) -> tuple[set[str], set[str]]:
    return (
        {row.get("canonical_record", row).get("record_id", "") for row in rows},
        {fingerprint(row.get("canonical_record", row).get("input", {})) for row in rows},
    )


def overlap_counts(rows: list[dict[str, Any]], references: list[dict[str, Any]]) -> dict[str, int]:
    ids, inputs = reference_fingerprints(references)
    return {
        "record_ids": sum(row["canonical_record"]["record_id"] in ids for row in rows),
        "canonical_inputs": sum(input_fingerprint(row) in inputs for row in rows),
    }


def scan_cross_split(train: list[dict[str, Any]], validation: list[dict[str, Any]]) -> dict[str, Any]:
    train_ids = {row["canonical_record"]["record_id"] for row in train}
    validation_ids = {row["canonical_record"]["record_id"] for row in validation}
    train_inputs = {input_fingerprint(row) for row in train}
    validation_inputs = {input_fingerprint(row) for row in validation}
    train_templates = {metadata(row)["template_family_id"] for row in train}
    validation_templates = {metadata(row)["template_family_id"] for row in validation}
    train_sources = {metadata(row)["source_family_id"] for row in train}
    validation_sources = {metadata(row)["source_family_id"] for row in validation}
    near, comparisons, examples = cross_split_near(train, validation)
    return {
        "exact_record_id_overlap": len(train_ids & validation_ids),
        "canonical_input_overlap": len(train_inputs & validation_inputs),
        "template_family_overlap": len(train_templates & validation_templates),
        "source_family_overlap": len(train_sources & validation_sources),
        "jaccard_threshold": 0.85,
        "near_duplicate_pairs": near,
        "candidate_pairs_compared_after_length_bound": comparisons,
        "near_duplicate_examples": examples,
    }


def redact(value: str) -> str:
    patterns = (
        (r"-----BEGIN [^-]+PRIVATE KEY-----[\s\S]*?-----END [^-]+PRIVATE KEY-----", "[REDACTED_PRIVATE_KEY]"),
        (r"(?i)\b(?:token|password|secret|api[_-]?key)\s*[:=]\s*\S+", "[REDACTED_SECRET]"),
        (r"(?i)\b[A-Z]:\\Users\\[^\\\s]+", "[REDACTED_USER_PATH]"),
        (r"/(?:home|root)/[^/\s]+", "[REDACTED_USER_PATH]"),
    )
    for pattern, replacement in patterns:
        value = re.sub(pattern, replacement, value)
    return value


def gate(train: list[dict[str, Any]], validation: list[dict[str, Any]], overlap: dict[str, Any], phase_e1: dict[str, int], controlled: dict[str, int], tokenization: dict[str, Any]) -> dict[str, Any]:
    train_dist, validation_dist = distributions(train), distributions(validation)
    all_rows = train + validation
    by_input: dict[str, set[str]] = defaultdict(set)
    for row in all_rows:
        by_input[input_fingerprint(row)].add(row["canonical_record"]["target"]["decision"])
    conflicts = sum(len(values) > 1 for values in by_input.values())
    checks = {
        "train_unique_inputs_at_least_6000": len({input_fingerprint(row) for row in train}) >= 6000,
        "validation_unique_inputs_at_least_750": len({input_fingerprint(row) for row in validation}) >= 750,
        "all_dataset_types_both_splits": all(name in train_dist["dataset_types"] and name in validation_dist["dataset_types"] for name in TYPES),
        "all_decisions_both_splits": all(name in train_dist["decisions"] and name in validation_dist["decisions"] for name in DECISIONS),
        "decision_floor_20_percent": all(item["fraction"] >= 0.20 for dist in (train_dist, validation_dist) for item in dist["decisions"].values()),
        "dataset_type_floor_10_percent": all(item["fraction"] >= 0.10 for dist in (train_dist, validation_dist) for item in dist["dataset_types"].values()),
        "injection_train_at_least_300": sum(item["count"] for item in train_dist["prompt_injection"].values()) >= 300,
        "injection_validation_at_least_100": sum(item["count"] for item in validation_dist["prompt_injection"].values()) >= 100,
        "template_family_cap_5_percent": all(item["fraction"] <= 0.05 for dist in (train_dist, validation_dist) for item in dist["template_families"].values()),
        "exact_cross_split_overlap_zero": overlap["exact_record_id_overlap"] == 0,
        "canonical_cross_split_overlap_zero": overlap["canonical_input_overlap"] == 0,
        "near_cross_split_overlap_zero": overlap["near_duplicate_pairs"] == 0,
        "template_family_overlap_zero": overlap["template_family_overlap"] == 0,
        "source_family_overlap_zero": overlap["source_family_overlap"] == 0,
        "eligible_conflicting_groups_zero": conflicts == 0,
        "strict_target_validity_100_percent": all(strict_target(row["canonical_record"]["target"]) for row in all_rows),
        "fully_masked_completions_zero": tokenization["fully_masked_completions"] == 0,
        "closing_brace_retention_100_percent": tokenization["supervised_closing_brace"] == len(all_rows),
        "termination_boundary_retention_100_percent": tokenization["supervised_turn_terminator"] == len(all_rows) and tokenization["supervised_eos"] == len(all_rows),
        "phase_e1_overlap_zero": phase_e1["record_ids"] + phase_e1["canonical_inputs"] == 0,
        "controlled_overfit_overlap_zero": controlled["record_ids"] + controlled["canonical_inputs"] == 0,
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "conflicting_groups": conflicts, "train": train_dist, "validation": validation_dist}
