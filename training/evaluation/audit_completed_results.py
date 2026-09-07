"""Independently audit completed Phase E1 incremental outputs without inference."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any

from e1_pipeline import DATASET_TYPES, aggregate, certify, load_jsonl, parse_and_normalize


MODEL_VARIANTS = ("base", "fine_tuned")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _threshold_checks(
    metrics: dict[str, Any], thresholds: dict[str, Any]
) -> dict[str, dict[str, dict[str, Any]]]:
    base = metrics["base"]
    tuned = metrics["fine_tuned"]
    accuracy_delta = tuned["accuracy"] - base["accuracy"]
    f1_delta = tuned["macro_f1"] - base["macro_f1"]

    def check(value: float, operator: str, limit: float) -> dict[str, Any]:
        passed = value >= limit if operator == ">=" else value <= limit
        return {"value": value, "operator": operator, "limit": limit, "passed": passed}

    pass_limits = thresholds["pass"]
    partial_limits = thresholds["partial"]
    per_dataset = min(value["accuracy"] for value in tuned["by_dataset_type"].values())
    return {
        "pass": {
            "fine_tuned_accuracy_min": check(tuned["accuracy"], ">=", pass_limits["fine_tuned_accuracy_min"]),
            "fine_tuned_macro_f1_min": check(tuned["macro_f1"], ">=", pass_limits["fine_tuned_macro_f1_min"]),
            "contract_validity_min": check(tuned["contract_validity_rate"], ">=", pass_limits["contract_validity_min"]),
            "parse_failure_max": check(tuned["parse_failure_rate"], "<=", pass_limits["parse_failure_max"]),
            "false_accept_rate_max": check(tuned["false_accept_rate"], "<=", pass_limits["false_accept_rate_max"]),
            "accuracy_or_macro_f1_improvement_min": check(max(accuracy_delta, f1_delta), ">=", pass_limits["accuracy_or_macro_f1_improvement_min"]),
            "per_dataset_accuracy_min": check(per_dataset, ">=", pass_limits["per_dataset_accuracy_min"]),
        },
        "partial": {
            "fine_tuned_accuracy_min": check(tuned["accuracy"], ">=", partial_limits["fine_tuned_accuracy_min"]),
            "fine_tuned_macro_f1_min": check(tuned["macro_f1"], ">=", partial_limits["fine_tuned_macro_f1_min"]),
            "contract_validity_min": check(tuned["contract_validity_rate"], ">=", partial_limits["contract_validity_min"]),
            "parse_failure_max": check(tuned["parse_failure_rate"], "<=", partial_limits["parse_failure_max"]),
            "false_accept_rate_max": check(tuned["false_accept_rate"], "<=", partial_limits["false_accept_rate_max"]),
            "accuracy_regression_tolerance": check(accuracy_delta, ">=", -partial_limits["material_regression_tolerance"]),
            "macro_f1_regression_tolerance": check(f1_delta, ">=", -partial_limits["material_regression_tolerance"]),
            "per_dataset_accuracy_min": check(per_dataset, ">=", partial_limits["per_dataset_accuracy_min"]),
        },
    }


def audit(output: Path, corpus_path: Path, overlap_path: Path, thresholds_path: Path) -> dict[str, Any]:
    raw_path = output / "logs" / "raw-responses.jsonl"
    results_path = output / "evaluation-results.json"
    raw_rows: list[dict[str, Any]] = []
    malformed_jsonl_lines: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw_path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            raw_rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            malformed_jsonl_lines.append({"line": line_number, "error": str(exc)})

    corpus = load_jsonl(corpus_path)
    corpus_by_id = {row["case_id"]: row for row in corpus}
    results = json.loads(results_path.read_text(encoding="utf-8"))
    result_cases = {
        (row["variant"], row["case_id"]): row
        for row in results["cases"]
        if row["variant"] in MODEL_VARIANTS
    }
    problems: list[dict[str, Any]] = []
    keys: list[tuple[str, str]] = []
    response_digests: list[dict[str, str]] = []

    for row in raw_rows:
        key = (row.get("variant"), row.get("case_id"))
        keys.append(key)
        normalized, parse_error = parse_and_normalize(row.get("raw_response", ""))
        predicted = normalized["decision"] if normalized else "invalid"
        if parse_error != row.get("parse_error") or predicted != row.get("predicted_decision"):
            problems.append({"variant": key[0], "case_id": key[1], "issue": "parse_recompute_mismatch"})
        stored = result_cases.get(key)
        if stored is None:
            problems.append({"variant": key[0], "case_id": key[1], "issue": "missing_results_case"})
        else:
            for field in (
                "dataset_type", "defect_categories", "expected_decision",
                "predicted_decision", "parse_error", "input_tokens", "output_tokens",
            ):
                if stored.get(field) != row.get(field):
                    problems.append({"variant": key[0], "case_id": key[1], "issue": "results_mismatch", "field": field})
        source = corpus_by_id.get(row.get("case_id"))
        if source is None:
            problems.append({"variant": key[0], "case_id": key[1], "issue": "unknown_case_id"})
        elif (
            source["dataset_type"], source["defect_categories"], source["expected_decision"]
        ) != (row["dataset_type"], row["defect_categories"], row["expected_decision"]):
            problems.append({"variant": key[0], "case_id": key[1], "issue": "corpus_metadata_mismatch"})
        response_digests.append({
            "variant": str(key[0]),
            "case_id": str(key[1]),
            "raw_response_sha256": hashlib.sha256(row.get("raw_response", "").encode()).hexdigest(),
        })

    duplicate_keys = [
        {"variant": variant, "case_id": case_id, "count": count}
        for (variant, case_id), count in collections.Counter(keys).items()
        if count > 1
    ]
    recomputed: dict[str, Any] = {}
    for variant in MODEL_VARIANTS:
        rows = [
            {key: value for key, value in row.items() if key != "raw_response"}
            for row in raw_rows if row.get("variant") == variant
        ]
        recomputed[variant] = aggregate(rows)
        if recomputed[variant] != results["metrics"][variant]:
            problems.append({"variant": variant, "issue": "aggregate_metrics_mismatch"})

    thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
    overlap = json.loads(overlap_path.read_text(encoding="utf-8"))
    outcome, reasons = certify(recomputed, overlap, thresholds, completed=True)
    if outcome != results["certification"]["outcome"] or reasons != results["certification"]["reasons"]:
        problems.append({"issue": "certification_mismatch"})

    corpus_ids = set(corpus_by_id)
    variant_case_sets_match = {
        variant: {row["case_id"] for row in raw_rows if row.get("variant") == variant} == corpus_ids
        for variant in MODEL_VARIANTS
    }
    audit_passed = (
        not malformed_jsonl_lines
        and not duplicate_keys
        and not problems
        and len(raw_rows) == 210
        and all(variant_case_sets_match.values())
    )
    return {
        "schema_version": "phase-e1-raw-response-audit-v1",
        "status": "passed" if audit_passed else "failed",
        "inference_rerun": False,
        "malformed_responses_preserved_as_failures": True,
        "raw_jsonl_sha256": _sha256(raw_path),
        "results_json_sha256": _sha256(results_path),
        "raw_line_count": len(raw_rows),
        "malformed_jsonl_lines": malformed_jsonl_lines,
        "unique_variant_case_pairs": len(set(keys)),
        "duplicate_variant_case_pairs": duplicate_keys,
        "variant_counts": dict(collections.Counter(row.get("variant") for row in raw_rows)),
        "variant_case_sets_match_corpus": variant_case_sets_match,
        "parse_error_counts": {
            variant: dict(collections.Counter(
                row.get("parse_error") or "NONE"
                for row in raw_rows if row.get("variant") == variant
            )) for variant in MODEL_VARIANTS
        },
        "predicted_decision_counts": {
            variant: dict(collections.Counter(
                row.get("predicted_decision")
                for row in raw_rows if row.get("variant") == variant
            )) for variant in MODEL_VARIANTS
        },
        "all_rows_recomputed_and_match": not problems,
        "problems": problems,
        "threshold_checks": _threshold_checks(recomputed, thresholds),
        "certification_recomputed": {"outcome": outcome, "reasons": reasons},
        "response_digests": response_digests,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--overlap", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.output, args.corpus, args.overlap, args.thresholds)
    destination = args.output / "logs" / "raw-response-audit.json"
    with destination.open("w", encoding="utf-8", newline="\n") as target:
        target.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in (
        "status", "raw_line_count", "unique_variant_case_pairs",
        "variant_counts", "parse_error_counts", "all_rows_recomputed_and_match",
        "certification_recomputed",
    )}, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
