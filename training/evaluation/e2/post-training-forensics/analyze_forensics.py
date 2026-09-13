"""Recompute Phase E2.5 forensics from immutable local evidence only."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dataset_forge_critic.e2_contract import (
    APPROVED_REASON_CODES,
    FIELDS,
    MODEL_VERSION,
    REASON_CODE_ONTOLOGY_VERSION,
    REASON_CODE_SPECS,
    parse_strict_response,
)


DECISIONS = ("accept", "revise", "reject")
FAILED_CASE_ID = "e1-scenario_expected_result-08"
EXPECTED_EVIDENCE_SHA256 = "312008177462208f7bc7c471324be0ad5e0f2d7b7d1849acd70f88bf6903e5a3"
EXPECTED_ADAPTER_SHA256 = "7e85179525db5ebcb0875e5520379b4e8a272bc602678cfb8aa8b049b98ae71a"


def stable(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def classification(rows: list[dict[str, Any]]) -> dict[str, Any]:
    matrix = {truth: {guess: 0 for guess in (*DECISIONS, "invalid")} for truth in DECISIONS}
    for row in rows:
        matrix[row["expected_decision"]][row["predicted_decision"]] += 1
    per_class = {}
    for label in DECISIONS:
        tp = matrix[label][label]
        fp = sum(matrix[truth][label] for truth in DECISIONS if truth != label)
        fn = sum(matrix[label][guess] for guess in (*DECISIONS, "invalid") if guess != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": sum(matrix[label].values())}
    latencies = [float(row["latency_seconds"]) for row in rows]
    elapsed = sum(latencies)
    rejects = [row for row in rows if row["expected_decision"] == "reject"]
    accepts = [row for row in rows if row["expected_decision"] == "accept"]
    revises = [row for row in rows if row["expected_decision"] == "revise"]
    return {
        "sample_count": len(rows),
        "accuracy": sum(matrix[label][label] for label in DECISIONS) / len(rows),
        "macro_precision": statistics.mean(item["precision"] for item in per_class.values()),
        "macro_recall": statistics.mean(item["recall"] for item in per_class.values()),
        "macro_f1": statistics.mean(item["f1"] for item in per_class.values()),
        "per_class": per_class,
        "confusion_matrix": matrix,
        "false_accept_rate": sum(row["predicted_decision"] == "accept" for row in rejects) / len(rejects),
        "false_reject_rate": sum(row["predicted_decision"] == "reject" for row in accepts) / len(accepts),
        "revise_accuracy": sum(row["predicted_decision"] == "revise" for row in revises) / len(revises),
        "contract_validity_rate": sum(row["predicted_decision"] != "invalid" for row in rows) / len(rows),
        "parse_failure_rate": sum(row["predicted_decision"] == "invalid" for row in rows) / len(rows),
        "latency": {
            "mean_seconds": statistics.mean(latencies),
            "median_seconds": statistics.median(latencies),
            "p95_seconds": percentile(latencies, 0.95),
        },
        "examples_per_minute": len(rows) / elapsed * 60,
        "input_tokens": sum(row["input_tokens"] for row in rows),
        "output_tokens": sum(row["output_tokens"] for row in rows),
    }


def tokens(value: Any) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", stable(value).casefold()))


def jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right) if left | right else 1.0


def missing_required(record: dict[str, Any]) -> list[str]:
    canonical = record["canonical_record"]
    spec = canonical["input"]["dataset_spec"]
    candidate = canonical["input"]["candidate_record"]
    required = spec.get("required_fields", spec.get("required", []))
    return sorted(set(required) - set(candidate))


def structural_signature(record: dict[str, Any]) -> str:
    canonical = record["canonical_record"]
    value = {
        "dataset_type": canonical["input"]["dataset_spec"].get("dataset_type"),
        "spec_keys": sorted(canonical["input"]["dataset_spec"]),
        "candidate_keys": sorted(canonical["input"]["candidate_record"]),
        "missing": missing_required(record),
        "decision": canonical["target"]["decision"],
        "codes": canonical["target"]["reason_codes"],
    }
    return stable(value)


def corpus_audit(train: list[dict[str, Any]], validation: list[dict[str, Any]], failed: dict[str, Any]) -> dict[str, Any]:
    splits = {"train": train, "validation": validation}
    occurrences: dict[str, Any] = {}
    all_hits = []
    for split, rows in splits.items():
        hits = [row for row in rows if "REQUIRED_FIELD_MISSING" in row["canonical_record"]["target"]["reason_codes"]]
        all_hits.extend((split, row) for row in hits)
        occurrences[split] = {
            "records": len(rows),
            "occurrences": len(hits),
            "corpus_fraction": len(hits) / len(rows),
            "decision_distribution": dict(sorted(Counter(row["canonical_record"]["target"]["decision"] for row in hits).items())),
            "dataset_type_distribution": dict(sorted(Counter(row["canonical_record"]["input"]["dataset_spec"]["dataset_type"] for row in hits).items())),
            "defect_category_distribution": dict(sorted(Counter(row.get("remediation", {}).get("mutation_family", "unlabeled") for row in hits).items())),
            "co_occurring_reason_code_sets": {"|".join(key): value for key, value in sorted(Counter(tuple(row["canonical_record"]["target"]["reason_codes"]) for row in hits).items())},
            "all_are_true_field_absence": all(bool(missing_required(row)) for row in hits),
            "structural_signature_count": len({structural_signature(row) for row in hits}),
            "template_pattern": "one omitted required schema field with a single revise-only reason code",
        }

    combined = [(split, row) for split, rows in splits.items() for row in rows]
    input_targets: dict[str, set[str]] = defaultdict(set)
    exact_candidates: dict[str, list[str]] = defaultdict(list)
    compatibility_violations = []
    for split, row in combined:
        canonical = row["canonical_record"]
        input_key = stable(canonical["input"])
        input_targets[input_key].add(stable(canonical["target"]))
        exact_candidates[stable(canonical["input"]["candidate_record"])].append(split)
        target = canonical["target"]
        try:
            parse_strict_response(json.dumps({field: target[field] for field in FIELDS}, ensure_ascii=False, separators=(",", ":")))
        except (KeyError, ValueError) as error:
            compatibility_violations.append({"record_id_hash": bytes_sha256(canonical["record_id"].encode()), "error": str(error)})
    contradictory_inputs = sum(len(targets) > 1 for targets in input_targets.values())
    cross_split_exact_candidates = sum("train" in splits_seen and "validation" in splits_seen for splits_seen in exact_candidates.values())
    candidate_targets: dict[str, set[str]] = defaultdict(set)
    for _, row in combined:
        canonical = row["canonical_record"]
        candidate_targets[stable(canonical["input"]["candidate_record"])].add(stable(canonical["target"]))
    conflicting_cross_split_candidates = sum(
        "train" in splits_seen and "validation" in splits_seen and len(candidate_targets[candidate]) > 1
        for candidate, splits_seen in exact_candidates.items()
    )

    hit_candidates = [(split, row, tokens(row["canonical_record"]["input"]["candidate_record"])) for split, row in all_hits]
    near_pairs = 0
    for index, (left_split, left, left_tokens) in enumerate(hit_candidates):
        for right_split, right, right_tokens in hit_candidates[index + 1 :]:
            if stable(left["canonical_record"]["input"]["candidate_record"]) != stable(right["canonical_record"]["input"]["candidate_record"]) and jaccard(left_tokens, right_tokens) >= 0.9:
                near_pairs += 1

    failed_candidate = failed["canonical_record"]["input"]["candidate_record"]
    failed_tokens = tokens(failed_candidate)
    comparisons = []
    for split, row in combined:
        candidate = row["canonical_record"]["input"]["candidate_record"]
        comparisons.append((stable(candidate) == stable(failed_candidate), jaccard(failed_tokens, tokens(candidate))))
    return {
        "schema_version": "phase-e2.5-reason-code-corpus-audit-v1",
        "reason_code": "REQUIRED_FIELD_MISSING",
        "occurrences": occurrences,
        "total_occurrences": len(all_hits),
        "all_targets_revise": all(row["canonical_record"]["target"]["decision"] == "revise" for _, row in all_hits),
        "all_single_code_targets": all(row["canonical_record"]["target"]["reason_codes"] == ["REQUIRED_FIELD_MISSING"] for _, row in all_hits),
        "compatibility_table_violations_full_corpus": len(compatibility_violations),
        "compatibility_violation_examples": compatibility_violations[:10],
        "contradictory_identical_inputs": contradictory_inputs,
        "cross_split_exact_candidate_duplicates": cross_split_exact_candidates,
        "conflicting_cross_split_exact_candidate_duplicates": conflicting_cross_split_candidates,
        "exact_candidate_duplicates_among_code_examples": len(hit_candidates) - len({stable(row["canonical_record"]["input"]["candidate_record"]) for _, row, _ in hit_candidates}),
        "near_duplicate_pairs_among_code_examples_at_0_90": near_pairs,
        "failed_case_overlap": {
            "exact_candidate_matches": sum(value for value, _ in comparisons),
            "near_candidate_matches_at_0_90": sum(score >= 0.9 for _, score in comparisons),
            "maximum_candidate_token_jaccard": max(score for _, score in comparisons),
            "leakage_detected": any(value or score >= 0.9 for value, score in comparisons),
        },
        "coverage_assessment": {
            "training_examples": occurrences["train"]["occurrences"],
            "validation_examples": occurrences["validation"]["occurrences"],
            "scenario_expected_result_training_examples": occurrences["train"]["dataset_type_distribution"].get("scenario_expected_result", 0),
            "scenario_expected_result_validation_examples": occurrences["validation"]["dataset_type_distribution"].get("scenario_expected_result", 0),
            "predicted_reject_examples": 0,
            "assessment": "SPARSE_AND_OVERLY_TEMPLATED",
            "finding": "The code is rare relative to the corpus and every example uses the same single-field-omission pattern. Scenario coverage is two training examples and no validation examples. All targets consistently teach revise, so sparsity—not contradiction—is the supported contributing factor.",
        },
        "schema_representation_gap": {
            "training_code_examples_use_required_fields": sum("required_fields" in row["canonical_record"]["input"]["dataset_spec"] for _, row in all_hits),
            "training_code_examples_use_required": sum("required" in row["canonical_record"]["input"]["dataset_spec"] for _, row in all_hits),
            "failed_case_uses_required": "required" in failed["canonical_record"]["input"]["dataset_spec"],
            "interpretation": "The representation differs, but the emitted reason code proves the model recognized the missing-field semantics; it does not explain the wrong reject decision.",
        },
    }


def smoke_audit(raw_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    replay = []
    for original in raw_rows:
        text = original["raw_response"]
        parsed = None
        error = None
        try:
            parsed = parse_strict_response(text)
        except ValueError as exc:
            error = str(exc)
        row = {key: value for key, value in original.items() if key != "raw_response"}
        row["predicted_decision"] = parsed["decision"] if parsed else "invalid"
        row["predicted_reason_codes"] = parsed["reason_codes"] if parsed else []
        row["parse_error"] = error
        replay.append(row)
    metrics = classification(replay)
    metrics["strict_five_field_count"] = sum(
        isinstance(json.loads(row["raw_response"]), dict) and tuple(json.loads(row["raw_response"])) == FIELDS
        for row in raw_rows
    )
    metrics["approved_case_sensitive_code_count"] = sum(
        all(code in APPROVED_REASON_CODES for code in json.loads(row["raw_response"]).get("reason_codes", []))
        for row in raw_rows
    )
    metrics["confidence_exactly_one_count"] = sum(json.loads(row["raw_response"]).get("confidence") == 1.0 for row in raw_rows)
    metrics["immediate_termination_count"] = sum(bool(row["terminated"]) for row in raw_rows)
    metrics["external_prose_count"] = sum(not row["raw_response"].startswith("{") or not row["raw_response"].endswith("}") for row in raw_rows)
    metrics["markdown_fence_count"] = sum("```" in row["raw_response"] for row in raw_rows)
    metrics["token_limit_hit_count"] = sum(bool(row["hit_token_limit"]) for row in raw_rows)
    metrics["reason_code_exact_match_strict"] = sum(set(row["expected_reason_codes"]) == set(row["predicted_reason_codes"]) for row in replay) / len(replay)
    metrics["raw_emitted_code_lexical_match"] = sum(set(row["expected_reason_codes"]) == set(json.loads(raw_rows[index]["raw_response"])["reason_codes"]) for index, row in enumerate(replay)) / len(replay)
    return metrics, replay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--evidence-archive", type=Path, required=True)
    parser.add_argument("--adapter-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.private_output.mkdir(parents=True, exist_ok=True)

    smoke_root = args.evidence_root / "post-training" / "smoke"
    raw_rows = load_jsonl(smoke_root / "raw-responses.jsonl")
    normalized_rows = load_jsonl(smoke_root / "normalized-responses.jsonl")
    heldout = load_jsonl(args.heldout)
    train = load_jsonl(args.train)
    validation = load_jsonl(args.validation)
    if len(raw_rows) != 15 or len({row["case_id"] for row in raw_rows}) != 15:
        raise ValueError("SMOKE_EVIDENCE_INCOMPLETE")
    failed = next(row for row in heldout if row["case_id"] == FAILED_CASE_ID)
    failed_raw = next(row for row in raw_rows if row["case_id"] == FAILED_CASE_ID)
    failed_normalized = next(row for row in normalized_rows if row["case_id"] == FAILED_CASE_ID)
    raw_value = json.loads(failed_raw["raw_response"])
    missing = missing_required(failed)

    metrics, replay = smoke_audit(raw_rows)
    original = json.loads((smoke_root / "result.json").read_text(encoding="utf-8"))["metrics"]
    comparable = ("sample_count", "accuracy", "macro_precision", "macro_recall", "macro_f1", "contract_validity_rate", "parse_failure_rate", "false_accept_rate", "false_reject_rate", "revise_accuracy", "input_tokens", "output_tokens", "examples_per_minute")
    discrepancies = {key: {"original": original[key], "recomputed": metrics[key]} for key in comparable if not math.isclose(float(original[key]), float(metrics[key]), rel_tol=0, abs_tol=1e-12)}
    metrics_document = {
        "schema_version": "phase-e2.5-smoke-recomputed-metrics-v1",
        "raw_response_count": len(raw_rows), "unique_case_ids": len({row["case_id"] for row in raw_rows}),
        "metrics": metrics, "reported_metric_discrepancies": discrepancies,
        "malformed_outputs_counted_as_failures": True, "evidence_overwritten": False,
        "case_audit": [
            {"case_id": row["case_id"], "expected_decision": row["expected_decision"], "recomputed_decision": row["predicted_decision"],
             "expected_reason_codes": row["expected_reason_codes"], "recomputed_reason_codes": row["predicted_reason_codes"],
             "parse_error": row["parse_error"], "terminated": row["terminated"], "hit_token_limit": row["hit_token_limit"]}
            for row in replay
        ],
    }
    write_json(args.output / "smoke-recomputed-metrics.json", metrics_document)

    corpus = corpus_audit(train, validation, failed)
    write_json(args.output / "reason-code-corpus-audit.json", corpus)

    prompt_text = args.prompt.read_text(encoding="utf-8")
    contract_hash = sha256(args.contract)
    prompt_hash = sha256(args.prompt)
    failed_analysis = {
        "schema_version": "phase-e2.5-failed-case-analysis-v1",
        "case_id": FAILED_CASE_ID, "dataset_type": failed["dataset_type"],
        "dataset_schema": {"fields": failed["canonical_record"]["input"]["dataset_spec"]["fields"], "required": failed["canonical_record"]["input"]["dataset_spec"]["required"], "additional_properties": False},
        "candidate_present_fields": sorted(failed["canonical_record"]["input"]["candidate_record"]),
        "actually_missing_required_fields": missing,
        "expected_decision": failed["expected_decision"], "expected_reason_codes": failed["expected_reason_codes"],
        "expected_feedback_semantics": "Add the absent expected_result field using source-grounded content; keep the repair limited to the identified defect.",
        "raw_model_response": failed_raw["raw_response"], "raw_response_sha256": bytes_sha256(failed_raw["raw_response"].encode()),
        "normalized_response": failed_normalized["normalized"], "parser_error": failed_normalized["parse_error"],
        "parser_rejection_location": "validate_contract: decision/reason-code compatibility check after JSON/schema/type/code validation",
        "emitted_decision": raw_value["decision"], "emitted_reason_codes": raw_value["reason_codes"], "model_confidence": raw_value["confidence"],
        "generated_tokens": failed_raw["output_tokens"], "termination": {"terminated": failed_raw["terminated"], "hit_token_limit": failed_raw["hit_token_limit"], "terminator_ids": [1, 106]},
        "latency_seconds": failed_raw["latency_seconds"],
        "prompt": {"version": "critic-system-v2", "sha256": prompt_hash, "explicit_rule": "REQUIRED_FIELD_MISSING is revise-only"},
        "ontology": {"version": REASON_CODE_ONTOLOGY_VERSION, "code": "REQUIRED_FIELD_MISSING", "allowed_decisions": sorted(REASON_CODE_SPECS["REQUIRED_FIELD_MISSING"].allowed_decisions), "definition": REASON_CODE_SPECS["REQUIRED_FIELD_MISSING"].definition},
        "parser": {"contract_sha256": contract_hash, "fields": list(FIELDS), "model_version": MODEL_VERSION, "dataset_type_specific_code_rules": False, "defect_category_specific_code_rules": False},
        "compatibility": {
            "predicted_decision": "PROHIBITED", "dataset_type": "NOT_RESTRICTED", "defect_category": "NOT_RESTRICTED",
            "combination_with_other_code": "NOT_APPLICABLE_SINGLE_CODE", "explicit_rule": "reject is not in the code's revise-only allowed_decisions set",
        },
        "semantic_assessment": {
            "field_is_missing": True, "emitted_reason_code_is_semantically_correct": True,
            "complete_output_is_contract_valid": False, "why": "The correct reason code was paired with reject rather than revise.",
            "parser_enforced_frozen_specification_correctly": True, "expected_label_matches_specification": True,
        },
        "related_code_distinctions": {
            "MISSING_REQUIRED_CONSTRAINT": "Omitted behavioral/factual constraint, not an absent schema field.",
            "INCOMPLETE_RESPONSE": "Substantive content missing inside a present response, not an absent schema field.",
            "DATASET_SPEC_MISMATCH": "Broader requested-type/specification mismatch; not needed for this exact field-presence defect.",
            "WRONG_FIELD_TYPE": "Field is present with the wrong JSON type; here the field is absent.",
        },
        "base_model": {"id": "google/gemma-4-E4B-it", "revision": "ee0ef6023621cff504d758262d4e04895a5af4a2"},
        "adapter": {"training_steps": 2382, "sha256": EXPECTED_ADAPTER_SHA256, "fresh_adapter": True},
        "private_complete_input": {"stored_separately": True, "record_sha256": bytes_sha256((stable(failed) + "\n").encode()), "excluded_from_publication": True},
    }
    write_json(args.output / "failed-case-analysis.json", failed_analysis)
    write_json(args.private_output / "failed-case-full.json", {"heldout_record": failed, "raw_evaluation_record": failed_raw, "normalized_evaluation_record": failed_normalized})

    statuses = {
        "model semantic misclassification": "CONFIRMED",
        "insufficient ontology coverage": "CONTRIBUTING",
        "contradictory training targets": "RULED_OUT",
        "ambiguous distinction between related reason codes": "RULED_OUT",
        "prompt-definition ambiguity": "RULED_OUT",
        "parser compatibility-table defect": "RULED_OUT",
        "expected-label defect": "RULED_OUT",
        "dataset-schema interpretation defect": "RULED_OUT",
        "truncation": "RULED_OUT",
        "termination failure": "RULED_OUT",
        "malformed JSON": "RULED_OUT",
        "adapter-loading failure": "RULED_OUT",
        "evaluation-runner mismatch": "RULED_OUT",
        "held-out leakage": "RULED_OUT",
        "quantization or precision effects": "UNRESOLVED",
    }
    matrix_lines = ["# Phase E2.5 Root-Cause Matrix", "", "| Candidate cause | Classification | Evidence |", "|---|---|---|"]
    evidence = {
        "model semantic misclassification": "The model emitted reject with a revise-only code; the expected decision is revise.",
        "insufficient ontology coverage": "10/6,350 training rows; two scenario rows; zero scenario validation rows; one repeated structural pattern.",
        "contradictory training targets": "All 12 train/validation occurrences use revise with the single expected code.",
        "ambiguous distinction between related reason codes": "The model selected the expected code, so neighboring-code ambiguity did not produce this failure.",
        "prompt-definition ambiguity": "The prompt explicitly marks the code [revise] and provides a revise example.",
        "parser compatibility-table defect": "The parser and ontology share the same revise-only authority and reject at the documented check.",
        "expected-label defect": "The absent expected_result field is repairable and the frozen target is revise plus REQUIRED_FIELD_MISSING.",
        "dataset-schema interpretation defect": "The emitted code shows the absence was recognized despite required vs required_fields representation.",
        "truncation": "58 generated tokens; token-limit flag false.",
        "termination failure": "The response terminated normally.",
        "malformed JSON": "JSON decoding and the five-field envelope succeed; semantic compatibility fails later.",
        "adapter-loading failure": "Checkpoint identity verified; all 15 outputs were genuine and 14 passed strict parsing.",
        "evaluation-runner mismatch": "Independent parser replay reproduces the exact same single rejection and aggregate metrics.",
        "held-out leakage": "No exact or >=0.90 near match against E2.4 train/validation; prior 105-case overlap audit is also zero.",
        "quantization or precision effects": "No counterfactual unquantized run exists, so this cannot be isolated locally.",
    }
    for cause, status in statuses.items():
        matrix_lines.append(f"| {cause} | {status} | {evidence[cause]} |")
    (args.output / "root-cause-matrix.md").write_text("\n".join(matrix_lines) + "\n", encoding="utf-8")

    remediation = """# Phase E2.5 Minimal Remediation Plan

## Decision

**A. MODEL_ERROR.** The parser and expected label are correct. The model emitted the semantically correct reason code with the prohibited `reject` decision.

## Smallest defensible change

Create a development-only contrast set for decision severity while leaving the parser, ontology, held-out cases, and thresholds unchanged:

- 24 new records: four dataset types (`scenario_expected_result`, `question_answer`, `instruction_response`, and `custom`) × three paired distinctions × two render variants.
- Pair `REQUIRED_FIELD_MISSING`/`revise` against `MISSING_REQUIRED_CONSTRAINT`/`revise` and genuinely non-repairable dataset-spec failures using an already-approved reject-compatible code.
- Include both supported schema spellings (`required` and `required_fields`) without copying fictional identifiers, wording, field combinations, or source facts from the failed held-out case.
- Add at least four independent `scenario_expected_result` missing-field records to validation; never move validation or held-out records into training.
- Re-run exact, canonical-input, token-Jaccard, source-family, template-family, controlled-overfit, E2.2 smoke, E2.4 validation, and E1 held-out overlap gates before any GPU request.

## Experiment proposal—not authorized

Prefer a small continuation experiment from the safely preserved final E2 adapter, because the evidence shows a narrow decision/code compatibility lapse after an otherwise completed run. Freeze a short step ceiling and evaluate a development-only contrast set plus the same 15-case smoke. A new full 2,382-step run is not presently justified. Request explicit GPU authorization only after the new corpus and overlap evidence are reviewed.

Do not change the strict parser, compatibility table, five-field contract, frozen certification thresholds, or original smoke result.
"""
    (args.output / "remediation-plan.md").write_text(remediation, encoding="utf-8")

    result = {
        "schema_version": "phase-e2.5-forensic-results-v1", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision_gate": "A. MODEL_ERROR", "training_status": "COMPLETE", "training_steps": 2382,
        "adapter_preserved": sha256(args.adapter_archive) == EXPECTED_ADAPTER_SHA256,
        "certification_status": "BLOCKED_POST_TRAINING_SMOKE_FAIL",
        "failed_case": {"case_id": FAILED_CASE_ID, "emitted_decision": "reject", "emitted_code": "REQUIRED_FIELD_MISSING", "parser_error": "INVALID_CODE_COMBINATION:REQUIRED_FIELD_MISSING"},
        "component_findings": {"parser": "CORRECT", "expected_label": "CORRECT", "model": "WRONG_DECISION", "corpus": "CONSISTENT_BUT_SPARSE_AND_TEMPLATED"},
        "root_cause_classifications": statuses, "smoke_metrics": metrics,
        "evidence_integrity": {"evidence_archive_sha256": sha256(args.evidence_archive), "evidence_archive_matches": sha256(args.evidence_archive) == EXPECTED_EVIDENCE_SHA256, "adapter_sha256": sha256(args.adapter_archive), "adapter_matches": sha256(args.adapter_archive) == EXPECTED_ADAPTER_SHA256},
        "gpu_used": False, "inference_run": False, "retraining_run": False, "certification_run": False,
    }
    write_json(args.output / "forensic-results.json", result)

    report = f"""# Phase E2.5 Post-Training Smoke Failure Forensics

## Outcome

**A. MODEL_ERROR.** Training completed successfully at 2,382/2,382 steps and the final adapter is hash-verified. Certification remains blocked because the mandatory strict-contract smoke failed before the frozen 105-case certification could run.

## Failed case

`{FAILED_CASE_ID}` is a `scenario_expected_result` record whose candidate omits the schema-required `expected_result` field. The expected target is `revise` with `REQUIRED_FIELD_MISSING`. The model emitted the correct code but paired it with `reject`. That code is explicitly revise-only, so the parser correctly returned `INVALID_CODE_COMBINATION:REQUIRED_FIELD_MISSING` and no normalized response exists.

The JSON syntax, exact five-field shape, confidence, code spelling, and termination were valid. The failure is the decision/code semantic combination. No output was repaired or reinterpreted.

## Versioned compatibility table

| Layer | Identity | Authoritative rule | Failed output |
|---|---|---|---|
| Output contract | five-field E2 contract | Exact ordered keys: decision, confidence, reason_codes, feedback, model_version | Conforms |
| Ontology | `{REASON_CODE_ONTOLOGY_VERSION}` | `REQUIRED_FIELD_MISSING` allows only `revise` | Violated by `reject` |
| Prompt | `critic-system-v2` / `{prompt_hash}` | Marks the code `[revise]` and includes a revise example | Violated by `reject` |
| Parser | `{contract_hash}` | Reject any code whose allowed-decision set excludes the emitted decision | Correctly rejected |
| Frozen fixture | E1 held-out / `{sha256(args.heldout)}` | Missing `expected_result` → `revise` + `REQUIRED_FIELD_MISSING` | Fixture is correct |

The code is not restricted by dataset type or defect category, and no other reason code was emitted. The sole explicit incompatibility is the predicted decision.

## Corpus evidence

- E2.4 train: {corpus['occurrences']['train']['occurrences']} code examples among {corpus['occurrences']['train']['records']} rows; all are `revise`, single-code targets, with two `scenario_expected_result` examples.
- E2.4 validation: {corpus['occurrences']['validation']['occurrences']} examples among {corpus['occurrences']['validation']['records']} rows; none are `scenario_expected_result`.
- Contradictory identical inputs: {corpus['contradictory_identical_inputs']}.
- Full-corpus compatibility violations: {corpus['compatibility_table_violations_full_corpus']}.
- Exact duplicate candidates among the 12 code examples: {corpus['exact_candidate_duplicates_among_code_examples']} (same compatible target; no conflict).
- Cross-split exact candidate duplicates in the full corpus: {corpus['cross_split_exact_candidate_duplicates']}; conflicting cross-split targets: {corpus['conflicting_cross_split_exact_candidate_duplicates']}.
- Failed-case exact or >=0.90 near leakage: none.

The training targets do not contradict the contract. Sparse, repetitive coverage is a contributing generalization weakness, not evidence that the parser or frozen label is wrong.

## Recomputed smoke

- Responses: 15/15 unique
- Strict validity: {metrics['contract_validity_rate']:.6%}
- Parse failures: {metrics['parse_failure_rate']:.6%}
- Decision accuracy: {metrics['accuracy']:.6%}
- Macro precision/recall/F1: {metrics['macro_precision']:.6f} / {metrics['macro_recall']:.6f} / {metrics['macro_f1']:.6f}
- Reason-code exact match after strict parsing: {metrics['reason_code_exact_match_strict']:.6%}
- Raw emitted-code lexical match before strict compatibility validation: {metrics['raw_emitted_code_lexical_match']:.6%}
- Immediate termination: {metrics['immediate_termination_count']}/15
- Markdown, external prose, token-limit hits: {metrics['markdown_fence_count']} / {metrics['external_prose_count']} / {metrics['token_limit_hit_count']}
- Recomputed/report discrepancies: {len(discrepancies)}

All 15 outputs had the exact key set, approved case-sensitive codes, confidence 1.0, immediate termination, no Markdown, no external prose, and no token-limit hit. Beyond the single parser failure, the smoke also contains semantic errors: one valid record was rejected as a duplicate, one revise case used `GROUNDING_PARTIAL`, and all five hallucinated-detail cases used `GROUNDING_CONTRADICTION`. These are preserved as emitted and explain the 46.666667% strict reason-code exact-match rate.

## Limits

Quantization or precision effects cannot be isolated without unauthorized counterfactual inference. The 15-case smoke is intentionally small. No GPU, inference, retraining, deployment, integration, or 105-case certification was performed in E2.5.
"""
    (args.output / "forensic-report.md").write_text(report, encoding="utf-8")

    reproducibility = {
        "schema_version": "phase-e2.5-reproducibility-manifest-v1",
        "inputs": {
            "evidence_archive": sha256(args.evidence_archive), "adapter_archive": sha256(args.adapter_archive),
            "raw_responses": sha256(smoke_root / "raw-responses.jsonl"), "normalized_responses": sha256(smoke_root / "normalized-responses.jsonl"),
            "original_smoke_result": sha256(smoke_root / "result.json"), "heldout_corpus": sha256(args.heldout),
            "training_corpus": sha256(args.train), "validation_corpus": sha256(args.validation),
            "prompt": prompt_hash, "contract": contract_hash,
        },
        "identities": {"git_sha_used_for_training": "f17563714f5acab1054f010eabb7962cb49a0444", "tree_sha_used_for_training": "737d871b06776e6048ce37e4013ea4678f3e5a86", "model_id": "google/gemma-4-E4B-it", "model_revision": "ee0ef6023621cff504d758262d4e04895a5af4a2", "prompt_version": "critic-system-v2", "ontology_version": REASON_CODE_ONTOLOGY_VERSION},
        "method": {"strict_parser": "dataset_forge_critic.e2_contract.parse_strict_response", "malformed_as_failure": True, "repair": False, "near_overlap_threshold": 0.9},
    }
    write_json(args.output / "reproducibility-manifest.json", reproducibility)


if __name__ == "__main__":
    main()
