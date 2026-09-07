"""Task-aware validation metrics which never infer absent annotations."""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Iterable

from .gemma import parse_response

DECISIONS = ("accept", "revise", "reject")


def _division(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


def evaluate_predictions(rows: Iterable[tuple[str, dict[str, Any], str]]) -> dict[str, Any]:
    """Evaluate (corpus, canonical record, generated text) triples."""
    counts = Counter()
    lossless: dict[str, Counter] = {label: Counter() for label in DECISIONS}
    score_errors: dict[str, list[float]] = defaultdict(list)
    preference = Counter()
    issues = Counter()
    by_corpus = defaultdict(Counter)
    for corpus, wrapper, text in rows:
        record = wrapper.get("canonical_record", wrapper)
        counts["examples"] += 1
        by_corpus[corpus]["examples"] += 1
        try:
            prediction = parse_response(text)
            counts["json_valid"] += 1
            counts["schema_valid"] += 1
            by_corpus[corpus]["valid"] += 1
        except ValueError:
            continue
        available = set(record["supervision"]["available_targets"])
        target = record.get("target", {})
        if "decision" in available:
            raw_truth = target.get("decision")
            truth = raw_truth.casefold() if isinstance(raw_truth, str) else raw_truth
            guess = prediction.get("decision")
            if truth in DECISIONS:
                lossless[truth][guess if guess in DECISIONS else "INVALID"] += 1
        if "preferred_candidate_index" in available:
            preference["count"] += 1
            preference["correct"] += prediction.get("preferred_candidate_index") == target.get("preferred_candidate_index")
        for path in available:
            if path.startswith("scores."):
                name = path.split(".", 1)[1]
                expected = target.get("scores", {}).get(name)
                actual = prediction.get("scores", {}).get(name)
                if expected is not None and actual is not None:
                    actual_value = actual.get("value") if isinstance(actual, dict) else actual
                    if isinstance(actual_value, (int, float)):
                        score_errors[name].append(abs(float(expected["value"]) - float(actual_value)))
        if "issue_codes" in available:
            truth_set = set(target.get("issue_codes", [])); guess_set = set(prediction.get("issue_codes", []))
            issues["tp"] += len(truth_set & guess_set); issues["fp"] += len(guess_set - truth_set); issues["fn"] += len(truth_set - guess_set)

    per_class: dict[str, Any] = {}
    f1_values: list[float] = []
    matrix = {truth: {guess: lossless[truth][guess] for guess in (*DECISIONS, "INVALID")} for truth in DECISIONS}
    for label in DECISIONS:
        tp = lossless[label][label]
        fp = sum(lossless[other][label] for other in DECISIONS if other != label)
        fn = sum(lossless[label][other] for other in (*DECISIONS, "INVALID") if other != label)
        precision, recall = _division(tp, tp + fp), _division(tp, tp + fn)
        f1 = None if precision is None or recall is None or precision + recall == 0 else 2 * precision * recall / (precision + recall)
        if f1 is not None: f1_values.append(f1)
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": sum(lossless[label].values())}
    reject_total = sum(lossless["reject"].values())
    accept_total = sum(lossless["accept"].values())
    return {
        "sample_count": counts["examples"], "json_parse_validity": _division(counts["json_valid"], counts["examples"]),
        "schema_validity": _division(counts["schema_valid"], counts["examples"]),
        "by_corpus": {key: {"sample_count": value["examples"], "validity": _division(value["valid"], value["examples"])} for key, value in by_corpus.items()},
        "native": {"confusion_matrix": matrix, "per_class": per_class, "macro_f1": sum(f1_values) / len(f1_values) if f1_values else None,
                   "false_acceptance_of_reject": _division(lossless["reject"]["accept"], reject_total),
                   "unnecessary_rejection_of_accept": _division(lossless["accept"]["reject"], accept_total)},
        "score_mae": {key: sum(values) / len(values) for key, values in score_errors.items()},
        "preference_accuracy": _division(preference["correct"], preference["count"]),
        "issue_codes": {"precision": _division(issues["tp"], issues["tp"] + issues["fp"]), "recall": _division(issues["tp"], issues["tp"] + issues["fn"]), "support": issues["tp"] + issues["fn"]},
        "uncertainty": "Point estimates only; interpret with the reported sample counts. No test split is used.",
    }
