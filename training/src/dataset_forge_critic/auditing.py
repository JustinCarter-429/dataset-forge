"""Bounded-memory data verification and tokenizer statistics."""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .integrity import audit_splits, iter_jsonl, sha256_file
from .tokenization import TokenizationExclusion, tokenize_record
from .training_config import TrainingConfiguration


def percentile(values: list[int], probability: float) -> int | None:
    if not values: return None
    ordered = sorted(values); return ordered[math.ceil(probability * len(ordered)) - 1]


def verify_training_data(config: TrainingConfiguration) -> dict[str, Any]:
    paths = config.data.model_dump()
    missing = [str(path) for path in paths.values() if not Path(path).is_file()]
    if missing: raise ValueError(f"MISSING_DATA_FILES:{missing}")
    splits = audit_splits([config.data.public_train, config.data.native_train], [config.data.public_validation, config.data.native_validation])
    if splits["exact_cross_split_duplicates"] or splits["normalized_cross_split_duplicates"]:
        raise ValueError(f"CROSS_SPLIT_DUPLICATES:{splits}")
    return {"status": "DATA_VERIFIED", "physical_sha256": {key: sha256_file(Path(path)) for key, path in paths.items()}, **splits,
            "test_split_accessed": False, "near_duplicate_note": "72,211 is the public curation report's analysis-only within-corpus grouping count; it is not itself a cross-split leak count."}


def audit_tokens(config: TrainingConfiguration, processor: Any, *, limit_per_file: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"corpora": {}, "fit_lengths": [1024, 2048, 4096], "packing": False}
    files = {"public_train": config.data.public_train, "native_train": config.data.native_train,
             "public_validation": config.data.public_validation, "native_validation": config.data.native_validation}
    for name, path in files.items():
        layer = name.split("_", 1)[0]; lengths: dict[str, list[int]] = defaultdict(list); exclusions = Counter(); tasks = Counter(); count = 0
        fit = Counter()
        for item in iter_jsonl(path):
            if limit_per_file is not None and count >= limit_per_file: break
            count += 1; record = item.get("canonical_record", item); tasks[record["task_family"]] += 1
            try:
                tokenized = tokenize_record(item, layer, processor, 4096)
            except TokenizationExclusion as exc:
                exclusions[str(exc).split(":", 1)[0]] += 1; continue
            total = len(tokenized.input_ids)
            lengths["total"].append(total); lengths["prompt"].append(tokenized.prompt_tokens); lengths["completion"].append(tokenized.completion_tokens); lengths["supervised"].append(tokenized.supervised_tokens)
            for boundary in (1024, 2048, 4096): fit[str(boundary)] += total <= boundary
        stats = {key: {"p50": percentile(value, .5), "p95": percentile(value, .95), "p99": percentile(value, .99), "max": max(value, default=None)} for key, value in lengths.items()}
        expected_padding = None
        if lengths["total"]:
            expected_padding = 1 - sum(lengths["total"]) / (len(lengths["total"]) * max(lengths["total"]))
        result["corpora"][name] = {"examined": count, "accepted_at_4096": len(lengths["total"]), "task_counts": dict(tasks), "lengths": stats,
                                    "fit_rates": {key: value / count for key, value in fit.items()}, "expected_dynamic_batch_padding_fraction_upper_bound": expected_padding, "exclusions": dict(exclusions)}
    return result
