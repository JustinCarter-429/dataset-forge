"""Deterministic validation for base-model and adapter comparisons."""
from __future__ import annotations

import hashlib
import heapq
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from .integrity import iter_jsonl
from .gemma import parse_response
from .metrics import evaluate_predictions
from .modeling import load_processor, load_qlora_model
from .tokenization import TokenizationExclusion, generation_terminator_ids, tokenize_record
from .training_config import TrainingConfiguration


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def controlled_overfit_gate(rows: list[dict[str, Any]], max_new_tokens: int) -> dict[str, Any]:
    total = len(rows)
    strict = sum(row["strict_valid"] for row in rows)
    terminated = sum(row["terminated"] and row["output_tokens"] < max_new_tokens for row in rows)
    correct = sum(row["predicted_decision"] == row["expected_decision"] for row in rows)
    fences = sum("```" in row["text"] for row in rows)
    external = sum(not row["strict_valid"] for row in rows)
    limit = sum(row["output_tokens"] >= max_new_tokens for row in rows)
    metrics = {
        "sample_count": total,
        "strict_json_validity": strict / total if total else 0.0,
        "immediate_termination_rate": terminated / total if total else 0.0,
        "decision_accuracy": correct / total if total else 0.0,
        "markdown_fence_count": fences,
        "external_prose_count": external,
        "hit_output_token_limit_count": limit,
    }
    metrics["status"] = "passed" if (
        metrics["strict_json_validity"] == 1.0
        and metrics["immediate_termination_rate"] == 1.0
        and metrics["decision_accuracy"] >= 0.95
        and fences == external == limit == 0
    ) else "failed"
    return metrics


def fixed_subset(path: Path, count: int) -> list[dict[str, Any]]:
    """Keep the N lowest record-id hashes without loading the whole corpus."""
    heap: list[tuple[int, dict[str, Any]]] = []
    for item in iter_jsonl(path):
        record = item.get("canonical_record", item); rank = int(hashlib.sha256(record["record_id"].encode()).hexdigest(), 16)
        entry = (-rank, item)
        if len(heap) < count: heapq.heappush(heap, entry)
        elif rank < -heap[0][0]: heapq.heapreplace(heap, entry)
    return [item for _, item in sorted(heap, key=lambda pair: -pair[0])]


def evaluate(config: TrainingConfiguration, *, adapter: Path | None, broad: bool = False) -> dict[str, Any]:
    import torch
    processor = load_processor(config); model, parameters = load_qlora_model(config, resume_adapter=adapter)
    model.eval(); predictions = []; losses: dict[str, list[float]] = defaultdict(list); exclusions: dict[str, int] = defaultdict(int)
    sources = [
        ("public", config.data.public_validation, config.evaluation.broad_public_count if broad else config.evaluation.frequent_public_count),
        ("native", config.data.native_validation, config.evaluation.broad_native_count if broad else config.evaluation.frequent_native_count),
    ]
    generation_rows: list[dict[str, Any]] = []; terminators = generation_terminator_ids(processor); max_new_tokens = 256
    artifact_value = os.environ.get("DATASET_FORGE_EVAL_ARTIFACT_DIR")
    artifact_root = Path(artifact_value).resolve() if artifact_value else None
    completed: dict[str, dict[str, Any]] = {}
    raw_handle = None
    if artifact_root:
        artifact_root.mkdir(parents=True, exist_ok=True)
        raw_path = artifact_root / "raw-outputs.jsonl"
        if raw_path.is_file():
            for line in raw_path.read_text(encoding="utf-8").splitlines():
                saved = json.loads(line); key = saved["corpus"] + ":" + saved["record_id"]
                if key in completed: raise ValueError(f"DUPLICATE_INCREMENTAL_EVALUATION:{key}")
                completed[key] = saved
        raw_handle = raw_path.open("a", encoding="utf-8", newline="\n")
        _atomic_json(artifact_root / "progress.json", {"status": "running", "completed": len(completed)})
    with torch.inference_mode():
        for corpus, path, requested in sources:
            count = requested or sum(1 for _ in iter_jsonl(path))
            for item in fixed_subset(path, count):
                record = item.get("canonical_record", item)
                key = corpus + ":" + record["record_id"]
                if key in completed:
                    saved = completed[key]; predictions.append((corpus, item, saved["text"])); generation_rows.append(saved)
                    continue
                try: tokenized = tokenize_record(item, corpus, processor, config.max_sequence_length)
                except TokenizationExclusion as exc: exclusions[str(exc).split(":", 1)[0]] += 1; continue
                values = {"input_ids": torch.tensor([tokenized.input_ids], device="cuda"), "attention_mask": torch.tensor([tokenized.attention_mask], device="cuda"), "labels": torch.tensor([tokenized.labels], device="cuda")}
                loss = model(**values).loss; losses[f"corpus:{corpus}"].append(float(loss)); losses[f"task:{record['task_family']}"] .append(float(loss))
                prompt = tokenized.input_ids[: tokenized.prompt_tokens]
                output = model.generate(input_ids=torch.tensor([prompt], device="cuda"), attention_mask=torch.ones((1, len(prompt)), dtype=torch.long, device="cuda"), do_sample=False, max_new_tokens=max_new_tokens,
                                        eos_token_id=terminators, pad_token_id=processor.tokenizer.pad_token_id)
                new_ids = output[0][len(prompt):]; text = processor.decode(new_ids, skip_special_tokens=True)
                predictions.append((corpus, item, text))
                try: parsed = parse_response(text); strict_valid = True
                except ValueError: parsed = {}; strict_valid = False
                truth = record.get("target", {}).get("decision"); truth = truth.casefold() if isinstance(truth, str) else truth
                generation_rows.append({"strict_valid": strict_valid, "terminated": bool(len(new_ids)) and int(new_ids[-1]) in terminators,
                                        "output_tokens": int(len(new_ids)), "predicted_decision": parsed.get("decision"),
                                        "expected_decision": truth, "text": text})
                if raw_handle:
                    saved = {**generation_rows[-1], "corpus": corpus, "record_id": record["record_id"]}
                    raw_handle.write(json.dumps(saved, ensure_ascii=False, sort_keys=True) + "\n"); raw_handle.flush(); os.fsync(raw_handle.fileno())
                    completed[key] = saved
                    _atomic_json(artifact_root / "progress.json", {"status": "running", "completed": len(completed)})
    if raw_handle: raw_handle.close()
    result = evaluate_predictions(predictions); result.update({"loss": {key: sum(value) / len(value) for key, value in losses.items()}, "exclusions": dict(exclusions),
                                                                    "adapter": str(adapter) if adapter else None, "base_model_baseline": adapter is None, "parameter_report": parameters,
                                                                    "subset": "broad" if broad else "frequent", "test_split_accessed": False,
                                                                    "controlled_overfit_gate": controlled_overfit_gate(generation_rows, max_new_tokens)})
    if artifact_root:
        (artifact_root / "evaluation-result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        _atomic_json(artifact_root / "progress.json", {"status": "complete", "completed": len(completed), "gate": result["controlled_overfit_gate"]})
    return result
