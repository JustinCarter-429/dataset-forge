"""Deterministic validation for base-model and adapter comparisons."""
from __future__ import annotations

import hashlib
import heapq
from collections import defaultdict
from pathlib import Path
from typing import Any

from .integrity import iter_jsonl
from .metrics import evaluate_predictions
from .modeling import load_processor, load_qlora_model
from .tokenization import TokenizationExclusion, tokenize_record
from .training_config import TrainingConfiguration


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
    with torch.inference_mode():
        for corpus, path, requested in sources:
            count = requested or sum(1 for _ in iter_jsonl(path))
            for item in fixed_subset(path, count):
                record = item.get("canonical_record", item)
                try: tokenized = tokenize_record(item, corpus, processor, config.max_sequence_length)
                except TokenizationExclusion as exc: exclusions[str(exc).split(":", 1)[0]] += 1; continue
                values = {"input_ids": torch.tensor([tokenized.input_ids], device="cuda"), "attention_mask": torch.tensor([tokenized.attention_mask], device="cuda"), "labels": torch.tensor([tokenized.labels], device="cuda")}
                loss = model(**values).loss; losses[f"corpus:{corpus}"].append(float(loss)); losses[f"task:{record['task_family']}"] .append(float(loss))
                prompt = tokenized.input_ids[: tokenized.prompt_tokens]
                output = model.generate(input_ids=torch.tensor([prompt], device="cuda"), attention_mask=torch.ones((1, len(prompt)), dtype=torch.long, device="cuda"), do_sample=False, max_new_tokens=768)
                text = processor.decode(output[0][len(prompt):], skip_special_tokens=True)
                predictions.append((corpus, item, text))
    result = evaluate_predictions(predictions); result.update({"loss": {key: sum(value) / len(value) for key, value in losses.items()}, "exclusions": dict(exclusions),
                                                                    "adapter": str(adapter) if adapter else None, "base_model_baseline": adapter is None, "parameter_report": parameters,
                                                                    "subset": "broad" if broad else "frequent", "test_split_accessed": False})
    return result
