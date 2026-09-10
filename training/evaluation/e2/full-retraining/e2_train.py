"""Fresh-base Phase E2 smoke and full QLoRA training with durable evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import signal
import sys
import time
from collections import Counter, defaultdict, deque
from functools import lru_cache
from pathlib import Path
from typing import Any

TRAINING_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(TRAINING_ROOT / "src"))

from dataset_forge_critic.e2_contract import APPROVED_REASON_CODES, FIELDS, canonical_json, parse_strict_response  # noqa: E402
from dataset_forge_critic.modeling import load_processor, load_qlora_model  # noqa: E402
from dataset_forge_critic.tokenization import generation_terminator_ids, pad_batch, tokenize_record  # noqa: E402
from dataset_forge_critic.training_config import TrainingConfiguration  # noqa: E402

STOP = False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def optimizer_step_count(record_count: int, epochs: int, gradient_accumulation_steps: int) -> int:
    """Return steps needed to consume every record in every epoch, including a partial final batch."""
    if record_count <= 0 or epochs <= 0 or gradient_accumulation_steps <= 0:
        raise ValueError("SAMPLER_DIMENSIONS_MUST_BE_POSITIVE")
    return math.ceil(record_count * epochs / gradient_accumulation_steps)


@lru_cache(maxsize=64)
def epoch_order(record_count: int, seed: int, epoch: int) -> tuple[int, ...]:
    """Create the runner's deterministic, no-replacement order for one epoch."""
    if record_count <= 0 or epoch < 0:
        raise ValueError("INVALID_EPOCH_COORDINATES")
    order = list(range(record_count))
    random.Random(seed + epoch).shuffle(order)
    return tuple(order)


def sampler_coordinate(record_count: int, seed: int, sample_index: int) -> tuple[int, int, int]:
    """Map a durable global sample position to epoch, position, and row index."""
    if sample_index < 0:
        raise ValueError("NEGATIVE_SAMPLER_POSITION")
    epoch, position = divmod(sample_index, record_count)
    return epoch, position, epoch_order(record_count, seed, epoch)[position]


def schedule_steps(total_steps: int, every_steps: int, *, baseline: bool) -> list[int]:
    if total_steps <= 0 or every_steps <= 0:
        raise ValueError("INVALID_SCHEDULE")
    values = list(range(every_steps, total_steps + 1, every_steps))
    if not values or values[-1] != total_steps:
        values.append(total_steps)
    return ([0] if baseline else []) + values


def accumulation_target(total_samples: int, sample_index: int, gradient_accumulation_steps: int) -> int:
    """Return the divisor for the current batch so a final partial batch is averaged correctly."""
    if not 0 <= sample_index < total_samples:
        raise ValueError("SAMPLE_INDEX_OUT_OF_RANGE")
    batch_start = (sample_index // gradient_accumulation_steps) * gradient_accumulation_steps
    return min(gradient_accumulation_steps, total_samples - batch_start)


def sampler_dry_run(record_ids: list[str], epochs: int, gradient_accumulation_steps: int, seed: int) -> dict[str, Any]:
    """Execute the same index mapping used by training and return record-level accounting."""
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("DUPLICATE_CORPUS_RECORD_ID")
    seen: list[list[str]] = []
    order_hashes: list[str] = []
    flattened: list[str] = []
    for epoch in range(epochs):
        order = epoch_order(len(record_ids), seed, epoch)
        values = [record_ids[index] for index in order]
        if len(values) != len(record_ids) or len(set(values)) != len(record_ids) or set(values) != set(record_ids):
            raise ValueError(f"SAMPLER_ACCOUNTING_FAILURE_EPOCH_{epoch}")
        seen.append(values)
        flattened.extend(values)
        order_hashes.append(hashlib.sha256("\n".join(values).encode()).hexdigest())
    steps = optimizer_step_count(len(record_ids), epochs, gradient_accumulation_steps)
    return {
        "records_per_epoch": len(record_ids), "epochs": epochs, "examples": len(flattened),
        "optimizer_steps": steps, "full_effective_batches": len(flattened) // gradient_accumulation_steps,
        "final_partial_batch_examples": len(flattened) % gradient_accumulation_steps,
        "epoch_order_sha256": order_hashes,
        "each_record_once_per_epoch": all(len(values) == len(set(values)) == len(record_ids) for values in seen),
    }


def stratified(items: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        record = item["canonical_record"]
        key = (record["target"]["decision"], record["input"]["dataset_spec"]["dataset_type"])
        buckets[key].append(item)
    rng = random.Random(seed)
    for values in buckets.values():
        values.sort(key=lambda value: value["canonical_record"]["record_id"]); rng.shuffle(values)
    selected: list[dict[str, Any]] = []
    keys = sorted(buckets)
    while len(selected) < count and keys:
        next_keys = []
        for key in keys:
            if buckets[key] and len(selected) < count:
                selected.append(buckets[key].pop())
            if buckets[key]:
                next_keys.append(key)
        keys = next_keys
    return selected


def macro_f1(truth: list[str], prediction: list[str | None]) -> float:
    scores = []
    for label in ("accept", "revise", "reject"):
        tp = sum(a == label and b == label for a, b in zip(truth, prediction))
        fp = sum(a != label and b == label for a, b in zip(truth, prediction))
        fn = sum(a == label and b != label for a, b in zip(truth, prediction))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)


def reason_code_macro_f1(truth: list[set[str]], prediction: list[set[str]]) -> float:
    scores = []
    for code in sorted(APPROVED_REASON_CODES):
        tp = sum(code in actual and code in predicted for actual, predicted in zip(truth, prediction))
        fp = sum(code not in actual and code in predicted for actual, predicted in zip(truth, prediction))
        fn = sum(code in actual and code not in predicted for actual, predicted in zip(truth, prediction))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)


def response_diagnostics(text: str) -> dict[str, Any]:
    result = {"unknown_reason_codes": [], "incorrectly_cased_reason_codes": [], "extra_keys": [], "missing_keys": list(FIELDS)}
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return result
    if not isinstance(value, dict):
        return result
    result["extra_keys"] = sorted(set(value) - set(FIELDS))
    result["missing_keys"] = sorted(set(FIELDS) - set(value))
    codes = value.get("reason_codes")
    if isinstance(codes, list):
        for code in codes:
            if not isinstance(code, str) or code in APPROVED_REASON_CODES:
                continue
            if code.upper() in APPROVED_REASON_CODES:
                result["incorrectly_cased_reason_codes"].append(code)
            else:
                result["unknown_reason_codes"].append(code)
    return result


def verify_inputs(config: dict[str, Any], data_dir: Path) -> tuple[Path, Path]:
    train, validation = data_dir / "train/corpus.jsonl", data_dir / "validation/corpus.jsonl"
    expected = config["data"]
    actual = {"train_sha256": sha256(train), "validation_sha256": sha256(validation)}
    if actual["train_sha256"] != expected["train_sha256"] or actual["validation_sha256"] != expected["validation_sha256"]:
        raise ValueError(f"CORPUS_HASH_MISMATCH:{actual}")
    train_rows, validation_rows = rows(train), rows(validation)
    if len(train_rows) != expected["train_records"] or len(validation_rows) != expected["validation_records"]:
        raise ValueError("CORPUS_COUNT_MISMATCH")
    for item in train_rows + validation_rows:
        parse_strict_response(canonical_json(item["canonical_record"]["target"]))
    return train, validation


def training_configuration(config: dict[str, Any], data_dir: Path, output_root: Path, mode: str) -> TrainingConfiguration:
    opt, model, monitoring = config["optimization"], config["model"], config["monitoring"]
    train, validation = data_dir / "train/corpus.jsonl", data_dir / "validation/corpus.jsonl"
    max_steps = config["smoke"]["optimizer_steps"] if mode == "smoke" else opt["max_optimizer_steps"]
    max_examples = max_steps * opt["gradient_accumulation_steps"] if mode == "smoke" else config["data"]["train_records"] * opt["epochs"]
    return TrainingConfiguration.model_validate({
        "config_version": "2.0.0", "profile": "gpu_smoke" if mode == "smoke" else "full", "seed": opt["seed"],
        "output_root": str(output_root), "max_sequence_length": opt["max_sequence_length"], "packing": False,
        "model": {"model_id": model["id"], "revision": model["revision"], "architecture": model["architecture"],
                  "renderer_version": model["renderer"], "require_bf16": True, "load_in_4bit": True, "quant_type": "nf4",
                  "double_quant": True, "gradient_checkpointing": True, "lora_rank": opt["lora_rank"], "lora_alpha": opt["lora_alpha"],
                  "lora_dropout": opt["lora_dropout"], "lora_target_regex": opt["lora_target_regex"]},
        "data": {"public_train": str(train), "native_train": str(train), "public_validation": str(validation), "native_validation": str(validation)},
        "mixture": {"public_probability": 0.5, "native_probability": 0.5, "with_replacement": True},
        "optimization": {"micro_batch_size": opt["micro_batch_size"], "gradient_accumulation_steps": opt["gradient_accumulation_steps"],
                         "learning_rate": opt["learning_rate"], "weight_decay": opt["weight_decay"], "optimizer": opt["optimizer"],
                         "scheduler": opt["scheduler"], "warmup_ratio": opt["warmup_ratio"], "max_grad_norm": opt["max_grad_norm"]},
        "limits": {"max_optimizer_steps": max_steps, "max_examples": max_examples,
                   "max_wall_seconds": monitoring["max_wall_seconds"], "min_free_disk_gb": monitoring["min_free_disk_gb"]},
        "checkpoint": {"every_steps": monitoring["checkpoint_every_steps"], "keep_latest": 3, "keep_best": 1, "selection_metric": "native_macro_f1"},
        "evaluation": {"every_steps": monitoring["validation_every_steps"], "frequent_public_count": monitoring["generation_records"],
                       "frequent_native_count": monitoring["generation_records"], "broad_public_count": 750, "broad_native_count": 750},
        "upload": {"destination": None, "visibility": None, "retries": 3},
    })


def evaluate(model: Any, processor: Any, items: list[dict[str, Any]], loss_count: int, generation_count: int, max_length: int, max_new_tokens: int) -> dict[str, Any]:
    import torch
    model.eval(); losses = []
    loss_rows = stratified(items, min(loss_count, len(items)), 7301)
    with torch.inference_mode():
        for row in loss_rows:
            item = tokenize_record(row, "native", processor, max_length)
            batch = pad_batch([item], processor.tokenizer.pad_token_id)
            values = {key: torch.tensor(value, device="cuda") for key, value in batch.items()}
            value = float(model(**values).loss.item())
            if not math.isfinite(value): raise FloatingPointError("NON_FINITE_VALIDATION_LOSS")
            losses.append(value)
        truths: list[str] = []; predictions: list[str | None] = []; truth_codes: list[set[str]] = []; prediction_codes: list[set[str]] = []; raw = []
        terminators = generation_terminator_ids(processor)
        for row in stratified(items, min(generation_count, len(items)), 7302):
            item = tokenize_record(row, "native", processor, max_length)
            prompt = item.input_ids[:item.prompt_tokens]
            output = model.generate(input_ids=torch.tensor([prompt], device="cuda"), attention_mask=torch.ones((1, len(prompt)), dtype=torch.long, device="cuda"),
                                    do_sample=False, max_new_tokens=max_new_tokens, eos_token_id=terminators, pad_token_id=processor.tokenizer.pad_token_id)
            generated = output[0][len(prompt):].tolist(); text = processor.decode(generated, skip_special_tokens=True)
            target = row["canonical_record"]["target"]; truth = target["decision"]; expected_codes = set(target["reason_codes"]); truths.append(truth); truth_codes.append(expected_codes)
            parser_error = None
            try:
                parsed = parse_strict_response(text); prediction = parsed["decision"]; predicted_codes = set(parsed["reason_codes"])
            except ValueError as error:
                prediction = None; predicted_codes = set(); parser_error = str(error)
            predictions.append(prediction); prediction_codes.append(predicted_codes)
            diagnostics = response_diagnostics(text)
            remediation = row.get("remediation", {})
            raw.append({"record_id": row["canonical_record"]["record_id"], "truth": truth, "prediction": prediction,
                        "truth_reason_codes": sorted(expected_codes), "prediction_reason_codes": sorted(predicted_codes), "parser_error": parser_error, "text": text,
                        "dataset_type": row["canonical_record"]["input"]["dataset_spec"]["dataset_type"], "semantic_group": remediation.get("semantic_group"),
                        "attack_category": remediation.get("mutation_family"), **diagnostics,
                        "output_tokens": len(generated), "terminated": bool(generated) and generated[-1] in terminators,
                        "markdown_fence": "```" in text, "hit_token_limit": len(generated) >= max_new_tokens})
    model.train()
    valid = sum(value is not None for value in predictions)
    reject_rows = [row for row in raw if row["truth"] == "reject"]
    malicious_rows = [row for row in raw if row["semantic_group"] == "malicious"]
    per_type = {}
    for dataset_type in sorted({row["dataset_type"] for row in raw}):
        subset = [row for row in raw if row["dataset_type"] == dataset_type]
        per_type[dataset_type] = {"records": len(subset), "accuracy": sum(row["truth"] == row["prediction"] for row in subset) / len(subset)}
    per_attack_category = {}
    for category in sorted({str(row["attack_category"]) for row in raw}):
        subset = [row for row in raw if str(row["attack_category"]) == category]
        per_attack_category[category] = {"records": len(subset), "accuracy": sum(row["truth"] == row["prediction"] for row in subset) / len(subset),
                                         "reason_code_exact_match": sum(set(row["truth_reason_codes"]) == set(row["prediction_reason_codes"]) for row in subset) / len(subset)}
    expected_coverage = set().union(*truth_codes) if truth_codes else set()
    return {"validation_loss": sum(losses) / len(losses), "generation_records": len(raw), "strict_json_validity": valid / len(raw),
            "parse_failure_rate": 1 - valid / len(raw), "accuracy": sum(a == b for a, b in zip(truths, predictions)) / len(raw),
            "macro_f1": macro_f1(truths, predictions), "immediate_termination_rate": sum(row["terminated"] and not row["hit_token_limit"] for row in raw) / len(raw),
            "grounding_false_accept_rate": sum(row["prediction"] == "accept" for row in reject_rows) / len(reject_rows) if reject_rows else 0.0,
            "malicious_injection_failure_rate": sum(row["prediction"] != "reject" for row in malicious_rows) / len(malicious_rows) if malicious_rows else 0.0,
            "reason_code_exact_match": sum(actual == predicted for actual, predicted in zip(truth_codes, prediction_codes)) / len(raw),
            "reason_code_macro_f1": reason_code_macro_f1(truth_codes, prediction_codes),
            "evaluation_reason_code_coverage": len(expected_coverage) / len(APPROVED_REASON_CODES),
            "evaluated_reason_codes": sorted(expected_coverage), "per_dataset_type": per_type, "per_attack_category": per_attack_category,
            "unknown_reason_codes": sum(len(row["unknown_reason_codes"]) for row in raw),
            "incorrectly_cased_reason_codes": sum(len(row["incorrectly_cased_reason_codes"]) for row in raw),
            "extra_keys": sum(len(row["extra_keys"]) for row in raw), "missing_keys": sum(len(row["missing_keys"]) for row in raw),
            "markdown_fences": sum(row["markdown_fence"] for row in raw), "external_prose": len(raw) - valid,
            "token_limit_hits": sum(row["hit_token_limit"] for row in raw), "raw": raw}


def smoke_gate_failures(config: dict[str, Any], metrics: dict[str, Any], step: int, adapter_changed: bool, finite_loss: bool, finite_gradients: bool) -> list[str]:
    thresholds = config.get("smoke", {}).get("gates")
    if not thresholds:
        thresholds = {"finite_loss": True, "finite_gradients": True, "adapter_parameters_changed": True,
                      "finite_validation_loss": True, "strict_json_validity_min": 1.0, "immediate_termination_min": 1.0,
                      "markdown_fences_max": 0, "external_prose_max": 0, "token_limit_hits_max": 0}
    actual = {"completed_optimizer_steps": step, "finite_loss": finite_loss, "finite_gradients": finite_gradients,
              "adapter_parameters_changed": adapter_changed, "finite_validation_loss": math.isfinite(metrics["validation_loss"]), **metrics}
    actual["decision_accuracy"] = metrics.get("accuracy")
    actual["decision_macro_f1"] = metrics.get("macro_f1")
    actual["immediate_termination"] = metrics.get("immediate_termination_rate")
    failures = []
    for key, expected in thresholds.items():
        if key == "process_exit_code":
            continue
        if key.endswith("_min"):
            metric = key[:-4]
            if actual.get(metric) is None or actual[metric] < expected:
                failures.append(f"{metric}={actual.get(metric)!r} < {expected!r}")
        elif key.endswith("_max"):
            metric = key[:-4]
            if actual.get(metric) is None or actual[metric] > expected:
                failures.append(f"{metric}={actual.get(metric)!r} > {expected!r}")
        elif actual.get(key) != expected:
            failures.append(f"{key}={actual.get(key)!r} != {expected!r}")
    return failures


def validation_regression_alerts(config: dict[str, Any], baseline: dict[str, Any], current: dict[str, Any], best_loss: float) -> list[str]:
    policy = config["regression_alerts"]
    alerts = []
    if math.isfinite(best_loss) and current["validation_loss"] > best_loss * (1 + policy["validation_loss_vs_best_relative_increase"]):
        alerts.append("VALIDATION_LOSS_RELATIVE_INCREASE")
    for metric, key in (("strict_json_validity", "strict_json_validity_drop_from_baseline"),
                        ("macro_f1", "decision_macro_f1_drop_from_baseline"),
                        ("reason_code_macro_f1", "reason_code_macro_f1_drop_from_baseline")):
        if baseline and baseline.get(metric, 0.0) - current.get(metric, 0.0) > policy[key]:
            alerts.append(f"{metric.upper()}_BASELINE_REGRESSION")
    return alerts


def checkpoint_file_hashes(directory: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for path in sorted(value for value in directory.rglob("*") if value.is_file() and value.name != "checkpoint-manifest.json"):
        relative = path.relative_to(directory).as_posix()
        result[relative] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    return result


def verify_checkpoint(path: Path, config_digest: str | None = None) -> dict[str, Any]:
    manifest_path = path / "checkpoint-manifest.json"
    if not manifest_path.is_file():
        raise ValueError("CHECKPOINT_MANIFEST_MISSING")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("complete") is not True:
        raise ValueError("CHECKPOINT_INCOMPLETE")
    if config_digest is not None and manifest.get("config_digest") != config_digest:
        raise ValueError("RESUME_CONFIG_MISMATCH")
    expected = manifest.get("files")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("CHECKPOINT_HASH_MANIFEST_MISSING")
    for relative, identity in expected.items():
        member = Path(relative)
        if member.is_absolute() or ".." in member.parts:
            raise ValueError("UNSAFE_CHECKPOINT_MEMBER")
        candidate = path / member
        if not candidate.is_file() or candidate.stat().st_size != identity["bytes"] or sha256(candidate) != identity["sha256"]:
            raise ValueError(f"CORRUPTED_CHECKPOINT:{relative}")
    if checkpoint_file_hashes(path) != expected:
        raise ValueError("CHECKPOINT_FILE_SET_MISMATCH")
    return manifest


def _checkpoint_destination(run_dir: Path, step: int, kind: str) -> Path:
    if kind == "best":
        return run_dir / f"best-checkpoints/step-{step:08d}"
    if kind == "final":
        return run_dir / f"final-checkpoint/step-{step:08d}"
    if kind == "periodic":
        return run_dir / f"checkpoints/step-{step:08d}"
    raise ValueError(f"UNKNOWN_CHECKPOINT_KIND:{kind}")


def save_checkpoint(run_dir: Path, model: Any, optimizer: Any, scheduler: Any, step: int, sample_index: int,
                    config_digest: str, metrics: dict[str, Any], *, kind: str = "periodic", keep_latest: int = 3) -> Path:
    import torch
    destination = _checkpoint_destination(run_dir, step, kind)
    if destination.exists():
        verify_checkpoint(destination, config_digest)
        return destination
    stage = destination.with_name(destination.name + f".tmp-{os.getpid()}")
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    model.save_pretrained(stage / "adapter", safe_serialization=True)
    torch.save({
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": None,
        "step": step, "sample_index": sample_index, "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
    }, stage / "training-state.pt")
    files = checkpoint_file_hashes(stage)
    atomic_json(stage / "checkpoint-manifest.json", {
        "schema_version": "phase-e2.4-checkpoint-v1", "complete": True, "kind": kind,
        "step": step, "sample_index": sample_index, "config_digest": config_digest,
        "metrics": {key: value for key, value in metrics.items() if key != "raw"}, "files": files,
    })
    verify_checkpoint(stage, config_digest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(stage, destination)
    verify_checkpoint(destination, config_digest)
    pointer = "best-checkpoint.json" if kind == "best" else "latest-checkpoint.json"
    atomic_json(run_dir / pointer, {"path": str(destination), "step": step, "kind": kind, "manifest_sha256": sha256(destination / "checkpoint-manifest.json")})
    if kind == "periodic":
        completed = sorted((run_dir / "checkpoints").glob("step-*"))
        for obsolete in completed[:-keep_latest]:
            verify_checkpoint(obsolete, config_digest)
            shutil.rmtree(obsolete)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--config", type=Path, required=True); parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True); parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--git-sha", required=True); parser.add_argument("--resume", type=Path)
    args = parser.parse_args(); config = json.loads(args.config.read_text(encoding="utf-8")); train_path, validation_path = verify_inputs(config, args.data_dir)
    args.run_dir.mkdir(parents=True, exist_ok=True); effective = training_configuration(config, args.data_dir, args.run_dir.parent, args.mode)
    digest = hashlib.sha256(json.dumps({"config": config, "mode": args.mode}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    resume_manifest = verify_checkpoint(args.resume, digest) if args.resume else None
    os.environ["DATASET_FORGE_MODEL_PATH"] = str(args.model_path.resolve())
    atomic_json(args.run_dir / "run-manifest.json", {"config": config, "effective_training_configuration": effective.canonical_dict(), "mode": args.mode,
                "config_digest": digest, "git_sha": args.git_sha, "train_sha256": sha256(train_path), "validation_sha256": sha256(validation_path),
                "fresh_base_adapter": args.resume is None, "resume_from_verified_fresh_run": args.resume is not None, "e1_adapter_used": False,
                "sampling": "deterministic_epoch_shuffle_without_replacement", "legacy_mixture_fields_used_by_runner": False,
                "checkpoint_selection": config["checkpoint_policy"]["best_selection"],
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    import torch
    from transformers import get_scheduler, set_seed
    set_seed(config["optimization"]["seed"]); processor = load_processor(effective)
    model, parameter_report = load_qlora_model(effective, resume_adapter=args.resume / "adapter" if args.resume else None)
    terminators = generation_terminator_ids(processor)
    if terminators != [1, 106]: raise ValueError(f"TERMINATOR_IDENTITY_MISMATCH:{terminators}")
    train_rows, validation_rows = rows(train_path), rows(validation_path)
    if args.mode == "smoke": train_rows = stratified(train_rows, config["smoke"]["training_records"], 7001)
    opt = config["optimization"]
    epochs = 1 if args.mode == "smoke" else opt["epochs"]
    total_samples = len(train_rows) * epochs
    calculated_steps = optimizer_step_count(len(train_rows), epochs, opt["gradient_accumulation_steps"])
    max_steps = config["smoke"]["optimizer_steps"] if args.mode == "smoke" else opt["max_optimizer_steps"]
    if calculated_steps != max_steps:
        raise ValueError(f"OPTIMIZER_STEP_COUNT_MISMATCH:calculated={calculated_steps}:configured={max_steps}")
    record_ids = [row["canonical_record"]["record_id"] for row in train_rows]
    dry_run = sampler_dry_run(record_ids, epochs, opt["gradient_accumulation_steps"], opt["seed"])
    atomic_json(args.run_dir / "sampler-accounting.json", dry_run)
    optimizer = __import__("bitsandbytes").optim.PagedAdamW8bit(model.parameters(), lr=opt["learning_rate"], weight_decay=opt["weight_decay"])
    scheduler = get_scheduler(opt["scheduler"], optimizer, int(max_steps * opt["warmup_ratio"]), max_steps)
    step = sample_index = 0
    if args.resume:
        state = torch.load(args.resume / "training-state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"]); step = state["step"]; sample_index = state["sample_index"]
        if resume_manifest["step"] != step or resume_manifest["sample_index"] != sample_index:
            raise ValueError("RESUME_PROGRESS_STATE_INCONSISTENCY")
        expected_resumed_steps = optimizer_step_count(sample_index, 1, opt["gradient_accumulation_steps"])
        if expected_resumed_steps != step or (sample_index % opt["gradient_accumulation_steps"] != 0 and sample_index != total_samples):
            raise ValueError("NON_DETERMINISTIC_RESUME_POSITION")
        random.setstate(state["python_rng"]); torch.set_rng_state(state["torch_rng"]); torch.cuda.set_rng_state_all(state["cuda_rng"])
    resumed_adapter_already_updated = bool(args.resume and step > 0)
    first_parameter = next(parameter for parameter in model.parameters() if parameter.requires_grad)
    initial_parameter = first_parameter.detach().float().cpu().clone(); model.train(); optimizer.zero_grad(set_to_none=True)
    started = time.monotonic(); recent = deque(maxlen=50); best_loss = math.inf; best_path = None; latest_validation: dict[str, Any] = {}
    best_pointer = args.run_dir / "best-checkpoint.json"
    if args.resume and best_pointer.is_file():
        pointer = json.loads(best_pointer.read_text(encoding="utf-8"))
        prior_best = Path(pointer["path"])
        prior_manifest = verify_checkpoint(prior_best, digest)
        best_path = prior_best
        best_loss = float(prior_manifest["metrics"]["validation_loss"])
    def request_stop(*_: Any) -> None:
        global STOP; STOP = True
    signal.signal(signal.SIGTERM, request_stop); signal.signal(signal.SIGINT, request_stop)
    baseline_validation: dict[str, Any] = {}
    if args.mode == "full" and config["monitoring"]["baseline_validation"] and not args.resume:
        latest_validation = evaluate(model, processor, validation_rows, config["monitoring"]["validation_loss_records"],
                                     config["monitoring"]["generation_records"], opt["max_sequence_length"], config["monitoring"]["max_new_tokens"])
        baseline_validation = latest_validation
        atomic_json(args.run_dir / "validation/step-00000000-baseline.json", latest_validation)
    validation_schedule = set(schedule_steps(max_steps, config["monitoring"]["validation_every_steps"], baseline=False))
    checkpoint_schedule = set(schedule_steps(max_steps, config["monitoring"]["checkpoint_every_steps"], baseline=False))
    while sample_index < total_samples and step < max_steps and not STOP:
        if time.monotonic() - started >= config["monitoring"]["max_wall_seconds"]: break
        if shutil.disk_usage(args.run_dir).free / 1024**3 < config["monitoring"]["min_free_disk_gb"]: raise RuntimeError("DISK_SAFETY_THRESHOLD")
        epoch, position, row_index = sampler_coordinate(len(train_rows), opt["seed"], sample_index)
        if epoch >= epochs:
            raise RuntimeError("SAMPLER_POSITION_OUT_OF_RANGE")
        item = tokenize_record(train_rows[row_index], "native", processor, opt["max_sequence_length"])
        batch = pad_batch([item], processor.tokenizer.pad_token_id); values = {key: torch.tensor(value, device="cuda") for key, value in batch.items()}
        loss = model(**values).loss
        if not torch.isfinite(loss): raise FloatingPointError("NON_FINITE_TRAINING_LOSS")
        divisor = accumulation_target(total_samples, sample_index, opt["gradient_accumulation_steps"])
        (loss / divisor).backward(); recent.append(float(loss.item())); sample_index += 1
        batch_complete = sample_index % opt["gradient_accumulation_steps"] == 0 or sample_index == total_samples
        if not batch_complete: continue
        gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad and parameter.grad is not None]
        if not gradients or any(not torch.isfinite(gradient).all() for gradient in gradients): raise FloatingPointError("NON_FINITE_OR_MISSING_GRADIENT")
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt["max_grad_norm"]); optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); step += 1
        if step == 1 and not resumed_adapter_already_updated and torch.equal(initial_parameter, first_parameter.detach().float().cpu()):
            raise RuntimeError("ADAPTER_PARAMETERS_NOT_UPDATING")
        expected_step = optimizer_step_count(sample_index, 1, opt["gradient_accumulation_steps"])
        if step != expected_step:
            raise RuntimeError("PROGRESS_STATE_INCONSISTENCY")
        due = step == max_steps if args.mode == "smoke" else step in validation_schedule
        if due:
            loss_count = config["smoke"]["validation_loss_records"] if args.mode == "smoke" else config["monitoring"]["validation_loss_records"]
            gen_count = config["smoke"]["generation_records"] if args.mode == "smoke" else config["monitoring"]["generation_records"]
            latest_validation = evaluate(model, processor, validation_rows, loss_count, gen_count, opt["max_sequence_length"], config["monitoring"]["max_new_tokens"])
            atomic_json(args.run_dir / f"validation/step-{step:08d}.json", latest_validation)
            alerts = validation_regression_alerts(config, baseline_validation, latest_validation, best_loss)
            if alerts:
                atomic_json(args.run_dir / f"validation/step-{step:08d}-alerts.json", {"step": step, "alerts": alerts, "certification_thresholds_modified": False})
            if latest_validation["validation_loss"] < best_loss:
                best_loss = latest_validation["validation_loss"]
                best_path = save_checkpoint(args.run_dir, model, optimizer, scheduler, step, sample_index, digest, latest_validation,
                                            kind="best", keep_latest=config["checkpoint_policy"]["keep_latest"])
        elapsed = time.monotonic() - started; rate = step / elapsed if elapsed else 0.0
        atomic_json(args.run_dir / "progress.json", {"status": "running", "mode": args.mode, "step": step, "total_steps": max_steps,
                    "percent": step / max_steps * 100, "elapsed_seconds": elapsed, "eta_seconds": (max_steps - step) / rate if rate else None,
                    "current_loss": recent[-1], "smoothed_loss": sum(recent) / len(recent), "validation": {k: v for k, v in latest_validation.items() if k != "raw"},
                    "latest_checkpoint": str(best_path) if best_path else None, "gpu": {"allocated_bytes": torch.cuda.memory_allocated(), "peak_bytes": torch.cuda.max_memory_allocated()},
                     "epoch": epoch, "epoch_position": position + 1, "samples_completed": sample_index, "samples_total": total_samples,
                     "adapter_parameters_updating": resumed_adapter_already_updated or not torch.equal(initial_parameter, first_parameter.detach().float().cpu())})
        if args.mode == "full" and step in checkpoint_schedule:
            save_checkpoint(args.run_dir, model, optimizer, scheduler, step, sample_index, digest, latest_validation,
                            kind="periodic", keep_latest=config["checkpoint_policy"]["keep_latest"])
    if best_path is None:
        latest_validation = evaluate(model, processor, validation_rows, config["smoke"]["validation_loss_records"], config["smoke"]["generation_records"], opt["max_sequence_length"], config["monitoring"]["max_new_tokens"])
        best_loss = latest_validation["validation_loss"]
        best_path = save_checkpoint(args.run_dir, model, optimizer, scheduler, step, sample_index, digest, latest_validation,
                                    kind="best", keep_latest=config["checkpoint_policy"]["keep_latest"])
    completed = step == max_steps and sample_index == total_samples
    final_path = None
    if completed:
        final_path = save_checkpoint(args.run_dir, model, optimizer, scheduler, step, sample_index, digest, latest_validation,
                                     kind="final", keep_latest=config["checkpoint_policy"]["keep_latest"])
    adapter_changed = resumed_adapter_already_updated or not torch.equal(initial_parameter, first_parameter.detach().float().cpu())
    finite_loss = all(math.isfinite(value) for value in recent)
    smoke_failures = smoke_gate_failures(config, latest_validation, step, adapter_changed, finite_loss, True)
    smoke_pass = not smoke_failures
    successful = completed and adapter_changed and finite_loss and (args.mode == "full" or smoke_pass)
    result = {"status": ("COMPLETE" if args.mode == "full" else "PASS") if successful else "FAIL", "mode": args.mode, "step": step,
              "total_steps": max_steps, "samples_completed": sample_index, "samples_total": total_samples,
              "best_checkpoint": str(best_path), "final_checkpoint": str(final_path) if final_path else None,
              "validation": latest_validation, "finite_loss": all(math.isfinite(value) for value in recent),
              "finite_gradients": True, "adapter_parameters_updating": adapter_changed, "smoke_gate_failures": smoke_failures if args.mode == "smoke" else [],
              "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    atomic_json(args.run_dir / "result.json", result); atomic_json(args.run_dir / "progress.json", {**result, "validation": {k: v for k, v in latest_validation.items() if k != "raw"}})
    return 0 if successful else 4


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    finally:
        if "--run-dir" in sys.argv:
            index = sys.argv.index("--run-dir") + 1
            if index < len(sys.argv):
                atomic_text(Path(sys.argv[index]) / "exit-code.txt", f"{exit_code}\n")
    raise SystemExit(exit_code)
