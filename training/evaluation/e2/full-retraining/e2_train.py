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
from pathlib import Path
from typing import Any

TRAINING_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(TRAINING_ROOT / "src"))

from dataset_forge_critic.e2_contract import parse_strict_response  # noqa: E402
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


def rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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
        parse_strict_response(json.dumps(item["canonical_record"]["target"], ensure_ascii=False, separators=(",", ":")))
    return train, validation


def training_configuration(config: dict[str, Any], data_dir: Path, output_root: Path, mode: str) -> TrainingConfiguration:
    opt, model, monitoring = config["optimization"], config["model"], config["monitoring"]
    train, validation = data_dir / "train/corpus.jsonl", data_dir / "validation/corpus.jsonl"
    max_steps = config["smoke"]["optimizer_steps"] if mode == "smoke" else opt["max_optimizer_steps"]
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
        "limits": {"max_optimizer_steps": max_steps, "max_examples": max_steps * opt["effective_batch_size"],
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
        truths: list[str] = []; predictions: list[str | None] = []; raw = []
        terminators = generation_terminator_ids(processor)
        for row in stratified(items, min(generation_count, len(items)), 7302):
            item = tokenize_record(row, "native", processor, max_length)
            prompt = item.input_ids[:item.prompt_tokens]
            output = model.generate(input_ids=torch.tensor([prompt], device="cuda"), attention_mask=torch.ones((1, len(prompt)), dtype=torch.long, device="cuda"),
                                    do_sample=False, max_new_tokens=max_new_tokens, eos_token_id=terminators, pad_token_id=processor.tokenizer.pad_token_id)
            generated = output[0][len(prompt):].tolist(); text = processor.decode(generated, skip_special_tokens=True)
            truth = row["canonical_record"]["target"]["decision"]; truths.append(truth)
            try: parsed = parse_strict_response(text); prediction = parsed["decision"]
            except ValueError: prediction = None
            predictions.append(prediction)
            remediation = row.get("remediation", {})
            raw.append({"record_id": row["canonical_record"]["record_id"], "truth": truth, "prediction": prediction, "text": text,
                        "dataset_type": row["canonical_record"]["input"]["dataset_spec"]["dataset_type"], "semantic_group": remediation.get("semantic_group"),
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
    return {"validation_loss": sum(losses) / len(losses), "generation_records": len(raw), "strict_json_validity": valid / len(raw),
            "parse_failure_rate": 1 - valid / len(raw), "accuracy": sum(a == b for a, b in zip(truths, predictions)) / len(raw),
            "macro_f1": macro_f1(truths, predictions), "immediate_termination_rate": sum(row["terminated"] and not row["hit_token_limit"] for row in raw) / len(raw),
            "grounding_false_accept_rate": sum(row["prediction"] == "accept" for row in reject_rows) / len(reject_rows) if reject_rows else 0.0,
            "malicious_injection_failure_rate": sum(row["prediction"] != "reject" for row in malicious_rows) / len(malicious_rows) if malicious_rows else 0.0,
            "per_dataset_type": per_type,
            "markdown_fences": sum(row["markdown_fence"] for row in raw), "external_prose": len(raw) - valid,
            "token_limit_hits": sum(row["hit_token_limit"] for row in raw), "raw": raw}


def save_checkpoint(run_dir: Path, model: Any, optimizer: Any, scheduler: Any, step: int, sample_index: int, config_digest: str, metrics: dict[str, Any], best: bool = False) -> Path:
    import torch
    destination = run_dir / ("best-checkpoint" if best else f"checkpoints/step-{step:08d}")
    stage = destination.with_name(destination.name + ".tmp")
    if stage.exists(): shutil.rmtree(stage)
    stage.mkdir(parents=True); model.save_pretrained(stage / "adapter", safe_serialization=True)
    torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "step": step, "sample_index": sample_index,
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}, stage / "training-state.pt")
    atomic_json(stage / "checkpoint-manifest.json", {"complete": True, "step": step, "sample_index": sample_index, "config_digest": config_digest, "metrics": metrics})
    if destination.exists(): shutil.rmtree(destination)
    os.replace(stage, destination)
    atomic_json(run_dir / "latest-checkpoint.json", {"path": str(destination), "step": step})
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--config", type=Path, required=True); parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True); parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--git-sha", required=True); parser.add_argument("--resume", type=Path)
    args = parser.parse_args(); config = json.loads(args.config.read_text(encoding="utf-8")); train_path, validation_path = verify_inputs(config, args.data_dir)
    args.run_dir.mkdir(parents=True, exist_ok=True); effective = training_configuration(config, args.data_dir, args.run_dir.parent, args.mode)
    digest = hashlib.sha256(json.dumps({"config": config, "mode": args.mode}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    os.environ["DATASET_FORGE_MODEL_PATH"] = str(args.model_path.resolve())
    atomic_json(args.run_dir / "run-manifest.json", {"config": config, "effective_training_configuration": effective.canonical_dict(), "mode": args.mode,
                "config_digest": digest, "git_sha": args.git_sha, "train_sha256": sha256(train_path), "validation_sha256": sha256(validation_path),
                "fresh_base_adapter": args.resume is None, "e1_adapter_used": False,
                "sampling": "deterministic_epoch_shuffle_without_replacement", "legacy_mixture_fields_used_by_runner": False,
                "checkpoint_selection": ["strict_json_validity", "accuracy", "macro_f1", "grounding_false_accept_rate", "malicious_injection_failure_rate", "parse_failure_rate", "validation_loss"],
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    import torch
    from transformers import get_scheduler, set_seed
    set_seed(config["optimization"]["seed"]); processor = load_processor(effective)
    model, parameter_report = load_qlora_model(effective, resume_adapter=args.resume / "adapter" if args.resume else None)
    terminators = generation_terminator_ids(processor)
    if terminators != [1, 106]: raise ValueError(f"TERMINATOR_IDENTITY_MISMATCH:{terminators}")
    train_rows, validation_rows = rows(train_path), rows(validation_path)
    if args.mode == "smoke": train_rows = stratified(train_rows, config["smoke"]["training_records"], 7001)
    opt = config["optimization"]; max_steps = config["smoke"]["optimizer_steps"] if args.mode == "smoke" else opt["max_optimizer_steps"]
    optimizer = __import__("bitsandbytes").optim.PagedAdamW8bit(model.parameters(), lr=opt["learning_rate"], weight_decay=opt["weight_decay"])
    scheduler = get_scheduler(opt["scheduler"], optimizer, int(max_steps * opt["warmup_ratio"]), max_steps)
    step = sample_index = 0
    if args.resume:
        state = torch.load(args.resume / "training-state.pt", map_location="cpu", weights_only=False)
        manifest = json.loads((args.resume / "checkpoint-manifest.json").read_text())
        if manifest["config_digest"] != digest: raise ValueError("RESUME_CONFIG_MISMATCH")
        optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"]); step = state["step"]; sample_index = state["sample_index"]
        torch.set_rng_state(state["torch_rng"]); torch.cuda.set_rng_state_all(state["cuda_rng"])
    first_parameter = next(parameter for parameter in model.parameters() if parameter.requires_grad)
    initial_parameter = first_parameter.detach().float().cpu().clone(); model.train(); optimizer.zero_grad(set_to_none=True)
    started = time.monotonic(); recent = deque(maxlen=50); best_score = None; best_path = None; latest_validation: dict[str, Any] = {}
    def request_stop(*_: Any) -> None:
        global STOP; STOP = True
    signal.signal(signal.SIGTERM, request_stop); signal.signal(signal.SIGINT, request_stop)
    while step < max_steps and not STOP:
        if time.monotonic() - started >= config["monitoring"]["max_wall_seconds"]: break
        if shutil.disk_usage(args.run_dir).free / 1024**3 < config["monitoring"]["min_free_disk_gb"]: raise RuntimeError("DISK_SAFETY_THRESHOLD")
        epoch = sample_index // len(train_rows); position = sample_index % len(train_rows); order = list(range(len(train_rows))); random.Random(opt["seed"] + epoch).shuffle(order)
        item = tokenize_record(train_rows[order[position]], "native", processor, opt["max_sequence_length"])
        batch = pad_batch([item], processor.tokenizer.pad_token_id); values = {key: torch.tensor(value, device="cuda") for key, value in batch.items()}
        loss = model(**values).loss
        if not torch.isfinite(loss): raise FloatingPointError("NON_FINITE_TRAINING_LOSS")
        (loss / opt["gradient_accumulation_steps"]).backward(); recent.append(float(loss.item())); sample_index += 1
        if sample_index % opt["gradient_accumulation_steps"]: continue
        gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad and parameter.grad is not None]
        if not gradients or any(not torch.isfinite(gradient).all() for gradient in gradients): raise FloatingPointError("NON_FINITE_OR_MISSING_GRADIENT")
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt["max_grad_norm"]); optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); step += 1
        due = step == max_steps or (args.mode == "full" and step % config["monitoring"]["validation_every_steps"] == 0)
        if due:
            loss_count = config["smoke"]["validation_loss_records"] if args.mode == "smoke" else config["monitoring"]["validation_loss_records"]
            gen_count = config["smoke"]["generation_records"] if args.mode == "smoke" else config["monitoring"]["generation_records"]
            latest_validation = evaluate(model, processor, validation_rows, loss_count, gen_count, opt["max_sequence_length"], config["monitoring"]["max_new_tokens"])
            atomic_json(args.run_dir / f"validation/step-{step:08d}.json", latest_validation)
            score = (latest_validation["strict_json_validity"], latest_validation["accuracy"], latest_validation["macro_f1"],
                     -latest_validation["grounding_false_accept_rate"], -latest_validation["malicious_injection_failure_rate"],
                     -latest_validation["parse_failure_rate"], -latest_validation["validation_loss"])
            if best_score is None or score > best_score:
                best_score = score; best_path = save_checkpoint(args.run_dir, model, optimizer, scheduler, step, sample_index, digest, latest_validation, best=True)
        elapsed = time.monotonic() - started; rate = step / elapsed if elapsed else 0.0
        atomic_json(args.run_dir / "progress.json", {"status": "running", "mode": args.mode, "step": step, "total_steps": max_steps,
                    "percent": step / max_steps * 100, "elapsed_seconds": elapsed, "eta_seconds": (max_steps - step) / rate if rate else None,
                    "current_loss": recent[-1], "smoothed_loss": sum(recent) / len(recent), "validation": {k: v for k, v in latest_validation.items() if k != "raw"},
                    "latest_checkpoint": str(best_path) if best_path else None, "gpu": {"allocated_bytes": torch.cuda.memory_allocated(), "peak_bytes": torch.cuda.max_memory_allocated()},
                    "adapter_parameters_updating": not torch.equal(initial_parameter, first_parameter.detach().float().cpu())})
        if args.mode == "full" and step % config["monitoring"]["checkpoint_every_steps"] == 0:
            save_checkpoint(args.run_dir, model, optimizer, scheduler, step, sample_index, digest, latest_validation)
    if best_path is None:
        latest_validation = evaluate(model, processor, validation_rows, config["smoke"]["validation_loss_records"], config["smoke"]["generation_records"], opt["max_sequence_length"], config["monitoring"]["max_new_tokens"])
        best_path = save_checkpoint(args.run_dir, model, optimizer, scheduler, step, sample_index, digest, latest_validation, best=True)
    smoke_pass = all((math.isfinite(latest_validation["validation_loss"]), latest_validation["strict_json_validity"] == 1.0,
                      latest_validation["immediate_termination_rate"] == 1.0, latest_validation["markdown_fences"] == 0,
                      latest_validation["external_prose"] == 0, latest_validation["token_limit_hits"] == 0,
                      not torch.equal(initial_parameter, first_parameter.detach().float().cpu())))
    result = {"status": "PASS" if (args.mode == "full" or smoke_pass) and step == max_steps else "FAIL", "mode": args.mode, "step": step,
              "total_steps": max_steps, "best_checkpoint": str(best_path), "validation": latest_validation, "finite_loss": all(math.isfinite(value) for value in recent),
              "finite_gradients": True, "adapter_parameters_updating": not torch.equal(initial_parameter, first_parameter.detach().float().cpu()),
              "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    atomic_json(args.run_dir / "result.json", result); atomic_json(args.run_dir / "progress.json", {**result, "validation": {k: v for k, v in latest_validation.items() if k != "raw"}})
    return 0 if result["status"] == "PASS" else 4


if __name__ == "__main__":
    raise SystemExit(main())
