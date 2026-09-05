"""Direct, resumable QLoRA optimizer loop for Dataset Forge Critic."""
from __future__ import annotations

import json
import os
import random
import shutil
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .checkpointing import atomic_json, compatibility_identity, validate_resume
from .integrity import sha256_file
from .mixture import MixedRecordStream
from .modeling import load_processor, load_qlora_model
from .runtime import RunLock
from .tokenization import TokenizationExclusion, pad_batch, tokenize_record
from .training_config import TrainingConfiguration


def data_hashes(config: TrainingConfiguration) -> dict[str, str]:
    return {key: sha256_file(Path(value)) for key, value in config.data.model_dump().items()}


def run_directory(config: TrainingConfiguration) -> Path:
    return config.output_root / f"{config.profile}-{config.digest()[:12]}"


def _save_checkpoint(run_dir: Path, model: Any, optimizer: Any, scheduler: Any, stream: MixedRecordStream,
                     config: TrainingConfiguration, step: int, counters: dict[str, Any], *, best: bool = False) -> Path:
    import torch
    destination = run_dir / "checkpoints" / f"step-{step:08d}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_dir() and (destination / "checkpoint-manifest.json").is_file():
        return destination
    stage = Path(tempfile.mkdtemp(prefix="checkpoint-", dir=destination.parent))
    try:
        model.save_pretrained(stage / "adapter", safe_serialization=True)
        torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all(), "python_rng": random.getstate()}, stage / "training-state.pt")
        identity = compatibility_identity(config.digest(), config.model.model_id, config.model.revision, data_hashes(config))
        atomic_json(stage / "checkpoint-manifest.json", {"format": "dataset-forge-checkpoint-v1", "complete": True, "step": step,
                   "compatibility": identity, "sampler_state": stream.state_dict(), "counters": counters, "best": best})
        os.replace(stage, destination)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True); raise
    atomic_json(run_dir / "latest-checkpoint.json", {"path": str(destination), "step": step})
    checkpoints = sorted((run_dir / "checkpoints").glob("step-*"))
    for expired in checkpoints[:-config.checkpoint.keep_latest]:
        shutil.rmtree(expired)
    return destination


def train(config: TrainingConfiguration, *, confirmation: str, resume: Path | None = None) -> dict[str, Any]:
    if config.profile == "cpu_synthetic": raise ValueError("USE_SYNTHETIC_INTEGRATION_COMMAND")
    if confirmation != config.digest(): raise ValueError(f"CONFIRMATION_REQUIRED:{config.digest()}")
    import torch
    from transformers import get_scheduler, set_seed
    from torch.utils.tensorboard import SummaryWriter
    run_dir = run_directory(config); lock = RunLock(run_dir, config.digest()); lock.acquire()
    try:
        free = shutil.disk_usage(run_dir).free / 1024 ** 3
        if free < config.limits.min_free_disk_gb: raise ValueError(f"INSUFFICIENT_DISK:{free:.2f}GB")
        set_seed(config.seed); processor = load_processor(config)
        processor.save_pretrained(run_dir / "processor")
        atomic_json(run_dir / "run-manifest.json", {"config": config.canonical_dict(), "config_digest": config.digest(),
                    "model_id": config.model.model_id, "revision": config.model.revision, "data_hashes": data_hashes(config)})
        writer = SummaryWriter(log_dir=run_dir / "tensorboard")
        resume_adapter = resume / "adapter" if resume else None
        model, parameter_report = load_qlora_model(config, resume_adapter=resume_adapter)
        optimizer = (torch.optim.AdamW(model.parameters(), lr=config.optimization.learning_rate, weight_decay=config.optimization.weight_decay)
                     if config.optimization.optimizer == "adamw_torch" else __import__("bitsandbytes").optim.PagedAdamW8bit(model.parameters(), lr=config.optimization.learning_rate, weight_decay=config.optimization.weight_decay))
        warmup = int(config.limits.max_optimizer_steps * config.optimization.warmup_ratio)
        scheduler = get_scheduler(config.optimization.scheduler, optimizer, warmup, config.limits.max_optimizer_steps)
        stream = MixedRecordStream(config.data.public_train, config.data.native_train, config.mixture.public_probability, config.seed)
        step = examples = total_tokens = supervised_tokens = 0; exclusions = Counter(); started = time.monotonic()
        compatibility = compatibility_identity(config.digest(), config.model.model_id, config.model.revision, data_hashes(config))
        if resume:
            manifest = json.loads((resume / "checkpoint-manifest.json").read_text(encoding="utf-8")); validate_resume(manifest, compatibility)
            state = torch.load(resume / "training-state.pt", map_location="cpu", weights_only=False)
            optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"]); torch.set_rng_state(state["torch_rng"]); torch.cuda.set_rng_state_all(state["cuda_rng"]); random.setstate(state["python_rng"])
            stream.load_state_dict(manifest["sampler_state"]); step = manifest["step"]
            counters = manifest["counters"]; examples = counters["examples"]; total_tokens = counters["total_tokens"]; supervised_tokens = counters["supervised_tokens"]
        model.train(); optimizer.zero_grad(set_to_none=True)
        while step < config.limits.max_optimizer_steps:
            if (run_dir / "STOP_REQUESTED").exists(): break
            if config.limits.max_wall_seconds and time.monotonic() - started >= config.limits.max_wall_seconds: break
            batch_items = []
            while len(batch_items) < config.optimization.micro_batch_size:
                corpus, record = stream.next()
                try: batch_items.append(tokenize_record(record, corpus, processor, config.max_sequence_length))
                except TokenizationExclusion as exc:
                    exclusions[str(exc).split(":", 1)[0]] += 1
                    if sum(exclusions.values()) > max(1000, examples * 10 + 100): raise ValueError("EXCESSIVE_TOKENIZATION_EXCLUSIONS")
            pad_id = processor.tokenizer.pad_token_id
            if pad_id is None: raise ValueError("TOKENIZER_PAD_TOKEN_REQUIRED")
            raw = pad_batch(batch_items, pad_id); batch = {key: torch.tensor(value, device="cuda") for key, value in raw.items()}
            loss = model(**batch).loss / config.optimization.gradient_accumulation_steps; loss.backward()
            examples += len(batch_items); total_tokens += sum(len(item.input_ids) for item in batch_items); supervised_tokens += sum(item.supervised_tokens for item in batch_items)
            if examples % (config.optimization.micro_batch_size * config.optimization.gradient_accumulation_steps) == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimization.max_grad_norm); optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); step += 1
                elapsed = max(time.monotonic() - started, 1e-6)
                current = {"step": step, "examples": examples, "total_tokens": total_tokens, "supervised_tokens": supervised_tokens,
                           "loss": float(loss.item() * config.optimization.gradient_accumulation_steps), "learning_rate": scheduler.get_last_lr()[0],
                           "tokens_per_second": total_tokens / elapsed, "supervised_tokens_per_second": supervised_tokens / elapsed,
                           "gpu_memory_current": torch.cuda.memory_allocated(), "gpu_memory_peak": torch.cuda.max_memory_allocated(),
                           "elapsed_seconds": elapsed, "exclusions": dict(exclusions), "sampler": stream.state_dict(), "parameter_report": parameter_report}
                atomic_json(run_dir / "status.json", current)
                writer.add_scalar("train/loss", current["loss"], step)
                writer.add_scalar("train/learning_rate", current["learning_rate"], step)
                writer.add_scalar("throughput/tokens_per_second", current["tokens_per_second"], step)
                writer.add_scalar("throughput/supervised_tokens_per_second", current["supervised_tokens_per_second"], step)
                writer.add_scalar("hardware/gpu_memory_current_bytes", current["gpu_memory_current"], step)
                writer.add_scalar("hardware/gpu_memory_peak_bytes", current["gpu_memory_peak"], step)
                writer.flush()
                if step % config.checkpoint.every_steps == 0: _save_checkpoint(run_dir, model, optimizer, scheduler, stream, config, step, current)
            if config.limits.max_examples and examples >= config.limits.max_examples: break
            if config.limits.max_total_tokens and total_tokens >= config.limits.max_total_tokens: break
        counters = {"examples": examples, "total_tokens": total_tokens, "supervised_tokens": supervised_tokens}
        checkpoint = _save_checkpoint(run_dir, model, optimizer, scheduler, stream, config, step, counters)
        atomic_json(run_dir / "result.json", {"status": "GRACEFULLY_STOPPED" if (run_dir / "STOP_REQUESTED").exists() else "TRAINING_BUDGET_COMPLETE", "checkpoint": str(checkpoint), **counters})
        writer.close()
        return json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    finally:
        lock.release()
