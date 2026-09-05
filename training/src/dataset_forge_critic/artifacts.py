"""Adapter export manifests and explicit, retryable artifact synchronization."""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from .checkpointing import atomic_json
from .integrity import sha256_file
from .training_config import TrainingConfiguration


def export_adapter(checkpoint: Path, destination: Path, config: TrainingConfiguration) -> dict[str, Any]:
    source = checkpoint / "adapter"
    if not source.is_dir() or not (checkpoint / "checkpoint-manifest.json").is_file(): raise ValueError("INVALID_CHECKPOINT")
    if destination.exists(): raise ValueError(f"EXPORT_DESTINATION_EXISTS:{destination}")
    shutil.copytree(source, destination / "adapter")
    manifest_source = json.loads((checkpoint / "checkpoint-manifest.json").read_text(encoding="utf-8"))
    files = []
    for path in sorted(item for item in destination.rglob("*") if item.is_file()):
        files.append({"path": path.relative_to(destination).as_posix(), "size": path.stat().st_size, "sha256": sha256_file(path)})
    manifest = {"format": "dataset-forge-adapter-export-v1", "adapter_only": True, "standalone_model": False,
                "base_model_id": config.model.model_id, "base_revision": config.model.revision, "config_digest": config.digest(),
                "checkpoint_step": manifest_source["step"], "source_data_hashes": manifest_source["compatibility"]["data_hashes"],
                "files": files, "reload_verification": "GPU_PENDING"}
    atomic_json(destination / "export-manifest.json", manifest); return manifest


def sync_artifacts(source: Path, destination: str, *, visibility: str | None, retries: int) -> dict[str, Any]:
    if not source.is_dir(): raise ValueError("SYNC_SOURCE_MISSING")
    if destination.startswith("hf://"):
        if visibility not in {"private", "public"}: raise ValueError("HF_VISIBILITY_REQUIRED")
        repo = destination.removeprefix("hf://")
        command = ["hf", "upload-large-folder", repo, str(source), "--repo-type", "model"]
        last = ""
        for attempt in range(retries + 1):
            result = subprocess.run(command, text=True, capture_output=True)
            if result.returncode == 0: return {"status": "REMOTE_SYNC_COMPLETE", "destination": destination, "attempts": attempt + 1}
            last = (result.stderr or result.stdout)[-2000:]
            if attempt < retries: time.sleep(min(2 ** attempt, 30))
        raise ValueError(f"REMOTE_SYNC_FAILED:{last}")
    target = Path(destination).resolve()
    if target.exists(): raise ValueError(f"SYNC_DESTINATION_EXISTS:{target}")
    shutil.copytree(source, target)
    hashes = {path.relative_to(target).as_posix(): sha256_file(path) for path in target.rglob("*") if path.is_file()}
    source_hashes = {path.relative_to(source).as_posix(): sha256_file(path) for path in source.rglob("*") if path.is_file()}
    if hashes != source_hashes: raise ValueError("SYNC_VERIFY_FAILED")
    return {"status": "LOCAL_SYNC_VERIFIED", "destination": str(target), "files": len(hashes)}
