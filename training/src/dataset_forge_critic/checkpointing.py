"""Atomic checkpoint metadata and compatibility validation."""
from __future__ import annotations

import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", delete=False, dir=path.parent) as staged:
        staged.write(json.dumps(value, sort_keys=True, indent=2) + "\n")
        staged_path = Path(staged.name)
    os.replace(staged_path, path)


def compatibility_identity(config_digest: str, model_id: str, revision: str, data_hashes: dict[str, str]) -> dict[str, Any]:
    return {"config_digest": config_digest, "model_id": model_id, "revision": revision, "data_hashes": data_hashes}


def validate_resume(checkpoint_manifest: dict[str, Any], expected: dict[str, Any]) -> None:
    actual = checkpoint_manifest.get("compatibility")
    if actual != expected:
        raise ValueError("CHECKPOINT_INCOMPATIBLE")


def save_synthetic_checkpoint(path: Path, *, step: int, compatibility: dict[str, Any], sampler_state: dict[str, Any], values: list[float]) -> None:
    """Dependency-light checkpoint used to test interruption semantics on CPU."""
    atomic_json(path, {"format": "dataset-forge-synthetic-v1", "step": step, "compatibility": compatibility,
                       "sampler_state": sampler_state, "rng_state_repr": repr(random.getstate()), "values": values})


def load_synthetic_checkpoint(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8")); validate_resume(value, expected); return value
