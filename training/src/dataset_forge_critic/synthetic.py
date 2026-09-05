"""Dependency-light deterministic training integration used only for local tests."""
from __future__ import annotations

from pathlib import Path

from .checkpointing import compatibility_identity, load_synthetic_checkpoint, save_synthetic_checkpoint
from .mixture import MixedRecordStream


def run_synthetic(public: Path, native: Path, checkpoint: Path, *, steps: int, seed: int = 42, resume: bool = False) -> dict:
    identity = compatibility_identity("synthetic-config-v1", "synthetic-linear", "v1", {"public": str(public), "native": str(native)})
    stream = MixedRecordStream(public, native, .5, seed); value = 0.0; start = 0; trajectory: list[float] = []
    if resume:
        state = load_synthetic_checkpoint(checkpoint, identity); start = state["step"]; trajectory = state["values"]; value = trajectory[-1] if trajectory else 0.0; stream.load_state_dict(state["sampler_state"])
    for _ in range(start, steps):
        corpus, row = stream.next(); target = float(row.get("synthetic_target", corpus == "native")); prediction = value
        value -= .1 * 2 * (prediction - target); trajectory.append(value)
    save_synthetic_checkpoint(checkpoint, step=steps, compatibility=identity, sampler_state=stream.state_dict(), values=trajectory)
    return {"status": "CPU_SYNTHETIC_COMPLETE", "step": steps, "value": value, "trajectory": trajectory, "sampler": stream.state_dict()}
