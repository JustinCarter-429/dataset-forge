"""Deterministic replacement sampler with serializable data-position state."""
from __future__ import annotations

import base64
import json
import pickle
import random
from collections import Counter
from pathlib import Path
from typing import Any


def jsonl_offsets(path: Path) -> list[int]:
    offsets: list[int] = []
    with path.open("rb") as handle:
        while True:
            position = handle.tell()
            line = handle.readline()
            if not line:
                break
            if line.strip():
                offsets.append(position)
    if not offsets:
        raise ValueError(f"EMPTY_DATASET: {path}")
    return offsets


class MixedRecordStream:
    def __init__(self, public_path: Path, native_path: Path, public_probability: float, seed: int):
        self.paths = {"public": public_path, "native": native_path}
        self.offsets = {name: jsonl_offsets(path) for name, path in self.paths.items()}
        self.public_probability = public_probability
        self.rng = random.Random(seed)
        self.draws: Counter[str] = Counter()
        self.unique_indices: dict[str, set[int]] = {"public": set(), "native": set()}

    def next(self) -> tuple[str, dict[str, Any]]:
        corpus = "public" if self.rng.random() < self.public_probability else "native"
        index = self.rng.randrange(len(self.offsets[corpus]))
        with self.paths[corpus].open("rb") as handle:
            handle.seek(self.offsets[corpus][index])
            payload = json.loads(handle.readline().decode("utf-8"))
        self.draws[corpus] += 1
        self.unique_indices[corpus].add(index)
        return corpus, payload

    def state_dict(self) -> dict[str, Any]:
        rng = base64.b64encode(pickle.dumps(self.rng.getstate())).decode("ascii")
        return {
            "rng_state": rng,
            "draws": dict(self.draws),
            "unique_indices": {key: sorted(value) for key, value in self.unique_indices.items()},
            "source_counts": {key: len(value) for key, value in self.offsets.items()},
            "epoch_definition": "optimizer-budget stream with replacement; exposure=draws/source_count",
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = {key: len(value) for key, value in self.offsets.items()}
        if state.get("source_counts") != expected:
            raise ValueError("SAMPLER_SOURCE_COUNT_MISMATCH")
        self.rng.setstate(pickle.loads(base64.b64decode(state["rng_state"])))
        self.draws = Counter(state["draws"])
        self.unique_indices = {key: set(value) for key, value in state["unique_indices"].items()}
