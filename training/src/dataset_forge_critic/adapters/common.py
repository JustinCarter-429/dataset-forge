from __future__ import annotations

from typing import Any

from ..provenance import stable_record_id
from ..schemas import CanonicalCriticRecord, CriticInput, CriticTarget, NormalizedScore, Provenance, ScoreProvenance, Supervision, TaskFamily


def score(field: str, value: int | float, scale: str, meaning: str) -> NormalizedScore:
    return NormalizedScore(value=float(value), provenance=ScoreProvenance(original_field_name=field, original_value=value, original_scale=scale, normalization_method=f"source_preserved:{meaning}", normalization_version="1.0.0"))


def record(*, dataset: str, split: str, source_id: str, snapshot_id: str, adapter: str, task_family: TaskFamily, input: CriticInput, target: CriticTarget | None = None, available: list[str] | None = None, license: str | None = None, candidate_index: int | None = None) -> CanonicalCriticRecord:
    version = "1.0.1"
    return CanonicalCriticRecord(record_id=stable_record_id(dataset, split, f"{snapshot_id}:{source_id}", version, candidate_index), task_family=task_family, input=input, target=target or CriticTarget(), supervision=Supervision(available_targets=available or []), provenance=Provenance(source_dataset=dataset, source_split=split, source_record_id=source_id, license=license, adapter_name=adapter, adapter_version=version, transform_version=version, source_fingerprint=snapshot_id))
