"""Contract for deterministic, source-specific canonical record transformations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..schemas import CanonicalCriticRecord, TaskFamily


class DatasetAdapter(ABC):
    """Adapters validate raw records, preserve provenance, and emit deterministic output."""

    dataset_name: str
    task_family: TaskFamily
    adapter_version: str
    transform_version: str

    @abstractmethod
    def validate_raw_record(self, raw_record: dict[str, Any]) -> None:
        """Raise ValueError with an actionable reason if the source record is unusable."""

    @abstractmethod
    def transform(self, raw_record: dict[str, Any]) -> list[CanonicalCriticRecord]:
        """Transform one raw record without I/O, randomness, or fabricated labels."""

    def transform_validated(self, raw_record: dict[str, Any]) -> list[CanonicalCriticRecord]:
        self.validate_raw_record(raw_record)
        records = self.transform(raw_record)
        if not records:
            raise ValueError("Adapter transform must emit at least one canonical record")
        for record in records:
            if record.provenance.source_dataset != self.dataset_name:
                raise ValueError("Adapter output provenance.source_dataset must match dataset_name")
            if record.provenance.adapter_version != self.adapter_version:
                raise ValueError("Adapter output provenance.adapter_version must match adapter_version")
            if record.provenance.transform_version != self.transform_version:
                raise ValueError("Adapter output provenance.transform_version must match transform_version")
            if record.task_family != self.task_family:
                raise ValueError("Adapter output task_family must match adapter task_family")
        return records
