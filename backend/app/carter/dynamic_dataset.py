"""Canonical dynamic records for Carter-backed datasets.

Legacy ``DatasetRecord`` remains intentionally separate: no generic projection
exists because DatasetSpec owns the semantic fields of a Carter record.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .runtime import CarterPromptPackage, CarterPromptPackageError


@dataclass(frozen=True)
class CarterCanonicalDataset:
    specification: dict[str, Any]
    records: tuple[dict[str, Any], ...]
    compiled_schema: dict[str, Any]

    @property
    def field_order(self) -> tuple[str, ...]:
        return tuple(field["name"] for field in self.specification["fields"]) + ("evidence",)


def validate_canonical_dataset(package: CarterPromptPackage, specification: dict[str, Any], records: list[dict[str, Any]], *, allowed_source_refs: set[str], batch_count: int | None = None) -> CarterCanonicalDataset:
    schema = package.compile_generation_schema(specification, batch_count or len(records))
    result = {"status": "generated", "records": records, "insufficiency": None}
    package.validate(schema, result)
    for record in records:
        for evidence in record["evidence"]:
            if evidence["source_ref"] not in allowed_source_refs:
                raise CarterPromptPackageError("Fabricated or unauthorized evidence source reference.")
    return CarterCanonicalDataset(specification, tuple(records), schema)


def export_canonical_json(dataset: CarterCanonicalDataset, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {"dataset_spec": dataset.specification, "records": list(dataset.records)}
    destination.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return destination


def _csv_cell(value: Any) -> str:
    if value is None: return ""
    if isinstance(value, str): return value
    if isinstance(value, bool): return "true" if value else "false"
    if isinstance(value, (int, float)): return str(value)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def export_canonical_csv(dataset: CarterCanonicalDataset, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = dataset.field_order
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for record in dataset.records:
            writer.writerow({field: _csv_cell(record[field]) for field in fields})
    return destination
