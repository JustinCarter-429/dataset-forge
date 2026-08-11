"""Canonical dynamic records for Carter-backed datasets.

Legacy ``DatasetRecord`` remains intentionally separate: no generic projection
exists because DatasetSpec owns the semantic fields of a Carter record.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
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


@dataclass(frozen=True)
class CarterQualityFinding:
    code: str
    severity: str
    record_id: str
    message: str
    validator: str


@dataclass(frozen=True)
class CarterQualityReport:
    total_records: int
    accepted_records: int
    quarantined_records: int
    rejected_records: int
    schema_failures: int
    grounding_failures: int
    duplicate_records: int
    security_findings: int
    findings: tuple[CarterQualityFinding, ...]

    @property
    def export_eligible(self) -> bool:
        return self.accepted_records > 0

    def as_dict(self) -> dict[str, Any]:
        return {"status": "passed" if self.export_eligible else "failed", "totalRecords": self.total_records,
                "acceptedRecords": self.accepted_records, "quarantinedRecords": self.quarantined_records,
                "rejectedRecords": self.rejected_records, "schemaFailures": self.schema_failures,
                "groundingFailures": self.grounding_failures, "duplicateRecords": self.duplicate_records,
                "securityFindings": self.security_findings, "exportEligible": self.export_eligible,
                "findings": [finding.__dict__ for finding in self.findings]}


_SECRET = re.compile(r"(?:-----BEGIN [A-Z ]*PRIVATE KEY-----|\b(?:sk|rk)_[A-Za-z0-9]{16,}|\bBearer\s+[A-Za-z0-9._-]{16,}|\b(?:api[_-]?key|password|secret)\s*[:=]\s*[^\s]{8,})", re.I)
_PII = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b|\b\d{3}-\d{2}-\d{4}\b|\b(?:\d{3}[-.\s]){2}\d{4}\b")


def _texts(value: Any) -> list[str]:
    if isinstance(value, str): return [value]
    if isinstance(value, dict): return [text for child in value.values() for text in _texts(child)]
    if isinstance(value, list): return [text for child in value for text in _texts(child)]
    return []


def quality_gate(package: CarterPromptPackage, dataset: CarterCanonicalDataset, *, allowed_source_refs: set[str]) -> tuple[CarterCanonicalDataset, CarterQualityReport]:
    """Disposition Carter records deterministically; no model result can bypass this gate."""
    accepted: list[dict[str, Any]] = []; findings: list[CarterQualityFinding] = []; seen: set[str] = set()
    counts = {"quarantined": 0, "rejected": 0, "schema": 0, "grounding": 0, "duplicate": 0, "security": 0}
    for index, record in enumerate(dataset.records, 1):
        record_id = f"record_{index:03d}"
        try:
            validate_canonical_dataset(package, dataset.specification, [record], allowed_source_refs=allowed_source_refs, batch_count=1)
        except CarterPromptPackageError as exc:
            code = "INVALID_EVIDENCE" if "evidence" in exc.detail.lower() or "source" in exc.detail.lower() else "SCHEMA_INVALID"
            counts["grounding" if code == "INVALID_EVIDENCE" else "schema"] += 1; counts["rejected"] += 1
            findings.append(CarterQualityFinding(code, "error", record_id, "Record failed deterministic schema or source validation.", "canonical_validator")); continue
        fingerprint = hashlib.sha256(json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        if fingerprint in seen:
            counts["rejected"] += 1; counts["duplicate"] += 1
            findings.append(CarterQualityFinding("EXACT_DUPLICATE", "warning", record_id, "Duplicate record excluded from export.", "duplicate_validator")); continue
        seen.add(fingerprint)
        text = "\n".join(_texts(record))
        if _SECRET.search(text) or _PII.search(text):
            counts["quarantined"] += 1; counts["security"] += 1
            findings.append(CarterQualityFinding("SENSITIVE_VALUE", "warning", record_id, "Possible sensitive value excluded from export.", "safety_validator")); continue
        accepted.append(record)
    report = CarterQualityReport(len(dataset.records), len(accepted), counts["quarantined"], counts["rejected"], counts["schema"], counts["grounding"], counts["duplicate"], counts["security"], tuple(findings))
    return CarterCanonicalDataset(dataset.specification, tuple(accepted), dataset.compiled_schema), report


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
    if isinstance(value, str):
        # Keep spreadsheet readers from interpreting generated content as a formula.
        return f"'{value}" if value[:1] in {"=", "+", "-", "@"} else value
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
