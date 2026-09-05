"""Physical-file integrity, safe archives, derived exports, and split checks."""
from __future__ import annotations

import hashlib
import json
import os
import tarfile
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from .schemas import CANONICAL_SCHEMA_VERSION
from .validation import validate_record


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_extract_tar(archive: Path, destination: Path, expected_sha256: str) -> None:
    if sha256_file(archive) != expected_sha256:
        raise ValueError("ARCHIVE_SHA256_MISMATCH")
    destination = destination.resolve()
    with tarfile.open(archive, "r:*") as source:
        for member in source.getmembers():
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or member.issym() or member.islnk():
                raise ValueError(f"UNSAFE_ARCHIVE_MEMBER: {member.name}")
            resolved = (destination / Path(*name.parts)).resolve()
            if destination not in resolved.parents and resolved != destination:
                raise ValueError(f"UNSAFE_ARCHIVE_MEMBER: {member.name}")
        source.extractall(destination, filter="data")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"INVALID_JSONL:{path}:{line_number}") from exc


def normalized_input(record: dict[str, Any]) -> str:
    value = record.get("canonical_record", record).get("input", {})
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return " ".join(text.casefold().split())


def audit_splits(train_paths: list[Path], validation_paths: list[Path]) -> dict[str, Any]:
    exact_train: set[str] = set(); normalized_train: set[str] = set(); counts = Counter()
    for path in train_paths:
        for item in iter_jsonl(path):
            record = item.get("canonical_record", item); validate_record(record)
            exact_train.add(record["record_id"]); normalized_train.add(normalized_input(record)); counts["train"] += 1
    exact_cross = normalized_cross = 0
    for path in validation_paths:
        for item in iter_jsonl(path):
            record = item.get("canonical_record", item); validate_record(record); counts["validation"] += 1
            exact_cross += record["record_id"] in exact_train
            normalized_cross += normalized_input(record) in normalized_train
    return {"counts": dict(counts), "exact_cross_split_duplicates": exact_cross, "normalized_cross_split_duplicates": normalized_cross}


def derive_jsonl(source: Path, destination: Path, *, source_identity: str) -> dict[str, Any]:
    """Create an LF-only export with physically serialized schema_version."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", delete=False, dir=destination.parent) as staged:
        staged_path = Path(staged.name); count = 0
        for item in iter_jsonl(source):
            wrapped = "canonical_record" in item
            record = item["canonical_record"] if wrapped else item
            record = dict(record); record["schema_version"] = CANONICAL_SCHEMA_VERSION
            validate_record(record)
            output = dict(item); output["canonical_record"] = record if wrapped else output.get("canonical_record", record)
            if not wrapped: output = record
            staged.write(json.dumps(output, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"); count += 1
    os.replace(staged_path, destination)
    return {"source_identity": source_identity, "source_path": str(source), "source_sha256": sha256_file(source),
            "output_path": str(destination), "output_sha256": sha256_file(destination), "records": count,
            "schema_version": CANONICAL_SCHEMA_VERSION, "encoding": "UTF-8", "newline": "LF", "checksum_semantics": "SHA-256 of physical file bytes"}
