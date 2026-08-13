"""Offline, untrusted-data-safe local source ingestion for WP2."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import shutil
import stat
import tarfile
import tempfile
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

EXPECTED_DATASETS = {
    "google_facts_grounding": "Facts-grounding-public",
    "halu_eval": "HaluEval-bucket",
    "nvidia_helpsteer2": "HelpSteer2",
    "openbmb_ultrafeedback": "UltraFeedBack",
}
PAYLOAD_SUFFIXES = {".csv", ".tsv", ".json", ".jsonl", ".parquet", ".arrow", ".gz"}
ARCHIVE_SUFFIXES = {".zip", ".tar", ".tgz", ".gz"}
INSPECTION_VERSION = "1.0.0"


@dataclass(frozen=True)
class SourceFile:
    relative_path: str
    size_bytes: int
    sha256: str
    format: str
    classification: str
    lfs_pointer: bool


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def is_lfs_pointer(path: Path) -> bool:
    try:
        return path.read_bytes()[:256].splitlines()[:1] == [b"version https://git-lfs.github.com/spec/v1"]
    except OSError:
        return False


def resolve_sources(source_root: Path) -> dict[str, Path | None]:
    if not source_root.is_dir():
        raise ValueError(f"Source root does not exist or is not a directory: {source_root}")
    children = {child.name.casefold(): child for child in source_root.iterdir()}
    return {key: children.get(name.casefold()) for key, name in EXPECTED_DATASETS.items()}


def _safe_member_path(name: str) -> PurePosixPath:
    member = PurePosixPath(name)
    if member.is_absolute() or ".." in member.parts or (member.parts and ":" in member.parts[0]):
        raise ValueError(f"Unsafe archive member path: {name}")
    return member


def safe_extract_zip(archive: Path, destination: Path, max_members: int = 100_000, max_bytes: int = 50 * 1024**3) -> None:
    with zipfile.ZipFile(archive) as bundle:
        infos = bundle.infolist()
        if len(infos) > max_members or sum(info.file_size for info in infos) > max_bytes:
            raise ValueError(f"Archive exceeds extraction safety limits: {archive.name}")
        for info in infos:
            _safe_member_path(info.filename)
            if stat.S_IFMT(info.external_attr >> 16) == stat.S_IFLNK:
                raise ValueError(f"Archive contains symbolic-link member: {info.filename}")
        bundle.extractall(destination)


def safe_extract_tar(archive: Path, destination: Path, max_members: int = 100_000, max_bytes: int = 50 * 1024**3) -> None:
    with tarfile.open(archive, "r:*") as bundle:
        members = bundle.getmembers()
        if len(members) > max_members or sum(member.size for member in members) > max_bytes:
            raise ValueError(f"Archive exceeds extraction safety limits: {archive.name}")
        for member in members:
            _safe_member_path(member.name)
            if member.issym() or member.islnk() or member.isdev():
                raise ValueError(f"Archive contains unsafe member: {member.name}")
        bundle.extractall(destination, filter="data")


def _format(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".jsonl.gz"):
        return "jsonl.gz"
    if name.endswith(".csv.gz"):
        return "csv.gz"
    return path.suffix.lower().lstrip(".") or "unknown"


def _classification(path: Path) -> str:
    if path.name.lower() in {"readme.md", "license", "license.md", ".gitattributes"}:
        return "metadata"
    return "payload" if path.suffix.lower() in PAYLOAD_SUFFIXES else "auxiliary"


def inventory(source: Path) -> list[SourceFile]:
    files: list[SourceFile] = []
    for path in sorted((item for item in source.rglob("*") if item.is_file()), key=lambda item: item.as_posix()):
        relative = path.relative_to(source).as_posix()
        if any(part in {".git", ".github", ".venv", "__pycache__"} for part in path.parts):
            continue
        files.append(SourceFile(relative, path.stat().st_size, sha256_file(path), _format(path), _classification(path), is_lfs_pointer(path)))
    return files


def snapshot_id(dataset_key: str, files: list[SourceFile]) -> str:
    payload = "\n".join(f"{file.relative_path}\0{file.sha256}\0{file.size_bytes}" for file in files)
    source_fingerprint = (dataset_key + "\n" + payload).encode()
    return f"snapshot-{hashlib.sha256(source_fingerprint).hexdigest()[:16]}"


def _open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8", newline="") if path.name.lower().endswith(".gz") else path.open("r", encoding="utf-8", newline="")


def _walk(value: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, nested in value.items():
            yield from _walk(nested, f"{prefix}.{key}" if prefix else str(key))
    else:
        yield prefix, value


def inspect_payload(path: Path) -> dict[str, Any]:
    fmt = _format(path)
    fields: dict[str, Counter[str]] = defaultdict(Counter)
    nulls: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    count = 0
    malformed: list[int] = []
    def observe(record: dict[str, Any]) -> None:
        nonlocal count
        count += 1
        if len(samples) < 2:
            samples.append({key: type(value).__name__ for key, value in record.items()})
        for key, value in _walk(record):
            fields[key][type(value).__name__] += 1
            if value is None:
                nulls[key] += 1
    if fmt in {"jsonl", "jsonl.gz"}:
        with _open_text(path) as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    malformed.append(line_number); continue
                if isinstance(value, dict): observe(value)
                else: malformed.append(line_number)
    elif fmt in {"csv", "tsv", "csv.gz"}:
        csv.field_size_limit(2**31 - 1)
        with _open_text(path) as handle:
            for row in csv.DictReader(handle, delimiter="\t" if fmt == "tsv" else ","):
                observe(dict(row))
    elif fmt == "json":
        value = json.loads(path.read_text(encoding="utf-8"))
        for record in value if isinstance(value, list) else [value]:
            if isinstance(record, dict): observe(record)
    elif fmt == "parquet":
        import pyarrow.parquet as pq
        parquet = pq.ParquetFile(path)
        count = parquet.metadata.num_rows
        for field in parquet.schema_arrow:
            fields[field.name][str(field.type)] += count
        samples = [{field.name: str(field.type) for field in parquet.schema_arrow}]
    else:
        return {"format": fmt, "record_count": None, "record_count_kind": "not_applicable"}
    return {"format": fmt, "record_count": count, "record_count_kind": "exact", "field_types": {key: dict(value) for key, value in fields.items()}, "null_counts": dict(nulls), "samples": samples, "malformed_lines": malformed}


def ingest_dataset(dataset_key: str, source: Path, raw_root: Path, manifests_root: Path, reports_root: Path) -> dict[str, Any]:
    files = inventory(source)
    if not files:
        raise ValueError(f"No usable files found for {dataset_key}: {source}")
    identifier = snapshot_id(dataset_key, files)
    destination = raw_root / dataset_key / identifier
    staging = raw_root / ".staging" / dataset_key / identifier
    completion_marker = destination / ".dataset-forge-snapshot.json"
    if destination.exists() and completion_marker.is_file():
        reused = True
    else:
        reused = False
        staging.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            shutil.rmtree(destination)
        if staging.exists(): shutil.rmtree(staging)
        shutil.copytree(source, staging, ignore=shutil.ignore_patterns(".git", ".github", ".venv", "__pycache__"))
        copied = inventory(staging)
        if [(item.relative_path, item.sha256) for item in copied] != [(item.relative_path, item.sha256) for item in files]:
            shutil.rmtree(staging); raise ValueError(f"Snapshot reconciliation failed for {dataset_key}")
        for file in files:
            if file.classification == "payload" and not file.lfs_pointer:
                inspect_payload(staging / file.relative_path)
        completion_marker_staging = staging / ".dataset-forge-snapshot.json"
        completion_marker_staging.write_text(json.dumps({"dataset_key": dataset_key, "snapshot_id": identifier}, sort_keys=True), encoding="utf-8")
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging.replace(destination)
    payload_inspections = {}
    for file in files:
        if file.classification == "payload" and not file.lfs_pointer:
            payload_inspections[file.relative_path] = inspect_payload(destination / file.relative_path)
    manifest = {"inspection_version": INSPECTION_VERSION, "dataset_key": dataset_key, "snapshot_id": identifier,
                "source_artifact_type": "directory", "files": [asdict(file) for file in files], "reused": reused,
                "integrity_status": "passed", "lfs_pointer_files": [file.relative_path for file in files if file.lfs_pointer],
                "payload_inspections": payload_inspections}
    manifests_root.mkdir(parents=True, exist_ok=True)
    (manifests_root / f"{dataset_key}-{identifier}.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    reports_root.mkdir(parents=True, exist_ok=True)
    lines = [f"# {dataset_key} inspection", "", f"Snapshot: `{identifier}`", "", "## Files", ""]
    lines += [f"- `{file.relative_path}` — {file.size_bytes} bytes; {file.format}; {file.classification}; SHA-256 `{file.sha256}`" for file in files]
    lines += ["", "## Observed payload schemas", "", "```json", json.dumps(payload_inspections, indent=2, sort_keys=True), "```", "", "## WP3 Adapter Readiness", "", "Use only fields observed above; preserve source scales and leave absent supervision unavailable. No mapping is implemented in WP2."]
    (reports_root / f"{dataset_key}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def inspect_local_datasets(source_root: Path, project_root: Path) -> dict[str, Any]:
    raw_root = project_root / "data" / "raw"
    manifests_root = project_root / "manifests"
    reports_root = project_root / "reports" / "datasets"
    results: dict[str, Any] = {}
    for dataset_key, source in resolve_sources(source_root).items():
        results[dataset_key] = {"status": "missing"} if source is None else ingest_dataset(dataset_key, source, raw_root, manifests_root, reports_root)
    return results
