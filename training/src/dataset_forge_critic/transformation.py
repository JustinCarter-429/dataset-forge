"""Deterministic, offline transformation of locally ingested critic datasets."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Iterator

import pyarrow.parquet as pq

from .adapters import FactsGroundingAdapter, HaluBenchAdapter, HelpSteer2Adapter, UltraFeedbackAdapter
from .schemas import CANONICAL_SCHEMA_VERSION, CanonicalCriticRecord
from .validation import validate_record

TRANSFORM_VERSION = "1.0.1"
SOURCES = {
    "google_facts_grounding": "snapshot-189e672d70f48d87",
    "halu_eval": "snapshot-29c4c6c6d2d4d532",
    "nvidia_helpsteer2": "snapshot-ff06772036b8dc1c",
    "openbmb_ultrafeedback": "snapshot-8fc47deccbd27dbc",
}


def canonical_json(record: CanonicalCriticRecord) -> str:
    return json.dumps(record.model_dump(mode="json", exclude_none=False), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _row_iter(path: Path, kind: str) -> Iterator[dict[str, Any]]:
    if kind == "csv":
        csv.field_size_limit(min(sys.maxsize, 2_147_483_647))
        with path.open("r", encoding="utf-8", newline="") as handle:
            yield from csv.DictReader(handle)
    elif kind == "gzip_jsonl":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
    elif kind == "jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
    elif kind == "parquet":
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1024):
            yield from batch.to_pylist()
    else:  # pragma: no cover - internal mapping guard
        raise ValueError(f"Unknown source kind: {kind}")


def _source_files(root: Path, dataset: str) -> list[tuple[Path, str, Any]]:
    if dataset == "google_facts_grounding":
        return [(root / "examples.csv", "csv", FactsGroundingAdapter(SOURCES[dataset]))]
    if dataset == "halu_eval":
        return [(root / "data" / "test-00000-of-00001.parquet", "parquet", HaluBenchAdapter(SOURCES[dataset]))]
    if dataset == "nvidia_helpsteer2":
        return [
            (root / "train.jsonl.gz", "gzip_jsonl", HelpSteer2Adapter(SOURCES[dataset], "train")),
            (root / "validation.jsonl.gz", "gzip_jsonl", HelpSteer2Adapter(SOURCES[dataset], "validation")),
            (root / "preference" / "preference.jsonl.gz", "gzip_jsonl", HelpSteer2Adapter(SOURCES[dataset], "preference")),
            (root / "disagreements" / "disagreements.jsonl.gz", "gzip_jsonl", HelpSteer2Adapter(SOURCES[dataset], "disagreements")),
        ]
    if dataset == "openbmb_ultrafeedback":
        return [(root / name, "jsonl", UltraFeedbackAdapter(SOURCES[dataset], name.removesuffix(".jsonl"))) for name in (
            "evol_instruct.jsonl", "false_qa.jsonl", "flan.jsonl", "sharegpt.jsonl", "truthful_qa.jsonl", "ultrachat.jsonl")]
    raise ValueError(f"Unsupported dataset: {dataset}")


def _percentile(values: list[int], pct: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * pct))]


@dataclass
class DatasetTransformResult:
    dataset: str
    manifest: dict[str, Any]
    report: dict[str, Any]


def transform_dataset(workspace: Path, dataset: str, *, validate_only: bool = False) -> DatasetTransformResult:
    if dataset not in SOURCES:
        raise ValueError(f"Unsupported dataset: {dataset}")
    training = workspace / "training"
    raw_root = training / "data" / "raw" / dataset / SOURCES[dataset]
    if not raw_root.is_dir():
        raise ValueError(f"Missing local snapshot: {raw_root}")
    content_key = hashlib.sha256(f"{SOURCES[dataset]}:{TRANSFORM_VERSION}:{CANONICAL_SCHEMA_VERSION}".encode()).hexdigest()[:16]
    version = f"processed-{content_key}"
    interim_dir = training / "data" / "interim" / dataset / version
    processed_dir = training / "data" / "processed" / dataset / version
    temp_context = tempfile.TemporaryDirectory() if validate_only else None
    stage_root = Path(temp_context.name) if temp_context else training / "data" / ".staging" / dataset / version
    if stage_root.exists():
        shutil.rmtree(stage_root)
    stage_root.mkdir(parents=True)
    stage_interim = stage_root / "interim.jsonl"
    stage_processed = stage_root / "canonical.jsonl"
    seen_ids: set[str] = set()
    raw_count = output_count = failed = excluded = source_rows_with_output = 0
    reasons: Counter[str] = Counter()
    source_hashes: dict[str, str] = {}
    task_families: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    scores: dict[str, list[float]] = defaultdict(list)
    lengths: list[int] = []
    fingerprints: Counter[str] = Counter()
    samples: list[tuple[str, dict[str, Any]]] = []
    with stage_interim.open("w", encoding="utf-8", newline="\n") as staged:
      for source_path, kind, adapter in _source_files(raw_root, dataset):
        if not source_path.is_file():
            raise ValueError(f"Missing expected source file: {source_path}")
        source_hashes[source_path.relative_to(raw_root).as_posix()] = file_hash(source_path)
        for row_number, source_row in enumerate(_row_iter(source_path, kind)):
            raw_count += 1
            row = dict(source_row)
            row["_row"] = row_number
            if dataset == "nvidia_helpsteer2" and getattr(adapter, "mode", "") == "disagreements":
                excluded += 1
                reasons["DISAGREEMENT_NONCONSENSUS_EXCLUDED"] += 1
                continue
            try:
                adapter.validate_raw_record(row)
                records = adapter.transform(row)
                if not records:
                    excluded += 1
                    reasons["NO_CANONICAL_CANDIDATE"] += 1
                    continue
                source_rows_with_output += 1
                for item in records:
                    if item.record_id in seen_ids:
                        raise ValueError("RECORD_ID_COLLISION")
                    seen_ids.add(item.record_id)
                    validated = validate_record(item.model_dump(mode="json"))
                    encoded = canonical_json(validated)
                    staged.write(encoded + "\n")
                    output_count += 1
                    task_families[validated.task_family.value] += 1
                    for target in validated.supervision.available_targets:
                        target_counts[target] += 1
                    for name, value in validated.target.scores.model_dump().items():
                        if value is not None:
                            scores[name].append(float(value["value"]))
                    text = " ".join(str(value) for value in (validated.input.user_request, validated.input.source_context, validated.input.candidate_record) if value)
                    lengths.append(len(" ".join(text.split())))
                    fingerprint = hashlib.sha256(json.dumps(validated.input.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
                    fingerprints[fingerprint] += 1
                    sample = (hashlib.sha256(validated.record_id.encode()).hexdigest(), {"record_id": validated.record_id, "task_family": validated.task_family.value, "available_targets": validated.supervision.available_targets, "source": validated.provenance.source_dataset, "split": validated.provenance.source_split})
                    samples.append(sample)
                    samples.sort(key=lambda entry: entry[0])
                    del samples[10:]
            except ValueError as exc:
                failed += 1
                reasons[str(exc)] += 1
    output_hash = file_hash(stage_interim)
    score_stats = {name: {"count": len(values), "min": min(values), "max": max(values), "mean": sum(values) / len(values), "median": median(values)} for name, values in sorted(scores.items())}
    report = {
        "dataset": dataset, "processed_version": version, "raw_rows": raw_count, "canonical_records": output_count,
        "excluded_rows": excluded, "failed_rows": failed, "exclusion_or_failure_reasons": dict(sorted(reasons.items())),
        "task_families": dict(sorted(task_families.items())), "supervision_target_counts": dict(sorted(target_counts.items())),
        "score_statistics": score_stats, "candidate_text_lengths_whitespace": {"p50": _percentile(lengths, .50), "p90": _percentile(lengths, .90), "p95": _percentile(lengths, .95), "p99": _percentile(lengths, .99), "max": max(lengths, default=0)},
        "duplicate_candidate_fingerprints": sum(count - 1 for count in fingerprints.values() if count > 1),
        "validation": {"canonical_records_valid": output_count, "canonical_records_invalid": 0},
        "reconciliation": {"raw_rows": raw_count, "source_rows_with_output": source_rows_with_output, "canonical_records": output_count, "excluded_rows": excluded, "failed_rows": failed, "source_rows_accounted": raw_count == source_rows_with_output + excluded + failed},
    }
    manifest = {"manifest_version": "1.0.0", "dataset": dataset, "source_snapshot_id": SOURCES[dataset], "adapter_version": TRANSFORM_VERSION, "canonical_schema_version": CANONICAL_SCHEMA_VERSION, "processed_version": version, "source_file_sha256": source_hashes, "raw_rows": raw_count, "canonical_records": output_count, "excluded_rows": excluded, "failed_rows": failed, "reason_counts": dict(sorted(reasons.items())), "canonical_jsonl_sha256": output_hash}
    if not validate_only:
        # Re-read the staged representation to ensure promotion only follows canonical validation.
        with stage_interim.open("r", encoding="utf-8") as handle, stage_processed.open("w", encoding="utf-8", newline="\n") as destination:
            for line in handle:
                item = validate_record(json.loads(line))
                destination.write(canonical_json(item) + "\n")
        if file_hash(stage_processed) != output_hash:
            raise ValueError("STAGED_OUTPUT_HASH_MISMATCH")
        for destination, staged in ((interim_dir, stage_interim), (processed_dir, stage_processed)):
            if destination.exists() and file_hash(destination / staged.name) != output_hash:
                raise ValueError(f"EXISTING_OUTPUT_HASH_MISMATCH: {destination}")
            if not destination.exists():
                destination.mkdir(parents=True)
                shutil.copy2(staged, destination / staged.name)
        sample_path = processed_dir / "review-sample.json"
        sample_path.write_text(json.dumps([entry for _, entry in sorted(samples)[:10]], sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
        shutil.rmtree(stage_root)
    else:
        temp_context.cleanup()
    return DatasetTransformResult(dataset, manifest, report)


def transform_all(workspace: Path, datasets: Iterable[str] | None = None, *, validate_only: bool = False) -> list[DatasetTransformResult]:
    return [transform_dataset(workspace, dataset, validate_only=validate_only) for dataset in (list(datasets) if datasets else list(SOURCES))]
