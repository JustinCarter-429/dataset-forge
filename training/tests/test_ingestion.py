from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from dataset_forge_critic.ingestion import (EXPECTED_DATASETS, ingest_dataset, is_lfs_pointer,
                                             safe_extract_zip, sha256_file, snapshot_id)


def test_hash_and_lfs_detection(tmp_path: Path):
    pointer = tmp_path / "data.parquet"
    pointer.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 4\n", encoding="utf-8")
    assert is_lfs_pointer(pointer)
    assert sha256_file(pointer) == sha256_file(pointer)


def test_safe_zip_rejects_traversal_and_extracts_nested(tmp_path: Path):
    archive = tmp_path / "safe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("nested/data.json", "[]")
    destination = tmp_path / "out"
    destination.mkdir()
    safe_extract_zip(archive, destination)
    assert (destination / "nested" / "data.json").exists()
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as bundle:
        bundle.writestr("../escape.txt", "no")
    with pytest.raises(ValueError, match="Unsafe archive"):
        safe_extract_zip(bad, destination)


def test_ingestion_is_idempotent_and_inspects_jsonl(tmp_path: Path):
    source = tmp_path / "source root with spaces" / "Example"
    source.mkdir(parents=True)
    (source / "records.jsonl").write_text('{"id": 1, "nested": {"x": null}}\n', encoding="utf-8")
    project = tmp_path / "project"
    first = ingest_dataset("example", source, project / "data" / "raw", project / "manifests", project / "reports" / "datasets")
    second = ingest_dataset("example", source, project / "data" / "raw", project / "manifests", project / "reports" / "datasets")
    assert first["snapshot_id"] == second["snapshot_id"]
    assert second["reused"] is True
    assert first["payload_inspections"]["records.jsonl"]["record_count"] == 1
