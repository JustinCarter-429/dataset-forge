"""Build a versioned, self-verifying GPU handoff without frozen test data."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from .integrity import derive_jsonl, sha256_file

LEGACY_ARCHIVE = "gemma4-e4b-wp5a-d5cf80fd96e88c55.tar.gz"
LEGACY_SHA256 = "39f30dc45c339b926c487a827757a4c261753082f1a6f302418657d273487e9a"


def _copy_tree(source: Path, destination: Path) -> None:
    shutil.copytree(
        source,
        destination,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest*", ".venv*", "*.egg-info"),
    )


def build_handoff(workspace: Path) -> dict[str, Any]:
    training = workspace / "training"; legacy = training / "bundles" / LEGACY_ARCHIVE
    if not legacy.is_file() or sha256_file(legacy) != LEGACY_SHA256: raise ValueError("LEGACY_BUNDLE_HASH_MISMATCH")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=workspace, text=True).strip()
    short = commit[:12]; root = training / "bundles" / f"critic-gpu-handoff-v2-{short}"
    if root.exists(): shutil.rmtree(root)
    root.mkdir(parents=True)
    transforms = []
    sources = [
        ("public-train", training / "data/curated/public-critic-v1/train.jsonl", root / "training/data/curated/public-critic-v1/train.jsonl"),
        ("public-validation", training / "data/curated/public-critic-v1/validation.jsonl", root / "training/data/curated/public-critic-v1/validation.jsonl"),
        ("native-train", training / "data/native/native-rule-v2/train.jsonl", root / "training/data/native/native-rule-v2/train.jsonl"),
        ("native-validation", training / "data/native/native-rule-v2/validation.jsonl", root / "training/data/native/native-rule-v2/validation.jsonl"),
    ]
    for identity, source, destination in sources: transforms.append(derive_jsonl(source, destination, source_identity=identity))
    for folder in ("src", "configs", "prompts", "requirements", "scripts", "docs", "notebooks", "tests", "manifests"):
        _copy_tree(training / folder, root / "training" / folder)
    for file in ("pyproject.toml", "README.md"):
        shutil.copy2(training / file, root / "training" / file)
    subprocess.run(["git", "bundle", "create", str(root / "source.bundle"), "HEAD"], cwd=workspace, check=True)
    files = []
    forbidden_names = {".env", "id_rsa", "id_ed25519", "vast_datasetforge", "credentials", "credentials.json", "token"}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        parts = {part.casefold() for part in Path(relative).parts}
        if any(part in forbidden_names or part.startswith(".env.") for part in parts):
            raise ValueError(f"SECRET_LIKE_PATH:{relative}")
        files.append({"path": relative, "size": path.stat().st_size, "sha256": sha256_file(path)})
    payload = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    manifest = {"handoff_version": "critic-gpu-v2", "source_git_commit": commit, "source_bundle": "source.bundle",
                "model_id": "google/gemma-4-E4B-it", "model_revision": "ee0ef6023621cff504d758262d4e04895a5af4a2",
                "architecture": "Gemma4ForConditionalGeneration", "legacy_input_archive_sha256_verified": LEGACY_SHA256,
                "data_transformations": transforms, "test_records_included": 0, "payload_sha256": payload, "files": files,
                "manifest_hash_scope": "files excludes HANDOFF-MANIFEST.json and CHECKSUMS.sha256 to avoid self-reference",
                "readiness": {"local_implementation": "COMPLETE", "local_tests": "PENDING_AT_BUILD", "gpu_smoke": "PENDING", "benchmark": "PENDING", "remote_backup": "PENDING"}}
    (root / "HANDOFF-MANIFEST.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
    (root / "CHECKSUMS.sha256").write_text("".join(f"{item['sha256']}  {item['path']}\n" for item in files), encoding="utf-8", newline="\n")
    archive = training / "bundles" / f"critic-gpu-handoff-v2-{short}-{payload[:12]}.tar.gz"
    with tarfile.open(archive, "w:gz", format=tarfile.PAX_FORMAT) as output:
        for path in sorted(root.rglob("*")):
            if path.is_file(): output.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
    checksum_path = archive.with_suffix(archive.suffix + ".sha256")
    checksum_path.write_text(f"{sha256_file(archive)}  {archive.name}\n", encoding="utf-8", newline="\n")
    return {"status": "HANDOFF_BUILT", "root": str(root), "archive": str(archive), "archive_sha256": sha256_file(archive),
            "payload_sha256": payload, "source_git_commit": commit, "test_records_included": 0}


def verify_handoff(root: Path) -> dict[str, Any]:
    manifest = json.loads((root / "HANDOFF-MANIFEST.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path = root / item["path"]
        if not path.is_file() or sha256_file(path) != item["sha256"]: raise ValueError(f"HANDOFF_HASH_MISMATCH:{item['path']}")
    payload = hashlib.sha256(json.dumps(manifest["files"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if payload != manifest["payload_sha256"]: raise ValueError("HANDOFF_PAYLOAD_MISMATCH")
    if manifest["test_records_included"] != 0: raise ValueError("HANDOFF_CONTAINS_TEST")
    return {"status": "HANDOFF_VERIFIED", "payload_sha256": payload, "files": len(manifest["files"]), "test_records_included": 0}
