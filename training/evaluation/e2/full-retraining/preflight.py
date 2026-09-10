"""Local-only Phase E2.4 audit, deterministic packaging, and retrieval verification."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import io
import json
import os
import re
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath
from typing import Any

HERE = Path(__file__).resolve().parent
TRAINING = HERE.parents[2]
WORKSPACE = TRAINING.parent
REMEDIATION = HERE.parent / "reason-code-remediation"
CORPUS = REMEDIATION / "corpus"
REPORTS = HERE / "evidence"
PACKAGE_DIR = HERE / "package"

SPEC = importlib.util.spec_from_file_location("e2_train", HERE / "e2_train.py")
RUNNER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RUNNER)

from dataset_forge_critic.e2_contract import APPROVED_REASON_CODES, FIELDS, canonical_json, parse_strict_response  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(value)
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _record_identity(row: dict[str, Any]) -> tuple[str, str]:
    record = row["canonical_record"]
    canonical_input = json.dumps(record["input"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return record["record_id"], hashlib.sha256(canonical_input.encode()).hexdigest()


def audit(rebuild_dir: Path | None = None) -> dict[str, Any]:
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    train_path, validation_path = CORPUS / "train.jsonl", CORPUS / "validation.jsonl"
    train, validation = read_jsonl(train_path), read_jsonl(validation_path)
    train_identity = [_record_identity(row) for row in train]
    validation_identity = [_record_identity(row) for row in validation]
    train_ids, validation_ids = {value[0] for value in train_identity}, {value[0] for value in validation_identity}
    train_inputs, validation_inputs = {value[1] for value in train_identity}, {value[1] for value in validation_identity}
    decisions, dataset_types, codes = set(), set(), set()
    exact_targets = True
    for row in train + validation:
        record = row["canonical_record"]
        target = record["target"]
        exact_targets &= set(target) == set(FIELDS)
        parse_strict_response(canonical_json(target))
        decisions.add(target["decision"])
        dataset_types.add(record["input"]["dataset_spec"]["dataset_type"])
        codes.update(target["reason_codes"])
    prior = json.loads((REMEDIATION / "reports/final-corpus-verification.json").read_text(encoding="utf-8"))
    smoke_validation = read_jsonl(REMEDIATION / "smoke/validation.jsonl")
    smoke_validation_ids = {_record_identity(row)[0] for row in smoke_validation}
    opt = config["optimization"]
    sampler = RUNNER.sampler_dry_run([value[0] for value in train_identity], opt["epochs"], opt["gradient_accumulation_steps"], opt["seed"])
    rebuild = None
    if rebuild_dir is not None:
        rebuild = {
            "train_byte_identical": sha256(train_path) == sha256(rebuild_dir / "corpus/train.jsonl"),
            "validation_byte_identical": sha256(validation_path) == sha256(rebuild_dir / "corpus/validation.jsonl"),
            "train_sha256": sha256(rebuild_dir / "corpus/train.jsonl"),
            "validation_sha256": sha256(rebuild_dir / "corpus/validation.jsonl"),
        }
    checks = {
        "train_count_6350": len(train) == config["data"]["train_records"] == 6350,
        "validation_count_806": len(validation) == config["data"]["validation_records"] == 806,
        "train_hash_frozen": sha256(train_path) == config["data"]["train_sha256"],
        "validation_hash_frozen": sha256(validation_path) == config["data"]["validation_sha256"],
        "train_ids_unique": len(train_ids) == len(train),
        "validation_ids_unique": len(validation_ids) == len(validation),
        "record_id_overlap_zero": not train_ids & validation_ids,
        "canonical_input_overlap_zero": not train_inputs & validation_inputs,
        "smoke_validation_not_in_training": not train_ids & smoke_validation_ids,
        "strict_five_field_targets": exact_targets,
        "all_decisions": decisions == {"accept", "revise", "reject"},
        "all_dataset_types": dataset_types == {"question_answer", "instruction_response", "classification", "scenario_expected_result", "custom"},
        "all_28_reason_codes": codes == set(APPROVED_REASON_CODES) and len(codes) == 28,
        "prior_independence_and_tokenization_gates": prior["status"] == "PASS" and all(prior["checks"].values()),
        "optimizer_steps_2382": sampler["optimizer_steps"] == opt["max_optimizer_steps"] == 2382,
        "examples_19050": sampler["examples"] == opt["total_training_examples"] == 19050,
        "partial_final_batch_2": sampler["final_partial_batch_examples"] == opt["final_partial_batch_examples"] == 2,
        "each_record_once_per_epoch": sampler["each_record_once_per_epoch"],
        "validation_schedule_frozen": RUNNER.schedule_steps(2382, 250, baseline=True) == [0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2382],
        "checkpoint_schedule_frozen": RUNNER.schedule_steps(2382, 250, baseline=False) == [250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2382],
        "rebuild_byte_identical": rebuild is not None and all(value for key, value in rebuild.items() if key.endswith("byte_identical")),
    }
    result = {
        "schema_version": "phase-e2.4-preflight-v1", "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks, "corpus": {"train_records": len(train), "validation_records": len(validation),
        "train_sha256": sha256(train_path), "validation_sha256": sha256(validation_path),
        "approved_reason_codes": len(codes), "dataset_types": sorted(dataset_types), "decisions": sorted(decisions)},
        "sampler": sampler, "rebuild": rebuild,
        "tokenization": prior["tokenization"],
        "validation_steps": RUNNER.schedule_steps(2382, 250, baseline=True),
        "checkpoint_steps": RUNNER.schedule_steps(2382, 250, baseline=False),
        "held_out_thresholds_sha256": sha256(TRAINING / "evaluation/thresholds.json"),
        "remote_or_gpu_work_performed": False,
    }
    atomic_bytes(REPORTS / "preflight-report.json", stable(result))
    if result["status"] != "PASS":
        raise ValueError(f"PREFLIGHT_FAILED:{[key for key, value in checks.items() if not value]}")
    return result


def package_inputs() -> dict[str, Path]:
    mapping = {
        "phase-e2.4/config.json": HERE / "config.json",
        "phase-e2.4/runner/e2_train.py": HERE / "e2_train.py",
        "phase-e2.4/runner/remote-runner.sh": HERE / "remote-runner.sh",
        "phase-e2.4/runner/monitor.sh": HERE / "monitor.sh",
        "phase-e2.4/preflight.py": HERE / "preflight.py",
        "phase-e2.4/OPERATOR.md": HERE / "OPERATOR.md",
        "phase-e2.4/post-training.json": HERE / "post-training.json",
        "phase-e2.4/corpus/train/corpus.jsonl": CORPUS / "train.jsonl",
        "phase-e2.4/corpus/validation/corpus.jsonl": CORPUS / "validation.jsonl",
        "phase-e2.4/prompt/critic-system-v2.txt": TRAINING / "prompts/critic-system-v2.txt",
        "phase-e2.4/contract/e2_contract.py": TRAINING / "src/dataset_forge_critic/e2_contract.py",
        "phase-e2.4/contract/modeling.py": TRAINING / "src/dataset_forge_critic/modeling.py",
        "phase-e2.4/contract/tokenization.py": TRAINING / "src/dataset_forge_critic/tokenization.py",
        "phase-e2.4/contract/training_config.py": TRAINING / "src/dataset_forge_critic/training_config.py",
        "phase-e2.4/post-training/e1_pipeline.py": TRAINING / "evaluation/e1_pipeline.py",
        "phase-e2.4/post-training/thresholds.json": TRAINING / "evaluation/thresholds.json",
        "phase-e2.4/tests/test_full_retraining.py": HERE / "tests/test_full_retraining.py",
        "phase-e2.4/tests/test_e2_contract.py": TRAINING / "tests/test_e2_contract.py",
        "phase-e2.4/tests/test_e2_reason_code_remediation.py": TRAINING / "tests/test_e2_reason_code_remediation.py",
        "phase-e2.4/tests/test_gemma.py": TRAINING / "tests/test_gemma.py",
        "phase-e2.4/evidence/preflight-report.json": REPORTS / "preflight-report.json",
        "phase-e2.4/evidence/e2.2-reproducibility-manifest.json": REMEDIATION / "manifests/reproducibility-manifest.json",
        "phase-e2.4/evidence/e2.2-final-corpus-verification.json": REMEDIATION / "reports/final-corpus-verification.json",
        "phase-e2.4/evidence/test-summary.json": REPORTS / "test-summary.json",
        "phase-e2.4/evidence/test-exclusions.json": REPORTS / "test-exclusions.json",
        "phase-e2.4/evidence/focused-tests.log": REPORTS / "focused-tests.log",
        "phase-e2.4/evidence/relevant-tests.log": REPORTS / "relevant-tests.log",
    }
    missing = [name for name, path in mapping.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"MISSING_PACKAGE_INPUTS:{missing}")
    return mapping


SENSITIVE = {
    "private_key": re.compile(rb"BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY"),
    "hugging_face_token": re.compile(rb"hf_[A-Za-z0-9]{20,}"),
    "api_key": re.compile(rb"(?:sk-|api[_-]?key[\"']?\s*[:=]\s*[\"'])[A-Za-z0-9_-]{20,}", re.I),
    "windows_user_path": re.compile(rb"[A-Za-z]:\\Users\\", re.I),
    "private_unix_path": re.compile(rb"/(?:home|Users|workspace)/[^\s\"']+"),
    "signed_url": re.compile(rb"https?://[^\s]+[?&](?:token|signature|sig|X-Amz-Signature)=", re.I),
    "ip_address": re.compile(rb"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)"),
}


def publication_scan(files: dict[str, bytes]) -> dict[str, Any]:
    findings = []
    for name, value in files.items():
        for category, pattern in SENSITIVE.items():
            if pattern.search(value):
                findings.append({"path": name, "category": category})
    return {"schema_version": "phase-e2.4-publication-scan-v1", "status": "PASS" if not findings else "FAIL", "files_scanned": len(files), "findings": findings}


def _tar_bytes(files: dict[str, bytes]) -> bytes:
    payload = io.BytesIO()
    with gzip.GzipFile(fileobj=payload, mode="wb", mtime=0, filename="") as zipped:
        with tarfile.open(fileobj=zipped, mode="w") as archive:
            for name in sorted(files):
                info = tarfile.TarInfo(name)
                info.size = len(files[name]); info.mtime = 0; info.mode = 0o644; info.uid = info.gid = 0; info.uname = info.gname = ""
                archive.addfile(info, io.BytesIO(files[name]))
    return payload.getvalue()


def create_test_summary() -> dict[str, Any]:
    runs = []
    for name in ("focused", "relevant"):
        root = ET.parse(REPORTS / f"{name}-tests.xml").getroot()
        suite = root.find("testsuite") if root.tag == "testsuites" else root
        assert suite is not None
        run = {key: int(suite.attrib.get(key, 0)) for key in ("tests", "failures", "errors", "skipped")}
        run.update({"name": name, "passed": run["tests"] - run["failures"] - run["errors"] - run["skipped"],
                    "duration_seconds": float(suite.attrib["time"]),
                    "exit_code": int((REPORTS / f"{name}-tests-exit-code.txt").read_text().strip())})
        runs.append(run)
    exclusions = json.loads((REPORTS / "test-exclusions.json").read_text(encoding="utf-8"))["excluded"]
    result = {"schema_version": "phase-e2.4-test-summary-v1", "status": "PASS" if all(run["exit_code"] == 0 and not run["failures"] and not run["errors"] for run in runs) else "FAIL",
              "runs": runs, "excluded": exclusions}
    atomic_bytes(REPORTS / "test-summary.json", stable(result))
    return result


def build_package() -> dict[str, Any]:
    summary = create_test_summary()
    if summary["status"] != "PASS":
        raise ValueError("TEST_EVIDENCE_NOT_PASSING")
    inputs = package_inputs()
    files = {name: path.read_bytes() for name, path in inputs.items()}
    scan = publication_scan(files)
    atomic_bytes(REPORTS / "publication-safety-scan.json", stable(scan))
    if scan["status"] != "PASS":
        raise ValueError(f"PUBLICATION_SCAN_FAILED:{scan['findings']}")
    files["phase-e2.4/evidence/publication-safety-scan.json"] = stable(scan)
    manifest = {"schema_version": "phase-e2.4-package-manifest-v1", "files": {
        name: {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()} for name, value in sorted(files.items())}}
    files["phase-e2.4/manifests/package-manifest.json"] = stable(manifest)
    scan_with_manifest = publication_scan(files)
    if scan_with_manifest["status"] != "PASS":
        raise ValueError("PACKAGE_MANIFEST_PUBLICATION_SCAN_FAILED")
    archive = PACKAGE_DIR / "phase-e2.4-full-retraining.tar.gz"
    atomic_bytes(archive, _tar_bytes(files))
    identity = verify_package(archive)
    atomic_bytes(archive.with_suffix(archive.suffix + ".sha256"), f"{identity['archive_sha256']}  {archive.name}\n".encode())
    atomic_bytes(PACKAGE_DIR / "package-verification.json", stable(identity))
    return identity


def verify_package(archive: Path) -> dict[str, Any]:
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        for member in members:
            pure = PurePosixPath(member.name)
            if member.name.startswith(("/", "\\")) or ".." in pure.parts or not member.isfile():
                raise ValueError(f"UNSAFE_ARCHIVE_MEMBER:{member.name}")
        values = {member.name: bundle.extractfile(member).read() for member in members}
    manifest_name = "phase-e2.4/manifests/package-manifest.json"
    manifest = json.loads(values.pop(manifest_name))
    actual = {name: {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()} for name, value in sorted(values.items())}
    if actual != manifest["files"]:
        raise ValueError("INTERNAL_MANIFEST_MISMATCH")
    scan = publication_scan({**values, manifest_name: stable(manifest)})
    if scan["status"] != "PASS":
        raise ValueError(f"ARCHIVE_PUBLICATION_SCAN_FAILED:{scan['findings']}")
    return {"schema_version": "phase-e2.4-package-verification-v1", "status": "PASS", "archive": archive.name,
            "archive_bytes": archive.stat().st_size, "archive_sha256": sha256(archive), "members": len(values) + 1,
            "internal_manifest_verified": True, "safe_members": True, "publication_scan": scan}


def write_retrieval_verified(ledger: Path, output: Path) -> dict[str, Any]:
    value = json.loads(ledger.read_text(encoding="utf-8"))
    items = value.get("items", [])
    valid = bool(items) and all(item.get("remote_sha256") == item.get("local_sha256") and item.get("bytes", 0) > 0 for item in items)
    result = {"schema_version": "phase-e2.4-retrieval-verification-v1", "status": "VERIFIED" if valid else "FAILED",
              "required_items": len(items), "all_hashes_match": valid, "instance_destruction_permitted": valid}
    atomic_bytes(output, stable(result))
    if not valid:
        raise ValueError("RETRIEVAL_NOT_VERIFIED")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    audit_parser = sub.add_parser("audit"); audit_parser.add_argument("--rebuild-dir", type=Path, required=True)
    sub.add_parser("package")
    verify_parser = sub.add_parser("verify-package"); verify_parser.add_argument("--archive", type=Path, required=True)
    retrieval = sub.add_parser("retrieval-marker"); retrieval.add_argument("--ledger", type=Path, required=True); retrieval.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "audit": result = audit(args.rebuild_dir)
    elif args.command == "package": result = build_package()
    elif args.command == "verify-package": result = verify_package(args.archive)
    else: result = write_retrieval_verified(args.ledger, args.output)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
