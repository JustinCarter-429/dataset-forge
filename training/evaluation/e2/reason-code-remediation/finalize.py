"""Build and verify the non-model Phase E2.2 remediation evidence package."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import tarfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[3]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def package_inputs() -> dict[str, Path]:
    files = {
        "phase-e2.2/build.py": HERE / "build.py",
        "phase-e2.2/finalize.py": HERE / "finalize.py",
        "phase-e2.2/scan_publication.py": HERE / "scan_publication.py",
        "phase-e2.2/record_tests.py": HERE / "record_tests.py",
        "phase-e2.2/verify_final.py": HERE / "verify_final.py",
        "phase-e2.2/smoke-config.json": HERE / "smoke-config.json",
        "phase-e2.2/EXECUTION.md": HERE / "EXECUTION.md",
        "phase-e2.2/smoke-data/train/corpus.jsonl": HERE / "smoke/train.jsonl",
        "phase-e2.2/smoke-data/validation/corpus.jsonl": HERE / "smoke/validation.jsonl",
        "phase-e2.2/reports/root-cause-report.md": HERE / "reports/root-cause-report.md",
        "phase-e2.2/reports/reason-code-audit.json": HERE / "reports/reason-code-audit.json",
        "phase-e2.2/reports/reason-code-coverage.csv": HERE / "reports/reason-code-coverage.csv",
        "phase-e2.2/reports/proposed-smoke-gates.json": HERE / "reports/proposed-smoke-gates.json",
        "phase-e2.2/reports/publication-safety-scan.json": HERE / "reports/publication-safety-scan.json",
        "phase-e2.2/reports/final-corpus-verification.json": HERE / "reports/final-corpus-verification.json",
        "phase-e2.2/reports/build.log": HERE / "reports/build.log",
        "phase-e2.2/reports/build-exit-code.txt": HERE / "reports/build-exit-code.txt",
        "phase-e2.2/logs/focused-tests.txt": HERE / "logs/public/focused-publication-final-tests.txt",
        "phase-e2.2/logs/focused-tests.xml": HERE / "logs/public/focused-publication-final-tests.xml",
        "phase-e2.2/logs/development/focused-tests-initial-failure.txt": HERE / "logs/public/focused-tests-initial-failure.txt",
        "phase-e2.2/logs/development/focused-tests-second-failure.txt": HERE / "logs/public/focused-tests-second-failure.txt",
        "phase-e2.2/logs/relevant-core-tests.txt": HERE / "logs/public/relevant-core-tests.txt",
        "phase-e2.2/logs/relevant-core-tests.xml": HERE / "logs/public/relevant-core-tests.xml",
        "phase-e2.2/logs/relevant-schemas-tests.txt": HERE / "logs/public/relevant-schemas-tests.txt",
        "phase-e2.2/logs/relevant-schemas-tests.xml": HERE / "logs/public/relevant-schemas-tests.xml",
        "phase-e2.2/logs/relevant-training-system-tests.txt": HERE / "logs/public/relevant-training-system-tests.txt",
        "phase-e2.2/logs/relevant-training-system-tests.xml": HERE / "logs/public/relevant-training-system-tests.xml",
        "phase-e2.2/logs/relevant-tail-tests.txt": HERE / "logs/public/relevant-tail-tests.txt",
        "phase-e2.2/logs/relevant-tail-tests.xml": HERE / "logs/public/relevant-tail-tests.xml",
        "phase-e2.2/logs/corpus-tests.txt": HERE / "logs/public/corpus-tests.txt",
        "phase-e2.2/logs/corpus-tests.xml": HERE / "logs/public/corpus-tests.xml",
        "phase-e2.2/manifests/reproducibility-manifest.json": HERE / "manifests/reproducibility-manifest.json",
        "phase-e2.2/manifests/final-file-hashes.json": HERE / "manifests/final-file-hashes.json",
        "phase-e2.2/manifests/test-evidence.json": HERE / "manifests/test-evidence.json",
        "phase-e2.2/runner/e2_train.py": WORKSPACE / "training/evaluation/e2/full-retraining/e2_train.py",
        "phase-e2.2/contract/e2_contract.py": WORKSPACE / "training/src/dataset_forge_critic/e2_contract.py",
        "phase-e2.2/contract/critic-system-v2.txt": WORKSPACE / "training/prompts/critic-system-v2.txt",
        "phase-e2.2/tests/test_e2_reason_code_remediation.py": WORKSPACE / "training/tests/test_e2_reason_code_remediation.py",
        "phase-e2.2/tests/test_e2_contract.py": WORKSPACE / "training/tests/test_e2_contract.py",
        "phase-e2.2/tests/test_gemma.py": WORKSPACE / "training/tests/test_gemma.py",
        "phase-e2.2/tests/test_full_retraining.py": WORKSPACE / "training/evaluation/e2/full-retraining/tests/test_full_retraining.py",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"MISSING_PACKAGE_INPUTS:{missing}")
    return files


def main() -> int:
    test_evidence = json.loads((HERE / "manifests/test-evidence.json").read_text(encoding="utf-8"))
    final_runs = [run for run in test_evidence["runs"] if run["classification"] == "final confirmation"]
    if test_evidence.get("status") != "PASS" or any(run["failed"] or run["errors"] or (run["numeric_process_exit_code"] not in (0, None)) for run in final_runs):
        raise ValueError("TEST_EVIDENCE_NOT_PASSING")
    hash_targets = {
        "training/evaluation/e2/reason-code-remediation/corpus/train.jsonl": HERE / "corpus/train.jsonl",
        "training/evaluation/e2/reason-code-remediation/corpus/validation.jsonl": HERE / "corpus/validation.jsonl",
        "training/evaluation/e2/reason-code-remediation/smoke/train.jsonl": HERE / "smoke/train.jsonl",
        "training/evaluation/e2/reason-code-remediation/smoke/validation.jsonl": HERE / "smoke/validation.jsonl",
        "training/evaluation/e2/reason-code-remediation/smoke-config.json": HERE / "smoke-config.json",
        "training/prompts/critic-system-v2.txt": WORKSPACE / "training/prompts/critic-system-v2.txt",
        "training/src/dataset_forge_critic/e2_contract.py": WORKSPACE / "training/src/dataset_forge_critic/e2_contract.py",
        "training/evaluation/e2/full-retraining/e2_train.py": WORKSPACE / "training/evaluation/e2/full-retraining/e2_train.py",
        "training/evaluation/e2/reason-code-remediation/reports/proposed-smoke-gates.json": HERE / "reports/proposed-smoke-gates.json",
        "training/evaluation/e2/reason-code-remediation/reports/reason-code-audit.json": HERE / "reports/reason-code-audit.json",
        "training/evaluation/e2/reason-code-remediation/reports/reason-code-coverage.csv": HERE / "reports/reason-code-coverage.csv",
        "training/evaluation/e2/reason-code-remediation/reports/root-cause-report.md": HERE / "reports/root-cause-report.md",
        "training/evaluation/e2/reason-code-remediation/manifests/reproducibility-manifest.json": HERE / "manifests/reproducibility-manifest.json",
    }
    final_hashes = {"schema_version": "phase-e2.2-final-file-hashes-v1", "files": {name: {"bytes": path.stat().st_size, "sha256": sha256(path)} for name, path in sorted(hash_targets.items())}}
    (HERE / "manifests/final-file-hashes.json").write_bytes(stable(final_hashes))
    inputs = package_inputs()
    payloads = {name: path.read_bytes() for name, path in inputs.items()}
    package_manifest = {
        "schema_version": "phase-e2.2-package-manifest-v1",
        "non_model_artifacts_only": True,
        "remote_execution_authorized": False,
        "files": {name: {"bytes": len(data), "sha256": sha256_bytes(data)} for name, data in sorted(payloads.items())},
    }
    manifest_bytes = stable(package_manifest)
    payloads["phase-e2.2/manifests/package-manifest.json"] = manifest_bytes
    manifest_path = HERE / "manifests/package-manifest.json"
    manifest_path.write_bytes(manifest_bytes)

    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for name, data in sorted(payloads.items()):
            info = tarfile.TarInfo(name); info.size = len(data); info.mtime = 0; info.mode = 0o644
            info.uid = info.gid = 0; info.uname = info.gname = ""
            archive.addfile(info, io.BytesIO(data))
    destination = HERE / "package/phase-e2.2-smoke-remediation.tar.gz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
        zipped.write(tar_buffer.getvalue())

    extraction = HERE / f"package/verification-extract-{int(time.time())}"
    extraction.mkdir(parents=True, exist_ok=False)
    with tarfile.open(destination, "r:gz") as archive:
        members = {member.name: member for member in archive.getmembers()}
        if set(members) != set(payloads):
            raise ValueError("PACKAGE_MEMBER_SET_MISMATCH")
        manifest_name = "phase-e2.2/manifests/package-manifest.json"
        manifest_member = members[manifest_name]
        manifest_handle = archive.extractfile(manifest_member)
        if manifest_handle is None or manifest_handle.read() != manifest_bytes:
            raise ValueError("INTERNAL_MANIFEST_MISMATCH")
        manifest_target = extraction / manifest_name
        manifest_target.parent.mkdir(parents=True, exist_ok=True); manifest_target.write_bytes(manifest_bytes)
        extracted: set[str] = set()
        for name, expected in package_manifest["files"].items():
            member = members[name]
            if not member.isfile() or member.issym() or member.islnk() or member.isdev() or member.isfifo() or Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError(f"UNSAFE_PACKAGE_MEMBER:{name}")
            handle = archive.extractfile(member)
            data = handle.read() if handle else b""
            if handle is None or sha256_bytes(data) != expected["sha256"]:
                raise ValueError(f"PACKAGE_HASH_MISMATCH:{name}")
            target = extraction / Path(name)
            target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data); extracted.add(name)
        internal = json.loads((extraction / "phase-e2.2/manifests/package-manifest.json").read_text(encoding="utf-8"))
        for name, expected in internal["files"].items():
            target = extraction / Path(name)
            if not target.is_file() or target.stat().st_size != expected["bytes"] or sha256(target) != expected["sha256"]:
                raise ValueError(f"EXTRACTED_HASH_MISMATCH:{name}")
        if extracted != set(internal["files"]):
            raise ValueError("EXTRACTED_MEMBER_SET_MISMATCH")
    digest = sha256(destination)
    (HERE / "package/phase-e2.2-smoke-remediation.tar.gz.sha256").write_text(
        f"{digest}  phase-e2.2-smoke-remediation.tar.gz\n", encoding="ascii", newline="\n"
    )
    verification = {"status": "PASS", "archive_members": len(payloads), "manifested_payload_files": len(package_manifest["files"]),
                    "missing": 0, "unexpected": 0, "mismatched": 0, "unsafe_members": 0,
                    "independent_extraction": extraction.relative_to(WORKSPACE).as_posix(), "sha256": digest}
    (HERE / "package/package-verification.json").write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": "PASS", "files": len(payloads), "package": str(destination.relative_to(WORKSPACE)), "sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
