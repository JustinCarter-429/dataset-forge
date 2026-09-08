"""Fail closed when proposed Phase E2.2 publication material contains private data."""
from __future__ import annotations

import json
import re
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[3]
TEXT_PATTERNS = {
    "ipv4_address": re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])"),
    "windows_user_path": re.compile(r"(?i)[a-z]:[\\/]users[\\/][^\\/\s]+"),
    "absolute_windows_path": re.compile(r"(?i)\b[a-z]:[\\/](?![<>])"),
    "private_remote_path": re.compile(r"(?i)(?:/home/[^/\s]+|/root|/workspace)(?=/|\s|[\"'])"),
    "private_key": re.compile(r"(?i)(?:-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----|id_(?:rsa|ed25519|ecdsa))"),
    "huggingface_token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    "credential_assignment": re.compile(r"(?i)\b(?:api[_-]?key|access[_-]?token|secret[_-]?key|password|cookie|authorization)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{8,}"),
    "authorization_header": re.compile(r"(?i)\bauthorization\s*:\s*(?:bearer|basic)\s+\S+"),
    "signed_url": re.compile(r"(?i)[?&](?:X-Amz-(?:Signature|Credential)|Signature|sig|token)="),
    "ssh_port": re.compile(r"(?i)(?:\bssh\b[^\r\n]{0,120}\s-p\s+\d{2,5}\b|\bssh_port\b\s*[:=]\s*\d{2,5})"),
    "full_checkpoint_path": re.compile(r"(?i)(?:[a-z]:[\\/]|/(?:home|root|workspace)/)[^\r\n\"']*checkpoint[^\r\n\"']*"),
    "raw_private_prompt_field": re.compile(r'(?i)"(?:input_prompt|rendered_conversation)"\s*:'),
}
FORBIDDEN_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".onnx", ".gguf", ".pem", ".key"}


def proposed_files() -> list[Path]:
    relative = [
        "training/evaluation/e2/corpus-remediation/src/corpus_remediation/validation.py",
        "training/evaluation/e2/full-retraining/e2_train.py", "training/prompts/critic-system-v2.txt",
        "training/src/dataset_forge_critic/e2_contract.py", "training/src/dataset_forge_critic/gemma.py",
        "training/tests/test_e2_contract.py", "training/tests/test_e2_reason_code_remediation.py", "training/tests/test_training_system.py",
    ]
    paths = [WORKSPACE / item for item in relative]
    for folder in (HERE / "reports", HERE / "manifests", HERE / "logs/public"):
        if folder.exists():
            paths.extend(path for path in folder.rglob("*") if path.is_file())
    paths.extend([HERE / name for name in ("build.py", "finalize.py", "record_tests.py", "scan_publication.py", "verify_final.py", "EXECUTION.md", "smoke-config.json", "smoke/train.jsonl", "smoke/validation.jsonl")])
    report = HERE / "reports/publication-safety-scan.json"
    return sorted(set(path for path in paths if path.is_file() and path != report))


def main() -> int:
    unresolved: list[dict[str, object]] = []; reviewed: list[dict[str, object]] = []; files = proposed_files(); corpus_records = 0; archive_members = 0
    for path in files:
        label = path.relative_to(WORKSPACE).as_posix()
        if path.suffix.lower() in FORBIDDEN_SUFFIXES or any(part in {"__pycache__", ".venv", "venv"} for part in path.parts):
            unresolved.append({"category": "model_weight_cache_or_credential_file", "file": label, "location": "member name", "remediation_status": "unresolved"})
            continue
        data = path.read_bytes()
        if b"\x00" in data:
            unresolved.append({"category": "unexpected_binary", "file": label, "location": "file content", "remediation_status": "unresolved"}); continue
        text = data.decode("utf-8", errors="replace")
        if path.name == "scan_publication.py":
            reviewed.append({"category": "scanner_signature_literals", "file": label, "location": "pattern declarations", "remediation_status": "cleared: executable scanner definitions, not credential values"})
        else:
            for category, pattern in TEXT_PATTERNS.items():
                for match in pattern.finditer(text):
                    unresolved.append({"category": category, "file": label, "location": f"line {text.count(chr(10), 0, match.start()) + 1}", "remediation_status": "unresolved"})
        if path.name == "corpus.jsonl" or path.parent.name == "smoke" and path.suffix == ".jsonl":
            for line_number, line in enumerate(text.splitlines(), 1):
                if not line.strip(): continue
                corpus_records += 1; row = json.loads(line); provenance = row["canonical_record"]["provenance"]
                if provenance.get("license") != "CC0-1.0" or provenance.get("source_dataset") not in {"phase-e2.2-deterministic-reason-codes", "phase-e2.1-deterministic-injection"}:
                    unresolved.append({"category": "unapproved_corpus_provenance", "file": label, "location": f"record line {line_number}", "remediation_status": "unresolved"})
            reviewed.append({"category": "proprietary_source_content", "file": label, "location": "all JSONL records", "remediation_status": "cleared: deterministic CC0-only provenance"})
    package = HERE / "package/phase-e2.2-smoke-remediation.tar.gz"
    if package.is_file():
        with tarfile.open(package, "r:gz") as archive:
            for member in archive.getmembers():
                archive_members += 1; name = member.name
                if not member.isfile() or member.issym() or member.islnk() or member.isdev() or member.isfifo() or Path(name).is_absolute() or ".." in Path(name).parts:
                    unresolved.append({"category": "unsafe_archive_member", "file": "phase-e2.2-smoke-remediation.tar.gz", "location": name, "remediation_status": "unresolved"}); continue
                if Path(name).suffix.lower() in FORBIDDEN_SUFFIXES or any(part in {"__pycache__", ".venv", "venv"} for part in Path(name).parts):
                    unresolved.append({"category": "model_weight_cache_or_credential_file", "file": "phase-e2.2-smoke-remediation.tar.gz", "location": name, "remediation_status": "unresolved"}); continue
                handle = archive.extractfile(member); data = handle.read() if handle else b""
                if b"\x00" in data:
                    unresolved.append({"category": "unexpected_binary", "file": "phase-e2.2-smoke-remediation.tar.gz", "location": name, "remediation_status": "unresolved"}); continue
                if name.endswith("scan_publication.py"):
                    continue
                text = data.decode("utf-8", errors="replace")
                for category, pattern in TEXT_PATTERNS.items():
                    for match in pattern.finditer(text):
                        unresolved.append({"category": category, "file": "phase-e2.2-smoke-remediation.tar.gz", "location": f"member {name}, line {text.count(chr(10), 0, match.start()) + 1}", "remediation_status": "unresolved"})
    report = {
        "schema_version": "phase-e2.2-publication-safety-scan-v2", "status": "PASS" if not unresolved else "FAIL",
        "files_scanned": len(files), "corpus_records_reviewed": corpus_records, "archive_member_names_preflighted": True,
        "archive_members_content_scanned": archive_members,
        "image_metadata_files": 0, "resolved_or_cleared_findings": reviewed, "unresolved_findings": unresolved,
        "categories_checked": sorted([*TEXT_PATTERNS, "model_or_adapter_weights", "cache_or_virtual_environment", "proprietary_source_content", "unexpected_binary"]),
    }
    destination = HERE / "reports/publication-safety-scan.json"
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": report["status"], "files_scanned": len(files), "corpus_records_reviewed": corpus_records, "archive_members_scanned": archive_members, "unresolved": len(unresolved)}, sort_keys=True))
    return 0 if not unresolved else 3


if __name__ == "__main__":
    raise SystemExit(main())
