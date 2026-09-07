"""Build the deterministic Phase E2.1 remediated corpus and evidence."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import shutil
import sys
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve()
WORKSPACE = HERE.parents[6]
sys.path.insert(0, str(WORKSPACE / "training" / "src"))

from dataset_forge_critic.e2_contract import FIELDS, canonical_json, parse_strict_response, target_from_record  # noqa: E402
from dataset_forge_critic.tokenization import tokenize_record  # noqa: E402

from .canonicalization import CANONICALIZATION_VERSION, fingerprint
from .clustering import input_fingerprint
from .conflicts import canonical_record, resolve
from .injection import generate
from .validation import gate, metadata, overlap_counts, redact, scan_cross_split

BUILD_VERSION = "phase-e2.1-builder-v1"
CONFIDENCE_NOTICE = "UNCALIBRATED_LABEL_AUTHORITY_PLACEHOLDER"
DEFAULT_OUTPUT = HERE.parents[2]


def stable(value: Any, *, pretty: bool = False) -> str:
    if pretty:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = "".join(stable(row) for row in sorted(rows, key=lambda row: row["canonical_record"]["record_id"]))
    atomic_write(path, payload.encode("utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def native_to_contract(row: dict[str, Any]) -> dict[str, Any]:
    converted = json.loads(json.dumps(row))
    item = converted["canonical_record"]
    original = item["target"]
    item["target"] = target_from_record(row)
    item["supervision"] = {"available_targets": list(FIELDS)}
    converted["remediation"] = {
        "injection_related": False,
        "semantic_group": "native",
        "dataset_type": item["input"].get("dataset_spec", {}).get("dataset_type"),
        "template_family_id": converted.get("native", {}).get("template_family_id"),
        "source_family_id": converted.get("native", {}).get("source_family_id"),
        "mutation_family": converted.get("native", {}).get("mutation_family"),
        "generator_seed": item.get("provenance", {}).get("source_record_id"),
        "original_native_target_fingerprint": fingerprint(original),
    }
    return converted


class Processor:
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def apply_chat_template(self, *args: Any, **kwargs: Any) -> Any:
        return self.tokenizer.apply_chat_template(*args, **kwargs)


def tokenization_audit(rows: list[dict[str, Any]], tokenizer_path: Path) -> dict[str, Any]:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    processor = Processor(tokenizer)
    counts: Counter[str] = Counter()
    lengths: list[int] = []
    for row in rows:
        tokenized = tokenize_record(row, "native", processor, 2048)
        target_text = canonical_json(row["canonical_record"]["target"])
        parse_strict_response(target_text)
        decoded = tokenizer.decode(tokenized.input_ids[tokenized.prompt_tokens:-2], skip_special_tokens=False)
        counts["strict_target"] += 1
        counts["supervised_closing_brace"] += decoded.endswith("}")
        counts["supervised_turn_terminator"] += tokenized.input_ids[-2] == 106 and tokenized.labels[-2] == 106
        counts["supervised_eos"] += tokenized.input_ids[-1] == 1 and tokenized.labels[-1] == 1
        counts["fully_masked_completions"] += tokenized.supervised_tokens == 0
        lengths.append(len(tokenized.input_ids))
    return {**dict(counts), "records": len(rows), "max_sequence_tokens": max(lengths, default=0), "tokenizer_json_sha256": sha256(tokenizer_path / "tokenizer.json"), "chat_template_sha256": sha256(tokenizer_path / "chat_template.jinja")}


def deterministic_tar_gz(root: Path, paths: list[Path], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
            name = path.relative_to(root).as_posix()
            info = tarfile.TarInfo(name)
            data = path.read_bytes()
            info.size = len(data); info.mtime = 0; info.mode = 0o644; info.uid = info.gid = 0; info.uname = info.gname = ""
            archive.addfile(info, io.BytesIO(data))
    with destination.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            zipped.write(buffer.getvalue())


def build(output: Path, *, package: bool = True) -> dict[str, Any]:
    source_paths = [WORKSPACE / "training/data/native/native-rule-v2/train.jsonl", WORKSPACE / "training/data/native/native-rule-v2/validation.jsonl"]
    original = [row for path in source_paths for row in read_jsonl(path)]
    retained, quarantine_groups = resolve(original)
    native = [native_to_contract(row) for row in retained]
    new_train = generate("train", {"benign": 1200, "repairable": 1200, "malicious": 936}, 210000)
    validation = generate("validation", {"benign": 250, "repairable": 250, "malicious": 250}, 910000)
    train = native + new_train

    quarantine_rows = sum(item["record_count"] for item in quarantine_groups)
    conflict_groups = [item for item in quarantine_groups if item["quarantine_reason"] == "CONFLICTING_DECISIONS"]
    exact_groups = [item for item in quarantine_groups if item["quarantine_reason"] == "EXACT_DUPLICATE_TARGET"]
    differing_groups = [item for item in quarantine_groups if item["quarantine_reason"] == "SAME_DECISION_MATERIAL_TARGET_DIFFERENCE"]
    write_jsonl(output / "train/corpus.jsonl", train)
    write_jsonl(output / "validation/corpus.jsonl", validation)
    atomic_write(output / "quarantine/groups.jsonl", "".join(stable(item) for item in quarantine_groups).encode("utf-8"))

    heldout = read_jsonl(WORKSPACE / "training/evaluation/heldout-corpus.jsonl")
    controlled = []
    for path in sorted((WORKSPACE / "training/evaluation/e2/controlled-overfit/data").glob("*.jsonl")):
        controlled.extend(read_jsonl(path))
    phase_e1_overlap = overlap_counts(train + validation, heldout)
    controlled_overlap = overlap_counts(train + validation, controlled)
    overlap = scan_cross_split(train, validation)
    tokenizer = WORKSPACE / "training/evaluation/e2/artifacts/tokenizer-reference"
    tokenization = tokenization_audit(train + validation, tokenizer)
    final_gate = gate(train, validation, overlap, phase_e1_overlap, controlled_overlap, tokenization)
    test_evidence_source = DEFAULT_OUTPUT / "manifests/focused-test-evidence.json"
    reproducibility_source = DEFAULT_OUTPUT / "manifests/reproducibility-check.json"
    test_evidence = json.loads(test_evidence_source.read_text(encoding="utf-8")) if test_evidence_source.exists() else {"status": "MISSING"}
    reproducibility = json.loads(reproducibility_source.read_text(encoding="utf-8")) if reproducibility_source.exists() else {"status": "MISSING"}
    final_gate["checks"]["focused_tests_pass"] = test_evidence.get("status") == "PASS" and test_evidence.get("total_tests") == 32
    final_gate["checks"]["reproducible_rebuild_byte_identical"] = reproducibility.get("status") == "PASS" and reproducibility.get("all_byte_identical") is True
    final_gate["status"] = "PASS" if all(final_gate["checks"].values()) else "FAIL"

    source_hashes = {path.relative_to(WORKSPACE).as_posix(): sha256(path) for path in source_paths}
    audit = {
        "schema_version": "phase-e2.1-audit-v1", "builder_version": BUILD_VERSION,
        "canonicalization_version": CANONICALIZATION_VERSION, "fixed_seeds": {"train": 210000, "validation": 910000},
        "confidence_notice": CONFIDENCE_NOTICE, "starting_records": len(original),
        "quarantine": {"groups": len(quarantine_groups), "records": quarantine_rows, "conflicting_groups": len(conflict_groups), "conflicting_records": sum(item["record_count"] for item in conflict_groups), "exact_duplicate_groups": len(exact_groups), "exact_duplicate_records": sum(item["record_count"] for item in exact_groups), "differing_target_groups": len(differing_groups), "differing_target_records": sum(item["record_count"] for item in differing_groups)},
        "remaining_defensible_native_records": len(native), "new_deterministic_records": len(new_train) + len(validation),
        "overlap": overlap, "phase_e1_overlap": phase_e1_overlap, "controlled_overfit_overlap": controlled_overlap,
        "tokenization": tokenization, "tests": {"status": test_evidence.get("status"), "total": test_evidence.get("total_tests", 0)},
        "reproducibility": {"status": reproducibility.get("status"), "byte_identical": reproducibility.get("all_byte_identical", False)},
        "gate": final_gate, "test_split_accessed": False,
    }
    atomic_write(output / "reports/audit.json", stable(audit, pretty=True).encode("utf-8"))

    files = [output / "train/corpus.jsonl", output / "validation/corpus.jsonl", output / "quarantine/groups.jsonl", output / "reports/audit.json"]
    manifest = {
        "schema_version": "phase-e2.1-reproducibility-manifest-v1", "builder_version": BUILD_VERSION,
        "canonicalization_version": CANONICALIZATION_VERSION, "source_files": source_hashes,
        "outputs": {path.relative_to(output).as_posix(): {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in files},
        "fixed_seeds": {"train": 210000, "validation": 910000}, "jaccard_threshold": 0.85,
        "target_contract_fields": list(FIELDS), "confidence_notice": CONFIDENCE_NOTICE,
        "phase_e1_immutable": True, "test_split_accessed": False,
    }
    atomic_write(output / "manifests/reproducibility-manifest.json", stable(manifest, pretty=True).encode("utf-8"))
    files.append(output / "manifests/reproducibility-manifest.json")
    if test_evidence_source.exists():
        atomic_write(output / "manifests/focused-test-evidence.json", test_evidence_source.read_bytes())
        files.append(output / "manifests/focused-test-evidence.json")
    if reproducibility_source.exists():
        atomic_write(output / "manifests/reproducibility-check.json", reproducibility_source.read_bytes())
        files.append(output / "manifests/reproducibility-check.json")
    report = f"""# Phase E2.1 corpus remediation audit\n\nStatus: **{final_gate['status']}**\n\nThe original 10,656 native train/development rows were re-evaluated under the authoritative whole-group quarantine policy. No Phase E1, controlled-overfit, or native test examples were used. Confidence `1.0` means `{CONFIDENCE_NOTICE}` and is not calibrated probability.\n\n- Conflicting groups: {len(conflict_groups)} ({sum(item['record_count'] for item in conflict_groups)} rows quarantined)\n- Exact duplicate copies removed: {sum(item['record_count'] for item in exact_groups)}\n- Material same-decision target conflicts: {sum(item['record_count'] for item in differing_groups)}\n- Defensible native survivors: {len(native)}\n- New deterministic records: {len(new_train) + len(validation)}\n- Training / validation: {len(train)} / {len(validation)}\n- Exact / canonical / near cross-split overlap: {overlap['exact_record_id_overlap']} / {overlap['canonical_input_overlap']} / {overlap['near_duplicate_pairs']}\n- Template / source-family overlap: {overlap['template_family_overlap']} / {overlap['source_family_overlap']}\n- Strict targets / termination boundaries: {tokenization['strict_target']} / {tokenization['supervised_turn_terminator']} of {len(train) + len(validation)}\n- Focused tests: {test_evidence.get('total_tests', 0)} ({test_evidence.get('status')})\n- Byte-identical independent rebuild: {reproducibility.get('all_byte_identical', False)}\n\nThe repaired corpus is development-only until explicit GPU/retraining authorization is given.\n"""
    atomic_write(output / "reports/pretraining-report.md", redact(report).encode("utf-8"))
    files.append(output / "reports/pretraining-report.md")

    if final_gate["status"] == "PASS" and package:
        destination = output / "package/phase-e2.1-clean-corpus-v1.tar.gz"
        deterministic_tar_gz(output, files, destination)
        atomic_write(output / "package/phase-e2.1-clean-corpus-v1.tar.gz.sha256", f"{sha256(destination)}  {destination.name}\n".encode())
        audit["package"] = {"path": destination.relative_to(WORKSPACE).as_posix() if output.is_relative_to(WORKSPACE) else destination.name, "bytes": destination.stat().st_size, "sha256": sha256(destination)}
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-package", action="store_true")
    parser.add_argument("--package-only", action="store_true")
    args = parser.parse_args()
    if args.package_only:
        output = args.output.resolve()
        audit = json.loads((output / "reports/audit.json").read_text(encoding="utf-8"))
        if audit.get("gate", {}).get("status") != "PASS":
            raise SystemExit("REFUSING_TO_PACKAGE_FAILED_GATE")
        paths = [
            output / "train/corpus.jsonl", output / "validation/corpus.jsonl", output / "quarantine/groups.jsonl",
            output / "reports/audit.json", output / "reports/pretraining-report.md",
            output / "manifests/reproducibility-manifest.json", output / "manifests/focused-test-evidence.json",
            output / "manifests/reproducibility-check.json",
        ]
        destination = output / "package/phase-e2.1-clean-corpus-v1.tar.gz"
        deterministic_tar_gz(output, paths, destination)
        digest = sha256(destination)
        atomic_write(output / "package/phase-e2.1-clean-corpus-v1.tar.gz.sha256", f"{digest}  {destination.name}\n".encode())
        print(stable({"status": "PASS", "bytes": destination.stat().st_size, "sha256": digest}, pretty=True), end="")
        return 0
    result = build(args.output.resolve(), package=not args.no_package)
    print(stable(result, pretty=True), end="")
    return 0 if result["gate"]["status"] == "PASS" else 4


if __name__ == "__main__":
    raise SystemExit(main())
