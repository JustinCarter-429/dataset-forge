"""Scan Phase E1 publication artifacts for secrets and private path/source leakage."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


TEXT_SUFFIXES = {".html", ".json", ".csv", ".md", ".txt"}
PATTERNS = {
    "private_key_material": re.compile(r"-----BEGIN (?:OPENSSH|RSA|EC|DSA) PRIVATE KEY-----", re.I),
    "openai_style_token": re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    "hugging_face_token": re.compile(r"\bhf_[A-Za-z0-9]{16,}\b"),
    "authorization_bearer": re.compile(r"\bAuthorization\s*:\s*Bearer\s+\S+", re.I),
    "signed_url": re.compile(r"(?:X-Amz-(?:Signature|Credential)|[?&](?:sig|signature|token)=)", re.I),
    "ssh_identity": re.compile(r"(?:root@|\.ssh[\\/])", re.I),
    "private_windows_path": re.compile(r"[A-Za-z]:[\\/](?:Users|Documents)[\\/]", re.I),
    "private_unix_path": re.compile(r"/(?:workspace|home|root)/", re.I),
}


def _heldout_source_fragments(corpus: Path) -> list[str]:
    fragments: list[str] = []
    for line in corpus.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        source = row.get("canonical_record", {}).get("input", {}).get("source_context")
        if isinstance(source, str) and len(source.strip()) >= 32:
            fragments.append(source.strip())
    return sorted(set(fragments))


def scan(root: Path, corpus: Path, forbidden_literals: list[str]) -> dict[str, Any]:
    excluded_logs = {
        "local-test-runner-stdout.txt",
        "local-test-runner-stderr.txt",
    }
    text_files = sorted(
        path for path in root.rglob("*")
        if path.is_file()
        and path.suffix.casefold() in TEXT_SUFFIXES
        and path.name not in excluded_logs
        and path.name != "publication-safety-scan.json"
    )
    findings: list[dict[str, str]] = []
    source_fragments = _heldout_source_fragments(corpus)
    for path in text_files:
        relative = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8", errors="replace")
        for name, pattern in PATTERNS.items():
            if pattern.search(text):
                findings.append({"file": relative, "finding": name})
        if any(literal.casefold() in text.casefold() for literal in forbidden_literals):
            findings.append({"file": relative, "finding": "operator_supplied_forbidden_literal"})
        if relative.startswith("reproducibility/"):
            continue
        if any(fragment in text for fragment in source_fragments):
            findings.append({"file": relative, "finding": "verbatim_private_source_text"})

    image_metadata: list[dict[str, Any]] = []
    try:
        from PIL import Image
        for path in sorted(root.rglob("*.png")):
            with Image.open(path) as image:
                metadata = {str(key): str(value) for key, value in image.info.items()}
                joined = "\n".join(metadata.values())
                relative = path.relative_to(root).as_posix()
                for name, pattern in PATTERNS.items():
                    if pattern.search(joined):
                        findings.append({"file": relative, "finding": f"png_metadata_{name}"})
                if any(literal.casefold() in joined.casefold() for literal in forbidden_literals):
                    findings.append({"file": relative, "finding": "png_metadata_operator_supplied_forbidden_literal"})
                image_metadata.append({
                    "file": relative,
                    "width": image.width,
                    "height": image.height,
                    "metadata_keys": sorted(metadata),
                })
    except ImportError:
        findings.append({"file": "screenshots/*.png", "finding": "png_metadata_not_checked_pillow_unavailable"})

    return {
        "schema_version": "phase-e1-publication-safety-scan-v1",
        "status": "passed" if not findings else "failed",
        "text_files_checked": [path.relative_to(root).as_posix() for path in text_files],
        "heldout_source_fragments_checked": len(source_fragments),
        "png_files_checked": image_metadata,
        "findings": findings,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--forbidden-literal", action="append", default=[])
    args = parser.parse_args()
    result = scan(args.root, args.corpus, args.forbidden_literal)
    destination = args.root / "logs" / "publication-safety-scan.json"
    with destination.open("w", encoding="utf-8", newline="\n") as target:
        target.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"],
        "text_file_count": len(result["text_files_checked"]),
        "heldout_source_fragments_checked": result["heldout_source_fragments_checked"],
        "png_file_count": len(result["png_files_checked"]),
        "findings": result["findings"],
    }, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
