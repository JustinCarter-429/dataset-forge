"""Strict, meaning-preserving canonicalization for Phase E2.1."""
from __future__ import annotations

import hashlib
import json
import unicodedata
from typing import Any

CANONICALIZATION_VERSION = "e2.1-nfc-json-v1"


def normalize(value: Any) -> Any:
    """Normalize representation without deleting semantic content."""
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n")).strip()
    if isinstance(value, list):
        return [normalize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): normalize(value[key]) for key in sorted(value)}
    return value


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(normalize(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def canonical_target(target: dict[str, Any]) -> dict[str, Any]:
    value = normalize(target)
    if isinstance(value.get("reason_codes"), list):
        value["reason_codes"] = sorted(dict.fromkeys(value["reason_codes"]))
    return value
