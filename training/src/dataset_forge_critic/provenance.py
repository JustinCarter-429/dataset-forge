"""Stable identifiers and provenance helpers; no external services are used."""

from __future__ import annotations

from hashlib import sha256


def stable_record_id(source_dataset: str, source_split: str | None, source_record_id: str | None,
                     transform_version: str, candidate_index: int | None = None) -> str:
    """Derive a deterministic non-content ID from source identity and transformation version."""
    parts = (source_dataset, source_split or "", source_record_id or "", transform_version,
             "" if candidate_index is None else str(candidate_index))
    digest = sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:32]
    return f"dffc_{digest}"
