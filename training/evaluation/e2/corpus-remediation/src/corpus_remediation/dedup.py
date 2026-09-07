"""Explicit exact-deduplication facade for audit tooling."""
from __future__ import annotations

from typing import Any

from .conflicts import resolve


def deduplicate(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply conflict quarantine before deterministic exact deduplication."""
    return resolve(rows)
