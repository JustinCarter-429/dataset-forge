"""Conflict quarantine and deterministic exact-target retention."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from .canonicalization import canonical_bytes, canonical_target, fingerprint


def canonical_record(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("canonical_record", row)


def stable_retention_key(row: dict[str, Any]) -> tuple[str, ...]:
    item = canonical_record(row)
    provenance = item.get("provenance", {})
    return (
        str(provenance.get("source_dataset", "")),
        str(provenance.get("source_split", "")),
        str(provenance.get("source_record_id", "")),
        str(item.get("record_id", "")),
    )


def resolve(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[fingerprint(canonical_record(row)["input"])].append(row)
    retained: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    for input_fp in sorted(groups):
        group = sorted(groups[input_fp], key=stable_retention_key)
        decisions = sorted({str(canonical_record(row)["target"]["decision"]).casefold() for row in group})
        common = {
            "canonical_input_fingerprint": input_fp,
            "record_count": len(group),
            "decisions": decisions,
            "dataset_types": sorted({str(canonical_record(row)["input"].get("dataset_spec", {}).get("dataset_type")) for row in group}),
            "mutation_categories": sorted({str(row.get("native", {}).get("mutation_family", "MISSING")) for row in group}),
            "provenance": [
                {
                    "record_id": canonical_record(row).get("record_id"),
                    "source_dataset": canonical_record(row).get("provenance", {}).get("source_dataset"),
                    "source_record_id": canonical_record(row).get("provenance", {}).get("source_record_id"),
                    "source_split": canonical_record(row).get("provenance", {}).get("source_split"),
                }
                for row in group
            ],
        }
        if any(not canonical_record(row).get("record_id") or not canonical_record(row).get("provenance", {}).get("source_record_id") for row in group):
            quarantined.append({**common, "quarantine_reason": "MISSING_PROVENANCE"})
            continue
        if len(decisions) > 1:
            quarantined.append({**common, "quarantine_reason": "CONFLICTING_DECISIONS"})
            continue
        targets = {canonical_bytes(canonical_target(canonical_record(row)["target"])) for row in group}
        if len(targets) > 1:
            quarantined.append({**common, "quarantine_reason": "SAME_DECISION_MATERIAL_TARGET_DIFFERENCE"})
            continue
        retained.append(group[0])
        if len(group) > 1:
            quarantined.append({
                **common,
                "retained_record_id": canonical_record(group[0]).get("record_id"),
                "quarantined_record_ids": [canonical_record(row).get("record_id") for row in group[1:]],
                "quarantine_reason": "EXACT_DUPLICATE_TARGET",
                "record_count": len(group) - 1,
            })
    return retained, quarantined
