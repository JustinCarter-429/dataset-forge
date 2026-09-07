"""Group-aware split assertions used by the deterministic builder."""
from __future__ import annotations

from typing import Any, Callable


def assert_cluster_atomic(train: list[dict[str, Any]], validation: list[dict[str, Any]], cluster_key: Callable[[dict[str, Any]], str]) -> None:
    train_clusters = {cluster_key(row) for row in train}
    validation_clusters = {cluster_key(row) for row in validation}
    overlap = train_clusters & validation_clusters
    if overlap:
        raise ValueError(f"CROSS_SPLIT_CLUSTER_LEAKAGE:{len(overlap)}")
