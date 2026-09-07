"""Exact/family/near-duplicate clustering primitives."""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from .canonicalization import canonical_bytes, fingerprint

THRESHOLD = 0.85


def token_set(value: Any) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z0-9]+", canonical_bytes(value).decode("utf-8").casefold()))


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def near_clusters(inputs: list[Any], threshold: float = THRESHOLD) -> list[list[int]]:
    """Transitive clusters using exact normalized-token Jaccard."""
    sets = [token_set(value) for value in inputs]
    lengths: dict[int, list[int]] = defaultdict(list)
    for index, words in enumerate(sets):
        lengths[len(words)].append(index)
    union = UnionFind(len(inputs))
    for right, right_words in enumerate(sets):
        size = len(right_words)
        lower = int(size * threshold)
        upper = int(size / threshold) + 1 if threshold else max(lengths, default=0)
        for length in range(lower, upper + 1):
            for left in lengths.get(length, []):
                if left >= right:
                    continue
                if jaccard(sets[left], right_words) >= threshold:
                    union.union(left, right)
    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(inputs)):
        groups[union.find(index)].append(index)
    return [groups[key] for key in sorted(groups)]


def cross_split_near(train: list[dict[str, Any]], validation: list[dict[str, Any]], threshold: float = THRESHOLD) -> tuple[int, int, list[dict[str, Any]]]:
    train_sets = [(row["canonical_record"]["record_id"], token_set(row["canonical_record"]["input"])) for row in train]
    by_length: dict[int, list[tuple[str, frozenset[str]]]] = defaultdict(list)
    for item in train_sets:
        by_length[len(item[1])].append(item)
    count = comparisons = 0
    examples: list[dict[str, Any]] = []
    for row in validation:
        record_id = row["canonical_record"]["record_id"]
        words = token_set(row["canonical_record"]["input"])
        lower, upper = int(len(words) * threshold), int(len(words) / threshold) + 1
        for length in range(lower, upper + 1):
            for train_id, train_words in by_length.get(length, []):
                comparisons += 1
                score = jaccard(train_words, words)
                if score >= threshold:
                    count += 1
                    if len(examples) < 20:
                        examples.append({"train_record_id": train_id, "validation_record_id": record_id, "jaccard": score})
    return count, comparisons, examples


def input_fingerprint(row: dict[str, Any]) -> str:
    return fingerprint(row["canonical_record"]["input"])
