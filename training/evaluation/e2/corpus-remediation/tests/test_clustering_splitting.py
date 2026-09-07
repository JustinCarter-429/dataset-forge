from __future__ import annotations

import pytest

from corpus_remediation.clustering import cross_split_near, near_clusters
from corpus_remediation.splitting import assert_cluster_atomic


def record(record_id, text, cluster):
    return {"canonical_record": {"record_id": record_id, "input": {"text": text}}, "cluster": cluster}


def test_near_duplicate_clustering():
    values = [{"text": "alpha beta gamma delta epsilon"}, {"text": "alpha beta gamma delta epsilon zeta"}, {"text": "river stone copper"}]
    groups = near_clusters(values, 0.80)
    assert [0, 1] in groups and [2] in groups


def test_cluster_atomic_splitting():
    assert_cluster_atomic([record("a", "one", "c1")], [record("b", "two", "c2")], lambda item: item["cluster"])


def test_no_cross_split_cluster_leakage():
    with pytest.raises(ValueError, match="CROSS_SPLIT_CLUSTER_LEAKAGE"):
        assert_cluster_atomic([record("a", "one", "c1")], [record("b", "two", "c1")], lambda item: item["cluster"])


def test_overlap_scanner_detects_near_pair():
    train = [record("a", "alpha beta gamma delta epsilon", "c1")]
    validation = [record("b", "alpha beta gamma delta epsilon zeta", "c2")]
    count, comparisons, examples = cross_split_near(train, validation, 0.80)
    assert count == 1 and comparisons == 1 and examples[0]["train_record_id"] == "a"
