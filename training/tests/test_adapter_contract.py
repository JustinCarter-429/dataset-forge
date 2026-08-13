from __future__ import annotations

from typing import Any

from dataset_forge_critic.adapters.base import DatasetAdapter
from dataset_forge_critic.provenance import stable_record_id
from dataset_forge_critic.schemas import CanonicalCriticRecord, TaskFamily


class FakeAdapter(DatasetAdapter):
    dataset_name = "fake"
    task_family = TaskFamily.GROUNDING
    adapter_version = "1.0.0"
    transform_version = "1.0.0"

    def validate_raw_record(self, raw_record: dict[str, Any]) -> None:
        if "id" not in raw_record:
            raise ValueError("raw record requires id")

    def transform(self, raw_record: dict[str, Any]) -> list[CanonicalCriticRecord]:
        return [CanonicalCriticRecord.model_validate({
            "record_id": stable_record_id(self.dataset_name, "train", raw_record["id"], self.transform_version),
            "task_family": self.task_family,
            "input": {"candidate_record": {"text": raw_record.get("text", "")}},
            "provenance": {"source_dataset": self.dataset_name, "source_split": "train",
                           "source_record_id": raw_record["id"], "adapter_name": "fake",
                           "adapter_version": self.adapter_version, "transform_version": self.transform_version},
        })]


def test_adapter_is_deterministic_and_populates_provenance():
    adapter = FakeAdapter()
    first = adapter.transform_validated({"id": "7", "text": "tiny fixture"})
    second = adapter.transform_validated({"id": "7", "text": "tiny fixture"})
    assert first == second
    assert first[0].provenance.adapter_version == "1.0.0"
    assert first[0].task_family is TaskFamily.GROUNDING
