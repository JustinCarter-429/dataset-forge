from __future__ import annotations

from dataset_forge_critic.provenance import stable_record_id


def valid_record() -> dict:
    return {
        "record_id": stable_record_id("synthetic", "train", "1", "1.0.0"),
        "task_family": "grounding",
        "input": {
            "source_context": "Password rotation is required every 90 days.",
            "candidate_record": {"question": "How often?", "answer": "Every 90 days."},
        },
        "target": {"scores": {"grounding": {
            "value": 0.0,
            "provenance": {"original_field_name": "grounded", "original_value": 0,
                           "original_scale": "0-1", "normalization_method": "identity",
                           "normalization_version": "1.0.0"},
        }}},
        "supervision": {"available_targets": ["scores.grounding"]},
        "provenance": {"source_dataset": "synthetic", "source_split": "train", "source_record_id": "1",
                       "adapter_name": "fixture", "adapter_version": "1.0.0", "transform_version": "1.0.0"},
    }
