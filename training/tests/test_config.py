from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from dataset_forge_critic.config import DatasetRegistry, ModelConfiguration, load_dataset_registry, load_model_configuration


ROOT = Path(__file__).resolve().parents[1]


def test_valid_model_configuration():
    config = load_model_configuration(ROOT / "configs" / "model.yaml")
    assert config.model_id == "google/gemma-4-E4B-it"


def test_missing_model_id_rejected():
    with pytest.raises(ValidationError, match="model_id"):
        ModelConfiguration.model_validate({"role": "dataset_forge_critic", "config_version": "1.0.0"})


def test_valid_dataset_registry_has_all_six_candidates():
    registry = load_dataset_registry(ROOT / "configs" / "datasets.yaml")
    assert len(registry.datasets) == 6
    assert all(not dataset.enabled for dataset in registry.datasets)


def test_malformed_registry_entry_rejected():
    with pytest.raises(ValidationError, match="huggingface_id"):
        DatasetRegistry.model_validate({"registry_version": "1.0.0", "datasets": [{"name": "x"}]})


def test_duplicate_dataset_configuration_rejected():
    entry = {"name": "x", "provider": "p", "huggingface_id": "p/x", "enabled": False,
             "intended_task_family": "grounding", "priority": 1, "adapter": "x",
             "license_status": "unverified", "notes": "x"}
    with pytest.raises(ValidationError, match="duplicate"):
        DatasetRegistry.model_validate({"registry_version": "1.0.0", "datasets": [entry, entry]})
