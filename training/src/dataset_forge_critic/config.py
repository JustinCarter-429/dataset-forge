"""Local YAML configuration contracts; loading never contacts remote services."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .schemas import TaskFamily


class ModelConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: str = Field(min_length=1)
    role: str = Field(min_length=1)
    config_version: str = Field(min_length=1)


class DatasetRegistryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    huggingface_id: str = Field(min_length=1)
    enabled: bool
    intended_task_family: TaskFamily
    priority: int = Field(ge=1)
    adapter: str = Field(min_length=1)
    license_status: str = Field(pattern=r"^(unverified|unknown|verified)$")
    notes: str = Field(min_length=1)


class DatasetRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    registry_version: str = Field(min_length=1)
    datasets: list[DatasetRegistryEntry] = Field(min_length=1)

    @field_validator("datasets")
    @classmethod
    def unique_names(cls, value: list[DatasetRegistryEntry]) -> list[DatasetRegistryEntry]:
        names = [entry.name for entry in value]
        if len(names) != len(set(names)):
            raise ValueError("datasets contains duplicate dataset names")
        return value


def _load_yaml(path: Path) -> dict:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Unable to read configuration file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Configuration file {path} must contain a YAML mapping")
    return value


def load_model_configuration(path: Path) -> ModelConfiguration:
    return ModelConfiguration.model_validate(_load_yaml(path))


def load_dataset_registry(path: Path) -> DatasetRegistry:
    return DatasetRegistry.model_validate(_load_yaml(path))
