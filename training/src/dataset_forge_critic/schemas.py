"""Typed, source-preserving canonical records for future critic training."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CANONICAL_SCHEMA_VERSION = "1.0.0"


class TaskFamily(str, Enum):
    """Supported critic supervision families."""

    PREFERENCE = "preference"
    RANKING = "ranking"
    MULTIDIMENSIONAL_QUALITY = "multidimensional_quality"
    GROUNDING = "grounding"
    HALLUCINATION_DETECTION = "hallucination_detection"
    DATASET_FORGE_NATIVE = "dataset_forge_native"


class CriticDecision(str, Enum):
    ACCEPT = "ACCEPT"
    REVISE = "REVISE"
    REJECT = "REJECT"


class ScoreProvenance(BaseModel):
    """Retains source-scale information behind an optional normalized score."""

    model_config = ConfigDict(extra="forbid")
    original_field_name: str = Field(min_length=1)
    original_value: str | int | float | bool | None
    original_scale: str | None = None
    normalization_method: str = Field(min_length=1)
    normalization_version: str = Field(min_length=1)


class NormalizedScore(BaseModel):
    """A normalized score; zero is a real value, not missing supervision."""

    model_config = ConfigDict(extra="forbid")
    value: float = Field(ge=0.0, le=1.0)
    provenance: ScoreProvenance


class CriticScores(BaseModel):
    model_config = ConfigDict(extra="forbid")
    grounding: NormalizedScore | None = None
    correctness: NormalizedScore | None = None
    helpfulness: NormalizedScore | None = None
    coherence: NormalizedScore | None = None
    complexity: NormalizedScore | None = None
    verbosity: NormalizedScore | None = None
    usefulness: NormalizedScore | None = None
    novelty: NormalizedScore | None = None
    difficulty_fit: NormalizedScore | None = None


class CriticInput(BaseModel):
    """Schema-agnostic candidate input; candidate_record may be any JSON-like object."""

    model_config = ConfigDict(extra="forbid")
    user_request: str | None = None
    source_context: str | None = None
    dataset_spec: dict[str, Any] | None = None
    candidate_record: dict[str, Any] = Field(default_factory=dict)
    comparison_candidates: list[dict[str, Any]] | None = None
    prior_dataset_state: dict[str, Any] | None = None


class CriticTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: CriticDecision | None = None
    preferred_candidate_index: int | None = Field(default=None, ge=0)
    scores: CriticScores = Field(default_factory=CriticScores)
    issue_codes: list[str] = Field(default_factory=list)
    critique: str | None = None
    revision_directive: str | None = None

    @field_validator("issue_codes")
    @classmethod
    def issue_codes_are_nonempty(cls, value: list[str]) -> list[str]:
        if any(not code.strip() for code in value):
            raise ValueError("issue_codes must not contain empty values")
        return value


class Provenance(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_dataset: str = Field(min_length=1)
    source_split: str | None = None
    source_record_id: str | None = None
    source_url: str | None = None
    license: str | None = None
    adapter_name: str = Field(min_length=1)
    adapter_version: str = Field(min_length=1)
    transform_version: str = Field(min_length=1)
    source_fingerprint: str | None = None
    transformed_at: datetime | None = None


class CanonicalCriticRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    record_id: str = Field(pattern=r"^dffc_[a-f0-9]{32}$")
    schema_version: str = CANONICAL_SCHEMA_VERSION
    task_family: TaskFamily
    input: CriticInput
    target: CriticTarget = Field(default_factory=CriticTarget)
    supervision: "Supervision" = Field(default_factory=lambda: Supervision())
    provenance: Provenance

    @staticmethod
    def supported_target_paths() -> set[str]:
        return {
            "decision", "preferred_candidate_index", "issue_codes", "critique", "revision_directive",
            *(f"scores.{name}" for name in CriticScores.model_fields),
        }

    def target_value(self, path: str) -> Any:
        if path.startswith("scores."):
            return getattr(self.target.scores, path.removeprefix("scores."))
        return getattr(self.target, path)

    @model_validator(mode="after")
    def supervised_values_are_present(self) -> "CanonicalCriticRecord":
        missing = [path for path in self.supervision.available_targets if self.target_value(path) is None]
        if missing:
            raise ValueError(f"supervision.available_targets lists target(s) without a value: {', '.join(missing)}")
        return self


class Supervision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    available_targets: list[str] = Field(default_factory=list)

    @field_validator("available_targets")
    @classmethod
    def supported_and_unique(cls, value: list[str]) -> list[str]:
        supported = CanonicalCriticRecord.supported_target_paths()
        unknown = sorted(set(value) - supported)
        if unknown:
            raise ValueError(f"available_targets contains unsupported target path(s): {', '.join(unknown)}")
        if len(value) != len(set(value)):
            raise ValueError("available_targets must not contain duplicates")
        return value
