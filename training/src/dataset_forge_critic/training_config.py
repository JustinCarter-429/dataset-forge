"""Strict, hashable configuration for critic training runs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .gemma import MODEL_ID, REVISION, RENDERER


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelSpec(StrictModel):
    model_id: str = MODEL_ID
    revision: str = REVISION
    architecture: Literal["Gemma4ForConditionalGeneration"] = "Gemma4ForConditionalGeneration"
    renderer_version: str = RENDERER
    require_bf16: bool = True
    load_in_4bit: bool = True
    quant_type: Literal["nf4"] = "nf4"
    double_quant: bool = True
    gradient_checkpointing: bool = True
    lora_rank: int = Field(default=16, ge=1)
    lora_alpha: int = Field(default=16, ge=1)
    lora_dropout: float = Field(default=0.05, ge=0, lt=1)
    lora_target_regex: str = r"(^|\.)(language_model|text_model)(\.|$)"

    @model_validator(mode="after")
    def identity_is_pinned(self) -> "ModelSpec":
        if self.model_id != MODEL_ID or self.revision != REVISION:
            raise ValueError("MODEL_IDENTITY_MISMATCH: model id and revision are immutable")
        if self.renderer_version != RENDERER:
            raise ValueError("RENDERER_IDENTITY_MISMATCH: training and inference must use the same renderer")
        return self


class CorpusFiles(StrictModel):
    public_train: Path
    native_train: Path
    public_validation: Path
    native_validation: Path


class MixtureSpec(StrictModel):
    public_probability: float = Field(default=0.5, gt=0, lt=1)
    native_probability: float = Field(default=0.5, gt=0, lt=1)
    with_replacement: Literal[True] = True

    @model_validator(mode="after")
    def probabilities_sum_to_one(self) -> "MixtureSpec":
        if abs(self.public_probability + self.native_probability - 1.0) > 1e-9:
            raise ValueError("MIXTURE_PROBABILITIES_MUST_SUM_TO_ONE")
        return self


class OptimizationSpec(StrictModel):
    micro_batch_size: int = Field(default=1, ge=1)
    gradient_accumulation_steps: int = Field(default=8, ge=1)
    learning_rate: float = Field(default=1e-4, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)
    optimizer: Literal["adamw_torch", "paged_adamw_8bit"] = "paged_adamw_8bit"
    scheduler: Literal["linear", "cosine", "constant_with_warmup"] = "cosine"
    warmup_ratio: float = Field(default=0.03, ge=0, lt=1)
    max_grad_norm: float = Field(default=1.0, gt=0)


class LimitsSpec(StrictModel):
    max_optimizer_steps: int = Field(ge=1)
    max_examples: int | None = Field(default=None, ge=1)
    max_total_tokens: int | None = Field(default=None, ge=1)
    max_wall_seconds: int | None = Field(default=None, ge=1)
    min_free_disk_gb: float = Field(default=30, ge=1)
    estimated_process_cost_usd: float | None = Field(default=None, ge=0)


class CheckpointSpec(StrictModel):
    every_steps: int = Field(default=100, ge=1)
    keep_latest: int = Field(default=2, ge=1)
    keep_best: int = Field(default=1, ge=1)
    selection_metric: Literal["native_macro_f1", "validation_loss"] = "native_macro_f1"


class EvaluationSpec(StrictModel):
    every_steps: int = Field(default=100, ge=1)
    frequent_public_count: int = Field(default=256, ge=1)
    frequent_native_count: int = Field(default=256, ge=1)
    broad_public_count: int | None = Field(default=None, ge=1)
    broad_native_count: int | None = Field(default=None, ge=1)


class UploadSpec(StrictModel):
    destination: str | None = None
    visibility: Literal["private", "public"] | None = None
    retries: int = Field(default=3, ge=0, le=10)


class TrainingConfiguration(StrictModel):
    config_version: Literal["2.0.0"] = "2.0.0"
    profile: Literal["cpu_synthetic", "gpu_smoke", "gpu_benchmark", "pilot", "full"]
    seed: int = Field(default=42, ge=0)
    output_root: Path = Path("outputs")
    max_sequence_length: Literal[1024, 2048, 4096] = 2048
    packing: Literal[False] = False
    model: ModelSpec = Field(default_factory=ModelSpec)
    data: CorpusFiles
    mixture: MixtureSpec = Field(default_factory=MixtureSpec)
    optimization: OptimizationSpec = Field(default_factory=OptimizationSpec)
    limits: LimitsSpec
    checkpoint: CheckpointSpec = Field(default_factory=CheckpointSpec)
    evaluation: EvaluationSpec = Field(default_factory=EvaluationSpec)
    upload: UploadSpec = Field(default_factory=UploadSpec)

    @model_validator(mode="after")
    def profile_guards(self) -> "TrainingConfiguration":
        if self.profile != "cpu_synthetic" and not self.model.require_bf16:
            raise ValueError("GPU_PROFILES_REQUIRE_BF16")
        if self.profile != "cpu_synthetic" and not self.model.load_in_4bit:
            raise ValueError("GPU_PROFILES_REQUIRE_PINNED_QLORA")
        if self.profile == "gpu_benchmark" and self.limits.max_wall_seconds is None:
            raise ValueError("BENCHMARK_REQUIRES_WALL_CLOCK_LIMIT")
        return self

    def canonical_dict(self) -> dict:
        return self.model_dump(mode="json", exclude_none=False)

    def digest(self) -> str:
        payload = json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_training_configuration(path: Path) -> TrainingConfiguration:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"CONFIG_READ_FAILED: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("CONFIG_MUST_BE_A_MAPPING")
    return TrainingConfiguration.model_validate(raw)
