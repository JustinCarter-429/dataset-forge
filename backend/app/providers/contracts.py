from dataclasses import dataclass, field
from typing import Any, Protocol


class ProviderError(RuntimeError):
    def __init__(self, code: str, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


@dataclass(frozen=True)
class ProviderConfig:
    endpoint_id: str
    api_key: str
    model: str
    max_model_len: int = 32768
    poll_interval_seconds: float = 1.0
    queue_timeout_seconds: float = 300.0
    execution_timeout_seconds: float = 600.0
    records_per_batch: int = 4
    max_dataset_records: int = 20

    def validate(self) -> None:
        missing = [name for name, value in (("RUNPOD_ENDPOINT_ID", self.endpoint_id), ("RUNPOD_API_KEY", self.api_key), ("RUNPOD_MODEL", self.model)) if not value]
        if missing:
            raise ProviderError("RUNPOD_CONFIGURATION_REQUIRED", "RunPod configuration is required before real generation can start.")
        if self.max_model_len < 1024:
            raise ProviderError("RUNPOD_CONFIGURATION_INVALID", "RUNPOD_MAX_MODEL_LEN must be at least 1024.")
        if self.poll_interval_seconds <= 0 or self.queue_timeout_seconds <= 0 or self.execution_timeout_seconds <= 0:
            raise ProviderError("RUNPOD_CONFIGURATION_INVALID", "RunPod polling and timeout values must be positive.")
        if self.records_per_batch < 1 or self.max_dataset_records < 1:
            raise ProviderError("RUNPOD_CONFIGURATION_INVALID", "RunPod batch and record limits must be positive.")


@dataclass(frozen=True)
class ProviderJob:
    external_id: str
    state: str
    output: Any = None
    provider_metadata: dict[str, Any] = field(default_factory=dict)


class DatasetGenerationProvider(Protocol):
    def generate(self, *, messages: list[dict[str, str]], schema: dict[str, Any], max_tokens: int) -> ProviderJob: ...
    def health(self) -> dict[str, Any]: ...
    def cancel(self, external_id: str) -> bool: ...
