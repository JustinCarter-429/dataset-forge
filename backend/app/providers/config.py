import os
from .contracts import ProviderConfig


def provider_config_from_env() -> ProviderConfig:
    def number(name: str, default: str, cast):
        try:
            return cast(os.getenv(name, default))
        except ValueError as exc:
            raise ValueError(f"{name} must be numeric.") from exc

    return ProviderConfig(
        endpoint_id=os.getenv("RUNPOD_ENDPOINT_ID", ""),
        api_key=os.getenv("RUNPOD_API_KEY", ""),
        model=os.getenv("RUNPOD_MODEL", "openai/gpt-oss-20b"),
        max_model_len=number("RUNPOD_MAX_MODEL_LEN", "32768", int),
        poll_interval_seconds=number("RUNPOD_POLL_INTERVAL_SECONDS", "1", float),
        queue_timeout_seconds=number("RUNPOD_QUEUE_TIMEOUT_SECONDS", "300", float),
        execution_timeout_seconds=number("RUNPOD_EXECUTION_TIMEOUT_SECONDS", "600", float),
        records_per_batch=number("RUNPOD_RECORDS_PER_BATCH", "4", int),
        max_dataset_records=number("MAX_DATASET_RECORDS", "20", int),
    )
