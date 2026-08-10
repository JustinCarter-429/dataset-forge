import hashlib
import json
import time
from typing import Any
import httpx
from .contracts import DatasetGenerationProvider, ProviderConfig, ProviderError, ProviderJob


def build_runpod_openai_chat_job(
    *,
    model: str,
    messages: list[dict[str, str]],
    schema: dict[str, Any] | None,
    max_tokens: int,
    temperature: float = 0.2,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str = "auto",
) -> dict[str, Any]:
    """Build the worker-vLLM OpenAI passthrough inside a native RunPod job."""
    openai_input: dict[str, Any] = {
        "model": model, "messages": messages, "max_tokens": max_tokens,
        "temperature": temperature, "stream": False,
    }
    if schema is not None:
        openai_input["structured_outputs"] = {"json": schema}
    if tools:
        openai_input["tools"] = tools
        openai_input["tool_choice"] = tool_choice
    return {
        "input": {
            "openai_route": "/v1/chat/completions",
            "openai_input": openai_input,
        }
    }


def _safe_output_snapshot(output: Any) -> dict[str, Any]:
    snapshot: dict[str, Any] = {"output_type": type(output).__name__}
    choice: dict[str, Any] | None = None
    if isinstance(output, list):
        snapshot["array_length"] = len(output)
        if output and isinstance(output[0], dict):
            snapshot["top_level_keys"] = sorted(output[0].keys())
            if isinstance(output[0].get("choices"), list):
                snapshot["choices_count"] = len(output[0]["choices"])
                if output[0]["choices"] and isinstance(output[0]["choices"][0], dict):
                    choice = output[0]["choices"][0]
    elif isinstance(output, dict):
        snapshot["top_level_keys"] = sorted(output.keys())
        snapshot["usage_present"] = "usage" in output
        if isinstance(output.get("choices"), list):
            snapshot["choices_count"] = len(output["choices"])
            if output["choices"] and isinstance(output["choices"][0], dict):
                choice = output["choices"][0]
    if choice is not None:
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        content = message.get("content") if isinstance(message.get("content"), str) else choice.get("text")
        snapshot["content_type"] = type(content).__name__ if content is not None else None
        snapshot["content_length"] = len(content) if isinstance(content, str) else 0
        snapshot["reasoning_present"] = bool(message.get("reasoning") or message.get("reasoning_content"))
        snapshot["finish_reason"] = choice.get("finish_reason")
        snapshot.setdefault("usage_present", False)
    return snapshot


class RunPodProvider(DatasetGenerationProvider):
    """Native RunPod Serverless /run + /status client for vLLM workers."""

    def __init__(self, config: ProviderConfig, client: httpx.Client | None = None):
        config.validate()
        self.config = config
        self.client = client or httpx.Client(timeout=config.execution_timeout_seconds + 30)
        self.base_url = f"https://api.runpod.ai/v2/{config.endpoint_id}"
        self.last_telemetry: dict[str, Any] = {}
        self.metrics: dict[str, int] = {
            "providerSubmitAttempts": 0, "providerJobsCreated": 0,
            "providerJobsCompleted": 0, "providerJobsFailed": 0,
            # Polling is transport bookkeeping, never an inference job.
            "providerStatusPolls": 0, "providerTransportRetries": 0,
            "providerCancelCalls": 0,
        }
        self.cancel_check = None
        self.on_job_created = None

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"}

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = self.client.request(method, f"{self.base_url}{path}", headers=self.headers, **kwargs)
        except httpx.RequestError as exc:
            raise ProviderError("RUNPOD_NETWORK_ERROR", "Unable to reach the RunPod endpoint.", retryable=True) from exc
        if response.status_code == 401:
            raise ProviderError("RUNPOD_AUTH_FAILED", "RunPod authentication failed.")
        if response.status_code == 429:
            raise ProviderError("RUNPOD_RATE_LIMITED", "RunPod rate limit reached.", retryable=True)
        if response.status_code >= 500:
            raise ProviderError("RUNPOD_PROVIDER_ERROR", "RunPod returned a provider error.", retryable=True)
        if response.status_code >= 400:
            raise ProviderError("RUNPOD_REQUEST_INVALID", "RunPod rejected the generation request.")
        try:
            return response.json()
        except ValueError as exc:
            raise ProviderError("RUNPOD_INVALID_RESPONSE", "RunPod returned invalid JSON.") from exc

    def generate(self, *, messages: list[dict[str, str]], schema: dict[str, Any], max_tokens: int) -> ProviderJob:
        return self.chat(messages=messages, schema=schema, max_tokens=max_tokens)

    def chat(self, *, messages: list[dict[str, Any]], max_tokens: int, schema: dict[str, Any] | None = None, tools: list[dict[str, Any]] | None = None, tool_choice: str = "auto") -> ProviderJob:
        started = time.monotonic()
        schema_digest = hashlib.sha256(json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()).hexdigest() if schema else None
        payload = build_runpod_openai_chat_job(model=self.config.model, messages=messages, schema=schema, max_tokens=max_tokens, tools=tools, tool_choice=tool_choice)
        # A /run submission is intentionally single-shot. Retrying an unknown
        # POST outcome could create a duplicate paid inference job.
        self.metrics["providerSubmitAttempts"] += 1
        submitted = self._request("POST", "/run", json=payload)
        external_id = submitted.get("id")
        if not external_id or not isinstance(external_id, str):
            raise ProviderError("RUNPOD_MISSING_JOB_ID", "RunPod did not return an external job ID.")
        self.metrics["providerJobsCreated"] += 1
        if self.on_job_created:
            self.on_job_created(external_id)
        queue_started = time.monotonic()
        status_sequence: list[str] = []
        while True:
            if self.cancel_check and self.cancel_check():
                self.cancel(external_id)
                self.metrics["providerJobsFailed"] += 1
                raise ProviderError("RUNPOD_CANCELLED", "Generation was cancelled.")
            poll_attempt = 0
            while True:
                try:
                    self.metrics["providerStatusPolls"] += 1
                    current = self._request("GET", f"/status/{external_id}")
                    break
                except ProviderError as exc:
                    if not exc.retryable or poll_attempt >= 2:
                        self.metrics["providerJobsFailed"] += 1
                        raise
                    poll_attempt += 1
                    self.metrics["providerTransportRetries"] += 1
                    time.sleep(min(0.25, self.config.poll_interval_seconds))
            state = str(current.get("status", "")).upper()
            status_sequence.append(state)
            if state == "COMPLETED":
                output = current.get("output")
                self.last_telemetry = {
                    "external_job_id": external_id,
                    "status_sequence": status_sequence,
                    "queue_duration_ms": round((time.monotonic() - queue_started) * 1000, 1),
                    "total_duration_ms": round((time.monotonic() - started) * 1000, 1),
                    "request": {
                        "input_mode": "openai_passthrough",
                        "openai_route": "/v1/chat/completions",
                        "model": self.config.model,
                        "message_count": len(messages),
                        "message_character_counts": [len(item.get("content", "")) for item in messages],
                        "stream": False,
                        "max_token_field_name": "max_tokens",
                        "max_tokens": max_tokens,
                        "temperature": 0.2,
                        "structured_output_field": "structured_outputs" if schema else None,
                        "structured_output_mode": "json" if schema else None,
                        "schema_sha256": schema_digest if schema else None,
                        "tool_count": len(tools or []),
                        "tool_choice": tool_choice if tools else None,
                    },
                    "output": _safe_output_snapshot(output),
                }
                if isinstance(output, dict) and isinstance(output.get("error"), dict):
                    self.metrics["providerJobsFailed"] += 1
                    raise ProviderError("RUNPOD_WORKER_ERROR", "RunPod worker returned an inference error.")
                self.metrics["providerJobsCompleted"] += 1
                return ProviderJob(external_id, "completed", output, {"status": state, "output": self.last_telemetry["output"]})
            if state in {"FAILED", "CANCELLED", "TIMED_OUT"}:
                self.metrics["providerJobsFailed"] += 1
                raise ProviderError(f"RUNPOD_{state}", f"RunPod job ended with status {state.lower()}.")
            if state not in {"IN_QUEUE", "IN_PROGRESS"}:
                raise ProviderError("RUNPOD_UNKNOWN_STATUS", "RunPod returned an unknown job status.")
            elapsed = time.monotonic() - started
            if state == "IN_QUEUE" and elapsed > self.config.queue_timeout_seconds:
                raise ProviderError("RUNPOD_QUEUE_TIMEOUT", "The RunPod job remained queued too long.")
            if state == "IN_PROGRESS" and elapsed > self.config.queue_timeout_seconds + self.config.execution_timeout_seconds:
                raise ProviderError("RUNPOD_EXECUTION_TIMEOUT", "The RunPod job exceeded its execution timeout.")
            time.sleep(self.config.poll_interval_seconds)

    def cancel(self, external_id: str) -> bool:
        self.metrics["providerCancelCalls"] += 1
        try:
            self._request("POST", f"/cancel/{external_id}")
            return True
        except ProviderError as exc:
            if exc.code == "RUNPOD_REQUEST_INVALID":
                return False
            raise

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")
