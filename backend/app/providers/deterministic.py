"""Explicitly opt-in deterministic providers for browser acceptance only."""
from __future__ import annotations

import json
import re
from typing import Any

from .contracts import ProviderError, ProviderJob
from ..services.carter import CarterInferenceRequest, CarterInferenceResponse


def enabled() -> bool:
    import os
    return os.getenv("APP_ENVIRONMENT", "development") == "test" and os.getenv("CARTER_TEST_PROVIDER", "") == "deterministic"


def scenario() -> str:
    import os
    return os.getenv("CARTER_TEST_SCENARIO", "success")


class DeterministicCarterProvider:
    runtime = "cloud"
    allow_empty_retrieval = True

    def __init__(self, runtime: str):
        self.runtime = runtime
        self.calls = 0

    def available(self) -> dict[str, Any]:
        if self.runtime == "local" and scenario() == "local_unavailable":
            return {"configured": True, "available": False, "model": "deterministic-carter"}
        return {"configured": True, "available": True, "model": "deterministic-carter"}

    def infer(self, request: CarterInferenceRequest) -> CarterInferenceResponse:
        self.calls += 1
        question = " ".join(str(message.get("content", "")) for message in request.messages)
        if scenario() in {"ask_failure", "cloud_failure"} or "TEST_ASK_FAILURE" in question or "TEST_CLOUD_FAILURE" in question:
            raise ProviderError("CARTER_TEST_FAILURE", "Carter could not complete the request safely.")
        return CarterInferenceResponse("The selected documents describe negative testing for invalid inputs and accessibility testing for keyboard operation, visible focus, labels, and readable status feedback.", [])


class DeterministicDatasetProvider:
    def __init__(self, config: Any):
        self.config = config
        self.metrics = {"providerSubmitAttempts": 0, "providerJobsCreated": 0, "providerJobsCompleted": 0, "providerJobsFailed": 0, "providerStatusPolls": 0, "providerTransportRetries": 0, "providerCancelCalls": 0}
        self.cancel_check = None
        self.on_job_created = None

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "provider": "deterministic-test"}

    def cancel(self, external_id: str) -> bool:
        self.metrics["providerCancelCalls"] += 1
        return True

    @staticmethod
    def _aliases(messages: list[dict[str, str]]) -> list[tuple[str, str]]:
        text = "\n".join(message.get("content", "") for message in messages)
        return re.findall(r"SOURCE UNIT: (source_\d+)\nTEXT: ([^\n]+)", text)

    def generate(self, *, messages: list[dict[str, str]], schema: dict[str, Any], max_tokens: int) -> ProviderJob:
        self.metrics["providerSubmitAttempts"] += 1
        self.metrics["providerJobsCreated"] += 1
        user_text = "\n".join(message.get("content", "") for message in messages)
        if "issues" in schema.get("properties", {}):
            if scenario() == "quality_warning" or "TEST_QUALITY_WARNING" in user_text:
                ids = re.findall(r'"record_id":\s*"([^"]+)"', messages[-1].get("content", ""))
                payload = {"issues": [{"code": "LOW_INSTRUCTION_DIVERSITY", "severity": "warning", "record_ids": ids[:1], "message": "A warning-only deterministic review finding.", "suggested_action": "Consider varying instruction phrasing."}] if ids else [], "summary": "Deterministic quality review completed with a warning."}
            else:
                payload = {"issues": [], "summary": "Deterministic quality review passed."}
        else:
            if scenario() == "generation_failure" or "TEST_GENERATION_FAILURE" in user_text:
                raise ProviderError("CARTER_TEST_GENERATION_FAILED", "Deterministic generation failed safely.")
            aliases = self._aliases(messages)
            alias, quote = aliases[0] if aliases else ("unknown", "")
            if scenario() == "validation_failure" or "TEST_VALIDATION_FAILURE" in user_text: alias = "unknown-source"
            user = messages[-1].get("content", "")
            match = re.search(r"\((\d+) records requested\)", user)
            count = min(int(match.group(1)) if match else 1, 4)
            payload = {"records": [{"instruction": "Explain the testing practice.", "context": quote, "expected_output": quote, "category": "testing", "difficulty": "easy", "source_refs": [alias], "evidence": [{"source_ref": alias, "quote": quote}]} for _ in range(count)]}
        self.metrics["providerJobsCompleted"] += 1
        if self.on_job_created:
            self.on_job_created(f"deterministic-{self.metrics['providerJobsCreated']}")
        return ProviderJob(f"deterministic-{self.metrics['providerJobsCreated']}", "COMPLETED", {"choices": [{"message": {"content": json.dumps(payload)}}]})
