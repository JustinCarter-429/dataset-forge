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


def _record_target(schema: Any) -> int:
    if isinstance(schema, dict):
        records = schema.get("properties", {}).get("records")
        own = records.get("minItems", 0) if isinstance(records, dict) else 0
        return max(own if isinstance(own, int) else 0, *(_record_target(value) for value in schema.values()))
    elif isinstance(schema, list):
        return max((_record_target(value) for value in schema), default=0)
    return 0


class DeterministicCarterProvider:
    runtime = "runpod"
    allow_empty_retrieval = True

    def __init__(self, runtime: str):
        self.runtime = runtime
        self.calls = 0

    def available(self) -> dict[str, Any]:
        if self.runtime == "local_lm_studio" and scenario() == "local_unavailable":
            return {"configured": True, "available": False, "model": "deterministic-carter"}
        return {"configured": True, "available": True, "model": "deterministic-carter"}

    def infer(self, request: CarterInferenceRequest) -> CarterInferenceResponse:
        self.calls += 1
        # Production-route deterministic fixture.  It deliberately follows the
        # same manifest-derived schemas as live adapters and keeps custom fields
        # intact, rather than exercising the retired fixed-record generator.
        schema = request.response_schema or {}
        properties = schema.get("properties", {})
        text = " ".join(str(message.get("content", "")) for message in request.messages)
        if "dataset_type" in properties:
            requested = re.search(r"(?:exactly|create|generate)\s+(\d+)\s+records", text, re.I)
            count = min(int(requested.group(1)) if requested else 1, 20)
            spec = {"status":"ready","dataset_type":"custom","dataset_name":"deterministic-custom","dataset_description":"Deterministic Carter custom dataset.","requested_record_count":count,"effective_record_count":count,"fields":[{"name":"customer_intent","type":"string","required":True,"description":"Intent."},{"name":"confidence_label","type":"enum","required":True,"description":"Confidence.","enum_values":["high","low"]},{"name":"reasoning_style","type":"array_string","required":False,"description":"Style."}],"source_policy":"selected_documents_only","grounding_required":True,"evidence_required":True,"generation_requirements":["source_grounded","avoid_exact_duplicates"],"user_constraints":[],"clarification":{"required":False,"reason_code":None,"question":None,"reason":None}}
            return CarterInferenceResponse(json.dumps(spec), [])
        if "recommendation" in properties:
            if "TEST_QUALITY_WARNING" in text:
                return CarterInferenceResponse(json.dumps({"status":"completed","recommendation":"accept","summary":"Deterministic advisory warning accepted after validation.","issues":[{"issue_id":"issue_001","category":"custom_schema_quality","severity":"warning","affected_record_refs":["review_record_001"],"affected_field":"customer_intent","description":"A warning-only deterministic finding.","recommended_correction":"Consider variation."}]}), [])
            return CarterInferenceResponse(json.dumps({"status":"completed","recommendation":"accept","summary":"Deterministic review accepted the validated candidate.","issues":[]}), [])
        if "records" in properties:
            if "TEST_GENERATION_FAILURE" in text:
                raise ProviderError("CARTER_TEST_GENERATION_FAILED", "Deterministic generation failed safely.")
            refs = re.findall(r'"allowed_source_refs"\s*:\s*\[\s*"([^"]+)"', text)
            ref = refs[0] if refs else "source_1"
            if "TEST_VALIDATION_FAILURE" in text:
                ref = "unknown-source"
            count = _record_target(schema) or 1
            records = [{"customer_intent":f"support request {index + 1}","confidence_label":"high","reasoning_style":["concise"],"evidence":[{"source_ref":ref,"quote":"Deterministic evidence."}]} for index in range(count)]
            return CarterInferenceResponse(json.dumps({"status":"generated","records":records,"insufficiency":None}), [])
        question = " ".join(str(message.get("content", "")) for message in request.messages)
        if scenario() in {"ask_failure", "cloud_failure"} or "TEST_ASK_FAILURE" in question or "TEST_CLOUD_FAILURE" in question:
            raise ProviderError("CARTER_TEST_FAILURE", "Carter could not complete the request safely.")
        if self.calls == 1 and request.tool_choice == "required":
            return CarterInferenceResponse("", [{"id": "deterministic-search", "function": {"name": "search_local_knowledge", "arguments": json.dumps({"query": "testing"})}}])
        refs = list(dict.fromkeys(re.findall(r'"sourceRef"\s*:\s*"([^"]+)"', question)))
        citations = [{"sourceRef": reference} for reference in refs] or [{"sourceRef": "missing"}]
        answer = "I could not find relevant information in the selected documents." if "unsupported astronomy" in question else "The selected documents describe negative testing for invalid inputs and accessibility testing for keyboard operation, visible focus, labels, and readable status feedback."
        return CarterInferenceResponse(json.dumps({"answer": answer, "citations": citations}), [])


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
