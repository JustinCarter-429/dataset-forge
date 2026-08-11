import json
import httpx
import pytest
from app.domain.extraction_models import CanonicalExtractedDocument, ExtractionElement, ExtractionElementType, ExtractionStatistics, ExtractionValidation
from app.providers.config import provider_config_from_env
from app.providers.contracts import ProviderError, ProviderJob
from app.providers.runpod import RunPodProvider, build_runpod_openai_chat_job
from app.services.context_projection import build_context_batches
from app.services.generation import RunPodDatasetGenerator, _extract_provider_content, _parse_provider_output


def document() -> CanonicalExtractedDocument:
    blocks = [ExtractionElement(elementId=f"e{i}", type=ExtractionElementType.PARAGRAPH, text=f"Source paragraph {i} explains software testing.", order=i) for i in range(8)]
    return CanonicalExtractedDocument(documentId="doc", sourceFileId="file", sourceFilename="source.txt", mimeType="text/plain", extractor="plain_text", extractorVersion="1", blocks=blocks, statistics=ExtractionStatistics(pageCount=None, characterCount=300, wordCount=40, elementCount=8, headingCount=0, paragraphCount=8, listItemCount=0, tableCount=0, tableRowCount=0, nonEmptyElementCount=8), validation=ExtractionValidation(valid=True, quality="good"))


def test_provider_configuration_is_hard_gate(monkeypatch):
    monkeypatch.delenv("RUNPOD_ENDPOINT_ID", raising=False)
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    with pytest.raises(ProviderError) as error:
        RunPodProvider(provider_config_from_env())
    assert error.value.code == "RUNPOD_CONFIGURATION_REQUIRED"


def test_runpod_native_lifecycle_and_status_mapping():
    seen = []
    statuses = iter([{"status": "IN_QUEUE"}, {"status": "IN_PROGRESS"}, {"status": "COMPLETED", "output": {"records": []}}])
    def handler(request: httpx.Request):
        seen.append((request.method, request.url.path, json.loads(request.content) if request.content else None))
        if request.method == "POST":
            return httpx.Response(200, json={"id": "external-1"})
        return httpx.Response(200, json=next(statuses))
    config = provider_config_from_env()
    config = config.__class__(endpoint_id="endpoint", api_key="secret", model="gpt-oss-20b", poll_interval_seconds=0.001, queue_timeout_seconds=1, execution_timeout_seconds=1)
    result = RunPodProvider(config, httpx.Client(transport=httpx.MockTransport(handler))).generate(messages=[{"role": "user", "content": "x"}], schema={"type": "object"}, max_tokens=100)
    assert result.external_id == "external-1"
    assert [item[1] for item in seen] == ["/v2/endpoint/run", "/v2/endpoint/status/external-1", "/v2/endpoint/status/external-1", "/v2/endpoint/status/external-1"]
    assert seen[0][2]["input"]["openai_route"] == "/v1/chat/completions"
    assert seen[0][2]["input"]["openai_input"]["model"] == "gpt-oss-20b"
    assert seen[0][2]["input"]["openai_input"]["structured_outputs"]["json"] == {"type": "object"}


def test_openai_passthrough_serialization_is_bounded_and_non_streaming():
    payload = build_runpod_openai_chat_job(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": "Return JSON."}],
        schema={"type": "object", "properties": {"status": {"enum": ["ok"]}}},
        max_tokens=96,
    )
    request = payload["input"]["openai_input"]
    assert payload["input"]["openai_route"] == "/v1/chat/completions"
    assert request["model"] == "openai/gpt-oss-20b"
    assert request["max_tokens"] == 96
    assert request["stream"] is False
    assert request["structured_outputs"]["json"]["type"] == "object"


def test_runpod_auth_and_unknown_status_are_safe():
    def auth_handler(request):
        return httpx.Response(401, json={"error": "no"})
    config = provider_config_from_env().__class__(endpoint_id="endpoint", api_key="secret", model="gpt-oss-20b")
    with pytest.raises(ProviderError) as error:
        RunPodProvider(config, httpx.Client(transport=httpx.MockTransport(auth_handler))).health()
    assert error.value.code == "RUNPOD_AUTH_FAILED"


def test_runpod_terminal_failure_keeps_only_safe_error_telemetry():
    config = provider_config_from_env().__class__(endpoint_id="endpoint", api_key="secret", model="gpt-oss-20b", poll_interval_seconds=0.001)
    statuses = iter([{"status": "FAILED", "error": {"type": "invalid_request", "code": "UNSUPPORTED_FIELD", "message": "raw worker trace must not persist"}}])
    def handler(request):
        return httpx.Response(200, json={"id": "external-1"} if request.method == "POST" else next(statuses))
    provider = RunPodProvider(config, httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(ProviderError, match="failed"):
        provider.generate(messages=[{"role": "user", "content": "x"}], schema={"type": "object"}, max_tokens=32)
    failure = provider.last_telemetry["terminal_failure"]
    assert failure == {"status_keys": ["error", "status"], "error_present": True, "error_type": "invalid_request", "error_code": "UNSUPPORTED_FIELD", "error_category": "unsupported_parameter", "error_keys": ["code", "message", "type"]}
    assert "raw worker trace" not in str(provider.last_telemetry)


def test_context_batches_are_bounded_and_ordered():
    batches = build_context_batches(document(), max_model_len=1024, records_per_batch=2, record_limit=5)
    assert len(batches) >= 2
    assert sum(len(batch.source_element_ids) for batch in batches) <= 8
    assert batches[0].source_element_ids[0] == "e0"
    assert all(batch.estimated_input_tokens <= 1024 for batch in batches)


class ValidProvider:
    def generate(self, *, messages, schema, max_tokens):
        return ProviderJob("job", "completed", {"records": [{"instruction": "What is tested?", "input": "Functional testing", "output": "It verifies workflows."}]})


class RepairProvider:
    def __init__(self): self.calls = 0
    def generate(self, *, messages, schema, max_tokens):
        self.calls += 1
        if self.calls == 1: return ProviderJob("job", "completed", "not json")
        return ProviderJob("repair", "completed", {"records": [{"instruction": "What is tested?", "input": "Functional testing", "output": "It verifies workflows."}]})


def test_generation_normalizes_application_metadata_and_never_placeholder():
    dataset = RunPodDatasetGenerator(ValidProvider(), max_model_len=4096, records_per_batch=2, max_dataset_records=4).generate(document(), "Create 1 record", generation_id="gen", file_id="file", model="gpt-oss-20b")
    assert dataset.placeholder is False
    assert dataset.records[0].metadata["generation_id"] == "gen"
    assert dataset.records[0].metadata["provider"] == "runpod_serverless"


def test_generation_uses_at_most_one_bounded_repair():
    provider = RepairProvider()
    dataset = RunPodDatasetGenerator(provider, max_model_len=4096, records_per_batch=2, max_dataset_records=4).generate(document(), "Create 1 record", generation_id="gen", file_id="file", model="gpt-oss-20b")
    assert len(dataset.records) == 1
    assert provider.calls == 2


def test_parser_accepts_wrapped_vllm_choices():
    payload = [{"choices": [{"text": json.dumps({"records": [{"instruction": "i", "input": "x", "output": "o"}]})}]}]
    assert _parse_provider_output(payload)[0]["instruction"] == "i"


def test_parser_accepts_current_chat_completion_message_content():
    payload = {"choices": [{"message": {"role": "assistant", "content": '{"records":[{"instruction":"i","input":"x","output":"o"}]}'}, "finish_reason": "stop"}]}
    assert _parse_provider_output(payload)[0]["output"] == "o"


def test_content_extractor_accepts_tiny_compatibility_schema():
    payload = [{"choices": [{"message": {"content": '{"status":"ok"}', "reasoning": "not parsed"}}]}]
    assert json.loads(_extract_provider_content(payload))["status"] == "ok"


@pytest.mark.parametrize("payload", [None, [], {"choices": []}, {"choices": [{"message": {}}]}, {"choices": [{"message": {"content": None}}]}, {"choices": [{"message": {"content": ""}}]}])
def test_parser_rejects_invalid_completion_shapes(payload):
    with pytest.raises(ValueError):
        _parse_provider_output(payload)


def test_parser_rejects_reasoning_only_completion():
    with pytest.raises(ValueError):
        _parse_provider_output({"choices": [{"message": {"reasoning": "thinking"}}]})


def test_parser_classifies_worker_error():
    with pytest.raises(ProviderError) as error:
        _parse_provider_output({"error": {"type": "worker_error", "message": "bad request"}})
    assert error.value.code == "RUNPOD_WORKER_ERROR"
