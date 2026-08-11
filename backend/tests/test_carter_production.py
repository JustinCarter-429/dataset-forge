import json
import inspect
from dataclasses import dataclass
import pytest

from app.carter.production import CarterDatasetGenerationService
from app.carter.runtime import CarterPromptPackage
from app.domain.extraction_models import (CanonicalExtractedDocument, ExtractionElement,
    ExtractionElementType, ExtractionStatistics, ExtractionValidation)
from app.providers.deterministic import DeterministicCarterProvider
from app.services.carter import CarterInferenceResponse, RunPodCarterProvider, exact_output_contract_instruction
from app.providers.contracts import ProviderError, ProviderJob
from app.carter.runtime import CarterPromptPackageError
from app.api.routes import generation as generation_route


def document():
    return CanonicalExtractedDocument(documentId="doc-1", sourceFileId="file-1", sourceFilename="source.txt", mimeType="text/plain", extractor="plain_text", statistics=ExtractionStatistics(characterCount=42, wordCount=6, elementCount=1, headingCount=0, paragraphCount=1, listItemCount=0, tableCount=0, tableRowCount=0, nonEmptyElementCount=1), blocks=[ExtractionElement(elementId="source_1", type=ExtractionElementType.PARAGRAPH, text="Customers can request support.", order=1)], validation=ExtractionValidation(valid=True))


def test_production_orchestrator_preserves_dynamic_records(tmp_path):
    run = CarterDatasetGenerationService(CarterPromptPackage.load(), DeterministicCarterProvider("runpod"), knowledge_path=tmp_path / "knowledge.sqlite3").generate(runtime="runpod", user_request="Create a custom support dataset", output_format="json", documents=[document()])
    assert run.specification["dataset_type"] == "custom"
    assert run.dataset.records[0]["customer_intent"] == "support request"
    assert run.calls == {"planner": 1, "generator": 1, "tool_continuation": 0, "review": 1, "revision": 0, "revision_repair": 0}


def test_generation_with_application_source_context_sends_no_native_tools(tmp_path):
    provider = ScriptedProvider([CarterInferenceResponse(json.dumps(_spec()), []), CarterInferenceResponse(json.dumps(_candidate()), []), CarterInferenceResponse(json.dumps(_review()), [])])
    CarterDatasetGenerationService(CarterPromptPackage.load(), provider, knowledge_path=tmp_path / "no-tools.sqlite3").generate(runtime="runpod", user_request="Use source", output_format="json", documents=[document()])
    generation_request = provider.requests[1]
    assert generation_request.tools == []
    rendered = "\n".join(message["content"] for message in generation_request.messages)
    assert "Customers can request support." in rendered
    assert "to=tool:" not in rendered


def test_production_orchestrator_generates_three_valid_batches_in_order(tmp_path):
    phases = []
    service = CarterDatasetGenerationService(CarterPromptPackage.load(), DeterministicCarterProvider("runpod"), knowledge_path=tmp_path / "batched.sqlite3", generation_batch_size=5, on_phase=lambda phase, batch=None: phases.append((phase, batch.copy() if batch else None)))
    run = service.generate(runtime="runpod", user_request="Create exactly 12 records", output_format="json", documents=[document()])
    assert run.calls["planner"] == 1 and run.calls["generator"] == 3 and len(run.dataset.records) == 12
    completed = {}
    for phase, batch in phases:
        if phase == "generating" and batch: completed[batch["currentBatch"]] = batch
    assert [completed[index]["currentBatchTarget"] for index in (1, 2, 3)] == [5, 5, 2]
    assert [completed[index]["recordsGenerated"] for index in (1, 2, 3)] == [5, 10, 12]
    assert [record["customer_intent"] for record in run.dataset.records] == [f"support request {index}" for index in (1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2)]
    assert not any({"batch", "batch_id", "batch_number", "batch_index", "current_batch"} & set(record) for record in run.dataset.records)


def test_generation_route_uses_carter_not_legacy_generator():
    source = inspect.getsource(generation_route._run_job)
    assert "CarterDatasetGenerationService" in source
    assert "RunPodDatasetGenerator(" not in source


def test_runpod_adapter_accepts_production_prompt_package_request():
    class Transport:
        class Config:
            max_model_len = 1024
        config = Config()
        def __init__(self): self.request = None
        def chat(self, **kwargs):
            self.request = kwargs
            return ProviderJob("job-1", "completed", {"choices": [{"message": {"content": "{}"}}]})

    transport = Transport()
    package = CarterPromptPackage.load()
    request = package.render(package.resolve_operation("dataset_planning"), {
        "user_request": "Create one example.", "requested_output_format": "json",
        "application_limits": {"maximum_dataset_records": 1}, "selected_document_metadata": [],
    }, runtime="cloud")
    RunPodCarterProvider(transport).infer(request)
    assert transport.request["tool_choice"] == "auto"
    assert transport.request["schema"] == {"type": "object"}
    assert len(transport.request["messages"]) == len(request.messages) + 2
    assert "AUTHORITATIVE_OUTPUT_CONTRACT=" in transport.request["messages"][-3]["content"]
    assert "Provider constraint is JSON object" in transport.request["messages"][-2]["content"]


@dataclass
class ScriptedProvider:
    replies: list[CarterInferenceResponse]
    runtime: str = "runpod"
    calls: int = 0
    requests: list = None
    on_infer: object = None
    def __post_init__(self): self.requests = []
    def available(self): return {"configured": True, "available": True, "model": "scripted"}
    def infer(self, _request):
        self.calls += 1
        self.requests.append(_request)
        if self.on_infer: self.on_infer(self.calls)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception): raise reply
        return reply


def _spec():
    return {"status":"ready","dataset_type":"custom","dataset_name":"custom","dataset_description":"custom","requested_record_count":1,"effective_record_count":1,"fields":[{"name":"customer_intent","type":"string","required":True,"description":"Intent."}],"source_policy":"selected_documents_only","grounding_required":True,"evidence_required":True,"generation_requirements":["source_grounded","avoid_exact_duplicates"],"user_constraints":[],"clarification":{"required":False,"reason_code":None,"question":None,"reason":None}}


def _candidate(value="request"):
    return {"status":"generated","records":[{"customer_intent":value,"evidence":[{"source_ref":"source_1","quote":"Customers can request support."}]}],"insufficiency":None}


def _candidate_many(count=10, *, valid=True):
    records = []
    for index in range(count):
        record = {"customer_intent": f"request {index + 1}", "evidence":[{"source_ref":"source_1","quote":"Customers can request support."}]}
        if not valid:
            record.pop("customer_intent")
        records.append(record)
    return {"status":"generated", "records":records, "insufficiency":None}


def test_dynamic_output_contract_exposes_exact_custom_fields_and_batch_count():
    package = CarterPromptPackage.load()
    specification = _spec()
    specification["fields"] = [
        {"name":"customer_intent","type":"string","required":True,"description":"Intent."},
        {"name":"confidence_label","type":"enum","required":True,"description":"Confidence.","enum_values":["low","high"]},
        {"name":"reasoning_style","type":"string","required":True,"description":"Style."},
    ]
    schema = package.compile_generation_schema(specification, 5)
    contract = json.loads(exact_output_contract_instruction(schema).split("=", 1)[1])
    assert contract["required_top_level_fields"] == ["status", "records", "insufficiency"]
    assert set(contract["top_level_field_shapes"]) == {"status", "records", "insufficiency"}
    assert contract["batch_record_count"] == {"min": 5, "max": 5}
    assert contract["dynamic_fields"] == ["confidence_label", "customer_intent", "reasoning_style"]
    assert set(contract["required_record_fields"]) == {"customer_intent", "confidence_label", "reasoning_style", "evidence"}
    assert contract["output_schema"] == schema


def test_dynamic_output_contract_uses_exact_partial_batch_cardinality():
    package = CarterPromptPackage.load()
    contract = json.loads(exact_output_contract_instruction(package.compile_generation_schema(_spec(), 2)).split("=", 1)[1])
    assert contract["batch_record_count"] == {"min": 2, "max": 2}


def test_safe_dynamic_schema_feedback_reports_count_extra_fields_and_types():
    package = CarterPromptPackage.load()
    schema = package.compile_generation_schema(_spec(), 5)
    invalid = _candidate_many(4)["records"]
    invalid[0]["customer_intent"] = {"not": "a string"}
    invalid[0]["summary"] = "unexpected"
    errors = package.safe_validation_errors(schema, {"status":"generated", "records":invalid, "insufficiency":None})
    assert any("$.records: expected 5 minItems; received 4" == error for error in errors)
    assert any("$.records[0]: unexpected field" == error for error in errors)
    assert any("$.records[0].customer_intent: expected type; received dict" == error for error in errors)


def _review(revise=False):
    return {"status":"completed","recommendation":"revise_recommended" if revise else "accept","summary":"Review.","issues":[{"issue_id":"issue_001","category":"custom_schema_quality","severity":"major","affected_record_refs":["review_record_001"],"affected_field":"customer_intent","description":"Revise.","recommended_correction":"Revise."}] if revise else []}


def test_generation_retries_one_transient_no_content_response(tmp_path):
    provider = ScriptedProvider([CarterInferenceResponse(json.dumps(_spec()), []), ProviderError("PROVIDER_NO_FINAL_CONTENT", "no final content"), CarterInferenceResponse(json.dumps(_candidate()), []), CarterInferenceResponse(json.dumps(_review()), [])])
    run = CarterDatasetGenerationService(CarterPromptPackage.load(), provider, knowledge_path=tmp_path / "retry.sqlite3").generate(runtime="runpod", user_request="Use source", output_format="json", documents=[document()])
    assert run.calls == {"planner": 1, "generator": 2, "tool_continuation": 0, "review": 1, "revision": 0, "revision_repair": 0}
    assert len(run.dataset.records) == 1


def test_generation_allows_a_second_no_content_retry(tmp_path):
    provider = ScriptedProvider([CarterInferenceResponse(json.dumps(_spec()), []), ProviderError("PROVIDER_NO_FINAL_CONTENT", "empty"), ProviderError("PROVIDER_NO_FINAL_CONTENT", "empty"), CarterInferenceResponse(json.dumps(_candidate()), []), CarterInferenceResponse(json.dumps(_review()), [])])
    run = CarterDatasetGenerationService(CarterPromptPackage.load(), provider, knowledge_path=tmp_path / "retry-three.sqlite3").generate(runtime="runpod", user_request="Use source", output_format="json", documents=[document()])
    assert run.calls["generator"] == 3 and len(run.dataset.records) == 1


def test_generation_stops_after_three_no_content_attempts(tmp_path):
    provider = ScriptedProvider([CarterInferenceResponse(json.dumps(_spec()), []), ProviderError("PROVIDER_NO_FINAL_CONTENT", "empty"), ProviderError("PROVIDER_NO_FINAL_CONTENT", "empty"), ProviderError("PROVIDER_NO_FINAL_CONTENT", "empty")])
    with pytest.raises(ProviderError) as error:
        CarterDatasetGenerationService(CarterPromptPackage.load(), provider, knowledge_path=tmp_path / "retry-exhausted.sqlite3").generate(runtime="runpod", user_request="Use source", output_format="json", documents=[document()])
    assert error.value.code == "PROVIDER_NO_FINAL_CONTENT" and provider.calls == 4


def test_review_retries_no_content_then_succeeds_without_regenerating_batches(tmp_path):
    specification = _spec(); specification["requested_record_count"] = specification["effective_record_count"] = 10
    provider = ScriptedProvider([
        CarterInferenceResponse(json.dumps(specification), []),
        CarterInferenceResponse(json.dumps(_candidate_many(5)), []),
        CarterInferenceResponse(json.dumps(_candidate_many(5)), []),
        ProviderError("PROVIDER_NO_FINAL_CONTENT", "empty review"),
        CarterInferenceResponse(json.dumps(_review()), []),
    ])
    service = CarterDatasetGenerationService(CarterPromptPackage.load(), provider, knowledge_path=tmp_path / "review-retry.sqlite3", generation_batch_size=5)
    run = service.generate(runtime="runpod", user_request="Use source", output_format="json", documents=[document()])
    assert len(run.dataset.records) == 10 and service.calls["generator"] == 2 and service.calls["review"] == 2
    assert service.post_generation_telemetry == [{"action": "review", "attempt_number": 2, "max_attempts": 3, "result": "PASS"}]


def test_review_allows_second_no_content_retry_then_succeeds(tmp_path):
    provider = ScriptedProvider([CarterInferenceResponse(json.dumps(_spec()), []), CarterInferenceResponse(json.dumps(_candidate()), []), ProviderError("PROVIDER_NO_FINAL_CONTENT", "empty"), ProviderError("PROVIDER_NO_FINAL_CONTENT", "empty"), CarterInferenceResponse(json.dumps(_review()), [])])
    run = CarterDatasetGenerationService(CarterPromptPackage.load(), provider, knowledge_path=tmp_path / "review-retry-three.sqlite3").generate(runtime="runpod", user_request="Use source", output_format="json", documents=[document()])
    assert len(run.dataset.records) == 1 and run.calls["generator"] == 1 and run.calls["review"] == 3


def test_review_stops_after_three_no_content_attempts_without_regenerating(tmp_path):
    provider = ScriptedProvider([CarterInferenceResponse(json.dumps(_spec()), []), CarterInferenceResponse(json.dumps(_candidate()), []), ProviderError("PROVIDER_NO_FINAL_CONTENT", "empty"), ProviderError("PROVIDER_NO_FINAL_CONTENT", "empty"), ProviderError("PROVIDER_NO_FINAL_CONTENT", "empty")])
    service = CarterDatasetGenerationService(CarterPromptPackage.load(), provider, knowledge_path=tmp_path / "review-exhausted.sqlite3")
    with pytest.raises(ProviderError) as error:
        service.generate(runtime="runpod", user_request="Use source", output_format="json", documents=[document()])
    assert error.value.code == "PROVIDER_NO_FINAL_CONTENT" and service.calls["generator"] == 1 and service.calls["review"] == 3 and provider.calls == 5


def test_review_mixed_no_content_and_malformed_json_shares_three_attempt_ceiling(tmp_path):
    provider = ScriptedProvider([CarterInferenceResponse(json.dumps(_spec()), []), CarterInferenceResponse(json.dumps(_candidate()), []), ProviderError("PROVIDER_NO_FINAL_CONTENT", "empty"), CarterInferenceResponse("not-json", []), CarterInferenceResponse(json.dumps(_review()), [])])
    run = CarterDatasetGenerationService(CarterPromptPackage.load(), provider, knowledge_path=tmp_path / "review-mixed.sqlite3").generate(runtime="runpod", user_request="Use source", output_format="json", documents=[document()])
    assert len(run.dataset.records) == 1 and run.calls["generator"] == 1 and run.calls["review"] == 3


def test_review_cancellation_after_first_no_content_does_not_retry(tmp_path):
    provider = ScriptedProvider([CarterInferenceResponse(json.dumps(_spec()), []), CarterInferenceResponse(json.dumps(_candidate()), []), ProviderError("PROVIDER_NO_FINAL_CONTENT", "empty")])
    service = CarterDatasetGenerationService(CarterPromptPackage.load(), provider, knowledge_path=tmp_path / "review-cancelled.sqlite3", cancelled=lambda: provider.calls >= 3)
    with pytest.raises(ProviderError) as error:
        service.generate(runtime="runpod", user_request="Use source", output_format="json", documents=[document()])
    assert error.value.code == "PROVIDER_NO_FINAL_CONTENT" and service.calls["review"] == 1 and provider.calls == 3


def test_generation_regenerates_malformed_json_once_and_preserves_the_batch(tmp_path):
    provider = ScriptedProvider([CarterInferenceResponse(json.dumps(_spec()), []), CarterInferenceResponse("not-json", []), CarterInferenceResponse(json.dumps(_candidate()), []), CarterInferenceResponse(json.dumps(_review()), [])])
    service = CarterDatasetGenerationService(CarterPromptPackage.load(), provider, knowledge_path=tmp_path / "malformed-once.sqlite3")
    run = service.generate(runtime="runpod", user_request="Use source", output_format="json", documents=[document()])
    assert run.calls["generator"] == 2 and len(run.dataset.records) == 1
    assert provider.calls == 4
    assert "previous response was not valid JSON" in provider.requests[2].messages[-1]["content"]


def test_generation_regenerates_malformed_json_twice_then_succeeds(tmp_path):
    provider = ScriptedProvider([CarterInferenceResponse(json.dumps(_spec()), []), CarterInferenceResponse("not-json", []), CarterInferenceResponse("still-not-json", []), CarterInferenceResponse(json.dumps(_candidate()), []), CarterInferenceResponse(json.dumps(_review()), [])])
    run = CarterDatasetGenerationService(CarterPromptPackage.load(), provider, knowledge_path=tmp_path / "malformed-twice.sqlite3").generate(runtime="runpod", user_request="Use source", output_format="json", documents=[document()])
    assert run.calls["generator"] == 3 and provider.calls == 5 and len(run.dataset.records) == 1


def test_generation_stops_after_three_malformed_json_attempts(tmp_path):
    provider = ScriptedProvider([CarterInferenceResponse(json.dumps(_spec()), []), CarterInferenceResponse("not-json", []), CarterInferenceResponse("not-json", []), CarterInferenceResponse("not-json", [])])
    service = CarterDatasetGenerationService(CarterPromptPackage.load(), provider, knowledge_path=tmp_path / "malformed-exhausted.sqlite3")
    with pytest.raises(ProviderError) as error:
        service.generate(runtime="runpod", user_request="Use source", output_format="json", documents=[document()])
    assert error.value.code == "STRUCTURED_OUTPUT_INVALID" and service.calls["generator"] == 3 and provider.calls == 4


def test_generation_recovers_first_batch_malformed_json_then_continues_batches(tmp_path):
    specification = _spec(); specification["requested_record_count"] = specification["effective_record_count"] = 10
    provider = ScriptedProvider([CarterInferenceResponse(json.dumps(specification), []), CarterInferenceResponse("not-json", []), CarterInferenceResponse(json.dumps(_candidate_many(5)), []), CarterInferenceResponse(json.dumps(_candidate_many(5)), []), CarterInferenceResponse(json.dumps(_review()), [])])
    run = CarterDatasetGenerationService(CarterPromptPackage.load(), provider, knowledge_path=tmp_path / "first-batch-malformed.sqlite3", generation_batch_size=5).generate(runtime="runpod", user_request="Use source", output_format="json", documents=[document()])
    assert run.calls["generator"] == 3 and provider.calls == 5 and len(run.dataset.records) == 10


def test_generation_mixed_no_content_and_malformed_json_never_exceeds_three_attempts(tmp_path):
    provider = ScriptedProvider([CarterInferenceResponse(json.dumps(_spec()), []), ProviderError("PROVIDER_NO_FINAL_CONTENT", "empty"), CarterInferenceResponse("not-json", []), CarterInferenceResponse(json.dumps(_candidate()), []), CarterInferenceResponse(json.dumps(_review()), [])])
    run = CarterDatasetGenerationService(CarterPromptPackage.load(), provider, knowledge_path=tmp_path / "mixed-retry.sqlite3").generate(runtime="runpod", user_request="Use source", output_format="json", documents=[document()])
    assert run.calls["generator"] == 3 and provider.calls == 5 and len(run.dataset.records) == 1


def test_generation_regenerates_batch_two_when_it_returns_the_full_dataset(tmp_path):
    specification = _spec(); specification["requested_record_count"] = specification["effective_record_count"] = 10
    provider = ScriptedProvider([CarterInferenceResponse(json.dumps(specification), []), CarterInferenceResponse(json.dumps(_candidate_many(5)), []), CarterInferenceResponse(json.dumps(_candidate_many(10)), []), CarterInferenceResponse(json.dumps(_candidate_many(5)), []), CarterInferenceResponse(json.dumps(_review()), [])])
    service = CarterDatasetGenerationService(CarterPromptPackage.load(), provider, knowledge_path=tmp_path / "batch-two-full-dataset.sqlite3", generation_batch_size=5)
    run = service.generate(runtime="runpod", user_request="Use source", output_format="json", documents=[document()])
    assert len(run.dataset.records) == 10 and run.calls["generator"] == 3
    batch_two_header = provider.requests[2].messages[-1]["content"]
    assert "TOTAL DATASET REQUEST: 10 records" in batch_two_header
    assert "CURRENT BATCH: 2 of 2" in batch_two_header
    assert "RECORDS ALREADY COMPLETED: 5" in batch_two_header
    assert "CURRENT BATCH TARGET: exactly 5 new records" in batch_two_header
    assert "CURRENT BATCH RECORD RANGE: 6 through 10" in batch_two_header
    assert "DO NOT return all 10 requested records" in batch_two_header
    assert provider.requests[2].response_schema == provider.requests[3].response_schema
    assert "prior response contained 10 records" in provider.requests[3].messages[-1]["content"]


def test_generation_allows_a_second_dynamic_schema_regeneration(tmp_path):
    specification = _spec(); specification["requested_record_count"] = specification["effective_record_count"] = 10
    provider = ScriptedProvider([CarterInferenceResponse(json.dumps(specification), []), CarterInferenceResponse(json.dumps(_candidate_many(5)), []), CarterInferenceResponse(json.dumps(_candidate_many(10)), []), CarterInferenceResponse(json.dumps(_candidate_many(4)), []), CarterInferenceResponse(json.dumps(_candidate_many(5)), []), CarterInferenceResponse(json.dumps(_review()), [])])
    run = CarterDatasetGenerationService(CarterPromptPackage.load(), provider, knowledge_path=tmp_path / "batch-two-second-schema-retry.sqlite3", generation_batch_size=5).generate(runtime="runpod", user_request="Use source", output_format="json", documents=[document()])
    assert len(run.dataset.records) == 10 and run.calls["generator"] == 4 and provider.calls == 6


def test_generation_stops_after_three_dynamic_schema_invalid_batch_attempts(tmp_path):
    specification = _spec(); specification["requested_record_count"] = specification["effective_record_count"] = 10
    provider = ScriptedProvider([CarterInferenceResponse(json.dumps(specification), []), CarterInferenceResponse(json.dumps(_candidate_many(5)), []), CarterInferenceResponse(json.dumps(_candidate_many(10)), []), CarterInferenceResponse(json.dumps(_candidate_many(10)), []), CarterInferenceResponse(json.dumps(_candidate_many(10)), [])])
    service = CarterDatasetGenerationService(CarterPromptPackage.load(), provider, knowledge_path=tmp_path / "batch-two-schema-exhausted.sqlite3", generation_batch_size=5)
    with pytest.raises(ProviderError) as error:
        service.generate(runtime="runpod", user_request="Use source", output_format="json", documents=[document()])
    assert error.value.code == "DYNAMIC_SCHEMA_INVALID" and service.calls["generator"] == 4 and service.calls["review"] == 0


def test_generation_regenerates_schema_invalid_first_batch_without_repeating_it(tmp_path):
    specification = _spec(); specification["requested_record_count"] = specification["effective_record_count"] = 10
    provider = ScriptedProvider([CarterInferenceResponse(json.dumps(specification), []), CarterInferenceResponse(json.dumps(_candidate_many(10)), []), CarterInferenceResponse(json.dumps(_candidate_many(5)), []), CarterInferenceResponse(json.dumps(_candidate_many(5)), []), CarterInferenceResponse(json.dumps(_review()), [])])
    run = CarterDatasetGenerationService(CarterPromptPackage.load(), provider, knowledge_path=tmp_path / "first-batch-schema-retry.sqlite3", generation_batch_size=5).generate(runtime="runpod", user_request="Use source", output_format="json", documents=[document()])
    assert len(run.dataset.records) == 10 and run.calls["generator"] == 3


def test_partial_final_batch_header_and_schema_retry_are_exact(tmp_path):
    specification = _spec(); specification["requested_record_count"] = specification["effective_record_count"] = 12
    provider = ScriptedProvider([CarterInferenceResponse(json.dumps(specification), []), CarterInferenceResponse(json.dumps(_candidate_many(5)), []), CarterInferenceResponse(json.dumps(_candidate_many(5)), []), CarterInferenceResponse(json.dumps(_candidate_many(12)), []), CarterInferenceResponse(json.dumps(_candidate_many(2)), []), CarterInferenceResponse(json.dumps(_review()), [])])
    run = CarterDatasetGenerationService(CarterPromptPackage.load(), provider, knowledge_path=tmp_path / "partial-final-schema-retry.sqlite3", generation_batch_size=5).generate(runtime="runpod", user_request="Use source", output_format="json", documents=[document()])
    assert len(run.dataset.records) == 12 and run.calls["generator"] == 4
    header = provider.requests[3].messages[-1]["content"]
    assert "TOTAL DATASET REQUEST: 12 records" in header
    assert "CURRENT BATCH: 3 of 3" in header
    assert "RECORDS ALREADY COMPLETED: 10" in header
    assert "CURRENT BATCH TARGET: exactly 2 new records" in header
    assert "CURRENT BATCH RECORD RANGE: 11 through 12" in header


def test_mixed_retry_categories_share_the_three_attempt_batch_ceiling(tmp_path):
    provider = ScriptedProvider([CarterInferenceResponse(json.dumps(_spec()), []), ProviderError("PROVIDER_NO_FINAL_CONTENT", "empty"), CarterInferenceResponse("not-json", []), CarterInferenceResponse(json.dumps(_candidate_many(10)), [])])
    service = CarterDatasetGenerationService(CarterPromptPackage.load(), provider, knowledge_path=tmp_path / "mixed-schema-ceiling.sqlite3")
    with pytest.raises(ProviderError) as error:
        service.generate(runtime="runpod", user_request="Use source", output_format="json", documents=[document()])
    assert error.value.code == "DYNAMIC_SCHEMA_INVALID" and service.calls["generator"] == 3 and provider.calls == 4


def test_cancellation_prevents_malformed_json_regeneration(tmp_path):
    provider = ScriptedProvider([CarterInferenceResponse(json.dumps(_spec()), []), CarterInferenceResponse("not-json", [])])
    service = CarterDatasetGenerationService(CarterPromptPackage.load(), provider, knowledge_path=tmp_path / "cancel-malformed.sqlite3", cancelled=lambda: provider.calls >= 2)
    with pytest.raises(CarterPromptPackageError):
        service.generate(runtime="runpod", user_request="Use source", output_format="json", documents=[document()])
    assert service.calls["generator"] == 1 and provider.calls == 2


def test_non_no_content_provider_error_is_not_retried(tmp_path):
    provider = ScriptedProvider([CarterInferenceResponse(json.dumps(_spec()), []), ProviderError("RUNPOD_AUTH_FAILED", "no")])
    with pytest.raises(ProviderError, match="no"):
        CarterDatasetGenerationService(CarterPromptPackage.load(), provider, knowledge_path=tmp_path / "no-retry.sqlite3").generate(runtime="runpod", user_request="Use source", output_format="json", documents=[document()])
    assert provider.calls == 2


def test_tool_continuation_and_revision_are_bounded(tmp_path):
    tool = {"id":"call-1","function":{"name":"get_source_units","arguments":json.dumps({"source_refs":["source_1"]})}}
    provider = ScriptedProvider([CarterInferenceResponse(json.dumps(_spec()), []), CarterInferenceResponse("", [tool]), CarterInferenceResponse(json.dumps(_candidate()), []), CarterInferenceResponse(json.dumps(_review(True)), []), CarterInferenceResponse(json.dumps(_candidate("revised")), [])])
    run = CarterDatasetGenerationService(CarterPromptPackage.load(), provider, knowledge_path=tmp_path / "tool.sqlite3").generate(runtime="runpod", user_request="Use source", output_format="json", documents=[document()])
    assert run.tools_executed == ["get_source_units"] and run.calls["tool_continuation"] == 1
    assert run.revisions == 1 and run.dataset.records[0]["customer_intent"] == "revised"


def test_invalid_bounded_revision_fails_closed(tmp_path):
    provider = ScriptedProvider([CarterInferenceResponse(json.dumps(_spec()), []), CarterInferenceResponse(json.dumps(_candidate()), []), CarterInferenceResponse(json.dumps(_review(True)), []), CarterInferenceResponse(json.dumps({"status":"generated","records":[],"insufficiency":None}), []), CarterInferenceResponse(json.dumps({"status":"generated","records":[],"insufficiency":None}), [])])
    service = CarterDatasetGenerationService(CarterPromptPackage.load(), provider, knowledge_path=tmp_path / "bad.sqlite3")
    with pytest.raises(CarterPromptPackageError):
        service.generate(runtime="runpod", user_request="Use source", output_format="json", documents=[document()])


def test_invalid_full_revision_uses_one_valid_structural_repair_without_regenerating_batches(tmp_path):
    specification = _spec(); specification["requested_record_count"] = specification["effective_record_count"] = 10
    provider = ScriptedProvider([
        CarterInferenceResponse(json.dumps(specification), []),
        CarterInferenceResponse(json.dumps(_candidate_many()), []),
        CarterInferenceResponse(json.dumps(_review(True)), []),
        CarterInferenceResponse(json.dumps(_candidate_many(9)), []),
        CarterInferenceResponse(json.dumps(_candidate_many()), []),
    ])
    service = CarterDatasetGenerationService(CarterPromptPackage.load(), provider, knowledge_path=tmp_path / "revision-repair.sqlite3", generation_batch_size=10)
    run = service.generate(runtime="runpod", user_request="Use source", output_format="json", documents=[document()])
    assert len(run.dataset.records) == 10
    assert run.calls == {"planner": 1, "generator": 1, "tool_continuation": 0, "review": 1, "revision": 1, "revision_repair": 1}
    assert [item["dynamic_schema_validation"] for item in run.revision_telemetry] == ["FAIL", "PASS"]
    assert all(entry["runtime"] == "runpod" for entry in service.invocation_ledger)
    assert provider.requests[3].response_schema["allOf"][0]["then"]["properties"]["records"]["maxItems"] == 10


def test_invalid_full_revision_and_repair_fail_closed_after_one_repair(tmp_path):
    specification = _spec(); specification["requested_record_count"] = specification["effective_record_count"] = 10
    provider = ScriptedProvider([
        CarterInferenceResponse(json.dumps(specification), []),
        CarterInferenceResponse(json.dumps(_candidate_many()), []),
        CarterInferenceResponse(json.dumps(_review(True)), []),
        CarterInferenceResponse(json.dumps(_candidate_many(9)), []),
        CarterInferenceResponse(json.dumps(_candidate_many(9)), []),
    ])
    service = CarterDatasetGenerationService(CarterPromptPackage.load(), provider, knowledge_path=tmp_path / "revision-repair-fail.sqlite3", generation_batch_size=10)
    with pytest.raises(CarterPromptPackageError):
        service.generate(runtime="runpod", user_request="Use source", output_format="json", documents=[document()])
    assert service.calls["generator"] == 1 and service.calls["revision"] == 1 and service.calls["revision_repair"] == 1
    assert len(service.revision_telemetry) == 2


def test_valid_full_revision_never_uses_repair(tmp_path):
    specification = _spec(); specification["requested_record_count"] = specification["effective_record_count"] = 10
    provider = ScriptedProvider([
        CarterInferenceResponse(json.dumps(specification), []),
        CarterInferenceResponse(json.dumps(_candidate_many()), []),
        CarterInferenceResponse(json.dumps(_review(True)), []),
        CarterInferenceResponse(json.dumps(_candidate_many()), []),
    ])
    service = CarterDatasetGenerationService(CarterPromptPackage.load(), provider, knowledge_path=tmp_path / "revision-valid.sqlite3", generation_batch_size=10)
    run = service.generate(runtime="runpod", user_request="Use source", output_format="json", documents=[document()])
    assert len(run.dataset.records) == 10 and run.calls["revision_repair"] == 0


def test_pre_revision_invalid_dataset_fails_before_review(tmp_path):
    specification = _spec(); specification["requested_record_count"] = specification["effective_record_count"] = 10
    provider = ScriptedProvider([CarterInferenceResponse(json.dumps(specification), []), CarterInferenceResponse(json.dumps(_candidate_many(valid=False)), []), CarterInferenceResponse(json.dumps(_candidate_many(valid=False)), []), CarterInferenceResponse(json.dumps(_candidate_many(valid=False)), [])])
    service = CarterDatasetGenerationService(CarterPromptPackage.load(), provider, knowledge_path=tmp_path / "pre-revision-invalid.sqlite3", generation_batch_size=10)
    with pytest.raises(ProviderError, match="DatasetSpec schema"):
        service.generate(runtime="runpod", user_request="Use source", output_format="json", documents=[document()])
    assert service.calls["generator"] == 3 and service.calls["review"] == service.calls["revision"] == service.calls["revision_repair"] == 0


@pytest.mark.parametrize("runtime", ["runpod", "local_lm_studio"])
def test_runtime_is_pinned_for_tool_continuation_review_and_revision(tmp_path, runtime):
    tool = {"id":"call-1","function":{"name":"get_source_units","arguments":json.dumps({"source_refs":["source_1"]})}}
    provider = ScriptedProvider([CarterInferenceResponse(json.dumps(_spec()), []), CarterInferenceResponse("", [tool]), CarterInferenceResponse(json.dumps(_candidate()), []), CarterInferenceResponse(json.dumps(_review(True)), []), CarterInferenceResponse(json.dumps(_candidate("revised")), [])], runtime=runtime)
    service = CarterDatasetGenerationService(CarterPromptPackage.load(), provider, knowledge_path=tmp_path / f"{runtime}.sqlite3")
    run = service.generate(runtime=runtime, user_request="Use source", output_format="json", documents=[document()])
    assert provider.runtime == runtime and run.calls == {"planner": 1, "generator": 1, "tool_continuation": 1, "review": 1, "revision": 1, "revision_repair": 0}
    assert [entry["phase"] for entry in service.invocation_ledger] == ["planner", "generator", "tool_continuation", "review", "revision"]
    assert [entry["runtime"] for entry in service.invocation_ledger] == [runtime] * 5


@pytest.mark.parametrize("runtime", ["runpod", "local_lm_studio"])
def test_selected_runtime_failure_has_no_fallback(tmp_path, runtime):
    provider = ScriptedProvider([CarterInferenceResponse(json.dumps(_spec()), []), CarterInferenceResponse("not-json", []), CarterInferenceResponse("not-json", []), CarterInferenceResponse("not-json", [])], runtime=runtime)
    service = CarterDatasetGenerationService(CarterPromptPackage.load(), provider, knowledge_path=tmp_path / f"{runtime}-failure.sqlite3")
    with pytest.raises(ProviderError) as exc_info:
        service.generate(runtime=runtime, user_request="Use source", output_format="json", documents=[document()])
    assert exc_info.value.code == "STRUCTURED_OUTPUT_INVALID"
    assert provider.calls == 4 and [entry["runtime"] for entry in service.invocation_ledger] == [runtime] * 4


def test_active_run_uses_captured_provider_after_future_selector_changes(tmp_path):
    future_selection = {"runtime": "runpod"}
    tool = {"id":"call-1","function":{"name":"get_source_units","arguments":json.dumps({"source_refs":["source_1"]})}}
    run_a_provider = ScriptedProvider([CarterInferenceResponse(json.dumps(_spec()), []), CarterInferenceResponse("", [tool]), CarterInferenceResponse(json.dumps(_candidate()), []), CarterInferenceResponse(json.dumps(_review(True)), []), CarterInferenceResponse(json.dumps(_candidate("revised")), [])], runtime="runpod")
    run_a_provider.on_infer = lambda call: future_selection.update(runtime="local_lm_studio") if call == 2 else None
    run_a = CarterDatasetGenerationService(CarterPromptPackage.load(), run_a_provider, knowledge_path=tmp_path / "run-a.sqlite3")
    run_a.generate(runtime=future_selection["runtime"], user_request="Run A", output_format="json", documents=[document()])
    assert future_selection["runtime"] == "local_lm_studio"
    assert [entry["runtime"] for entry in run_a.invocation_ledger] == ["runpod"] * 5
    run_b_provider = ScriptedProvider([CarterInferenceResponse(json.dumps(_spec()), []), CarterInferenceResponse(json.dumps(_candidate()), []), CarterInferenceResponse(json.dumps(_review(False)), [])], runtime="local_lm_studio")
    run_b = CarterDatasetGenerationService(CarterPromptPackage.load(), run_b_provider, knowledge_path=tmp_path / "run-b.sqlite3")
    run_b.generate(runtime=future_selection["runtime"], user_request="Run B", output_format="json", documents=[document()])
    assert [entry["runtime"] for entry in run_b.invocation_ledger] == ["local_lm_studio"] * 3
