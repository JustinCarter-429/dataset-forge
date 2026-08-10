import json
import inspect
from dataclasses import dataclass

from app.carter.production import CarterDatasetGenerationService
from app.carter.runtime import CarterPromptPackage
from app.domain.extraction_models import (CanonicalExtractedDocument, ExtractionElement,
    ExtractionElementType, ExtractionStatistics, ExtractionValidation)
from app.providers.deterministic import DeterministicCarterProvider
from app.services.carter import CarterInferenceResponse
from app.carter.runtime import CarterPromptPackageError
from app.api.routes import generation as generation_route


def document():
    return CanonicalExtractedDocument(documentId="doc-1", sourceFileId="file-1", sourceFilename="source.txt", mimeType="text/plain", extractor="plain_text", statistics=ExtractionStatistics(characterCount=42, wordCount=6, elementCount=1, headingCount=0, paragraphCount=1, listItemCount=0, tableCount=0, tableRowCount=0, nonEmptyElementCount=1), blocks=[ExtractionElement(elementId="source_1", type=ExtractionElementType.PARAGRAPH, text="Customers can request support.", order=1)], validation=ExtractionValidation(valid=True))


def test_production_orchestrator_preserves_dynamic_records(tmp_path):
    run = CarterDatasetGenerationService(CarterPromptPackage.load(), DeterministicCarterProvider("runpod"), knowledge_path=tmp_path / "knowledge.sqlite3").generate(runtime="runpod", user_request="Create a custom support dataset", output_format="json", documents=[document()])
    assert run.specification["dataset_type"] == "custom"
    assert run.dataset.records[0]["customer_intent"] == "support request"
    assert run.calls == {"planner": 1, "generator": 1, "tool_continuation": 0, "review": 1, "revision": 0}


def test_generation_route_uses_carter_not_legacy_generator():
    source = inspect.getsource(generation_route._run_job)
    assert "CarterDatasetGenerationService" in source
    assert "RunPodDatasetGenerator(" not in source


@dataclass
class ScriptedProvider:
    replies: list[CarterInferenceResponse]
    runtime: str = "runpod"
    calls: int = 0
    def available(self): return {"configured": True, "available": True, "model": "scripted"}
    def infer(self, _request):
        self.calls += 1
        return self.replies.pop(0)


def _spec():
    return {"status":"ready","dataset_type":"custom","dataset_name":"custom","dataset_description":"custom","requested_record_count":1,"effective_record_count":1,"fields":[{"name":"customer_intent","type":"string","required":True,"description":"Intent."}],"source_policy":"selected_documents_only","grounding_required":True,"evidence_required":True,"generation_requirements":["source_grounded","avoid_exact_duplicates"],"user_constraints":[],"clarification":{"required":False,"reason_code":None,"question":None,"reason":None}}


def _candidate(value="request"):
    return {"status":"generated","records":[{"customer_intent":value,"evidence":[{"source_ref":"source_1","quote":"Customers can request support."}]}],"insufficiency":None}


def _review(revise=False):
    return {"status":"completed","recommendation":"revise_recommended" if revise else "accept","summary":"Review.","issues":[{"issue_id":"issue_001","category":"custom_schema_quality","severity":"major","affected_record_refs":["review_record_001"],"affected_field":"customer_intent","description":"Revise.","recommended_correction":"Revise."}] if revise else []}


def test_tool_continuation_and_revision_are_bounded(tmp_path):
    tool = {"id":"call-1","function":{"name":"get_source_units","arguments":json.dumps({"source_refs":["source_1"]})}}
    provider = ScriptedProvider([CarterInferenceResponse(json.dumps(_spec()), []), CarterInferenceResponse("", [tool]), CarterInferenceResponse(json.dumps(_candidate()), []), CarterInferenceResponse(json.dumps(_review(True)), []), CarterInferenceResponse(json.dumps(_candidate("revised")), [])])
    run = CarterDatasetGenerationService(CarterPromptPackage.load(), provider, knowledge_path=tmp_path / "tool.sqlite3").generate(runtime="runpod", user_request="Use source", output_format="json", documents=[document()])
    assert run.tools_executed == ["get_source_units"] and run.calls["tool_continuation"] == 1
    assert run.revisions == 1 and run.dataset.records[0]["customer_intent"] == "revised"


def test_invalid_bounded_revision_fails_closed(tmp_path):
    provider = ScriptedProvider([CarterInferenceResponse(json.dumps(_spec()), []), CarterInferenceResponse(json.dumps(_candidate()), []), CarterInferenceResponse(json.dumps(_review(True)), []), CarterInferenceResponse(json.dumps({"status":"generated","records":[],"insufficiency":None}), [])])
    service = CarterDatasetGenerationService(CarterPromptPackage.load(), provider, knowledge_path=tmp_path / "bad.sqlite3")
    import pytest
    with pytest.raises(CarterPromptPackageError):
        service.generate(runtime="runpod", user_request="Use source", output_format="json", documents=[document()])
