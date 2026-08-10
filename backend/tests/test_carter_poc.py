from pathlib import Path
import json

import pytest

from app.domain.extraction_models import CanonicalExtractedDocument, ExtractionElement, ExtractionElementType, ExtractionStatistics, ExtractionValidation
from app.providers.contracts import ProviderError
from app.services.carter import CarterAskService, CarterInferenceResponse, KnowledgeStore, MAX_DOCUMENTS, MAX_TOOL_ROUNDS
from app.providers.runpod import _safe_output_snapshot, build_runpod_openai_chat_job
from app.api.routes.carter import AskRequest
from pydantic import ValidationError


def document(identifier: str, text: str = "Regression testing checks existing behavior after change."):
    return CanonicalExtractedDocument(documentId=identifier, sourceFileId=identifier, sourceFilename=f"{identifier}.txt", mimeType="text/plain", extractor="plain_text", statistics=ExtractionStatistics(characterCount=len(text), wordCount=len(text.split()), elementCount=1, headingCount=0, paragraphCount=1, listItemCount=0, tableCount=0, tableRowCount=0, nonEmptyElementCount=1), blocks=[ExtractionElement(elementId=f"ref-{identifier}", type=ExtractionElementType.PARAGRAPH, text=text, order=1)], validation=ExtractionValidation(valid=True))


def test_knowledge_store_indexes_and_filters_documents(tmp_path: Path):
    store = KnowledgeStore(tmp_path / "knowledge.sqlite3")
    store.ingest(document("one")); store.ingest(document("two", "API testing validates request boundaries."))
    assert [item["documentId"] for item in store.documents()] == ["one", "two"]
    assert store.search("regression")[0]["sourceRef"] == "ref-one"
    assert store.search("testing", ["two"])[0]["documentId"] == "two"
    assert store.source_units(["ref-one"])[0]["documentName"] == "one.txt"
    with pytest.raises(ValueError): store.source_units(["missing"])


def test_knowledge_store_caps_documents(tmp_path: Path):
    store = KnowledgeStore(tmp_path / "knowledge.sqlite3")
    for index in range(MAX_DOCUMENTS): store.ingest(document(str(index)))
    with pytest.raises(ValueError, match="at most 3"): store.ingest(document("four"))


class ScriptedProvider:
    runtime = "local"
    def __init__(self, responses): self.responses, self.calls, self.requests = list(responses), 0, []
    def available(self): return {"configured": True, "available": True}
    def infer(self, request):
        self.calls += 1; self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception): raise response
        return response


def service(tmp_path, responses):
    store = KnowledgeStore(tmp_path / "knowledge.sqlite3")
    store.ingest(document("doc-a", "Functional testing checks feature workflows. Negative testing checks invalid input."))
    store.ingest(document("doc-b", "Accessibility testing includes keyboard operation. Regression testing checks existing behavior."))
    return CarterAskService(store, ScriptedProvider(responses))


def call(name, arguments, call_id="call-1"):
    return CarterInferenceResponse("", [{"id": call_id, "function": {"name": name, "arguments": arguments}}])


def test_carter_final_answer_without_tool(tmp_path):
    subject = service(tmp_path, [call("search_local_knowledge", {"query": "functional", "documentIds": ["doc-a"]}), CarterInferenceResponse(json.dumps({"answer": "Grounded answer", "citations": [{"sourceRef": "ref-doc-a"}]}), [])])
    result = subject.ask("functional testing", ["doc-a"])
    assert result["answer"] == "Grounded answer" and result["toolRounds"] == 1 and result["retrievalResults"] == 1


def test_carter_one_two_and_three_tool_rounds_execute_handlers(tmp_path):
    subject = service(tmp_path, [
        call("list_documents", {}),
        call("search_local_knowledge", {"query": "accessibility", "documentIds": ["doc-b"]}, "call-2"),
        call("get_source_units", {"sourceRefs": ["ref-doc-b"]}, "call-3"),
        CarterInferenceResponse(json.dumps({"answer": "Keyboard operation is covered.", "citations": [{"sourceRef": "ref-doc-b"}]}), []),
    ])
    result = subject.ask("accessibility", ["doc-a", "doc-b"])
    assert result["toolRounds"] == 3 and subject.provider.calls == 4


def test_carter_rejects_fourth_tool_round(tmp_path):
    subject = service(tmp_path, [call("list_documents", {}, str(index)) for index in range(4)])
    with pytest.raises(ProviderError, match="three tool rounds"):
        subject.ask("functional testing")
    assert subject.provider.calls == MAX_TOOL_ROUNDS + 1


@pytest.mark.parametrize("name,arguments", [
    ("unknown", {}), ("list_documents", {"path": "../../secret.txt"}),
    ("search_local_knowledge", "not-json"), ("search_local_knowledge", {"query": 3}),
    ("search_local_knowledge", {"query": "x", "documentIds": ["../../secret.txt"]}),
    ("get_source_units", {"sourceRefs": ["..\\..\\secret.txt"]}),
    ("get_source_units", {"sourceRefs": ["C:\\Users\\example\\secret.txt"]}),
    ("get_source_units", {"sourceRefs": ["file:///etc/passwd"]}),
    ("get_source_units", {"sourceRefs": ["\\\\server\\share"]}),
])
def test_carter_rejects_malformed_unknown_and_path_like_tool_arguments(tmp_path, name, arguments):
    subject = service(tmp_path, [call(name, arguments)])
    with pytest.raises(ValueError): subject.ask("functional testing")


def test_carter_rejects_unknown_document_duplicate_ids_and_provider_failure(tmp_path):
    subject = service(tmp_path, [CarterInferenceResponse("", [])])
    with pytest.raises(ValueError, match="not found"): subject.ask("functional", ["missing"])
    with pytest.raises(ValueError, match="Duplicate"): subject.ask("functional", ["doc-a", "doc-a"])
    subject = service(tmp_path, [ProviderError("TIMEOUT", "provider timeout")])
    with pytest.raises(ProviderError, match="provider timeout"): subject.ask("functional")


@pytest.mark.parametrize("runtime", ["local_lm_studio", "runpod"])
def test_explicit_runtime_contract_accepts_only_supported_values(runtime):
    assert AskRequest(question="question", runtime=runtime).runtime == runtime


def test_explicit_runtime_contract_rejects_unknown_value():
    with pytest.raises(ValidationError):
        AskRequest(question="question", runtime="banana-provider")


def test_runpod_tool_transport_preserves_canonical_tools_without_structured_answer_wrapper():
    tools = [{"type": "function", "function": {"name": "search_local_knowledge", "parameters": {"type": "object"}}}]
    payload = build_runpod_openai_chat_job(model="openai/gpt-oss-20b", messages=[{"role": "user", "content": "find evidence"}], schema=None, tools=tools, max_tokens=4096)
    request = payload["input"]["openai_input"]
    assert request["tools"] == tools and request["tool_choice"] == "auto"
    assert "structured_outputs" not in request


def test_runpod_safe_protocol_snapshot_records_tool_shape_without_content():
    snapshot = _safe_output_snapshot({"choices": [{"finish_reason": "tool_calls", "message": {"content": None, "reasoning": "never retained", "tool_calls": [{"id": "call-1", "function": {"name": "search_local_knowledge", "arguments": "{}"}}]}}]})
    assert snapshot["tool_call_count"] == 1
    assert snapshot["tool_calls"] == [{"id_present": True, "name": "search_local_knowledge", "arguments_type": "str"}]
    assert snapshot["reasoning_present"] is True and "never retained" not in str(snapshot)


def test_tool_continuation_keeps_runtime_pinned(tmp_path):
    subject = service(tmp_path, [call("search_local_knowledge", {"query": "functional"}), CarterInferenceResponse(json.dumps({"answer": "Grounded answer", "citations": [{"sourceRef": "ref-doc-a"}]}), [])])
    result = CarterAskService(subject.store, subject.provider, "runpod").ask("functional testing")
    assert result["runtime"] == "runpod" and subject.provider.calls == 2


@pytest.mark.parametrize("runtime", ["runpod", "local_lm_studio"])
def test_tool_loop_pins_the_selected_runtime_for_initial_and_continuation(tmp_path, runtime):
    subject = service(tmp_path, [call("search_local_knowledge", {"query": "functional"}), CarterInferenceResponse(json.dumps({"answer": "Grounded answer", "citations": [{"sourceRef": "ref-doc-a"}]}), [])])
    result = CarterAskService(subject.store, subject.provider, runtime).ask("functional testing", ["doc-a"])
    assert result["runtime"] == runtime and subject.provider.calls == 2
    assert [request.tool_choice for request in subject.provider.requests] == ["required", "auto"]


def test_carter_rejects_unstructured_or_ungrounded_final_response(tmp_path):
    subject = service(tmp_path, [call("search_local_knowledge", {"query": "functional"}), CarterInferenceResponse("A prose answer", [])])
    with pytest.raises(ProviderError, match="invalid structured"): subject.ask("functional testing", ["doc-a"])
    subject = service(tmp_path, [call("search_local_knowledge", {"query": "functional"}), CarterInferenceResponse(json.dumps({"answer": "Wrong citation", "citations": [{"sourceRef": "not-real"}]}), [])])
    with pytest.raises(ProviderError, match="citations not grounded"): subject.ask("functional testing", ["doc-a"])
