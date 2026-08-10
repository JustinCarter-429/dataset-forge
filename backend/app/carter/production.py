"""Production Carter dataset orchestration.

This module is deliberately the only application-level path from a generation
job to the Carter prompt package.  Provider implementations remain below the
``CarterProvider`` boundary in :mod:`app.services.carter`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..domain.extraction_models import CanonicalExtractedDocument
from ..services.carter import CarterInferenceRequest as TransportRequest, CarterProvider, KnowledgeStore
from .dynamic_dataset import CarterCanonicalDataset, validate_canonical_dataset
from .runtime import (CarterAgentTurnState, CarterPromptPackage, CarterPromptPackageError,
                      CarterToolRegistry, authorize_revision, build_knowledge_tool_handlers,
                      validate_quality_review)


@dataclass
class CarterRunResult:
    dataset: CarterCanonicalDataset
    specification: dict[str, Any]
    review: dict[str, Any]
    calls: dict[str, int] = field(default_factory=dict)
    tools_executed: list[str] = field(default_factory=list)
    revisions: int = 0


class CarterDatasetGenerationService:
    """Provider-neutral, fail-closed Carter production orchestrator."""
    def __init__(self, package: CarterPromptPackage, provider: CarterProvider, *, knowledge_path: Path,
                 on_phase: Callable[[str], None] | None = None, cancelled: Callable[[], bool] | None = None):
        self.package, self.provider = package, provider
        self.knowledge_path, self.on_phase = knowledge_path, on_phase or (lambda _phase: None)
        self.cancelled = cancelled or (lambda: False)
        self.calls: dict[str, int] = {name: 0 for name in ("planner", "generator", "tool_continuation", "review", "revision")}

    def _infer(self, request: Any, phase: str):
        if self.cancelled():
            raise CarterPromptPackageError("Generation cancelled.")
        self.calls[phase] += 1
        response = self.provider.infer(TransportRequest(messages=list(request.messages), tools=list(request.tools),
            response_schema=request.response_schema, tool_choice="auto"))
        return response

    @staticmethod
    def _json(content: Any, error: str) -> dict[str, Any]:
        try:
            value = content if isinstance(content, dict) else json.loads(content)
        except (TypeError, json.JSONDecodeError) as exc:
            raise CarterPromptPackageError(error) from exc
        if not isinstance(value, dict):
            raise CarterPromptPackageError(error)
        return value

    def generate(self, *, runtime: str, user_request: str, output_format: str,
                 documents: list[CanonicalExtractedDocument]) -> CarterRunResult:
        # Loading here, before any provider invocation, keeps package corruption fail-closed.
        if not self.package.package_version.startswith("1.0"):
            raise CarterPromptPackageError("Unsupported Carter package version.")
        store = KnowledgeStore(self.knowledge_path)
        store.reset()
        for document in documents:
            store.ingest(document)
        document_ids = {document.document_id for document in documents}
        allowed_refs = {element.element_id for document in documents for element in document.elements if element.text.strip()}
        if not allowed_refs:
            raise CarterPromptPackageError("No usable source units are available.")

        self.on_phase("planning")
        planner = self.package.resolve_operation("dataset_planning")
        logical_runtime = "cloud" if runtime == "runpod" else "local"
        plan_request = self.package.render(planner, {"user_request": user_request,
            "requested_output_format": output_format, "application_limits": {"maximum_dataset_records": 100},
            "selected_document_metadata": [{"document_id": d.document_id, "name": d.source_filename} for d in documents]}, runtime=logical_runtime)
        specification = self._json(self._infer(plan_request, "planner").content, "Planner returned malformed JSON.")
        self.package.validate(planner.output_schema, specification)
        # Compile is both semantic validation and the single canonical compiler.
        compiled = self.package.compile_generation_schema(specification, specification["effective_record_count"])

        self.on_phase("generating")
        generator = self.package.resolve_operation("dataset_generation", compiled)
        request = self.package.render(generator, {"dataset_spec": specification, "user_request": user_request, "selected_document_ids": sorted(document_ids),
            "allowed_source_refs": sorted(allowed_refs)}, runtime=logical_runtime)
        registry = CarterToolRegistry(self.package, build_knowledge_tool_handlers(store, document_ids, allowed_refs))
        state = CarterAgentTurnState(self.package, compiled)
        messages = list(request.messages)
        tools_executed: list[str] = []
        while True:
            current = request.__class__(request.runtime, request.operation, tuple(messages), request.tools,
                                        request.response_schema, request.package_fingerprint)
            response = self._infer(current, "generator" if not tools_executed else "tool_continuation")
            content = response.content
            # Native tool calls are normalized by Part 1; a structured terminal result is
            # wrapped only at this boundary to fit the canonical agent-action contract.
            if not response.tool_calls:
                content = {"action": "final_response", "final_response": self._json(content, "Generator returned malformed JSON.")}
            action = state.normalize(response.tool_calls, content)
            if action["action"] == "final_response":
                candidate = validate_canonical_dataset(self.package, specification, action["final_response"]["records"], allowed_source_refs=allowed_refs)
                break
            self.on_phase("tool_use")
            call = action["tool_call"]
            result = registry.execute(call["name"], call["arguments"])
            tools_executed.append(call["name"])
            messages.append({"role": "tool", "name": call["name"], "content": json.dumps(result, separators=(",", ":"))})

        self.on_phase("reviewing")
        review_op = self.package.resolve_operation("quality_review")
        refs = {f"review_record_{index:03d}" for index, _ in enumerate(candidate.records, 1)}
        review_request = self.package.render(review_op, {"dataset_spec": specification, "user_request": user_request,
            "records": [{"record_ref": ref, "record": record} for ref, record in zip(sorted(refs), candidate.records)]}, runtime=logical_runtime)
        review = self._json(self._infer(review_request, "review").content, "Quality review returned malformed JSON.")
        validate_quality_review(self.package, review, refs, {field["name"] for field in specification["fields"]})
        revisions = 0
        if review["recommendation"] == "revise_recommended":
            authorize_revision(revisions, True); revisions = 1
            self.on_phase("revising")
            # A revision is a fresh bounded generator turn using the same compiled contract/runtime.
            revision_request = self.package.render(generator, {"dataset_spec": specification, "candidate_records": list(candidate.records),
                "review": review, "selected_document_ids": sorted(document_ids), "allowed_source_refs": sorted(allowed_refs)}, runtime=logical_runtime)
            revised = self._json(self._infer(revision_request, "revision").content, "Revision returned malformed JSON.")
            candidate = validate_canonical_dataset(self.package, specification, revised["records"], allowed_source_refs=allowed_refs)
        return CarterRunResult(candidate, specification, review, dict(self.calls), tools_executed, revisions)
