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
from ..providers.contracts import ProviderError
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
                 on_phase: Callable[[str, dict[str, int] | None], None] | None = None, cancelled: Callable[[], bool] | None = None,
                 generation_batch_size: int = 5):
        self.package, self.provider = package, provider
        if not 1 <= generation_batch_size <= 20: raise ValueError("generation_batch_size must be between 1 and 20")
        self.knowledge_path, self.on_phase, self.generation_batch_size = knowledge_path, on_phase or (lambda _phase, _batch=None: None), generation_batch_size
        self.cancelled = cancelled or (lambda: False)
        self.calls: dict[str, int] = {name: 0 for name in ("planner", "generator", "tool_continuation", "review", "revision")}
        self.invocation_ledger: list[dict[str, str]] = []

    def _phase(self, name: str, batch: dict[str, int] | None = None) -> None:
        try:
            self.on_phase(name, batch)
        except TypeError:
            self.on_phase(name)

    def _infer(self, request: Any, phase: str):
        if self.cancelled():
            raise CarterPromptPackageError("Generation cancelled.")
        self.calls[phase] += 1
        self.invocation_ledger.append({"phase": phase, "runtime": self.provider.runtime})
        transport = getattr(self.provider, "provider", None)
        if transport is not None:
            transport.current_phase = phase
        response = self.provider.infer(TransportRequest(messages=list(request.messages), tools=list(request.tools),
            response_schema=request.response_schema, tool_choice="auto"))
        return response

    @staticmethod
    def _generation_source_units(documents: list[CanonicalExtractedDocument]) -> tuple[list[dict[str, Any]], set[str]]:
        """Provide the bounded source context required by the generation contract.

        A normal generation turn with this context has no retrieval requirement,
        so it must not advertise unrelated OpenAI tools to a GPT-OSS transport.
        """
        units: list[dict[str, Any]] = []
        budget = 12_000
        for document in documents:
            for element in document.elements:
                text = element.text.strip()
                if not text or budget <= 0:
                    continue
                text = text[:min(1_200, budget)]
                budget -= len(text)
                units.append({"source_ref": element.element_id, "document_name": document.source_filename,
                              "section": " / ".join(element.section_path), "page": element.page_number,
                              "text": text})
        return units, {unit["source_ref"] for unit in units}

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
        source_units, allowed_refs = self._generation_source_units(documents)
        if not allowed_refs:
            raise CarterPromptPackageError("No usable source units are available.")

        self._phase("planning")
        planner = self.package.resolve_operation("dataset_planning")
        logical_runtime = "cloud" if runtime == "runpod" else "local"
        plan_request = self.package.render(planner, {"user_request": user_request,
            "requested_output_format": output_format, "application_limits": {"maximum_dataset_records": 100},
            "selected_document_metadata": [{"document_id": d.document_id, "name": d.source_filename} for d in documents]}, runtime=logical_runtime)
        specification = self._json(self._infer(plan_request, "planner").content, "Planner returned malformed JSON.")
        try:
            self.package.validate(planner.output_schema, specification)
        except CarterPromptPackageError as exc:
            raise ProviderError("DYNAMIC_SCHEMA_INVALID", "Planner result did not match the application schema.") from exc
        # Compile is both semantic validation and the single canonical compiler.
        requested_records = specification["effective_record_count"]
        batch_targets = [min(self.generation_batch_size, requested_records - offset) for offset in range(0, requested_records, self.generation_batch_size)]

        registry = CarterToolRegistry(self.package, build_knowledge_tool_handlers(store, document_ids, allowed_refs))
        tools_executed: list[str] = []
        merged_records: list[dict[str, Any]] = []
        for batch_index, target in enumerate(batch_targets, 1):
            batch = {"currentBatch": batch_index, "totalBatches": len(batch_targets), "recordsGenerated": len(merged_records), "recordsRequested": requested_records, "currentBatchTarget": target}
            self._phase("generating", batch)
            compiled = self.package.compile_generation_schema(specification, target)
            generator = self.package.resolve_operation("dataset_generation", compiled)
            request = self.package.render(generator, {"dataset_spec": specification, "user_request": user_request,
                "selected_document_ids": sorted(document_ids), "source_units": source_units,
                "allowed_source_refs": sorted(allowed_refs), "batch_number": batch_index,
                "total_batches": len(batch_targets), "batch_record_target": target}, runtime=logical_runtime)
            # The frozen operation allows retrieval only when the application
            # enables it.  This bounded source payload is sufficient, therefore
            # no native tools are sent and vLLM owns no Harmony tool conversion.
            request = request.__class__(request.runtime, request.operation, request.messages, (),
                                        request.response_schema, request.package_fingerprint)
            state = CarterAgentTurnState(self.package, compiled); messages = list(request.messages); batch_tools: list[str] = []
            while True:
                current = request.__class__(request.runtime, request.operation, tuple(messages), request.tools, request.response_schema, request.package_fingerprint)
                response = self._infer(current, "generator" if not batch_tools else "tool_continuation")
                content = response.content if response.tool_calls else {"action": "final_response", "final_response": self._json(response.content, "Generator returned malformed JSON.")}
                action = state.normalize(response.tool_calls, content)
                if action["action"] == "final_response":
                    records = action["final_response"]["records"]
                    if len(records) != target: raise CarterPromptPackageError("Generator returned an incorrect batch record count.")
                    try:
                        validated = validate_canonical_dataset(self.package, specification, records, allowed_source_refs=allowed_refs, batch_count=target)
                    except CarterPromptPackageError as exc:
                        raise ProviderError("DYNAMIC_SCHEMA_INVALID", "Generated records did not match the DatasetSpec schema.") from exc
                    merged_records.extend(validated.records); break
                self._phase("tool_use", batch); call = action["tool_call"]; result = registry.execute(call["name"], call["arguments"])
                batch_tools.append(call["name"]); tools_executed.append(call["name"]); messages.append({"role": "tool", "name": call["name"], "content": json.dumps(result, separators=(",", ":"))})
            batch["recordsGenerated"] = len(merged_records); self._phase("generating", batch)
        candidate = validate_canonical_dataset(self.package, specification, merged_records, allowed_source_refs=allowed_refs, batch_count=requested_records)

        self._phase("reviewing")
        review_op = self.package.resolve_operation("quality_review")
        refs = {f"review_record_{index:03d}" for index, _ in enumerate(candidate.records, 1)}
        review_request = self.package.render(review_op, {"dataset_spec": specification, "user_request": user_request,
            "records": [{"record_ref": ref, "record": record} for ref, record in zip(sorted(refs), candidate.records)]}, runtime=logical_runtime)
        review = self._json(self._infer(review_request, "review").content, "Quality review returned malformed JSON.")
        validate_quality_review(self.package, review, refs, {field["name"] for field in specification["fields"]})
        revisions = 0
        if review["recommendation"] == "revise_recommended":
            authorize_revision(revisions, True); revisions = 1
            self._phase("revising")
            # A revision is a fresh bounded generator turn using the same compiled contract/runtime.
            revision_request = self.package.render(generator, {"dataset_spec": specification, "candidate_records": list(candidate.records),
                "review": review, "selected_document_ids": sorted(document_ids), "allowed_source_refs": sorted(allowed_refs)}, runtime=logical_runtime)
            revised = self._json(self._infer(revision_request, "revision").content, "Revision returned malformed JSON.")
            candidate = validate_canonical_dataset(self.package, specification, revised["records"], allowed_source_refs=allowed_refs)
        return CarterRunResult(candidate, specification, review, dict(self.calls), tools_executed, revisions)
