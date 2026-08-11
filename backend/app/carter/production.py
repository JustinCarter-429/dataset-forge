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
    revision_telemetry: list[dict[str, Any]] = field(default_factory=list)


class CarterDatasetGenerationService:
    """Provider-neutral, fail-closed Carter production orchestrator."""
    def __init__(self, package: CarterPromptPackage, provider: CarterProvider, *, knowledge_path: Path,
                 on_phase: Callable[[str, dict[str, int] | None], None] | None = None, cancelled: Callable[[], bool] | None = None,
                 generation_batch_size: int = 5, generation_no_content_retries: int = 2):
        self.package, self.provider = package, provider
        if not 1 <= generation_batch_size <= 20: raise ValueError("generation_batch_size must be between 1 and 20")
        if not 0 <= generation_no_content_retries <= 2: raise ValueError("generation_no_content_retries must be between 0 and 2")
        self.knowledge_path, self.on_phase, self.generation_batch_size, self.generation_no_content_retries = knowledge_path, on_phase or (lambda _phase, _batch=None: None), generation_batch_size, generation_no_content_retries
        self.cancelled = cancelled or (lambda: False)
        self.calls: dict[str, int] = {name: 0 for name in ("planner", "generator", "tool_continuation", "review", "revision", "revision_repair")}
        self.invocation_ledger: list[dict[str, str]] = []
        self.revision_telemetry: list[dict[str, Any]] = []

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

    def _revision_snapshot(self, *, phase: str, attempt: int, response: Any | None = None,
                           json_parse: str = "NOT_RUN", schema: str = "NOT_RUN",
                           record_count: int | None = None, safe_error_code: str | None = None) -> dict[str, Any]:
        """Preserve bounded revision telemetry without retaining provider content."""
        transport = getattr(self.provider, "provider", None)
        provider = getattr(transport, "last_telemetry", {}) if transport is not None else {}
        item: dict[str, Any] = {
            "phase": phase,
            "attempt": attempt,
            "provider_job_id": provider.get("external_job_id"),
            "terminal_state": provider.get("terminal_state"),
            "content_present": bool(response and isinstance(response.content, str) and response.content.strip()),
            "reasoning_present": provider.get("reasoning_present"),
            "finish_reason": provider.get("finish_reason"),
            "json_parse": json_parse,
            "dynamic_schema_validation": schema,
            "record_count": record_count,
            "safe_error_code": safe_error_code,
        }
        return item

    def _revision_request(self, *, generator: Any, specification: dict[str, Any], requested_records: int,
                          candidate: CarterCanonicalDataset, review: dict[str, Any], user_request: str,
                          document_ids: set[str], source_units: list[dict[str, Any]], allowed_refs: set[str],
                          runtime: str, repair_errors: list[str] | None = None) -> Any:
        """Build a full-dataset revision request from the same authoritative DatasetSpec."""
        inputs: dict[str, Any] = {
            "dataset_spec": specification,
            "user_request": user_request,
            "selected_document_ids": sorted(document_ids),
            "source_units": source_units,
            "allowed_source_refs": sorted(allowed_refs),
            "generation_batch": {"batch_id": "revision", "batch_index": 1, "total_batches": 1,
                                 "target_record_count": requested_records},
            "batch_number": 1,
            "total_batches": 1,
            "batch_record_target": requested_records,
            "candidate_records": list(candidate.records),
            "review": review,
            "revision_contract": {
                "mode": "full_canonical_dataset",
                "required_record_count": requested_records,
                "instruction": "Return only the complete canonical dataset object. Preserve DatasetSpec field names, types, evidence, and record count.",
            },
        }
        if repair_errors:
            inputs["revision_repair"] = {
                "structural_validation_errors": repair_errors,
                "instruction": "Repair only the structural errors. Return only the required complete canonical dataset object with no commentary.",
            }
        return self.package.render(generator, inputs, runtime=runtime)

    def _validate_revision(self, *, content: Any, specification: dict[str, Any], requested_records: int,
                           allowed_refs: set[str]) -> CarterCanonicalDataset:
        revised = self._json(content, "Revision returned malformed JSON.")
        records = revised.get("records")
        if not isinstance(records, list):
            raise CarterPromptPackageError("Revision did not return a records array.")
        if len(records) != requested_records:
            raise CarterPromptPackageError("Revision returned an incorrect full-dataset record count.")
        return validate_canonical_dataset(self.package, specification, records,
                                          allowed_source_refs=allowed_refs,
                                          batch_count=requested_records)

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
                phase = "generator" if not batch_tools else "tool_continuation"
                for attempt in range(1, self.generation_no_content_retries + 2):
                    try:
                        response = self._infer(current, phase)
                        break
                    except ProviderError as exc:
                        if phase != "generator" or exc.code != "PROVIDER_NO_FINAL_CONTENT" or attempt > self.generation_no_content_retries or self.cancelled():
                            raise
                        # Reuse this exact batch request; only a completed
                        # no-content provider response is transient here.
                        self._phase("generating", batch)
                final_response = None
                try:
                    content = response.content if response.tool_calls else {"action": "final_response", "final_response": self._json(response.content, "Generator returned malformed JSON.")}
                    final_response = content.get("final_response") if isinstance(content, dict) else None
                    action = state.normalize(response.tool_calls, content)
                except CarterPromptPackageError as exc:
                    errors = self.package.safe_validation_errors(compiled, final_response) if isinstance(final_response, dict) else []
                    transport = getattr(self.provider, "provider", None)
                    code = "DYNAMIC_SCHEMA_INVALID" if isinstance(final_response, dict) else "STRUCTURED_OUTPUT_INVALID"
                    if transport is not None and hasattr(transport, "_publish_telemetry"):
                        transport._publish_telemetry(json_parse="PASS" if isinstance(final_response, dict) else "FAIL",
                                                     dynamic_schema_validation="FAIL" if isinstance(final_response, dict) else "NOT_RUN", structural_errors=errors,
                                                     record_count=len(final_response.get("records", [])) if isinstance(final_response, dict) and isinstance(final_response.get("records"), list) else None,
                                                     safe_error_code=code)
                    raise ProviderError(code, "Generated records did not match the DatasetSpec schema.") from exc
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
            # Revision always returns the complete candidate set, so it must not
            # reuse the final generation batch's (possibly smaller) schema.
            revision_schema = self.package.compile_generation_schema(specification, requested_records)
            revision_generator = self.package.resolve_operation("dataset_generation", revision_schema)
            revision_request = self._revision_request(generator=revision_generator, specification=specification,
                requested_records=requested_records, candidate=candidate, review=review, user_request=user_request,
                document_ids=document_ids, source_units=source_units, allowed_refs=allowed_refs, runtime=logical_runtime)
            response = self._infer(revision_request, "revision")
            try:
                candidate = self._validate_revision(content=response.content, specification=specification,
                    requested_records=requested_records, allowed_refs=allowed_refs)
                self.revision_telemetry.append(self._revision_snapshot(phase="revision", attempt=1, response=response,
                    json_parse="PASS", schema="PASS", record_count=len(candidate.records)))
            except CarterPromptPackageError as exc:
                self.revision_telemetry.append(self._revision_snapshot(phase="revision", attempt=1, response=response,
                    json_parse="PASS", schema="FAIL", safe_error_code="DYNAMIC_SCHEMA_INVALID"))
                # One bounded structural repair is distinct from the one
                # application-authorized quality revision and never regenerates batches.
                repair_request = self._revision_request(generator=revision_generator, specification=specification,
                    requested_records=requested_records, candidate=candidate, review=review, user_request=user_request,
                    document_ids=document_ids, source_units=source_units, allowed_refs=allowed_refs, runtime=logical_runtime,
                    repair_errors=[exc.detail])
                repair = self._infer(repair_request, "revision_repair")
                try:
                    candidate = self._validate_revision(content=repair.content, specification=specification,
                        requested_records=requested_records, allowed_refs=allowed_refs)
                    self.revision_telemetry.append(self._revision_snapshot(phase="revision_repair", attempt=1, response=repair,
                        json_parse="PASS", schema="PASS", record_count=len(candidate.records)))
                except CarterPromptPackageError:
                    self.revision_telemetry.append(self._revision_snapshot(phase="revision_repair", attempt=1, response=repair,
                        json_parse="PASS", schema="FAIL", safe_error_code="DYNAMIC_SCHEMA_INVALID"))
                    raise
        return CarterRunResult(candidate, specification, review, dict(self.calls), tools_executed, revisions, self.revision_telemetry)
