"""Production Carter dataset orchestration.

This module is deliberately the only application-level path from a generation
job to the Carter prompt package.  Provider implementations remain below the
``CarterProvider`` boundary in :mod:`app.services.carter`.
"""
from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..domain.extraction_models import CanonicalExtractedDocument
from ..services.carter import CarterInferenceRequest as TransportRequest, CarterProvider, KnowledgeStore
from ..providers.contracts import ProviderError
from .dynamic_dataset import CarterCanonicalDataset, validate_canonical_dataset
from .count_policy import CountPlan, resolve_count
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
    count_plan: CountPlan | None = None
    auto_stop_reason: str | None = None
    review_record_map: list[dict[str, str]] = field(default_factory=list)


class CarterDatasetGenerationService:
    """Provider-neutral, fail-closed Carter production orchestrator."""
    MAX_BATCH_PROVIDER_ATTEMPTS = 3
    MAX_STRUCTURED_OUTPUT_REGENERATIONS = 2
    MAX_DYNAMIC_SCHEMA_REGENERATIONS = 2
    MAX_POST_GENERATION_PROVIDER_ATTEMPTS = 3
    def __init__(self, package: CarterPromptPackage, provider: CarterProvider, *, knowledge_path: Path,
                 on_phase: Callable[[str, dict[str, int] | None], None] | None = None, cancelled: Callable[[], bool] | None = None,
                 generation_batch_size: int = 5, generation_no_content_retries: int = 2):
        self.package, self.provider = package, provider
        if not 1 <= generation_batch_size <= 20: raise ValueError("generation_batch_size must be between 1 and 20")
        if not 0 <= generation_no_content_retries <= 2: raise ValueError("generation_no_content_retries must be between 0 and 2")
        self.knowledge_path, self.on_phase, self.generation_batch_size, self.generation_no_content_retries = knowledge_path, on_phase or (lambda _phase, _batch=None: None), generation_batch_size, generation_no_content_retries
        self.cancelled = cancelled or (lambda: False)
        self.auto_max_records = 100
        self.calls: dict[str, int] = {name: 0 for name in ("planner", "generator", "tool_continuation", "review", "revision", "revision_repair")}
        self.invocation_ledger: list[dict[str, str]] = []
        self.revision_telemetry: list[dict[str, Any]] = []
        self.post_generation_telemetry: list[dict[str, Any]] = []

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

    def _publish_batch_telemetry(self, *, batch_number: int, attempt_number: int,
                                 retry_category: str | None = None, **changes: Any) -> None:
        """Attach bounded batch-attempt facts without retaining provider content."""
        transport = getattr(self.provider, "provider", None)
        if transport is not None and hasattr(transport, "_publish_telemetry"):
            transport._publish_telemetry(batch_number=batch_number, attempt_number=attempt_number,
                                         maximum_attempts=self.MAX_BATCH_PROVIDER_ATTEMPTS,
                                         retry_category=retry_category, **changes)

    def _publish_post_generation_telemetry(self, *, action: str, attempt_number: int,
                                           **changes: Any) -> None:
        """Record bounded post-generation attempt facts without provider output."""
        transport = getattr(self.provider, "provider", None)
        if transport is not None and hasattr(transport, "_publish_telemetry"):
            transport._publish_telemetry(action=action, attempt_number=attempt_number,
                                         maximum_attempts=self.MAX_POST_GENERATION_PROVIDER_ATTEMPTS,
                                         **changes)

    def _review_with_retries(self, request: Any, *, refs: set[str], fields: set[str]) -> dict[str, Any]:
        """Retry only review output-contract failures, preserving generated candidates."""
        for attempt in range(1, self.MAX_POST_GENERATION_PROVIDER_ATTEMPTS + 1):
            if self.cancelled():
                raise CarterPromptPackageError("Generation cancelled.")
            try:
                response = self._infer(request, "review")
            except ProviderError as exc:
                self._publish_post_generation_telemetry(action="review", attempt_number=attempt,
                                                        json_parse="NOT_RUN", dynamic_schema_validation="NOT_RUN",
                                                        safe_error_code=exc.code)
                if (exc.code != "PROVIDER_NO_FINAL_CONTENT" or self.cancelled()
                        or attempt == self.MAX_POST_GENERATION_PROVIDER_ATTEMPTS):
                    raise
                self._phase("reviewing")
                continue
            try:
                review = self._json(response.content, "Quality review returned malformed JSON.")
                # Carter 1.0 examples used three-digit references. Normalize
                # that legacy spelling only when it identifies an existing
                # six-digit application-owned reference; all other refs still
                # fail closed.
                for issue in review.get("issues", []) if isinstance(review, dict) else []:
                    if isinstance(issue, dict) and isinstance(issue.get("affected_record_refs"), list):
                        issue["affected_record_refs"] = [
                            f"review_record_{ref.rsplit('_', 1)[-1].zfill(6)}" if isinstance(ref, str) and ref.startswith("review_record_") and ref.rsplit("_", 1)[-1].isdigit() and f"review_record_{ref.rsplit('_', 1)[-1].zfill(6)}" in refs else ref
                            for ref in issue["affected_record_refs"]
                        ]
            except CarterPromptPackageError as exc:
                code = "STRUCTURED_OUTPUT_INVALID"
                self._publish_post_generation_telemetry(action="review", attempt_number=attempt,
                                                        json_parse="FAIL", dynamic_schema_validation="NOT_RUN",
                                                        content_length=len(response.content) if isinstance(response.content, str) else 0,
                                                        safe_error_code=code)
                if self.cancelled() or attempt == self.MAX_POST_GENERATION_PROVIDER_ATTEMPTS:
                    raise ProviderError(code, "Quality review returned malformed JSON.") from exc
                self._phase("reviewing")
                continue
            try:
                validate_quality_review(self.package, review, refs, fields)
            except CarterPromptPackageError as exc:
                code = "DYNAMIC_SCHEMA_INVALID"
                self._publish_post_generation_telemetry(action="review", attempt_number=attempt,
                                                        json_parse="PASS", dynamic_schema_validation="FAIL",
                                                        content_length=len(response.content) if isinstance(response.content, str) else 0,
                                                        safe_error_code=code)
                if self.cancelled() or attempt == self.MAX_POST_GENERATION_PROVIDER_ATTEMPTS:
                    raise ProviderError(code, "Quality review did not match its output contract.") from exc
                self._phase("reviewing")
                continue
            self._publish_post_generation_telemetry(action="review", attempt_number=attempt,
                                                    json_parse="PASS", dynamic_schema_validation="PASS",
                                                    content_length=len(response.content) if isinstance(response.content, str) else 0)
            self.post_generation_telemetry.append({"action": "review", "attempt_number": attempt,
                                                   "max_attempts": self.MAX_POST_GENERATION_PROVIDER_ATTEMPTS,
                                                   "result": "PASS"})
            return review
        raise AssertionError("review retry loop exhausted without returning or raising")

    @staticmethod
    def _verified_review(review: dict[str, Any], record_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Keep model review advisory unless duplicate claims are reproducible."""
        verified, advisory = [], []
        for issue in review.get("issues", []):
            refs = issue.get("affected_record_refs", [])
            invalid = len(refs) != len(set(refs)) or any(ref not in record_map for ref in refs)
            duplicate_claim = issue.get("category") == "semantic_repetition"
            hashes = [record_map[ref]["normalized_hash"] for ref in refs if ref in record_map]
            confirmed = not duplicate_claim or (len(hashes) > 1 and len(set(hashes)) < len(hashes))
            if invalid or not confirmed:
                advisory.append({**issue, "verification": "unverified_advisory_finding"})
            else:
                verified.append({**issue, "verification": "verified"})
        result = dict(review)
        result["issues"] = verified + advisory
        result["verifiedIssues"] = verified
        result["unverifiedAdvisoryFindings"] = advisory
        result["reviewFindingVerification"] = "failed" if advisory else "passed"
        if review.get("recommendation") == "revise_recommended" and not any(item.get("severity") == "major" for item in verified):
            result["recommendation"] = "accept"
            result["revisionSkipReason"] = "unverified_ai_finding"
        return result

    @staticmethod
    def _batch_execution_header(*, requested_records: int, batch_number: int, total_batches: int,
                                records_completed: int, batch_target: int) -> str:
        """Make the authoritative batch assignment explicit without changing frozen prompts."""
        first = records_completed + 1
        last = records_completed + batch_target
        return (
            "BATCH_EXECUTION_HEADER\n"
            f"TOTAL DATASET REQUEST: {requested_records} records\n"
            f"CURRENT BATCH: {batch_number} of {total_batches}\n"
            f"RECORDS ALREADY COMPLETED: {records_completed}\n"
            f"CURRENT BATCH TARGET: exactly {batch_target} new records\n"
            f"CURRENT BATCH RECORD RANGE: {first} through {last}\n"
            f"RESPONSE REQUIREMENT: Return exactly {batch_target} records in this response.\n"
            f"DO NOT return all {requested_records} requested records, repeat records from prior batches, "
            f"return records belonging to another batch, or return more or fewer than {batch_target} records."
        )

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
            "requested_output_format": output_format, "application_limits": {"maximum_dataset_records": self.auto_max_records},
            "selected_document_metadata": [{"document_id": d.document_id, "name": d.source_filename} for d in documents]}, runtime=logical_runtime)
        specification = self._json(self._infer(plan_request, "planner").content, "Planner returned malformed JSON.")
        try:
            self.package.validate(planner.output_schema, specification)
        except CarterPromptPackageError as exc:
            raise ProviderError("DYNAMIC_SCHEMA_INVALID", "Planner result did not match the application schema.") from exc
        # Carter 1.0's planner contract has a frozen fallback.  Count policy is
        # deliberately applied after validating that contract, so no silent
        # default can determine production generation.
        combined = documents[0].model_copy(update={"elements": [element for document in documents for element in document.elements]})
        count_plan = resolve_count(user_request, combined, self.auto_max_records)
        # Old internal callers used terse non-dataset commands such as "Use
        # source". They are retained only as a compatibility shim; ordinary
        # natural-language dataset requests always enter auto mode.
        if count_plan.mode == "auto" and not any(word in user_request.lower() for word in ("dataset", "record", "example", "question", "answer", "classification", "scenario", "instruction")):
            count_plan = CountPlan("explicit", specification["effective_record_count"], None, None, None, specification["effective_record_count"], False)
        specification = dict(specification)
        specification["requested_record_count"] = count_plan.requested
        specification["effective_record_count"] = count_plan.target
        requested_records = count_plan.target
        batch_targets = [min(self.generation_batch_size, requested_records - offset) for offset in range(0, requested_records, self.generation_batch_size)]

        registry = CarterToolRegistry(self.package, build_knowledge_tool_handlers(store, document_ids, allowed_refs))
        tools_executed: list[str] = []
        merged_records: list[dict[str, Any]] = []
        for batch_index, target in enumerate(batch_targets, 1):
            batch = {"currentBatch": batch_index, "totalBatches": len(batch_targets), "recordsGenerated": len(merged_records), "recordsRequested": requested_records, "currentBatchTarget": target}
            batch_header = self._batch_execution_header(requested_records=requested_records, batch_number=batch_index,
                                                        total_batches=len(batch_targets), records_completed=len(merged_records),
                                                        batch_target=target)
            self._phase("generating", batch)
            compiled = self.package.compile_generation_schema(specification, target)
            generator = self.package.resolve_operation("dataset_generation", compiled)
            request = self.package.render(generator, {"dataset_spec": specification, "user_request": user_request,
                "selected_document_ids": sorted(document_ids), "source_units": source_units,
                "allowed_source_refs": sorted(allowed_refs), "batch_number": batch_index,
                "total_batches": len(batch_targets), "batch_record_target": target,
                "records_completed_before_batch": len(merged_records),
                "current_batch_record_range": {"first": len(merged_records) + 1, "last": len(merged_records) + target}}, runtime=logical_runtime)
            # The frozen operation allows retrieval only when the application
            # enables it.  This bounded source payload is sufficient, therefore
            # no native tools are sent and vLLM owns no Harmony tool conversion.
            request = request.__class__(request.runtime, request.operation, request.messages + ({"role": "system", "content": batch_header},), (),
                                        request.response_schema, request.package_fingerprint)
            state = CarterAgentTurnState(self.package, compiled); messages = list(request.messages); batch_tools: list[str] = []
            batch_provider_attempts = 0; no_content_retries = 0; structured_regenerations = 0; dynamic_schema_regenerations = 0
            while True:
                current = request.__class__(request.runtime, request.operation, tuple(messages), request.tools, request.response_schema, request.package_fingerprint)
                phase = "generator" if not batch_tools else "tool_continuation"
                if self.cancelled(): raise CarterPromptPackageError("Generation cancelled.")
                if batch_provider_attempts >= self.MAX_BATCH_PROVIDER_ATTEMPTS:
                    raise ProviderError("STRUCTURED_OUTPUT_INVALID", "Generation batch exhausted its provider-attempt budget.")
                batch_provider_attempts += 1
                try:
                    response = self._infer(current, phase)
                except ProviderError as exc:
                    if (phase != "generator" or exc.code != "PROVIDER_NO_FINAL_CONTENT" or self.cancelled()
                            or no_content_retries >= self.generation_no_content_retries
                            or batch_provider_attempts >= self.MAX_BATCH_PROVIDER_ATTEMPTS):
                        raise
                    no_content_retries += 1
                    # Reuse this exact batch request; the total ceiling prevents
                    # no-content and malformed-output retries from stacking.
                    self._phase("generating", batch)
                    continue
                final_response = None
                try:
                    content = response.content if response.tool_calls else {"action": "final_response", "final_response": self._json(response.content, "Generator returned malformed JSON.")}
                    final_response = content.get("final_response") if isinstance(content, dict) else None
                    action = state.normalize(response.tool_calls, content)
                except CarterPromptPackageError as exc:
                    errors = self.package.safe_validation_errors(compiled, final_response) if isinstance(final_response, dict) else []
                    code = "DYNAMIC_SCHEMA_INVALID" if isinstance(final_response, dict) else "STRUCTURED_OUTPUT_INVALID"
                    self._publish_batch_telemetry(batch_number=batch_index, attempt_number=batch_provider_attempts,
                                                  retry_category="structured_output" if code == "STRUCTURED_OUTPUT_INVALID" else "dynamic_schema",
                                                  json_parse="PASS" if isinstance(final_response, dict) else "FAIL",
                                                  dynamic_schema_validation="FAIL" if isinstance(final_response, dict) else "NOT_RUN",
                                                  structural_errors=errors,
                                                  record_count=len(final_response.get("records", [])) if isinstance(final_response, dict) and isinstance(final_response.get("records"), list) else None,
                                                  expected_record_count=target,
                                                  content_length=len(response.content) if isinstance(response.content, str) else 0,
                                                  safe_error_code=code)
                    if self.cancelled(): raise CarterPromptPackageError("Generation cancelled.")
                    retryable = code in {"STRUCTURED_OUTPUT_INVALID", "DYNAMIC_SCHEMA_INVALID"}
                    exhausted_category = (structured_regenerations >= self.MAX_STRUCTURED_OUTPUT_REGENERATIONS if code == "STRUCTURED_OUTPUT_INVALID"
                                          else dynamic_schema_regenerations >= self.MAX_DYNAMIC_SCHEMA_REGENERATIONS)
                    if (not retryable or phase != "generator" or exhausted_category
                            or batch_provider_attempts >= self.MAX_BATCH_PROVIDER_ATTEMPTS):
                        raise ProviderError(code, "Generated records did not match the DatasetSpec schema.") from exc
                    if code == "STRUCTURED_OUTPUT_INVALID":
                        structured_regenerations += 1
                        correction = "The previous response was not valid JSON. Return a completely new response containing exactly one valid JSON object that matches the supplied output contract. Do not include markdown, prose, comments, code fences, or any text outside the JSON object. Safe parser feedback: invalid JSON syntax."
                    else:
                        dynamic_schema_regenerations += 1
                        feedback = "; ".join(errors) or "output-contract validation failed"
                        actual = len(final_response.get("records", [])) if isinstance(final_response, dict) and isinstance(final_response.get("records"), list) else "an invalid count"
                        correction = f"Your prior response contained {actual} records, but this batch requires exactly {target} records. Generate a completely new response containing exactly {target} records for the current batch only. Safe structural feedback: {feedback}."
                    messages.append({"role": "system", "content": correction})
                    self._phase("generating", batch)
                    continue
                self._publish_batch_telemetry(batch_number=batch_index, attempt_number=batch_provider_attempts,
                                              json_parse="PASS" if isinstance(final_response, dict) else "NOT_RUN",
                                              dynamic_schema_validation="PASS" if isinstance(final_response, dict) else "NOT_RUN",
                                              record_count=len(final_response.get("records", [])) if isinstance(final_response, dict) and isinstance(final_response.get("records"), list) else None,
                                              expected_record_count=target,
                                              content_length=len(response.content) if isinstance(response.content, str) else 0)
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
        review_map = []
        for index, record in enumerate(candidate.records, 1):
            normalized_hash = hashlib.sha256(json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
            review_map.append({"review_ref": f"review_record_{index:06d}", "canonical_index": str(index), "normalized_hash": normalized_hash})
        refs = {item["review_ref"] for item in review_map}
        review_lookup = {item["review_ref"]: item for item in review_map}
        review_request = self.package.render(review_op, {"dataset_spec": specification, "user_request": user_request,
            "records": [{"record_ref": item["review_ref"], "canonical_index": item["canonical_index"], "normalized_hash": item["normalized_hash"], "record": record} for item, record in zip(review_map, candidate.records)]}, runtime=logical_runtime)
        review = self._review_with_retries(review_request, refs=refs,
                                            fields={field["name"] for field in specification["fields"]})
        review = self._verified_review(review, review_lookup)
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
        stop = "explicit_target_reached" if count_plan.mode == "explicit" else ("hard_cap_reached" if count_plan.hard_cap_limited else "source_coverage_complete")
        return CarterRunResult(candidate, specification, review, dict(self.calls), tools_executed, revisions, self.revision_telemetry, count_plan, stop, review_map)
