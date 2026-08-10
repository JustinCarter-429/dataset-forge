import json
import re
from datetime import datetime, timezone
from typing import Any, Callable
from ..domain.extraction_models import CanonicalExtractedDocument
from ..domain.models import CanonicalDataset, DatasetRecord, EvidenceReference
from ..providers.contracts import DatasetGenerationProvider, ProviderError
from ..prompts.dataset_author_v2 import PROMPT_VERSION, SYSTEM_PROMPT, repair_prompt, user_prompt
from .context_projection import ContextBatch, build_context_batches
from .validation import Phase4ValidationService, StructuralDatasetValidator, normalize_source_text


class DatasetGenerator:
    """Compatibility protocol for pipeline type boundaries."""
    def generate(self, document: CanonicalExtractedDocument, dataset_prompt: str) -> CanonicalDataset:
        raise NotImplementedError


# Permit an intentional modifier between the number and record noun (for
# example, "8 source-grounded records") while remaining bounded to one line.
RECORD_COUNT_PATTERN = re.compile(r"\b(\d{1,3})\b[^\n]{0,40}?\b(?:records?|examples?|items?)\b", re.IGNORECASE)


def _normalize_provider_records(raw: list[dict[str, Any]], generation_id: str, file_id: str, model: str) -> list[DatasetRecord]:
    normalized: list[DatasetRecord] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("A generated record was not an object.")
        context = str(item.get("context", item.get("input", "")))
        expected_output = str(item.get("expected_output", item.get("output", "")))
        raw_evidence = item.get("evidence") or []
        evidence = [EvidenceReference(source_ref=str(value.get("source_ref", "")), quote=str(value.get("quote", ""))) for value in raw_evidence if isinstance(value, dict)]
        source_refs = [str(value) for value in (item.get("source_refs") or [])]
        record = DatasetRecord(instruction=str(item.get("instruction", "")), input=context, output=expected_output, context=context, expected_output=expected_output, category=str(item.get("category", "general")), difficulty=str(item.get("difficulty", "medium")), source_refs=source_refs, evidence=evidence, metadata={str(k): str(v) for k, v in (item.get("metadata") or {}).items() if k not in {"record_id", "generation_id", "file_id", "source_refs", "evidence"}})
        record.metadata.update({"record_id": f"{generation_id}-{len(normalized) + 1}", "generation_id": generation_id, "file_id": file_id, "provider": "runpod_serverless", "model": model, "prompt_version": PROMPT_VERSION, "created_at": datetime.now(timezone.utc).isoformat()})
        normalized.append(record)
    return normalized


def requested_record_count(prompt: str, limit: int) -> int:
    match = RECORD_COUNT_PATTERN.search(prompt)
    return min(int(match.group(1)) if match else 4, limit)


def dataset_schema() -> dict[str, Any]:
    return {"type": "object", "properties": {"records": {"type": "array", "items": {"type": "object", "properties": {"instruction": {"type": "string"}, "context": {"type": "string"}, "expected_output": {"type": "string"}, "category": {"type": "string"}, "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]}, "source_refs": {"type": "array", "items": {"type": "string"}}, "evidence": {"type": "array", "items": {"type": "object", "properties": {"source_ref": {"type": "string"}, "quote": {"type": "string"}}, "required": ["source_ref", "quote"], "additionalProperties": False}}}, "required": ["instruction", "context", "expected_output", "category", "difficulty", "source_refs", "evidence"], "additionalProperties": False}}}, "required": ["records"], "additionalProperties": False}


def _extract_provider_content(output: Any) -> Any:
    """Extract assistant content from the documented RunPod/vLLM result shapes."""
    if isinstance(output, list) and len(output) == 1 and isinstance(output[0], dict) and "choices" in output[0]:
        output = output[0]
    if isinstance(output, dict) and isinstance(output.get("error"), dict):
        raise ProviderError("RUNPOD_WORKER_ERROR", "RunPod worker returned an inference error.")
    if isinstance(output, dict) and "choices" in output:
        choices = output["choices"]
        if not isinstance(choices, list) or not choices:
            raise ValueError("Provider output contained no choices.")
        choice = choices[0]
        if not isinstance(choice, dict):
            raise ValueError("Provider choice was not an object.")
        message = choice.get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str) and message["content"].strip():
            output = message["content"]
        elif isinstance(choice.get("text"), str) and choice["text"].strip():
            output = choice["text"]
        elif isinstance(choice.get("output"), str) and choice["output"].strip():
            output = choice["output"]
        else:
            raise ValueError("Provider choice contained no assistant content.")
    return output


def _parse_provider_output(output: Any) -> list[dict[str, Any]]:
    """Parse assistant content as a dataset result; never parse reasoning."""
    output = _extract_provider_content(output)
    if isinstance(output, dict) and "records" in output:
        records = output["records"]
    elif isinstance(output, list):
        records = output
    elif isinstance(output, str):
        cleaned = output.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            # vLLM workers can append a short non-JSON suffix after a valid
            # structured object. Keep parsing bounded to the first object.
            start = cleaned.find("{")
            if start < 0: raise
            parsed, _ = json.JSONDecoder().raw_decode(cleaned[start:])
        records = parsed.get("records") if isinstance(parsed, dict) else parsed
    else:
        raise ValueError("Provider output was empty or not JSON.")
    if not isinstance(records, list) or not records:
        raise ValueError("Provider output did not contain records.")
    return records


class RunPodDatasetGenerator:
    def __init__(self, provider: DatasetGenerationProvider, *, max_model_len: int, records_per_batch: int, max_dataset_records: int):
        self.provider = provider
        self.max_model_len = max_model_len
        self.records_per_batch = records_per_batch
        self.max_dataset_records = max_dataset_records
        self.validator = StructuralDatasetValidator()
        self.grounding_validator = Phase4ValidationService()
        self.last_run: dict[str, Any] = {"initial_generation_attempts": 0, "revision_attempts": 0, "review_attempts": 0, "provider_jobs": 0, "repair_reason": None, "revision_reason": []}

    @staticmethod
    def _canonicalize_source_refs(raw: list[dict[str, Any]], batch: ContextBatch) -> list[dict[str, Any]]:
        def resolve(value: Any) -> str:
            ref = str(value)
            if ref in batch.alias_to_canonical:
                return batch.alias_to_canonical[ref]
            raise ValueError(f"UNKNOWN_OR_CROSS_BATCH_SOURCE_REF: {ref} is not an alias allowed for {batch.batch_id}")
        normalized: list[dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict):
                normalized.append(item)
                continue
            candidate = dict(item)
            evidence = []
            for entry in item.get("evidence") or []:
                if not isinstance(entry, dict):
                    evidence.append(entry)
                    continue
                evidence.append({**entry, "source_ref": resolve(entry.get("source_ref", ""))})
            source_refs = [resolve(value) for value in (item.get("source_refs") or [])]
            evidence_refs = [entry["source_ref"] for entry in evidence if isinstance(entry, dict) and entry.get("source_ref")]
            if source_refs and evidence_refs and set(source_refs) != set(evidence_refs):
                raise ValueError("SOURCE_REF_EVIDENCE_MISMATCH: record source_refs and evidence disagree")
            for entry in evidence:
                if not isinstance(entry, dict):
                    continue
                canonical_ref = entry.get("source_ref", "")
                quoteable = batch.quoteable_text_by_canonical.get(canonical_ref, "")
                if normalize_source_text(str(entry.get("quote", ""))) not in normalize_source_text(quoteable):
                    raise ValueError(f"BATCH_EVIDENCE_NOT_FOUND: quote is not in the exact projection for {canonical_ref}")
            candidate["evidence"] = evidence
            candidate["source_refs"] = sorted(set(evidence_refs or source_refs))
            normalized.append(candidate)
        return normalized

    def _normalize(self, raw: list[dict[str, Any]], batch: ContextBatch, generation_id: str, file_id: str, model: str) -> list[DatasetRecord]:
        return _normalize_provider_records(self._canonicalize_source_refs(raw, batch), generation_id, file_id, model)

    def generate(self, document: CanonicalExtractedDocument, dataset_prompt: str, *, generation_id: str, file_id: str, model: str, on_progress: Callable[[int, str], None] | None = None) -> CanonicalDataset:
        self.last_run = {"initial_generation_attempts": 0, "revision_attempts": 0, "review_attempts": 0, "provider_jobs": 0, "repair_reason": None, "revision_reason": []}
        target = requested_record_count(dataset_prompt, self.max_dataset_records)
        batches = build_context_batches(document, max_model_len=self.max_model_len, records_per_batch=self.records_per_batch, record_limit=target)
        self.last_run["generation_batch_count"] = len(batches)
        records: list[DatasetRecord] = []
        for batch_index, batch in enumerate(batches):
            if on_progress:
                on_progress(int(35 + batch_index / max(1, len(batches)) * 40), f"batch {batch_index + 1} of {len(batches)}")
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt(dataset_prompt, batch.text, batch.batch_id, batch.requested_records)}]
            response = self.provider.generate(messages=messages, schema=dataset_schema(), max_tokens=min(8192, self.max_model_len // 4))
            self.last_run["initial_generation_attempts"] += 1
            self.last_run["provider_jobs"] += 1
            try:
                batch_records = self._normalize(_parse_provider_output(response.output), batch, generation_id, file_id, model)
                validation = self.validator.validate(CanonicalDataset(records=batch_records))
                grounding = self.grounding_validator.validate(CanonicalDataset(records=batch_records), document)[1] if any(isinstance(item, dict) and ("source_refs" in item or "evidence" in item) for item in _parse_provider_output(response.output)) else None
                if not validation.valid or grounding is not None and grounding.status == "failed":
                    raise ValueError("; ".join(validation.errors) or "BATCH_GROUNDING_FAILED")
            except (ValueError, json.JSONDecodeError) as exc:
                if self.last_run["revision_attempts"] >= 1:
                    raise ProviderError("STRUCTURED_OUTPUT_INVALID", "RunPod output failed canonical dataset validation after the single global revision budget was used.") from exc
                self.last_run["revision_attempts"] += 1
                self.last_run["repair_reason"] = str(exc)[:240]
                repair_messages = messages + [{"role": "user", "content": repair_prompt(str(response.output), str(exc))}]
                repaired = self.provider.generate(messages=repair_messages, schema=dataset_schema(), max_tokens=min(8192, self.max_model_len // 4))
                self.last_run["provider_jobs"] += 1
                try:
                    batch_records = self._normalize(_parse_provider_output(repaired.output), batch, generation_id, file_id, model)
                    validation = self.validator.validate(CanonicalDataset(records=batch_records))
                    grounding = self.grounding_validator.validate(CanonicalDataset(records=batch_records), document)[1] if any(isinstance(item, dict) and ("source_refs" in item or "evidence" in item) for item in _parse_provider_output(repaired.output)) else None
                    if not validation.valid or grounding is not None and grounding.status == "failed":
                        raise ValueError("; ".join(validation.errors) or "BATCH_GROUNDING_FAILED")
                except (ValueError, json.JSONDecodeError) as repair_exc:
                    raise ProviderError("STRUCTURED_OUTPUT_INVALID", f"RunPod output failed canonical dataset validation after one bounded repair: {str(repair_exc)[:160]}") from repair_exc
            # IDs belong to the application, not the provider batch. Rebase
            # each accepted record onto the global assembly position so a
            # multi-batch dataset cannot collide or confuse quality review.
            accepted = batch_records[: max(0, target - len(records))]
            for offset, record in enumerate(accepted, start=len(records) + 1):
                record.metadata["record_id"] = f"{generation_id}-{offset}"
            records.extend(accepted)
        if not records:
            raise ProviderError("EMPTY_DATASET", "RunPod returned no usable dataset records.")
        return CanonicalDataset(records=records[: self.max_dataset_records])
