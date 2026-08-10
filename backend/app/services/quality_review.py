import json
from collections import Counter
from typing import Any

from ..domain.models import CanonicalDataset, DatasetRecord, QualityReview, QualityReviewIssue, ReviewSummary, ValidationReport
from ..providers.contracts import DatasetGenerationProvider, ProviderError
from ..prompts.dataset_quality_review_v1 import PROMPT_VERSION, SYSTEM_PROMPT, revision_prompt, user_prompt
from .context_projection import estimate_tokens
from .generation import _extract_provider_content, _parse_provider_output, _normalize_provider_records, dataset_schema

MAX_REVIEW_ISSUES = 25
MAX_REVIEW_MESSAGE_CHARS = 320
MAX_REVIEW_ACTION_CHARS = 240
REVIEW_VERSION = "1.0"
ISSUE_CODES = {
    "REPETITIVE_RECORDS", "LOW_INSTRUCTION_DIVERSITY", "WEAK_CONTEXT", "WEAK_EXPECTED_OUTPUT",
    "AMBIGUOUS_INSTRUCTION", "DIFFICULTY_MISMATCH", "CATEGORY_INCONSISTENCY", "REDUNDANT_TEST_CASE",
    "SOURCE_SUPPORT_CONCERN", "OVERLY_BROAD_ANSWER", "OVERLY_TRIVIAL_RECORD", "POOR_DATASET_COVERAGE",
    "OTHER_BOUNDED_QUALITY_ISSUE",
}
BLOCKING_CODES = {"AMBIGUOUS_INSTRUCTION", "SOURCE_SUPPORT_CONCERN"}


class RevisionPolicy:
    max_revision_attempts = 1
    allow_structural_repair = True
    allow_grounding_repair = True
    allow_quality_revision = True


def review_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "issues": {"type": "array", "maxItems": MAX_REVIEW_ISSUES, "items": {"type": "object", "properties": {
                "code": {"type": "string", "enum": sorted(ISSUE_CODES)},
                "severity": {"type": "string", "enum": ["warning", "blocking"]},
                "record_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 20},
                "message": {"type": "string", "minLength": 1, "maxLength": MAX_REVIEW_MESSAGE_CHARS},
                "suggested_action": {"type": "string", "minLength": 1, "maxLength": MAX_REVIEW_ACTION_CHARS},
            }, "required": ["code", "severity", "record_ids", "message", "suggested_action"], "additionalProperties": False}},
            "summary": {"type": "string", "maxLength": 500},
        },
        "required": ["issues", "summary"],
        "additionalProperties": False,
    }


def _record_id(record: DatasetRecord) -> str:
    return record.metadata.get("record_id", "")


def _review_input(dataset: CanonicalDataset, report: ValidationReport, records: list[DatasetRecord]) -> str:
    category_counts = Counter(record.category for record in dataset.records)
    difficulty_counts = Counter(record.difficulty for record in dataset.records)
    payload = [{
        "record_id": _record_id(record),
        "instruction": record.instruction,
        "context": record.context,
        "expected_output": record.expected_output,
        "category": record.category,
        "difficulty": record.difficulty,
        "source_refs": record.source_refs,
        "evidence": [item.model_dump(mode="json") for item in record.evidence],
    } for record in records]
    summary = {
        "schema_version": report.schema_version,
        "grounding_status": report.grounding.status,
        "grounded_records": report.grounding.grounded_records,
        "category_counts": dict(sorted(category_counts.items())),
        "difficulty_distribution": dict(sorted(difficulty_counts.items())),
        "duplicate_warnings": report.duplicates,
    }
    return json.dumps({"records": payload, "validation_summary": summary}, ensure_ascii=False, sort_keys=True)


def _decode_json_object(value: str) -> Any:
    cleaned = value.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        if start < 0:
            raise
        parsed, _ = json.JSONDecoder().raw_decode(cleaned[start:])
        return parsed


def _parse_review_output(output: Any, known_ids: set[str]) -> tuple[list[dict[str, Any]], str]:
    # Quality review is an object, while the dataset parser returns its records array.
    output = _extract_provider_content(output)
    if isinstance(output, str):
        cleaned = output.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
        parsed = _decode_json_object(cleaned)
    elif isinstance(output, dict):
        parsed = output
    else:
        raise ProviderError("QUALITY_REVIEW_INVALID", "Quality review returned an invalid structured result.")
    if not isinstance(parsed, dict) or set(parsed) - {"issues", "summary"} or not isinstance(parsed.get("issues"), list) or not isinstance(parsed.get("summary"), str):
        raise ProviderError("QUALITY_REVIEW_INVALID", "Quality review did not match the application-owned review schema.")
    if len(parsed["issues"]) > MAX_REVIEW_ISSUES:
        raise ProviderError("QUALITY_REVIEW_INVALID", "Quality review exceeded the bounded issue limit.")
    for issue in parsed["issues"]:
        if not isinstance(issue, dict) or set(issue) != {"code", "severity", "record_ids", "message", "suggested_action"}:
            raise ProviderError("QUALITY_REVIEW_INVALID", "Quality review contained an invalid issue shape.")
        if issue["code"] not in ISSUE_CODES or issue["severity"] not in {"warning", "blocking"}:
            raise ProviderError("QUALITY_REVIEW_INVALID", "Quality review contained an unknown issue code or severity.")
        if not isinstance(issue["record_ids"], list) or any(record_id not in known_ids for record_id in issue["record_ids"]):
            raise ProviderError("QUALITY_REVIEW_INVALID", "Quality review referenced an unknown dataset record.")
        if not isinstance(issue["message"], str) or not issue["message"].strip() or len(issue["message"]) > MAX_REVIEW_MESSAGE_CHARS:
            raise ProviderError("QUALITY_REVIEW_INVALID", "Quality review issue text exceeded the bounded limit.")
        if not isinstance(issue["suggested_action"], str) or not issue["suggested_action"].strip() or len(issue["suggested_action"]) > MAX_REVIEW_ACTION_CHARS:
            raise ProviderError("QUALITY_REVIEW_INVALID", "Quality review corrective action exceeded the bounded limit.")
    return parsed["issues"], parsed["summary"][:500]


class QualityReviewService:
    def __init__(self, provider: DatasetGenerationProvider, *, model: str, max_model_len: int, issue_cap: int = MAX_REVIEW_ISSUES):
        self.provider = provider
        self.model = model
        self.max_model_len = max_model_len
        self.issue_cap = min(issue_cap, MAX_REVIEW_ISSUES)
        self.last_review_batches = 0
        self.provider_jobs = 0

    def review(self, dataset: CanonicalDataset, dataset_prompt: str, report: ValidationReport) -> QualityReview:
        records = dataset.records
        known_ids = {_record_id(record) for record in records}
        # Deterministic partitions use a conservative token budget and preserve record order.
        batches: list[list[DatasetRecord]] = []
        current: list[DatasetRecord] = []
        current_tokens = estimate_tokens(SYSTEM_PROMPT + dataset_prompt) + estimate_tokens(json.dumps(review_schema(), sort_keys=True))
        budget = max(1024, int(self.max_model_len * 0.5))
        for record in records:
            cost = estimate_tokens(json.dumps(record.model_dump(mode="json"), ensure_ascii=False))
            if current and current_tokens + cost > budget:
                batches.append(current); current = []; current_tokens = estimate_tokens(SYSTEM_PROMPT + dataset_prompt)
            current.append(record); current_tokens += cost
        if current: batches.append(current)
        batches = batches or [[]]
        all_issues: list[dict[str, Any]] = []
        summaries: list[str] = []
        completed = 0
        for index, batch in enumerate(batches, 1):
            review_text = _review_input(dataset, report, batch)
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt(dataset_prompt, review_text, f"review-batch-{index}", len(batch))}]
            response = self.provider.generate(messages=messages, schema=review_schema(), max_tokens=min(2048, self.max_model_len // 8))
            self.provider_jobs += 1
            issues, summary = _parse_review_output(response.output, known_ids)
            all_issues.extend(issues)
            summaries.append(summary)
            completed += 1
        normalized: list[QualityReviewIssue] = []
        for issue in all_issues[: self.issue_cap]:
            # Severity is application-owned: model severity is deliberately ignored.
            severity = "blocking" if issue["code"] in BLOCKING_CODES else "warning"
            normalized.append(QualityReviewIssue(code=issue["code"], severity=severity, recordIds=issue["record_ids"], message=issue["message"], suggestedAction=issue["suggested_action"]))
        blocking = sum(issue.severity == "blocking" for issue in normalized)
        warnings = sum(issue.severity == "warning" for issue in normalized)
        self.last_review_batches = len(batches)
        return QualityReview(reviewVersion=REVIEW_VERSION, model=self.model, promptVersion=PROMPT_VERSION, reviewBatchCount=len(batches), completedReviewBatches=completed, issuesFound=len(normalized), blockingIssues=blocking, warnings=warnings, revisionRequired=blocking > 0, revisionAttempted=False, revisionSucceeded=False, summary=" ".join(summaries)[:500] or "No actionable quality issues were identified.", issues=normalized)


class QualityRevisionService:
    def __init__(self, provider: DatasetGenerationProvider, *, model: str):
        self.provider = provider
        self.model = model
        self.provider_jobs = 0

    def revise(self, records: list[DatasetRecord], issues: list[QualityReviewIssue], dataset_prompt: str, generation_id: str, file_id: str, source_context: str) -> list[DatasetRecord]:
        affected_ids = {record_id for issue in issues for record_id in issue.record_ids}
        affected = [record for record in records if _record_id(record) in affected_ids]
        if not affected:
            raise ProviderError("QUALITY_REVISION_INVALID", "Quality review did not identify revisable records.")
        issue_payload = [issue.model_dump(mode="json", by_alias=True) for issue in issues if any(record_id in affected_ids for record_id in issue.record_ids)]
        record_payload = json.dumps([record.model_dump(mode="json") for record in affected], ensure_ascii=False, sort_keys=True)
        messages = [{"role": "system", "content": "You are Dataset Forge's bounded dataset revision model. Return only the supplied canonical dataset schema. Never include reasoning or self-grading."}, {"role": "user", "content": revision_prompt(dataset_prompt, record_payload, json.dumps(issue_payload, sort_keys=True), source_context)}]
        response = self.provider.generate(messages=messages, schema=dataset_schema(), max_tokens=min(4096,  self.provider.config.max_model_len // 4) if hasattr(self.provider, "config") else 4096)
        self.provider_jobs += 1
        provider_output = _extract_provider_content(response.output)
        if isinstance(provider_output, str):
            provider_output = _decode_json_object(provider_output)
        raw = _parse_provider_output(provider_output)
        if len(raw) == len(records):
            affected_positions = [index for index, record in enumerate(records) if _record_id(record) in affected_ids]
            raw = [raw[index] for index in affected_positions]
        if len(raw) != len(affected):
            raise ProviderError("QUALITY_REVISION_INVALID", "Quality revision did not return one replacement for each affected record.")
        revised = _normalize_provider_records(raw, generation_id, file_id, self.model)
        for original, replacement in zip(affected, revised):
            replacement.metadata = dict(original.metadata)
            replacement.metadata.update({"record_id": _record_id(original), "original_record_id": _record_id(original), "revision_attempt": "1", "revision_reason_codes": ",".join(sorted({issue.code for issue in issues if _record_id(original) in issue.record_ids})), "replaced_by": _record_id(original)})
        return [revised_by_id if _record_id(revised_by_id) in affected_ids else original for original, revised_by_id in zip(affected, revised)]
