import hashlib
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Protocol

from ..domain.extraction_models import CanonicalExtractedDocument
from ..domain.models import (
    CanonicalDataset,
    DatasetRecord,
    GroundingSummary,
    QualitySummary,
    ValidationIssue,
    ValidationReport,
    ValidationResult,
)

MIN_EVIDENCE_CHARS = 5
MAX_EVIDENCE_CHARS = 500
MAX_FIELD_CHARS = 4000
NEAR_DUPLICATE_THRESHOLD = 0.92


def quoteable_source_text(element) -> str:
    """The sole source-text projection used for prompts and evidence checks."""
    if element.type.value == "table" and element.rows:
        return "\n".join(" | ".join(row) for row in element.rows)
    return element.text or ""


class DatasetValidator(Protocol):
    def validate(self, dataset: CanonicalDataset) -> ValidationResult: ...


class StructuralDatasetValidator:
    def validate(self, dataset: CanonicalDataset) -> ValidationResult:
        errors: list[str] = []
        if not dataset.records:
            errors.append("Dataset must contain at least one record.")
        for index, record in enumerate(dataset.records):
            for field in ("instruction", "input", "output"):
                if not getattr(record, field).strip():
                    errors.append(f"Record {index} is missing {field}.")
        return ValidationResult(valid=not errors, errors=errors)


def normalize_source_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).replace("\u00a0", " ")
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = value.replace("“", '"').replace("”", '"').replace("’", "'")
    return re.sub(r"\s+", " ", value).strip().casefold()


def _issue(code: str, severity: str, message: str, record_id: str | None = None, field: str | None = None) -> ValidationIssue:
    return ValidationIssue(code=code, severity=severity, message=message, record_id=record_id, field=field)


def _record_id(record: DatasetRecord, index: int) -> str:
    return record.metadata.get("record_id") or f"record-{index + 1}"


def _semantic_text(record: DatasetRecord) -> str:
    return "|".join([record.instruction, record.context or record.input, record.expected_output or record.output, record.category, record.difficulty])


class Phase4ValidationService:
    """Single deterministic schema, grounding, duplicate, and quality gate."""

    def validate(self, dataset: CanonicalDataset, document: CanonicalExtractedDocument) -> tuple[CanonicalDataset, ValidationReport]:
        issues: list[ValidationIssue] = []
        source_text = {element.element_id: quoteable_source_text(element) for element in document.elements}
        records = list(dataset.records)
        if not records:
            issues.append(_issue("ZERO_RECORDS", "error", "Dataset must contain at least one record."))

        grounded_records = 0
        total_evidence = 0
        verified_evidence = 0
        duplicate_fingerprints: set[str] = set()
        deduplicated: list[DatasetRecord] = []
        exact_duplicates_removed = 0
        record_summaries: list[dict[str, object]] = []

        for index, record in enumerate(records):
            rid = _record_id(record, index)
            context = record.context or record.input
            expected = record.expected_output or record.output
            record_issues: list[ValidationIssue] = []
            for field, value in (("instruction", record.instruction), ("context", context), ("expected_output", expected), ("category", record.category)):
                if not value.strip():
                    record_issues.append(_issue("EMPTY_REQUIRED_FIELD", "error", f"Required field is empty: {field}.", rid, field))
                elif len(value) > MAX_FIELD_CHARS:
                    record_issues.append(_issue("FIELD_TOO_LONG", "error", f"Field exceeds the maximum length: {field}.", rid, field))
            if record.difficulty not in {"easy", "medium", "hard"}:
                record_issues.append(_issue("INVALID_DIFFICULTY", "error", "Difficulty must be easy, medium, or hard.", rid, "difficulty"))
            if context.strip().casefold() == record.instruction.strip().casefold():
                record_issues.append(_issue("CONTEXT_EQUALS_INSTRUCTION", "warning", "Context duplicates the instruction.", rid, "context"))
            refs = list(dict.fromkeys(record.source_refs))
            if not refs:
                record_issues.append(_issue("MISSING_SOURCE_REF", "error", "Record does not contain a source reference.", rid, "source_refs"))
            invalid_refs = [ref for ref in refs if ref not in source_text]
            for ref in invalid_refs:
                record_issues.append(_issue("INVALID_SOURCE_REF", "error", "Record references an unavailable extraction unit.", rid, "source_refs"))
            total_evidence += len(record.evidence)
            if not record.evidence:
                record_issues.append(_issue("MISSING_EVIDENCE", "error", "Record must contain at least one evidence item.", rid, "evidence"))
            verified_for_record = 0
            for evidence in record.evidence:
                quote = evidence.quote.strip()
                if not quote:
                    record_issues.append(_issue("EVIDENCE_EMPTY", "error", "Evidence quote is empty.", rid, "evidence"))
                    continue
                if len(quote) < MIN_EVIDENCE_CHARS:
                    record_issues.append(_issue("EVIDENCE_TOO_SHORT", "error", "Evidence quote is too short to verify.", rid, "evidence"))
                    continue
                if len(quote) > MAX_EVIDENCE_CHARS:
                    record_issues.append(_issue("EVIDENCE_TOO_LONG", "error", "Evidence quote exceeds the maximum length.", rid, "evidence"))
                    continue
                normalized_source = normalize_source_text(source_text.get(evidence.source_ref, ""))
                if evidence.source_ref not in source_text or normalize_source_text(quote) not in normalized_source:
                    record_issues.append(_issue("EVIDENCE_NOT_FOUND", "error", "Evidence quote was not found in its referenced extraction unit.", rid, "evidence"))
                    continue
                verified_for_record += 1
            verified_evidence += verified_for_record
            if not record_issues or not any(item.severity == "error" for item in record_issues):
                if verified_for_record > 0 and refs and not invalid_refs:
                    grounded_records += 1
                else:
                    record_issues.append(_issue("RECORD_UNGROUNDED", "error", "Record could not be grounded in verified source evidence.", rid))
            fingerprint = hashlib.sha256(normalize_source_text(_semantic_text(record)).encode()).hexdigest()
            if fingerprint in duplicate_fingerprints:
                exact_duplicates_removed += 1
                issues.append(_issue("EXACT_DUPLICATE", "warning", "Exact duplicate record removed; the first occurrence was retained.", rid))
                continue
            duplicate_fingerprints.add(fingerprint)
            deduplicated.append(record)
            issues.extend(record_issues)
            record_summaries.append({"recordId": rid, "groundingStatus": "passed" if verified_for_record and not any(item.severity == "error" for item in record_issues) else "failed", "sourceRefs": len(refs), "evidenceItems": len(record.evidence), "verifiedEvidenceItems": verified_for_record, "issueCodes": [item.code for item in record_issues]})

        near_pairs = 0
        fingerprints = [_semantic_text(record) for record in deduplicated]
        for left in range(len(fingerprints)):
            for right in range(left + 1, len(fingerprints)):
                if SequenceMatcher(None, normalize_source_text(fingerprints[left]), normalize_source_text(fingerprints[right])).ratio() >= NEAR_DUPLICATE_THRESHOLD:
                    near_pairs += 1
        if near_pairs:
            issues.append(_issue("NEAR_DUPLICATE", "warning", "Potential near-duplicate examples detected."))
        difficulties = {level: sum(record.difficulty == level for record in deduplicated) for level in ("easy", "medium", "hard")}
        categories = {record.category.strip() for record in deduplicated if record.category.strip()}
        warnings = [item for item in issues if item.severity == "warning"]
        errors = [item for item in issues if item.severity == "error"]
        grounding_status = "passed" if deduplicated and grounded_records == len(deduplicated) and verified_evidence == total_evidence else "failed"
        grounding = GroundingSummary(status=grounding_status, total_records=len(deduplicated), grounded_records=grounded_records, ungrounded_records=max(0, len(deduplicated) - grounded_records), total_evidence_items=total_evidence, verified_evidence_items=verified_evidence, failed_evidence_items=max(0, total_evidence - verified_evidence), grounding_percent=round(grounded_records / len(deduplicated) * 100, 1) if deduplicated else 0)
        quality_status = "failed" if errors or grounding_status == "failed" else "passed_with_warnings" if warnings else "passed"
        quality = QualitySummary(status=quality_status, schema_valid=not errors, total_generated_records=len(records), final_record_count=len(deduplicated), valid_records=grounded_records, invalid_records=max(0, len(deduplicated) - grounded_records), category_count=len(categories), difficulty_distribution=difficulties, exact_duplicates_removed=exact_duplicates_removed, near_duplicate_pairs=near_pairs, warnings=warnings)
        report = ValidationReport(schema_version="2.0", status=quality_status, schema_valid=not errors, records={"generated": len(records), "final": len(deduplicated), "valid": grounded_records, "invalid": max(0, len(deduplicated) - grounded_records)}, grounding=grounding, duplicates={"exactDuplicatesRemoved": exact_duplicates_removed, "nearDuplicatePairs": near_pairs, "nearDuplicateThreshold": NEAR_DUPLICATE_THRESHOLD, "method": "SequenceMatcher"}, quality=quality, issues=issues)
        report.records["recordSummaries"] = record_summaries  # type: ignore[assignment]
        return CanonicalDataset(records=deduplicated), report
