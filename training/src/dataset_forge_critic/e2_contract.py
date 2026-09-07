"""Phase E2 versioned structured-output contract.

This module is the single authority for target serialization and strict response
validation.  It deliberately does not recover JSON from prose or Markdown.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any

MODEL_VERSION = "dataset-forge-critic-v1"
CONFIDENCE_SEMANTICS = "UNCALIBRATED_LABEL_AUTHORITY_PLACEHOLDER"
MAPPING_VERSION = "phase-e2-native-mapping-v1"
DECISIONS = ("accept", "revise", "reject")
FIELDS = ("decision", "confidence", "reason_codes", "feedback", "model_version")
APPROVED_REASON_CODES = frozenset(
    {
        "ANSWER_LEAKAGE",
        "CONTRADICTION",
        "CRITIC_CONTRACT_OVERRIDE",
        "DATASET_SPEC_MISMATCH",
        "DIFFICULTY_TOO_LOW",
        "EMPTY_OR_VAGUE",
        "EXACT_DUPLICATE",
        "FORBIDDEN_EXTRA_FIELD",
        "GROUNDING_CONTRADICTION",
        "GROUNDING_PARTIAL",
        "HALLUCINATED_DETAIL",
        "INCOMPLETE_RESPONSE",
        "INVALID_ENUM",
        "LOW_COVERAGE_GAIN",
        "MALFORMED_JSON",
        "MALICIOUS_SOURCE_INSTRUCTION",
        "MISSING_REQUIRED_CONSTRAINT",
        "NEAR_DUPLICATE",
        "NO_COVERAGE_GAIN",
        "OFF_TOPIC",
        "OVERBROAD_RESPONSE",
        "PROMPT_INJECTION",
        "REQUIRED_FIELD_MISSING",
        "SOURCE_SCOPE_ERROR",
        "UNSUPPORTED_CLAIM",
        "WHITESPACE_ONLY",
        "WRONG_CONCEPT",
        "WRONG_FIELD_TYPE",
    }
)


class ContractError(ValueError):
    """A stable, machine-readable strict-contract failure."""


def _reject_constant(value: str) -> None:
    raise ContractError(f"NON_FINITE_NUMBER:{value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ContractError(f"DUPLICATE_KEY:{key}")
        value[key] = item
    return value


def validate_contract(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError("ROOT_NOT_OBJECT")
    missing = sorted(set(FIELDS) - set(value))
    unknown = sorted(set(value) - set(FIELDS))
    if missing:
        raise ContractError(f"MISSING_REQUIRED_FIELD:{','.join(missing)}")
    if unknown:
        raise ContractError(f"UNKNOWN_FIELD:{','.join(unknown)}")
    if value["decision"] not in DECISIONS:
        raise ContractError("INVALID_DECISION")
    confidence = value["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ContractError("WRONG_TYPE:confidence")
    if not math.isfinite(float(confidence)) or not 0.0 <= float(confidence) <= 1.0:
        raise ContractError("INVALID_CONFIDENCE")
    codes = value["reason_codes"]
    if not isinstance(codes, list) or any(not isinstance(code, str) for code in codes):
        raise ContractError("WRONG_TYPE:reason_codes")
    unapproved = sorted(set(codes) - APPROVED_REASON_CODES)
    if unapproved:
        raise ContractError(f"UNAPPROVED_REASON_CODE:{','.join(unapproved)}")
    if len(codes) != len(set(codes)):
        raise ContractError("DUPLICATE_REASON_CODE")
    if value["feedback"] is not None and not isinstance(value["feedback"], str):
        raise ContractError("WRONG_TYPE:feedback")
    if value["model_version"] != MODEL_VERSION:
        raise ContractError("INVALID_MODEL_VERSION")
    return value


def parse_strict_response(text: str) -> dict[str, Any]:
    if not isinstance(text, str) or not text:
        raise ContractError("EMPTY_OUTPUT")
    if text.strip() != text or not text.startswith("{") or not text.endswith("}"):
        raise ContractError("NOT_EXACT_JSON_OBJECT")
    try:
        value = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except ContractError:
        raise
    except json.JSONDecodeError as exc:
        raise ContractError("INVALID_JSON_SYNTAX") from exc
    return validate_contract(value)


def canonical_json(value: dict[str, Any]) -> str:
    validated = validate_contract(value)
    ordered = {field: validated[field] for field in FIELDS}
    return json.dumps(ordered, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def target_from_record(record: dict[str, Any]) -> dict[str, Any]:
    """Map authoritative native decision supervision into the E2 contract."""
    canonical = record.get("canonical_record", record)
    target = canonical.get("target", {})
    # Phase E2.1 corpora store the approved five-field target directly.  Keep
    # the legacy native mapper below for historical source rows, but never
    # reinterpret an already-versioned contract target.
    if isinstance(target, dict) and set(target) == set(FIELDS):
        return validate_contract(target)
    available = set(canonical.get("supervision", {}).get("available_targets", []))
    decision = target.get("decision")
    if "decision" not in available or not isinstance(decision, str) or decision.casefold() not in DECISIONS:
        raise ContractError("DECISION_SUPERVISION_REQUIRED")
    decision = decision.casefold()

    source_codes = target.get("reason_codes", target.get("issue_codes", [])) or []
    code_field = "reason_codes" if "reason_codes" in target else "issue_codes"
    if source_codes and code_field not in available:
        raise ContractError("REASON_CODE_SUPERVISION_REQUIRED")
    codes: list[str] = []
    for source_code in source_codes:
        if not isinstance(source_code, str):
            raise ContractError("WRONG_TYPE:source_reason_code")
        normalized = re.sub(r"[^A-Z0-9]+", "_", source_code.strip().upper()).strip("_")
        if normalized not in APPROVED_REASON_CODES:
            raise ContractError(f"UNAPPROVED_REASON_CODE:{normalized}")
        if normalized not in codes:
            codes.append(normalized)
    if decision != "accept" and not codes:
        raise ContractError("NONACCEPT_REASON_CODE_REQUIRED")

    def supervised_text(field: str) -> str | None:
        value = target.get(field)
        if value is None or field not in available:
            return None
        if not isinstance(value, str):
            raise ContractError(f"WRONG_TYPE:{field}")
        normalized = re.sub(r"\s+", " ", value).strip()
        return normalized or None

    if decision == "accept":
        feedback = None
    elif decision == "revise":
        feedback = supervised_text("revision_directive")
        if feedback is None:
            critique = supervised_text("critique")
            concrete = re.match(r"(?i)^(add|remove|replace|revise|correct|include|cite|use|change|clarify|specify|ensure)\b", critique or "")
            feedback = critique if concrete else None
        if feedback is None:
            raise ContractError("REVISION_FEEDBACK_REQUIRED")
    else:
        feedback = supervised_text("critique")
        if feedback is None:
            raise ContractError("REJECTION_FEEDBACK_REQUIRED")

    return validate_contract(
        {
            "decision": decision,
            "confidence": 1.0,
            "reason_codes": list(codes),
            "feedback": feedback,
            "model_version": MODEL_VERSION,
        }
    )
