"""Application-owned count policy layered over the frozen Carter 1.0 prompts."""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..domain.extraction_models import CanonicalExtractedDocument

_COUNT = re.compile(r"\b(?:exactly|only|create|generate|give\s+me|make)?\s*(\d{1,4})\s+(?:source-grounded\s+)?(?:question[- ]answer\s+)?(?:records?|examples?|items?|pairs?)\b", re.I)


@dataclass(frozen=True)
class CountPlan:
    mode: str
    requested: int | None
    recommended: int | None
    minimum: int | None
    maximum: int | None
    target: int
    hard_cap_limited: bool


def explicit_record_count(prompt: str) -> int | None:
    match = _COUNT.search(prompt)
    return int(match.group(1)) if match else None


def estimate_coverage(document: CanonicalExtractedDocument, hard_cap: int) -> CountPlan:
    """Estimate distinct learning targets from extractable units, not word count.

    Each meaningful source unit earns one opportunity. Long/table-like units may
    earn a second independent opportunity; this deliberately avoids sentence or
    paraphrase inflation while letting structured documents exceed ten records.
    """
    opportunities = 0
    for element in document.elements:
        text = element.text.strip()
        if not text:
            continue
        opportunities += 1
        if element.type.value == "table" or len(re.findall(r"[.!?]\s+", text)) >= 4:
            opportunities += 1
    opportunities = max(1, opportunities)
    maximum = min(opportunities, hard_cap)
    minimum = max(1, min(maximum, int(maximum * 0.8)))
    return CountPlan("auto", None, maximum, minimum, maximum, maximum, opportunities > hard_cap)


def resolve_count(prompt: str, document: CanonicalExtractedDocument, hard_cap: int) -> CountPlan:
    requested = explicit_record_count(prompt)
    if requested is not None:
        # Do not silently lower an explicit request. The controller can report
        # source insufficiency after quality validation.
        return CountPlan("explicit", requested, None, None, None, min(requested, hard_cap), requested > hard_cap)
    return estimate_coverage(document, hard_cap)
