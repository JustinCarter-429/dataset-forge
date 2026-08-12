"""Application-owned count policy layered over the frozen Carter 1.0 prompts."""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..domain.extraction_models import CanonicalExtractedDocument

_COUNT = re.compile(r"\b(?:exactly|only|create|generate|give\s+me|make)?\s*(\d{1,4})\s+(?:(?:source-grounded|cybersecurity|question[- ]answer|fine-tuning)\s+){0,4}(?:records?|examples?|items?|pairs?)\b", re.I)


@dataclass(frozen=True)
class CountPlan:
    mode: str
    requested: int | None
    recommended: int | None
    minimum: int | None
    maximum: int | None
    target: int
    hard_cap_limited: bool
    supported_count: int


def explicit_record_count(prompt: str) -> int | None:
    match = _COUNT.search(prompt)
    return int(match.group(1)) if match else None


def estimate_coverage(document: CanonicalExtractedDocument, hard_cap: int) -> CountPlan:
    """Estimate distinct learning targets from extractable units, not word count.

    Each meaningful source unit earns one opportunity. Long/table-like units may
    earn a second independent opportunity; this deliberately avoids sentence or
    paraphrase inflation while letting structured documents exceed ten records.
    """
    # Extraction blocks are presentation fragments, not learning targets. A
    # PDF can contain hundreds of small blocks for a few dozen concepts. Build
    # a bounded coverage estimate from document volume, page structure, and
    # independently useful tables instead of inflating one record per block.
    text_units = [element for element in document.elements if element.text.strip() and element.type.value != "table"]
    table_units = [element for element in document.elements if element.type.value == "table" and element.text.strip()]
    text_characters = sum(len(element.text.strip()) for element in text_units)
    page_count = document.statistics.page_count or max((element.page_number or 1 for element in text_units), default=1)
    # At least one opportunity per populated page; approximately one further
    # target per 900 characters remains conservative for dense reference PDFs.
    text_opportunities = max(page_count, (text_characters + 899) // 900)
    # Small, deliberately structured sources often use one extraction unit per
    # independent fact. Preserve that precision without letting large PDF
    # layout fragments inflate the estimate.
    if len(text_units) <= 50 or page_count <= 1:
        text_opportunities = max(text_opportunities, sum(1 + (len(re.findall(r"[.!?]\s+", element.text)) >= 4) for element in text_units))
    # A table can supply one additional distinct comparison/lookup target, but
    # no row/cell expansion occurs during planning.
    opportunities = max(1, text_opportunities + len(table_units))
    maximum = min(opportunities, hard_cap)
    minimum = max(1, min(maximum, int(maximum * 0.8)))
    return CountPlan("auto", None, maximum, minimum, maximum, maximum, opportunities > hard_cap, opportunities)


def resolve_count(prompt: str, document: CanonicalExtractedDocument, hard_cap: int) -> CountPlan:
    requested = explicit_record_count(prompt)
    if requested is not None:
        # Do not silently lower an explicit request. The controller can report
        # source insufficiency after quality validation.
        coverage = estimate_coverage(document, hard_cap)
        # Preserve what the user asked for, but never schedule records that the
        # application cannot ground in distinct source opportunities.
        target = min(requested, coverage.maximum or 0, hard_cap)
        return CountPlan("explicit", requested, None, None, None, target, requested > hard_cap, coverage.supported_count)
    return estimate_coverage(document, hard_cap)
