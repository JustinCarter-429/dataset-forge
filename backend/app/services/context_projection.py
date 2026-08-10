from dataclasses import dataclass
from ..domain.extraction_models import CanonicalExtractedDocument, ExtractionElement
from .validation import quoteable_source_text


@dataclass(frozen=True)
class ContextBatch:
    batch_id: str
    source_element_ids: tuple[str, ...]
    text: str
    estimated_input_tokens: int
    requested_records: int
    alias_to_canonical: dict[str, str]
    canonical_to_alias: dict[str, str]
    quoteable_text_by_canonical: dict[str, str]


def estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def _element_text(element: ExtractionElement) -> str:
    return quoteable_source_text(element)


def build_context_batches(document: CanonicalExtractedDocument, *, max_model_len: int, records_per_batch: int, record_limit: int) -> list[ContextBatch]:
    budget = max(512, int(max_model_len * 0.65))
    batches: list[ContextBatch] = []
    current_ids: list[str] = []
    current_text: list[str] = []
    current_tokens = 0
    for element in document.elements:
        text = _element_text(element).strip()
        if not text:
            continue
        element_tokens = estimate_tokens(text)
        if current_text and (current_tokens + element_tokens > budget or (record_limit > records_per_batch and len(current_ids) >= max(1, records_per_batch * 2))):
            aliases = {f"source_{i + 1}": value for i, value in enumerate(current_ids)}
            batches.append(ContextBatch(f"batch-{len(batches) + 1}", tuple(current_ids), "\n\n".join(current_text), current_tokens, max(1, min(records_per_batch, record_limit - len(batches) * records_per_batch)), aliases, {value: key for key, value in aliases.items()}, {item.element_id: _element_text(item).strip() for item in document.elements if item.element_id in current_ids}))
            current_ids, current_text, current_tokens = [], [], 0
        if element_tokens > budget:
            text = text[: budget * 4]
            element_tokens = estimate_tokens(text)
        current_ids.append(element.element_id)
        alias = f"source_{len(current_ids)}"
        current_text.append(f"SOURCE UNIT: {alias}\nTEXT: {text}")
        current_tokens += element_tokens
    if current_text:
        aliases = {f"source_{i + 1}": value for i, value in enumerate(current_ids)}
        batches.append(ContextBatch(f"batch-{len(batches) + 1}", tuple(current_ids), "\n\n".join(current_text), current_tokens, min(records_per_batch, record_limit), aliases, {value: key for key, value in aliases.items()}, {item.element_id: _element_text(item).strip() for item in document.elements if item.element_id in current_ids}))
    empty = ContextBatch("batch-1", tuple(), "No extractable source text was available.", estimate_tokens("No extractable source text was available."), min(records_per_batch, record_limit), {}, {}, {})
    return batches or [empty]
