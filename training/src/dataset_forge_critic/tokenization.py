"""Chat-template application and completion-only label construction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .gemma import render


class TokenizationExclusion(ValueError):
    """An explicit, reportable reason an example cannot be trained."""


@dataclass(frozen=True)
class TokenizedExample:
    record_id: str
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    prompt_tokens: int
    completion_tokens: int
    supervised_tokens: int


def _messages(rendered: dict[str, Any], *, completion: str | None = None) -> list[dict[str, Any]]:
    # Gemma's official instruction template is user/assistant based.  Preserve the
    # critic system text as prompt content without inventing an unsupported role.
    messages = [
        {"role": "user", "content": [{"type": "text", "text": rendered["system_prompt"] + "\n\n" + rendered["user_payload"]}]},
    ]
    if completion is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": completion}]})
    return messages


def _templated(processor: Any, messages: list[dict[str, Any]], *, add_generation_prompt: bool) -> list[int]:
    value = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_dict=True,
    )
    try:
        ids = value["input_ids"]
    except (KeyError, TypeError) as exc:
        raise TokenizationExclusion("CHAT_TEMPLATE_INPUT_IDS_UNAVAILABLE") from exc
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise TokenizationExclusion("TOKENIZER_RETURNED_UNEXPECTED_BATCH")
        ids = ids[0]
    return [int(item) for item in ids]


def _token_text(processor: Any, token_id: int) -> str:
    return str(processor.tokenizer.decode([token_id], skip_special_tokens=False))


def _with_supervised_eos(ids: list[int], prompt_length: int, processor: Any) -> list[int]:
    """Remove template-only trailing whitespace and terminate with explicit EOS."""
    eos = processor.tokenizer.eos_token_id
    if eos is None:
        raise TokenizationExclusion("TOKENIZER_EOS_REQUIRED")
    values = list(ids)
    while len(values) > prompt_length and values[-1] != eos and not _token_text(processor, values[-1]).strip():
        values.pop()
    if not values or values[-1] != eos:
        values.append(int(eos))
    return values


def generation_terminator_ids(processor: Any) -> list[int]:
    """Return EOS plus the non-whitespace assistant turn terminator, if present."""
    eos = processor.tokenizer.eos_token_id
    if eos is None:
        raise TokenizationExclusion("TOKENIZER_EOS_REQUIRED")
    rendered = {"system_prompt": "system", "user_payload": "{}"}
    prompt = _templated(processor, _messages(rendered), add_generation_prompt=True)
    full = _templated(processor, _messages(rendered, completion=""), add_generation_prompt=False)
    suffix = full[len(prompt):] if full[:len(prompt)] == prompt else []
    terminators = [int(eos)]
    for token_id in reversed(suffix):
        text = _token_text(processor, token_id)
        if not text.strip():
            continue
        if token_id != eos:
            terminators.append(int(token_id))
        break
    return list(dict.fromkeys(terminators))


def tokenize_record(record: dict[str, Any], layer: str, processor: Any, max_length: int) -> TokenizedExample:
    """Render the prompt and full conversation separately, then supervise only their suffix."""
    rendered = render(record, layer)
    prompt_ids = _templated(processor, _messages(rendered), add_generation_prompt=True)
    full_ids = _templated(
        processor,
        _messages(rendered, completion=rendered["completion"]),
        add_generation_prompt=False,
    )
    full_ids = _with_supervised_eos(full_ids, len(prompt_ids), processor)
    if len(full_ids) > max_length:
        raise TokenizationExclusion(f"SEQUENCE_TOO_LONG:{len(full_ids)}>{max_length}")
    if len(prompt_ids) >= len(full_ids):
        raise TokenizationExclusion("NO_SUPERVISED_TOKENS")
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise TokenizationExclusion("CHAT_TEMPLATE_PROMPT_PREFIX_MISMATCH")
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    supervised = len(full_ids) - len(prompt_ids)
    return TokenizedExample(
        record_id=rendered["record_id"], input_ids=full_ids, attention_mask=[1] * len(full_ids),
        labels=labels, prompt_tokens=len(prompt_ids), completion_tokens=supervised,
        supervised_tokens=sum(value != -100 for value in labels),
    )


def pad_batch(items: list[TokenizedExample], pad_token_id: int) -> dict[str, list[list[int]]]:
    if not items:
        raise ValueError("EMPTY_BATCH")
    width = max(len(item.input_ids) for item in items)
    return {
        "input_ids": [item.input_ids + [pad_token_id] * (width - len(item.input_ids)) for item in items],
        "attention_mask": [item.attention_mask + [0] * (width - len(item.input_ids)) for item in items],
        "labels": [item.labels + [-100] * (width - len(item.labels)) for item in items],
    }
