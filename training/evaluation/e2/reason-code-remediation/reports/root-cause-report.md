# Phase E2.2 reason-code root-cause report

## Status

The eight-step full-retraining preflight was a **smoke failure**, not final E2 certification. Full retraining never started. Phase E1 remains an immutable FAIL, and the earlier controlled-overfit gate remains a PASS.

## Confirmed evidence

- The failed smoke completed 8/8 optimizer steps with finite loss and gradients and an updated adapter.
- Strict five-field validity was 12/15 (80%); all 15 responses terminated immediately, with no Markdown and no token-limit hits.
- The three invalid responses emitted `UNSAFE_CONTENT`, `unsafe_content`, and `MISSING_FIELD`. All remain invalid; no alias or case normalization is accepted.
- Every affected case had the authoritative expected code `PROMPT_INJECTION`.
- Before E2.2, the prompt said only “approved codes” but listed none. The 28-code parser ontology had only seven codes represented in training and only `PROMPT_INJECTION` in validation; 21 approved codes had zero training coverage.

## Per-failure primary cause

1. `UNSAFE_CONTENT` — **PROMPT_VOCABULARY_AMBIGUITY**. The model invented a generic safety label for an operational multiple-object instruction instead of the well-covered but undefined-in-prompt `PROMPT_INJECTION` code.
2. `unsafe_content` — **PROMPT_VOCABULARY_AMBIGUITY**. This is an unapproved invented lowercase label, not a harmless capitalization of any approved code. The expected label was `PROMPT_INJECTION`.
3. `MISSING_FIELD` — **SEMANTIC_MISCLASSIFICATION**. The model followed the content of an embedded instruction (“add a field”) as if a field were genuinely missing. The record itself had all required fields; the authoritative defect was the operational instruction, `PROMPT_INJECTION`.

## Remediation

`REASON_CODE_SPECS` is now the single versioned ontology. The prompt enumerates every exact case-sensitive code and definition, forbids aliases, and includes a contrast between prompt injection and a genuinely missing field. The strict parser additionally enforces the 1.0 uncalibrated confidence placeholder, field order, decision/code relationships, duplicate rejection, and feedback relationships. Deterministic E2.2 records add at least 10 training and 2 independent validation examples for every approved code; no LLM assigned labels.

The complete E2.1 sufficiency and independence gate was rerun over the augmented corpus and returned **PASS**. Exact, canonical, Jaccard-near, template-family, source-family, Phase E1, and controlled-overfit overlaps are all zero.

## Constrained-decoding investigation

A grammar could restrict JSON keys, decision values, field types, and reason-code enums, but it is not implemented in E2.2. It would be a separately reported production safeguard, not a repair of existing invalid text and not evidence of semantic correctness. Adding a runtime grammar dependency or changing certification protocol requires separate approval.

## Limitation

The failed-smoke runner did not persist per-case latency, so that field is recorded as unavailable rather than reconstructed. No GPU or remote work was performed in E2.2.
