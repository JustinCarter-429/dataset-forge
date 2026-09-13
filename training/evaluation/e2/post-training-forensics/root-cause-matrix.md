# Phase E2.5 Root-Cause Matrix

| Candidate cause | Classification | Evidence |
|---|---|---|
| model semantic misclassification | CONFIRMED | The model emitted reject with a revise-only code; the expected decision is revise. |
| insufficient ontology coverage | CONTRIBUTING | 10/6,350 training rows; two scenario rows; zero scenario validation rows; one repeated structural pattern. |
| contradictory training targets | RULED_OUT | All 12 train/validation occurrences use revise with the single expected code. |
| ambiguous distinction between related reason codes | RULED_OUT | The model selected the expected code, so neighboring-code ambiguity did not produce this failure. |
| prompt-definition ambiguity | RULED_OUT | The prompt explicitly marks the code [revise] and provides a revise example. |
| parser compatibility-table defect | RULED_OUT | The parser and ontology share the same revise-only authority and reject at the documented check. |
| expected-label defect | RULED_OUT | The absent expected_result field is repairable and the frozen target is revise plus REQUIRED_FIELD_MISSING. |
| dataset-schema interpretation defect | RULED_OUT | The emitted code shows the absence was recognized despite required vs required_fields representation. |
| truncation | RULED_OUT | 58 generated tokens; token-limit flag false. |
| termination failure | RULED_OUT | The response terminated normally. |
| malformed JSON | RULED_OUT | JSON decoding and the five-field envelope succeed; semantic compatibility fails later. |
| adapter-loading failure | RULED_OUT | Checkpoint identity verified; all 15 outputs were genuine and 14 passed strict parsing. |
| evaluation-runner mismatch | RULED_OUT | Independent parser replay reproduces the exact same single rejection and aggregate metrics. |
| held-out leakage | RULED_OUT | No exact or >=0.90 near match against E2.4 train/validation; prior 105-case overlap audit is also zero. |
| quantization or precision effects | UNRESOLVED | No counterfactual unquantized run exists, so this cannot be isolated locally. |
