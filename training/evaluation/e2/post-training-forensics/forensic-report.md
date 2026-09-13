# Phase E2.5 Post-Training Smoke Failure Forensics

## Outcome

**A. MODEL_ERROR.** Training completed successfully at 2,382/2,382 steps and the final adapter is hash-verified. Certification remains blocked because the mandatory strict-contract smoke failed before the frozen 105-case certification could run.

## Failed case

`e1-scenario_expected_result-08` is a `scenario_expected_result` record whose candidate omits the schema-required `expected_result` field. The expected target is `revise` with `REQUIRED_FIELD_MISSING`. The model emitted the correct code but paired it with `reject`. That code is explicitly revise-only, so the parser correctly returned `INVALID_CODE_COMBINATION:REQUIRED_FIELD_MISSING` and no normalized response exists.

The JSON syntax, exact five-field shape, confidence, code spelling, and termination were valid. The failure is the decision/code semantic combination. No output was repaired or reinterpreted.

## Versioned compatibility table

| Layer | Identity | Authoritative rule | Failed output |
|---|---|---|---|
| Output contract | five-field E2 contract | Exact ordered keys: decision, confidence, reason_codes, feedback, model_version | Conforms |
| Ontology | `dataset-forge-reason-codes-v1` | `REQUIRED_FIELD_MISSING` allows only `revise` | Violated by `reject` |
| Prompt | `critic-system-v2` / `0c4a0bd3ce3a4563e0acc19ef6c60574ead5385913ab99e17e81d3d097d969a8` | Marks the code `[revise]` and includes a revise example | Violated by `reject` |
| Parser | `d05bac02e7bfb19c211eb7fad0646b115b23207b16a37f5e95bf9c559d21caff` | Reject any code whose allowed-decision set excludes the emitted decision | Correctly rejected |
| Frozen fixture | E1 held-out / `d9b046ebc9a73f39860c241329417f6606976939e03ac53e4f7ae4599ada2500` | Missing `expected_result` → `revise` + `REQUIRED_FIELD_MISSING` | Fixture is correct |

The code is not restricted by dataset type or defect category, and no other reason code was emitted. The sole explicit incompatibility is the predicted decision.

## Corpus evidence

- E2.4 train: 10 code examples among 6350 rows; all are `revise`, single-code targets, with two `scenario_expected_result` examples.
- E2.4 validation: 2 examples among 806 rows; none are `scenario_expected_result`.
- Contradictory identical inputs: 0.
- Full-corpus compatibility violations: 0.
- Exact duplicate candidates among the 12 code examples: 1 (same compatible target; no conflict).
- Cross-split exact candidate duplicates in the full corpus: 1; conflicting cross-split targets: 0.
- Failed-case exact or >=0.90 near leakage: none.

The training targets do not contradict the contract. Sparse, repetitive coverage is a contributing generalization weakness, not evidence that the parser or frozen label is wrong.

## Recomputed smoke

- Responses: 15/15 unique
- Strict validity: 93.333333%
- Parse failures: 6.666667%
- Decision accuracy: 86.666667%
- Macro precision/recall/F1: 0.944444 / 0.866667 / 0.895623
- Reason-code exact match after strict parsing: 46.666667%
- Raw emitted-code lexical match before strict compatibility validation: 53.333333%
- Immediate termination: 15/15
- Markdown, external prose, token-limit hits: 0 / 0 / 0
- Recomputed/report discrepancies: 0

All 15 outputs had the exact key set, approved case-sensitive codes, confidence 1.0, immediate termination, no Markdown, no external prose, and no token-limit hit. Beyond the single parser failure, the smoke also contains semantic errors: one valid record was rejected as a duplicate, one revise case used `GROUNDING_PARTIAL`, and all five hallucinated-detail cases used `GROUNDING_CONTRADICTION`. These are preserved as emitted and explain the 46.666667% strict reason-code exact-match rate.

## Limits

Quantization or precision effects cannot be isolated without unauthorized counterfactual inference. The 15-case smoke is intentionally small. No GPU, inference, retraining, deployment, integration, or 105-case certification was performed in E2.5.
