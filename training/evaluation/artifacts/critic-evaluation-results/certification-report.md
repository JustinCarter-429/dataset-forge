# Dataset Forge Critic Phase E1 Certification

## Outcome

**FAIL** — One or more predefined PARTIAL thresholds were not satisfied

## Identity

- Model: `google/gemma-4-E4B-it`
- Revision: `ee0ef6023621cff504d758262d4e04895a5af4a2`
- Checkpoint SHA-256: `cbb96f85c252273eda5beccbcd1bf622964deb11a85241b1d952884a1751906a`
- Held-out cases: 105
- Exact overlap: 0
- Near overlap: 0

## Comparison

| Metric | Base | Fine-tuned |
|---|---:|---:|
| Accuracy | 0.0% | 0.0% |
| Macro F1 | 0.0% | 0.0% |
| Contract validity | 0.0% | 0.0% |
| False-accept rate | 0.0% | 0.0% |
| False-reject rate | 0.0% | 0.0% |
| Revise accuracy | 0.0% | 0.0% |

## Dataset types

- `question_answer`: accuracy 0.0%, macro F1 0.0%
- `instruction_response`: accuracy 0.0%, macro F1 0.0%
- `classification`: accuracy 0.0%, macro F1 0.0%
- `scenario_expected_result`: accuracy 0.0%, macro F1 0.0%
- `custom`: accuracy 0.0%, macro F1 0.0%

## Limitations

- Authored synthetic held-out cases use fictional facts and cannot establish production-domain generalization.
- Point estimates have no confidence intervals.
- Only available training and validation corpora were checked for overlap; frozen test splits were not accessed.
