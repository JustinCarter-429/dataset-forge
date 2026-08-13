# WP3 supervision mapping

All transformations are local, deterministic, and source-preserving. Canonical JSONL contains no inferred decisions, rankings, or revisions. Source score values remain on their original scale with field, scale, and mapping provenance.

| Source | Input mapping | Native supervision | Exclusions |
| --- | --- | --- | --- |
| FACTS Grounding | `user_request` and `context_document`; system instruction and full prompt in candidate metadata | None. Examples are grounding inputs only, with an empty `available_targets` list. | `evaluation_prompts.csv` is counted as `EVALUATION_PROMPT_ONLY`; it is not a training example. |
| HaluBench | `question`, `passage`, and `answer` | `score` maps to `scores.hallucination`: PASS/no hallucination is 1 and FAIL/hallucination is 0. | Schema drift, inconsistent polarity, or an empty answer. |
| HelpSteer2 | `prompt` and response | The five 0–4 dimensions map directly to same-named score fields. Preference rows preserve both candidates; negative `preference_strength` selects candidate 0 and positive selects candidate 1. A genuine `preference_elaboration` maps to critique. | Every disagreement row is `DISAGREEMENT_NONCONSENSUS_EXCLUDED`; strength zero has no preferred index. |
| UltraFeedback | `instruction` and each completion response | A supplied `overall_score` maps only to `scores.usefulness`, preserving source scale. A supplied completion critique maps to critique. `annotations` and `fine-grained_score` are preserved as candidate metadata. | Missing or unusable completions become `NO_CANONICAL_CANDIDATE`. |

The `transform-datasets --all` command writes ignored interim and processed payloads after validation, then commits only portable manifests and aggregate reports. Record identifiers incorporate source snapshot, source row, and completion index where applicable. Duplicate fingerprints are reported but never removed.
