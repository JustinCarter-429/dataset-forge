# WP2 local inspection summary

All four user-supplied directories were physically copied into Git-ignored raw snapshots, hashed, reconciled, and inspected without changing their source copies. No Git LFS pointers or malformed payload records were found.

| Dataset | Snapshot | Observed form and exact records | WP3 evidence |
| --- | --- | --- | --- |
| FACTS Grounding | `snapshot-189e672d70f48d87` | CSV: `examples` 860, evaluation prompts 8 | `system_instruction`, `user_request`, `context_document`, and evaluator prompts; no candidate answer or judgment labels in this public copy, so it cannot directly supervise a critic decision without later generated candidates. |
| HaluEval local HaluBench subset | `snapshot-29c4c6c6d2d4d532` | one Parquet test split, 10,000 | `passage`, `question`, `answer`, `label`, `score`, `source_ds`; suitable for grounding/hallucination supervision after WP3 verifies label semantics. It has no native accept/revise/reject, novelty, or critique. |
| HelpSteer2 | `snapshot-ff06772036b8dc1c` | JSONL.GZ: train 20,324; validation 1,038; preference 9,125; disagreements 23,652 | scalar five-dimension scores, preference pairs/strength and explanations, plus lists of individual ratings. Preserve original 0–4 score scale and preference strength; do not infer grounding or native decisions. |
| UltraFeedback | `snapshot-8fc47deccbd27dbc` | six JSONL source subsets (64k prompts total by observed file counts) | `instruction`, `models`, `completions`, answer lists, and nested annotations. One raw row contains multiple candidate responses and likely emits multiple WP3 examples; preserve nested ratings/rationales. |

Local evidence verifies CC-BY-4.0 for FACTS and HelpSteer2, CC-BY-NC-2.0 for the local HaluBench subset, and MIT for UltraFeedback. The upstream HF identity of FACTS remained unverified because the Hub request failed; the other three were verified with their local cards and Hub metadata. All sources remain `inspected`/`adapter_pending`, never train-ready.
