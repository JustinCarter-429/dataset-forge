# Canonical record contract

Schema version is `1.0.0`. A record has a deterministic `record_id` (`dffc_` plus a 32-character SHA-256-derived digest), `task_family`, schema-agnostic `input`, optional `target`, explicit `supervision`, and mandatory `provenance`.

Task families are `preference`, `ranking`, `multidimensional_quality`, `grounding`, `hallucination_detection`, and `dataset_forge_native`. Native decisions may be `ACCEPT`, `REVISE`, or `REJECT`; they are optional for public records.

`input.candidate_record` is an arbitrary structured mapping, not a Q&A contract. `comparison_candidates`, `dataset_spec`, and `prior_dataset_state` are optional mappings/lists. Score fields are optional normalized objects in `[0, 1]`. Every present normalized score retains its original field name, original value, original scale (when known), and normalization method/version. A score of zero is present; a null score is unavailable.

`supervision.available_targets` lists only genuinely supplied targets (for example `scores.grounding` or `decision`). Every listed target must have a non-null value; unlisted targets may be absent. Never synthesize a label simply to populate the schema.

Provenance requires source dataset, adapter name/version, and transform version. Split, original ID, source URL, license, fingerprint, and timestamp remain nullable when truly unknown. Stable IDs derive from dataset, split, source record ID, candidate index, and transform version—not raw content.

Valid shape: `{"record_id":"dffc_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","task_family":"grounding","input":{"candidate_record":{"instruction":"When rotate passwords?"}},"target":{"scores":{"grounding":{"value":1,"provenance":{"original_field_name":"label","original_value":1,"original_scale":"binary","normalization_method":"identity","normalization_version":"1.0.0"}}}},"supervision":{"available_targets":["scores.grounding"]},"provenance":{"source_dataset":"synthetic","adapter_name":"example","adapter_version":"1.0.0","transform_version":"1.0.0"}}`.

Invalid examples include a free-text task family, `decision: "KEEP"`, a score with value `1.5`, or listing `scores.grounding` while its value is null.
