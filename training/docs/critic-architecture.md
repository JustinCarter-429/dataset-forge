# Critic architecture

The eventual, not-yet-integrated flow is:

`Carter → candidate records → deterministic validation → Dataset Forge Critic → ACCEPT / REVISE / REJECT + feedback → future feedback loop`

Carter authors records. Deterministic validation remains the final authority for technical validity and safety. The critic will assess qualities such as grounding, usefulness, novelty, adherence, and completeness.

Future batch operation is: generate a batch, validate it, critique it, accept/revise/reject records, update dataset state, then generate the next batch using that feedback. `prior_dataset_state` is intentionally a small optional mapping in WP1, reserving space for coverage, source-region, difficulty, cluster, counts, and next-batch priority signals. WP1 does not connect this flow to the live application.
