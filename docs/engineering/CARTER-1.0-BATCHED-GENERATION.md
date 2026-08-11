# Carter 1.0 batched generation

Dataset Forge splits a validated DatasetSpec's requested record count into
sequential batches of at most `CARTER_GENERATION_BATCH_SIZE` (default 5).
Each provider turn must return exactly its target count and pass the dynamic
schema/evidence structure validation before its records merge in batch order.
A failed batch stops the job; later batches and export do not begin.

RunPod is captured for the whole job and used for planner, every batch, review,
and bounded revision. No runtime fallback occurs. The existing Part 3 quality
gate runs once after merging, so cross-batch exact duplicates and likely
PII/secrets are handled before accepted-only JSON, CSV, and ZIP export.

The existing Live Pipeline has a real `Dataset planned` phase and uses the
polled job's typed batch progress to display the current batch and structurally
completed candidate-record count. Batch size is an environment reliability
setting, not a user-facing control. Smaller batches make more provider calls;
larger batches make fewer, larger generation turns. The GPT-OSS/vLLM final
channel limitation for live large generation remains deferred.
