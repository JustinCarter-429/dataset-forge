# Phase E2.4 full-retraining operator contract

This package is a local preflight artifact. It does not authorize GPU use. A future operator must obtain explicit authorization for the verified archive hash before transferring it or starting training.

## Frozen execution

- Start a fresh QLoRA adapter on `google/gemma-4-E4B-it` revision `ee0ef6023621cff504d758262d4e04895a5af4a2`.
- Never initialize from Phase E1, either failed smoke, or the passing E2.2 recovery-smoke adapter.
- Run 6,350 records for three deterministic no-replacement epochs: 19,050 examples, 2,381 complete effective batches of eight, then one correctly averaged partial batch of two. The final optimizer step is 2,382.
- Set `E2_FULL_TRAINING_AUTHORIZED=YES` only after the specific package hash is approved. The wrapper fails closed without it.
- Supply a resumable `E2_SYNC_COMMAND`. It must copy progress, logs, checkpoints, and raw evaluation responses off-instance and verify destination hashes.

Validation runs at baseline step 0, steps 250 through 2,250, and final step 2,382. Periodic checkpoints run at steps 250 through 2,250 and 2,382. The best checkpoint is the lowest validation loss, with earliest step winning a tie. Best and final checkpoints are versioned and separate.

Checkpoint manifests hash every adapter and state file. A resume is accepted only after hash verification and only at an optimizer-step boundary with a consistent sampler position. BF16 uses no gradient scaler; the checkpoint explicitly records `scaler: null`.

## Durability and lifecycle

Use an attached persistent volume or a resumable off-instance destination. The wrapper syncs at least every 60 seconds and after termination. Each completed checkpoint and incrementally flushed evaluation response must be included. Keep a local retrieval ledger with source hash, destination hash, bytes, and timestamp. Create local `RETRIEVAL_VERIFIED.json` only after every required evidence and adapter member matches.

Stopping a Vast instance preserves container storage but can continue storage billing. Destroying it permanently deletes temporary container storage. Never destroy before the local retrieval marker is valid.

## Post-training sequence

After verifying the final checkpoint, run the independent strict-contract smoke. Only if it passes, run the unchanged 105-case Phase E1 held-out certification against the pinned base, failed E1 adapter, and final E2 adapter, plus deterministic validators where applicable. Keep raw and normalized outputs separate; malformed output remains failure. Return exactly `PASS`, `PARTIAL`, or `FAIL`. Integration and deployment require separate approval regardless of outcome.

## Estimate

The recovery smoke's end-to-end observed window was about 1.6 hours for 340 steps plus setup and 97-case validation. A conservative planning range for 2,382 steps with repeated validation is 9–13 GPU-hours on comparable hardware. This is not a quote. Cost is `provider hourly GPU price × actual billable hours`, plus any storage/volume and network charges. Use the live instance price when authorization is granted.
