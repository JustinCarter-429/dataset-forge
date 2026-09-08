# Phase E2.2 proposed smoke execution

This package prepares but does not authorize the Phase E2.2 GPU smoke. Do not execute it until Justin gives explicit authorization.

Use the exact pinned `google/gemma-4-E4B-it` revision `ee0ef6023621cff504d758262d4e04895a5af4a2`. Start from a fresh adapter. Do not use the E1 adapter or the failed eight-step smoke adapter. The packaged data directory is `phase-e2.2/smoke-data`, and the frozen configuration is `phase-e2.2/smoke-config.json`.

The proposed run contains 340 training records, 97 independent validation records, eight deterministic epochs, effective batch size eight, and exactly 340 optimizer steps. The runner exits nonzero unless every predefined gate in `reports/proposed-smoke-gates.json` passes. Full 2,250-step retraining remains unauthorized regardless of the smoke result.

Before any authorized execution, verify the archive SHA-256, every entry in the internal package manifest, the exact Git commit, sufficient disk space, BF16 support, token identities 1 and 106, and a clean fresh-adapter output directory. Preserve all outputs incrementally and stop after the smoke result.
