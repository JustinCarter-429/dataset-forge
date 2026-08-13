# Training data storage

`raw/` contains original downloaded dataset artifacts or immutable snapshots. `interim/` contains dataset-specific staged or transformed forms. `processed/` contains canonical Dataset Forge Critic records ready for a future training job.

Large data must never be committed to Git. The directory markers are tracked solely to preserve this layout. WP1 does not download or create any public-dataset artifacts.

WP2 copies untrusted local sources into content-addressed snapshots under `raw/` only after streaming SHA-256 reconciliation. It uses `.staging/` and promotes complete snapshots atomically. `interim/` and `processed/` remain empty until later work packages.
