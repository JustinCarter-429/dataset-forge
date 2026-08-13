# Training data storage

`raw/` contains original downloaded dataset artifacts or immutable snapshots. `interim/` contains dataset-specific staged or transformed forms. `processed/` contains canonical Dataset Forge Critic records ready for a future training job.

Large data must never be committed to Git. The directory markers are tracked solely to preserve this layout. WP1 does not download or create any public-dataset artifacts.
