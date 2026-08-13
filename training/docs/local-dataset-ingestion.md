# Local dataset ingestion

Run `python -m dataset_forge_critic.cli inspect-local-datasets --source-root "<local dataset root>"`. The command treats sources as untrusted passive data: it never runs repository scripts, changes source files, follows symlinks, or downloads from the network.

Snapshots are content-addressed from relative paths, file hashes, and sizes. Files are copied to `data/raw/.staging/`, reconciled with the source inventory, and only then promoted to `data/raw/<dataset>/<snapshot>/`. A repeated unchanged run reuses that snapshot.

ZIP and TAR extraction helpers reject traversal paths, absolute paths, link/device members, and resource-limit violations. Inspection streams JSONL and compressed JSONL, parses CSV/TSV as data, reads Parquet metadata with PyArrow, detects Git LFS pointers, and records malformed records rather than changing them. Reports and manifests contain portable relative paths, not user-specific source locations.

Licenses and upstream IDs are recorded only when supported by local or authoritative evidence. `raw` is source snapshots; `interim` is future dataset-specific staging; `processed` is future canonical data. No canonical transformation or score normalization occurs here.
