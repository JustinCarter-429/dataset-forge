# Dataset Forge Critic training foundation

This isolated Python project defines training-data contracts for a future Dataset Forge Critic targeting `google/gemma-4-E4B-it`. Carter remains the author of candidate records. Existing deterministic validation remains authoritative for schema, safety, duplicate, export, and packaging checks; the future critic will judge training-data quality.

WP1 provides no model download, public-dataset download, training, Hugging Face authentication, GPU use, Carter integration, or runtime behavior change.

## Layout

`configs/` contains local-only registries; `src/` contains the package; `data/` separates raw, interim, and canonical processed data; `checkpoints/` and `outputs/` are reserved for later work and ignored by Git. `docs/` describes the contracts and future design.

## Local setup and checks

Use Python 3.11 or newer. From this directory, install only the small validation/test dependency set with `python -m pip install -e ".[dev]"`. Then run `python -m dataset_forge_critic.cli validate-config` and `python -m pytest`.

`validate-record path/to/record.json` validates one local canonical JSON record. Neither CLI command makes network calls.
