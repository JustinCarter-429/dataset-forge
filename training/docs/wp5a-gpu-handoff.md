# WP5A GPU handoff

Build the local handoff with `python -m dataset_forge_critic.cli build-gpu-bundle`; it exports only public/native train and validation JSONL plus exact code/configuration. `verify-gpu-bundle` validates all hashes and protected-data counts before GPU work. Transfer the generated archive with `send-wp5a-bundle.ps1`, verify its SHA-256 remotely, extract under `/workspace/dataset-forge-wp5a`, then run the bootstrap script in a durable terminal. The scripts never create, terminate, or otherwise manage a GPU pod.
