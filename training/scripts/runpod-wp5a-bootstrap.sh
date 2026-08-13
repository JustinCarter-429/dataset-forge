#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-/workspace/dataset-forge-wp5a}"
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
python3 -m venv /workspace/venvs/dataset-forge-critic
source /workspace/venvs/dataset-forge-critic/bin/activate
python -m pip install --upgrade pip
python -m pip install -r "$ROOT/training/requirements/gemma4-qlora.txt"
export PYTHONPATH="$ROOT/training/src"
python -m dataset_forge_critic.cli verify-gpu-bundle --bundle-root "$ROOT"
python -m dataset_forge_critic.cli training-preflight
