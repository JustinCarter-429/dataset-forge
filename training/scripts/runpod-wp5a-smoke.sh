#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-/workspace/dataset-forge-wp5a}"
source /workspace/venvs/dataset-forge-critic/bin/activate
export PYTHONPATH="$ROOT/training/src"
python -m dataset_forge_critic.cli verify-gpu-bundle --bundle-root "$ROOT"
python -m dataset_forge_critic.cli training-preflight
echo "GPU certification implementation must stop unless CUDA preflight passes; no production training is launched."
