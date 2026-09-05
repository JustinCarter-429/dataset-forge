#!/usr/bin/env bash
set -euo pipefail

# This script never installs or replaces a system NVIDIA driver.
DRY_RUN=0
WORKSPACE="${DATASET_FORGE_WORKSPACE:-/workspace/dataset-forge}"
VENV_PATH="${DATASET_FORGE_VENV:-${WORKSPACE}/.venv-gpu}"
HF_CACHE_PATH="${HF_HOME:-${WORKSPACE}/hf-cache}"
MIN_FREE_GB="${DATASET_FORGE_MIN_FREE_GB:-120}"
if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=1; fi

echo "Dataset Forge GPU setup inspection"
uname -a
python3 --version
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "nvidia-smi: unavailable (GPU readiness pending)"
FREE_GB="$(df -Pk "$(dirname "$WORKSPACE")" | awk 'NR==2 {printf "%d", $4/1024/1024}')"
echo "workspace=$WORKSPACE"
echo "venv=$VENV_PATH"
echo "hf_cache=$HF_CACHE_PATH"
echo "free_disk_gb=$FREE_GB required_before_download_gb=$MIN_FREE_GB"
if (( FREE_GB < MIN_FREE_GB )); then echo "INSUFFICIENT_DISK" >&2; exit 20; fi
if (( DRY_RUN )); then echo "DRY_RUN_OK: no files or packages changed"; exit 0; fi

mkdir -p "$WORKSPACE" "$HF_CACHE_PATH"
if [[ ! -x "$VENV_PATH/bin/python" ]]; then python3 -m venv "$VENV_PATH"; fi
"$VENV_PATH/bin/python" -m pip install --upgrade pip==26.2.1
# RTX 4090/Ada candidate wheel. Verify the host driver before this step; do not replace it here.
"$VENV_PATH/bin/python" -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
"$VENV_PATH/bin/python" -m pip install --requirement "$WORKSPACE/training/requirements/gemma4-qlora.txt"
"$VENV_PATH/bin/python" -m pip install --editable "$WORKSPACE/training"
HF_HOME="$HF_CACHE_PATH" "$VENV_PATH/bin/python" - <<'PY'
import json, platform
import torch, transformers, peft, bitsandbytes, accelerate
print(json.dumps({"python": platform.python_version(), "torch": torch.__version__, "torch_cuda": torch.version.cuda,
 "cuda_available": torch.cuda.is_available(), "transformers": transformers.__version__, "peft": peft.__version__,
 "bitsandbytes": bitsandbytes.__version__, "accelerate": accelerate.__version__}, sort_keys=True))
PY
echo "SETUP_COMPLETE_GPU_CERTIFICATION_PENDING"
