# RTX 4090 handoff

The packaged archive is self-contained except for the pinned base model. It includes source, profiles, train and validation data, and no frozen test records. Keep the SSH private key on the Windows laptop.

## Transfer from Windows

Replace `<HOST>`, `<PORT>`, and `<ARCHIVE>` with the current values. The target needs 140–150 GB allocated storage and at least 120 GB free before model synchronization.

```powershell
ssh -i "C:\Users\Justin\.ssh\vast_datasetforge" -p <PORT> root@<HOST> "mkdir -p /workspace/dataset-forge"
scp -i "C:\Users\Justin\.ssh\vast_datasetforge" -P <PORT> "<ARCHIVE>" root@<HOST>:/workspace/
scp -i "C:\Users\Justin\.ssh\vast_datasetforge" -P <PORT> "<ARCHIVE>.sha256" root@<HOST>:/workspace/
```

## Remote setup

```bash
export DATASET_FORGE_WORKSPACE=/workspace/dataset-forge
mkdir -p "$DATASET_FORGE_WORKSPACE"
cd /workspace
sha256sum -c <ARCHIVE_FILENAME>.sha256
tar -xzf /workspace/<ARCHIVE_FILENAME> -C "$DATASET_FORGE_WORKSPACE"
cd "$DATASET_FORGE_WORKSPACE"
bash training/scripts/setup-linux-gpu.sh
source "$DATASET_FORGE_WORKSPACE/.venv-gpu/bin/activate"
export HF_HOME="$DATASET_FORGE_WORKSPACE/hf-cache"
export DATASET_FORGE_MODEL_PATH="$DATASET_FORGE_WORKSPACE/model"
hf auth whoami
hf buckets sync hf://buckets/Jc429/gemma-4-E4B-it-bucket "$DATASET_FORGE_MODEL_PATH"
nvidia-smi
df -h "$DATASET_FORGE_WORKSPACE"
free -h
dataset-forge-critic preflight --config training/configs/profiles/gpu-smoke.yaml
dataset-forge-critic verify-data --config training/configs/profiles/gpu-smoke.yaml
dataset-forge-critic token-audit --config training/configs/profiles/gpu-smoke.yaml --limit-per-file 8
```

The bucket sync creates exactly one working model copy. Do not copy the laptop's SSH key to the server. Do not upload adapters or checkpoints to the public base-model bucket.

## Detached training and monitoring

Calculate the immutable confirmation digest, then launch the 8-step real smoke profile through the detached launcher:

```bash
cd /workspace/dataset-forge
source .venv-gpu/bin/activate
export HF_HOME=/workspace/dataset-forge/hf-cache
export DATASET_FORGE_MODEL_PATH=/workspace/dataset-forge/model
CONFIG=training/configs/profiles/gpu-smoke.yaml
DIGEST="$(python -c 'from pathlib import Path; from dataset_forge_critic.cli import _configuration; print(_configuration(Path("training/configs/profiles/gpu-smoke.yaml")).digest())')"
dataset-forge-critic launch --config "$CONFIG" --confirm "$DIGEST"
```

Start private monitors and forward them from the laptop:

```bash
dataset-forge-critic launch-monitor --root /workspace/dataset-forge --jupyter-port 8888 --tensorboard-port 6006
```

```powershell
ssh -i "C:\Users\Justin\.ssh\vast_datasetforge" -p <PORT> -L 8888:127.0.0.1:8888 -L 6006:127.0.0.1:6006 root@<HOST>
```

Open `http://127.0.0.1:8888` for Jupyter or `http://127.0.0.1:6006` for TensorBoard. The trainer and monitors run in detached sessions and survive an SSH/browser disconnect.

## Checkpoint, resume, and export

`status.json`, TensorBoard events, and adapter-only checkpoints are stored under `training/outputs/gpu_smoke-<digest-prefix>/`. Requesting stop is graceful and produces a final checkpoint.

```bash
RUN_DIR="$(find training/outputs -maxdepth 1 -type d -name 'gpu_smoke-*' | head -n 1)"
dataset-forge-critic status --run-dir "$RUN_DIR"
dataset-forge-critic stop --run-dir "$RUN_DIR"
CHECKPOINT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["path"])' "$RUN_DIR/latest-checkpoint.json")"
dataset-forge-critic launch --config "$CONFIG" --confirm "$DIGEST" --resume "$CHECKPOINT"
dataset-forge-critic export --config "$CONFIG" --checkpoint "$CHECKPOINT" --destination /workspace/dataset-forge/exports/gpu-smoke-adapter
```

After smoke verification, substitute `pilot.yaml` and compute its distinct digest before manually launching the pilot. Do not launch the full profile automatically.
