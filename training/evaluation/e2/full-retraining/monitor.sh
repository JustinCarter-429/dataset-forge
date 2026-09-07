#!/usr/bin/env bash
set -euo pipefail
: "${E2_BASE:?E2_BASE is required}"
progress="$E2_BASE/full/progress.json"
if [[ ! -f "$progress" ]]; then progress="$E2_BASE/smoke/progress.json"; fi
if [[ -f "$progress" ]]; then cat "$progress"; else printf '{"status":"STARTING"}\n'; fi
printf '\nGPU\n'
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits
printf '\nLOG\n'
tail -n 20 "$E2_BASE/evidence/stdout.log" 2>/dev/null || true
printf '\nEXIT\n'
if [[ -f "$E2_BASE/evidence/exit-code" ]]; then cat "$E2_BASE/evidence/exit-code"; else printf 'RUNNING\n'; fi
