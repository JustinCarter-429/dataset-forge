#!/usr/bin/env bash
set -Eeuo pipefail

: "${E2_BASE:?E2_BASE is required}"
: "${E2_GIT_SHA:?E2_GIT_SHA is required}"
ROOT="$E2_BASE/source/training"
DATA="$E2_BASE/corpus"
CONFIG="$ROOT/evaluation/e2/full-retraining/config.json"
RUNNER="$ROOT/evaluation/e2/full-retraining/e2_train.py"
MODEL="$E2_BASE/model"
EVIDENCE="$E2_BASE/evidence"
LOCK="$E2_BASE/e2-retraining.launch-lock"
mkdir -p "$EVIDENCE"
if ! mkdir "$LOCK" 2>/dev/null; then printf 'DUPLICATE_LAUNCH:%s\n' "$LOCK" >&2; exit 9; fi
exec > >(tee -a "$EVIDENCE/stdout.log") 2> >(tee -a "$EVIDENCE/stderr.log" >&2)

atomic_stage() {
  printf '{"status":"%s","phase":"%s","updated_utc":"%s"}\n' "$1" "$2" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$EVIDENCE/progress.json.tmp"
  mv "$EVIDENCE/progress.json.tmp" "$EVIDENCE/progress.json"
}
finish() {
  code=$?
  printf '%s\n' "$code" > "$EVIDENCE/exit-code.tmp"; mv "$EVIDENCE/exit-code.tmp" "$EVIDENCE/exit-code"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$EVIDENCE/finished-utc.txt"
  if [[ $code -eq 0 ]]; then atomic_stage complete training_complete; else atomic_stage failed "exit_$code"; fi
  rmdir "$LOCK" 2>/dev/null || true
  exit "$code"
}
trap finish EXIT
trap 'touch "$E2_BASE/full/STOP_REQUESTED" 2>/dev/null || true' TERM INT

source /venv/main/bin/activate
export PYTHONPATH="$ROOT/src"
date -u +%Y-%m-%dT%H:%M:%SZ > "$EVIDENCE/started-utc.txt"
printf '%s\n' "$E2_GIT_SHA" > "$EVIDENCE/pretraining-git-sha.txt"

atomic_stage running smoke
if [[ ! -f "$E2_BASE/smoke/result.json" ]]; then
  python "$RUNNER" --mode smoke --config "$CONFIG" --data-dir "$DATA" --run-dir "$E2_BASE/smoke" --model-path "$MODEL" --git-sha "$E2_GIT_SHA"
fi
python - "$E2_BASE/smoke/result.json" <<'PY'
import json,sys
value=json.load(open(sys.argv[1]))
raise SystemExit(0 if value.get('status') == 'PASS' else 4)
PY

atomic_stage running full_training
resume=()
if [[ -f "$E2_BASE/full/latest-checkpoint.json" && ! -f "$E2_BASE/full/result.json" ]]; then
  checkpoint="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["path"])' "$E2_BASE/full/latest-checkpoint.json")"
  resume=(--resume "$checkpoint")
fi
if [[ ! -f "$E2_BASE/full/result.json" ]]; then
  python "$RUNNER" --mode full --config "$CONFIG" --data-dir "$DATA" --run-dir "$E2_BASE/full" --model-path "$MODEL" --git-sha "$E2_GIT_SHA" "${resume[@]}"
fi

atomic_stage complete training_complete
