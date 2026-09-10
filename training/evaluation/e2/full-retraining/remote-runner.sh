#!/usr/bin/env bash
set -Eeuo pipefail

: "${E2_BASE:?E2_BASE is required}"
: "${E2_GIT_SHA:?E2_GIT_SHA is required}"
: "${E2_FULL_TRAINING_AUTHORIZED:?E2_FULL_TRAINING_AUTHORIZED is required}"
: "${E2_PACKAGE_ARCHIVE:?E2_PACKAGE_ARCHIVE is required}"
: "${E2_PACKAGE_SHA256:?E2_PACKAGE_SHA256 is required}"
: "${E2_SYNC_COMMAND:?E2_SYNC_COMMAND must name a resumable off-instance sync command}"
[[ "$E2_FULL_TRAINING_AUTHORIZED" == "YES" ]] || { printf 'FULL_TRAINING_NOT_AUTHORIZED\n' >&2; exit 13; }

ROOT="$E2_BASE/source/training"
DATA="$E2_BASE/corpus"
CONFIG="$ROOT/evaluation/e2/full-retraining/config.json"
RUNNER="$ROOT/evaluation/e2/full-retraining/e2_train.py"
MODEL="$E2_BASE/model"
EVIDENCE="$E2_BASE/evidence"
RUN="$E2_BASE/full"
LOCK="$E2_BASE/e2-full-retraining.launch-lock"
mkdir -p "$EVIDENCE"
if ! mkdir "$LOCK" 2>/dev/null; then printf 'DUPLICATE_LAUNCH:%s\n' "$LOCK" >&2; exit 9; fi
exec > >(tee -a "$EVIDENCE/stdout.log") 2> >(tee -a "$EVIDENCE/stderr.log" >&2)

atomic_stage() {
  printf '{"status":"%s","phase":"%s","updated_utc":"%s"}\n' "$1" "$2" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$EVIDENCE/progress.json.tmp"
  mv "$EVIDENCE/progress.json.tmp" "$EVIDENCE/progress.json"
}
sync_now() {
  "$E2_SYNC_COMMAND" "$E2_BASE" || { atomic_stage failed off_instance_sync_failed; return 1; }
}
sync_loop() {
  while [[ ! -f "$EVIDENCE/exit-code" ]]; do sleep 60; sync_now; done
}
finish() {
  code=$?
  [[ -n "${SYNC_PID:-}" ]] && kill "$SYNC_PID" 2>/dev/null || true
  sync_now || code=14
  printf '%s\n' "$code" > "$EVIDENCE/exit-code.tmp"; mv "$EVIDENCE/exit-code.tmp" "$EVIDENCE/exit-code"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$EVIDENCE/finished-utc.txt"
  if [[ $code -eq 0 ]]; then atomic_stage complete training_complete; else atomic_stage failed "exit_$code"; fi
  rmdir "$LOCK" 2>/dev/null || true
  exit "$code"
}
trap finish EXIT
trap 'touch "$RUN/STOP_REQUESTED" 2>/dev/null || true' TERM INT

actual_package_sha256="$(sha256sum "$E2_PACKAGE_ARCHIVE" | awk '{print $1}')"
[[ "$actual_package_sha256" == "$E2_PACKAGE_SHA256" ]] || { printf 'PACKAGE_HASH_MISMATCH\n' >&2; exit 10; }
python "$ROOT/evaluation/e2/full-retraining/preflight.py" verify-package --archive "$E2_PACKAGE_ARCHIVE"
source /venv/main/bin/activate
export PYTHONPATH="$ROOT/src"
date -u +%Y-%m-%dT%H:%M:%SZ > "$EVIDENCE/started-utc.txt"
printf '%s\n' "$E2_GIT_SHA" > "$EVIDENCE/pretraining-git-sha.txt"
atomic_stage running full_training
sync_loop & SYNC_PID=$!

resume=()
if [[ -f "$RUN/latest-checkpoint.json" && ! -f "$RUN/result.json" ]]; then
  checkpoint="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["path"])' "$RUN/latest-checkpoint.json")"
  resume=(--resume "$checkpoint")
fi
python "$RUNNER" --mode full --config "$CONFIG" --data-dir "$DATA" --run-dir "$RUN" --model-path "$MODEL" --git-sha "$E2_GIT_SHA" "${resume[@]}"
python - "$RUN/result.json" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if value.get("status") == "COMPLETE" else 4)
PY
