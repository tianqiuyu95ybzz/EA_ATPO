#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
OUT_DIR="$SCRIPT_DIR/outputs/qwen32_defer_validation_20261119_attempt01"
VENV="/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/.venv-real-pilot"
STARTED_EXPERT=0 STARTED_BASE=0 MON_PID=""

# Refuse before opening GPUs or truncating telemetry if this run directory has
# ever entered an inference/service attempt. Preserve old outputs by archiving
# the complete directory before creating a fresh run.
for marker in \
  "$OUT_DIR/gpu2_usage_attempt03.csv" \
  "$OUT_DIR/services/attempt03/base/service.log" \
  "$OUT_DIR/services/attempt03/expert/service.log" \
  "$OUT_DIR/smoke_20/run_manifest.json" \
  "$OUT_DIR/paired_rollouts/run_manifest.json" \
  "$OUT_DIR/experiment_complete.txt"; do
  if [[ -e "$marker" ]]; then
    echo "Existing run artifact found; refusing to start or overwrite: $marker" >&2
    exit 20
  fi
done

if [[ (-e "$OUT_DIR/case_selection.json" && ! -e "$OUT_DIR/model_fingerprint.json") || (! -e "$OUT_DIR/case_selection.json" && -e "$OUT_DIR/model_fingerprint.json") ]]; then
  echo "Partial preflight artifacts found; inspect and archive this run directory before retrying." >&2
  exit 22
fi
if [[ ! -e "$OUT_DIR/case_selection.json" ]]; then
  "$VENV/bin/python" "$SCRIPT_DIR/fingerprint_qwen32_defer_models.py"
  "$VENV/bin/python" "$SCRIPT_DIR/select_qwen32_defer_validation_cases.py"
fi

cleanup() {
  if [[ -n "$MON_PID" ]]; then kill "$MON_PID" 2>/dev/null || true; fi
  if (( STARTED_BASE )) && [[ -s "$OUT_DIR/services/attempt03/base/service.pid" ]]; then
    kill -TERM "$(cat "$OUT_DIR/services/attempt03/base/service.pid")" 2>/dev/null || true
  fi
  if (( STARTED_EXPERT )) && [[ -s "$OUT_DIR/services/attempt03/expert/service.pid" ]]; then
    kill -TERM "$(cat "$OUT_DIR/services/attempt03/expert/service.pid")" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

mkdir -p "$OUT_DIR"
printf 'timestamp_utc,memory_used_mib,memory_free_mib,utilization_pct\n' > "$OUT_DIR/gpu2_usage_attempt03.csv"
(
  while true; do
    ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    vals="$(nvidia-smi --id=2 --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')"
    [[ -n "$vals" ]] && printf '%s,%s\n' "$ts" "$vals" >> "$OUT_DIR/gpu2_usage_attempt03.csv"
    sleep 1
  done
) &
MON_PID=$!

bash "$SCRIPT_DIR/start_qwen32_defer_validation_service.sh"
STARTED_EXPERT=1
bash "$SCRIPT_DIR/start_qwen8_defer_validation_service.sh"
STARTED_BASE=1

if [[ -e "$OUT_DIR/smoke_20/run_manifest.json" ]]; then
  echo "Smoke output already exists; refusing to overwrite it." >&2
  exit 20
fi
"$VENV/bin/python" "$SCRIPT_DIR/run_qwen32_defer_validation.py" --smoke
"$VENV/bin/python" "$SCRIPT_DIR/check_qwen32_defer_smoke.py"

if [[ -e "$OUT_DIR/paired_rollouts/run_manifest.json" ]]; then
  echo "Main rollout manifest already exists; use explicit resume after auditing the partial result." >&2
  exit 21
fi
"$VENV/bin/python" "$SCRIPT_DIR/run_qwen32_defer_validation.py"
"$VENV/bin/python" "$SCRIPT_DIR/fit_qwen32_public_feature_router.py"
"$VENV/bin/python" "$SCRIPT_DIR/analyze_qwen32_defer_validation.py"

printf 'completed_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$OUT_DIR/experiment_complete.txt"
echo "Qwen3-32B-AWQ DEFER validation complete: $OUT_DIR"
