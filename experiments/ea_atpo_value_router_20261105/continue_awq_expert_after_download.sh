#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODEL_DIR="$PROJECT_DIR/experiments/real_case_pilot_20261103/assets/models/Qwen3-32B-AWQ"
DOWNLOAD_STATUS="$MODEL_DIR/range_download_status.json"
STEPWISE_MANIFEST="$SCRIPT_DIR/outputs/stepwise_test_v2/run_manifest.json"
PYTHON="$PROJECT_DIR/.venv-real-pilot/bin/python"
cd "$PROJECT_DIR"

read_status() {
  "$PYTHON" - "$1" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
try:
    print(json.loads(p.read_text()).get("status", "missing"))
except Exception:
    print("missing")
PY
}

while [[ "$(read_status "$STEPWISE_MANIFEST")" != "completed" ]]; do sleep 20; done
while [[ "$(read_status "$DOWNLOAD_STATUS")" != "all_shards_verified" ]]; do
  download_state="$(read_status "$DOWNLOAD_STATUS")"
  if [[ "$download_state" == "interrupted_or_failed_resumable" ]]; then
    echo "Range transfer was interrupted; restarting from verified part files."
    "$PYTHON" "$SCRIPT_DIR/download_official_qwen3_32b_awq_ranged.py" \
      --model-dir "$MODEL_DIR" --max-workers "${EA_ATPO_AWQ_MAX_WORKERS:-24}" --chunk-mib 16 || true
  elif [[ "$download_state" == "running" ]] && ! pgrep -f 'download_official_qwen3_32b_awq_ranged.py' >/dev/null 2>&1; then
    echo "No range downloader process is active; resuming from saved part files."
    "$PYTHON" "$SCRIPT_DIR/download_official_qwen3_32b_awq_ranged.py" \
      --model-dir "$MODEL_DIR" --max-workers "${EA_ATPO_AWQ_MAX_WORKERS:-24}" --chunk-mib 16 || true
  else
    sleep 30
  fi
done
"$PYTHON" "$SCRIPT_DIR/write_strong_expert_recovery_report.py"

# The manifest records the exact AWQ Hub revision; service startup performs an
# index/shard presence check and the downloader verifies every official SHA-256.
for file in config.json model.safetensors.index.json tokenizer.json tokenizer_config.json; do
  [[ -f "$MODEL_DIR/$file" ]] || { echo "Missing verified model file: $MODEL_DIR/$file" >&2; exit 2; }
done

# Release only the Router vLLM server started by this experiment.
BASE_PID_FILE="$SCRIPT_DIR/outputs/model_service.pid"
if [[ -f "$BASE_PID_FILE" ]]; then
  base_pid="$(cat "$BASE_PID_FILE")"
  base_cmd="$(ps -p "$base_pid" -o args= 2>/dev/null || true)"
  if [[ "$base_cmd" == *"$PROJECT_DIR/experiments/real_case_pilot_20261103/assets/models/Qwen3-8B"* ]]; then
    kill -TERM "$base_pid"
    for _ in $(seq 1 60); do kill -0 "$base_pid" 2>/dev/null || break; sleep 1; done
  fi
fi

while :; do
  best_gpu=""
  best_free_mib=0
  while IFS=, read -r gpu_index free_mib; do
    gpu_index="${gpu_index//[[:space:]]/}"
    free_mib="${free_mib//[[:space:]]/}"
    if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib > best_free_mib )); then
      best_gpu="$gpu_index"
      best_free_mib="$free_mib"
    fi
  done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
  if [[ -n "$best_gpu" ]] && (( best_free_mib >= 45000 )); then break; fi
  echo "Waiting for a GPU with 45000 MiB free; best available is ${best_free_mib} MiB." >&2
  sleep 60
done

if EA_ATPO_EXPERT_GPU="$best_gpu" EA_ATPO_EXPERT_VARIANT=AWQ bash "$SCRIPT_DIR/run_quantized_expert_eval.sh"; then
  :
else
  run_status=$?
  "$PYTHON" - "$SCRIPT_DIR/outputs/strong_expert_awq/deployment_failure.json" "$run_status" "$best_gpu" <<'PY'
import json,sys
from datetime import datetime,timezone
from pathlib import Path
p=Path(sys.argv[1])
p.write_text(json.dumps({"stage":"vLLM deployment or fixed-sample evaluation","exit_code":int(sys.argv[2]),"gpu":int(sys.argv[3]),"time_utc":datetime.now(timezone.utc).isoformat()},indent=2)+"\n")
PY
  "$PYTHON" "$SCRIPT_DIR/write_strong_expert_recovery_report.py"
  "$PYTHON" "$SCRIPT_DIR/refresh_reproducibility_manifest.py"
  exit "$run_status"
fi
"$PYTHON" "$SCRIPT_DIR/write_strong_expert_recovery_report.py"

# Stop only the AWQ vLLM process launched by its service wrapper.
EXPERT_PID_FILE="$SCRIPT_DIR/outputs/strong_expert_awq/model_service.pid"
if [[ -f "$EXPERT_PID_FILE" ]]; then
  expert_pid="$(cat "$EXPERT_PID_FILE")"
  expert_cmd="$(ps -p "$expert_pid" -o args= 2>/dev/null || true)"
  if [[ "$expert_cmd" == *"$MODEL_DIR"* ]]; then
    kill -TERM "$expert_pid"
  fi
fi
"$PYTHON" "$SCRIPT_DIR/refresh_reproducibility_manifest.py"
