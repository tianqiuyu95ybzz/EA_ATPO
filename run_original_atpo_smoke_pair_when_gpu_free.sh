#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WAIT_SECONDS="${EA_ATPO_GPU_WAIT_SECONDS:-900}"
POLL_SECONDS="${EA_ATPO_GPU_POLL_SECONDS:-15}"
MAX_UTIL="${EA_ATPO_MAX_GPU_UTIL_PCT:-100}"
MIN_FREE_MIB="${EA_ATPO_MIN_FREE_MIB:-100000}"
STABLE_SAMPLES="${EA_ATPO_GPU_STABLE_SAMPLES:-5}"
STABLE_INTERVAL="${EA_ATPO_GPU_STABLE_INTERVAL:-15}"
GPU_LIST="${EA_ATPO_GPUS:-4,6}"
LOG_DIR="$ROOT/experiments/ea_atpo_v1_smoke"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/gpu_pair_wait_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG_FILE") 2>&1

if [[ ! "$GPU_LIST" =~ ^[0-7](,[0-7])+$ ]]; then
  echo "Set EA_ATPO_GPUS to a comma-separated pair or group, such as 4,6." >&2
  exit 2
fi
IFS=',' read -r -a GPU_IDS <<< "$GPU_LIST"
SEEN_GPU_IDS=()
for GPU_INDEX in "${GPU_IDS[@]}"; do
  if [[ "$GPU_INDEX" == "1" || "$GPU_INDEX" == "7" ]]; then
    echo "GPU $GPU_INDEX is excluded from this project run." >&2
    exit 2
  fi
  for SEEN_GPU in "${SEEN_GPU_IDS[@]}"; do
    if [[ "$GPU_INDEX" == "$SEEN_GPU" ]]; then
      echo "GPU $GPU_INDEX occurs more than once in EA_ATPO_GPUS." >&2
      exit 2
    fi
  done
  SEEN_GPU_IDS+=("$GPU_INDEX")
done

START_TIME="$(date +%s)"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) Watching GPU set $GPU_LIST. Each GPU must have >=${MIN_FREE_MIB}MiB free for ${STABLE_SAMPLES} samples; measured utilization cap is ${MAX_UTIL}%."
echo "GPU 1 and 7 are excluded. Utilization is recorded because this is a shared-GPU run. Wait log: $LOG_FILE"

check_selected_gpus() {
  local gpu_index gpu_status free_mib util_pct
  for gpu_index in "${GPU_IDS[@]}"; do
    gpu_status="$(nvidia-smi -i "$gpu_index" --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits)"
    free_mib="${gpu_status%%,*}"
    util_pct="${gpu_status##*, }"
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) GPU $gpu_index free=${free_mib}MiB utilization=${util_pct}%"
    if (( free_mib < MIN_FREE_MIB || util_pct > MAX_UTIL )); then
      return 1
    fi
  done
  return 0
}

while true; do
  if check_selected_gpus; then
    STABLE=1
    for ((SAMPLE = 1; SAMPLE < STABLE_SAMPLES; SAMPLE++)); do
      sleep "$STABLE_INTERVAL"
      if ! check_selected_gpus; then
        echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) GPU set $GPU_LIST lost the resource gate during stability checks."
        STABLE=0
        break
      fi
    done

    if (( STABLE == 1 )); then
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) Starting baseline smoke on GPU set $GPU_LIST."
      set +e
      EA_ATPO_GPUS="$GPU_LIST" \
      EA_ATPO_MIN_FREE_MIB="$MIN_FREE_MIB" \
      EA_ATPO_MAX_GPU_UTIL_PCT="$MAX_UTIL" \
        "$ROOT/run_original_atpo_smoke.sh"
      RUN_STATUS=$?
      set -e
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) Baseline smoke exited with code $RUN_STATUS."
      exit "$RUN_STATUS"
    fi
  fi

  NOW="$(date +%s)"
  if (( NOW - START_TIME >= WAIT_SECONDS )); then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) No eligible GPU set met the gate within ${WAIT_SECONDS}s; no smoke was started."
    exit 3
  fi
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) Waiting for GPU set $GPU_LIST to meet the resource gate."
  sleep "$POLL_SECONDS"
done
