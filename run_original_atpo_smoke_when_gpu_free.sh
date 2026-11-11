#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WAIT_SECONDS="${EA_ATPO_GPU_WAIT_SECONDS:-3600}"
POLL_SECONDS="${EA_ATPO_GPU_POLL_SECONDS:-15}"
MAX_UTIL="${EA_ATPO_MAX_GPU_UTIL_PCT:-15}"
MIN_FREE_MIB="${EA_ATPO_MIN_FREE_MIB:-155000}"
STABLE_SAMPLES="${EA_ATPO_GPU_STABLE_SAMPLES:-5}"
STABLE_INTERVAL="${EA_ATPO_GPU_STABLE_INTERVAL:-15}"
START_TIME="$(date +%s)"
ELIGIBLE_GPUS=(0 2 3 4 5 6)

while true; do
  CANDIDATE_GPU=""
  for GPU_INDEX in "${ELIGIBLE_GPUS[@]}"; do
    GPU_STATUS="$(nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits)"
    FREE_MIB="${GPU_STATUS%%,*}"
    UTIL_PCT="${GPU_STATUS##*, }"
    if (( FREE_MIB >= MIN_FREE_MIB && UTIL_PCT <= MAX_UTIL )); then
      CANDIDATE_GPU="$GPU_INDEX"
      break
    fi
  done

  if [[ -n "$CANDIDATE_GPU" ]]; then
    STABLE=1
    for ((SAMPLE = 1; SAMPLE < STABLE_SAMPLES; SAMPLE++)); do
      sleep "$STABLE_INTERVAL"
      GPU_STATUS="$(nvidia-smi -i "$CANDIDATE_GPU" --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits)"
      FREE_MIB="${GPU_STATUS%%,*}"
      UTIL_PCT="${GPU_STATUS##*, }"
      if (( FREE_MIB < MIN_FREE_MIB || UTIL_PCT > MAX_UTIL )); then
        echo "GPU $CANDIDATE_GPU did not stay eligible across the stability window; will keep waiting."
        STABLE=0
        break
      fi
    done
    if (( STABLE == 1 )); then
      echo "Starting original ATPO smoke on stable GPU $CANDIDATE_GPU (free=${FREE_MIB}MiB, utilization=${UTIL_PCT}%)."
      EA_ATPO_GPU="$CANDIDATE_GPU" EA_ATPO_MAX_GPU_UTIL_PCT="$MAX_UTIL" EA_ATPO_MIN_FREE_MIB="$MIN_FREE_MIB" \
        "$ROOT/run_original_atpo_smoke.sh"
      exit $?
    fi
  fi

  NOW="$(date +%s)"
  if (( NOW - START_TIME >= WAIT_SECONDS )); then
    echo "No eligible GPU became available within ${WAIT_SECONDS}s; GPUs 1 and 7 were excluded." >&2
    exit 3
  fi
  echo "Waiting for an eligible GPU; checking only 0,2,3,4,5,6."
  sleep "$POLL_SECONDS"
done
