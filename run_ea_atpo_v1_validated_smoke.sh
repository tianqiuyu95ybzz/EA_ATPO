#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

GPU_LIST="${EA_ATPO_GPUS:-4,6}"
MIN_FREE_MIB="${EA_ATPO_MIN_FREE_MIB:-105000}"
STABLE_SAMPLES="${EA_ATPO_STABLE_GPU_SAMPLES:-5}"
SAMPLE_INTERVAL="${EA_ATPO_GPU_SAMPLE_INTERVAL_SEC:-15}"
RUN_DIR="${EA_ATPO_RUN_DIR:-$ROOT/experiments/ea_atpo_v1_smoke/ea_atpo_v1_attempt07}"
RUN_DIR="$(realpath -m "$RUN_DIR")"
TEST_LOG="$RUN_DIR/validation/four_action_protocol_tests.log"
GPU_LOG="$RUN_DIR/validation/gpu_memory_gate.log"

if [[ "$GPU_LIST" != "4,6" ]]; then
  echo "This validated reproduction uses only GPUs 4 and 6; got '$GPU_LIST'." >&2
  exit 2
fi
if (( MIN_FREE_MIB < 100000 )); then
  echo "MIN_FREE_MIB must be >= 100000; 105000 MiB provides about 14 GiB above Attempt 06 peak reserved memory." >&2
  exit 2
fi

mkdir -p "$RUN_DIR/validation"
PYTHONPATH="$ROOT/ea_atpo_stage:$ROOT/ATPO-official/codes${PYTHONPATH:+:$PYTHONPATH}" \
  "$ROOT/.venv/bin/python" -m unittest recipe.ea_atpo.test_ea_agent_loop -v \
  2>&1 | tee "$TEST_LOG"

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) Watching GPUs $GPU_LIST; each must keep >=${MIN_FREE_MIB} MiB free for ${STABLE_SAMPLES} samples ${SAMPLE_INTERVAL}s apart. Utilization is recorded but VRAM headroom is the admission gate." | tee "$GPU_LOG"
for ((sample = 1; sample <= STABLE_SAMPLES; sample++)); do
  sample_ok=1
  for gpu in 4 6; do
    IFS=',' read -r free_mib used_mib util_pct < <(
      nvidia-smi -i "$gpu" --query-gpu=memory.free,memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' '
    )
    printf '%s sample=%s gpu=%s free_mib=%s used_mib=%s util_pct=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$sample" "$gpu" "$free_mib" "$used_mib" "$util_pct" | tee -a "$GPU_LOG"
    if (( free_mib < MIN_FREE_MIB )); then
      sample_ok=0
      echo "Free VRAM below safety threshold on GPU $gpu; no training process was started." | tee -a "$GPU_LOG" >&2
      exit 2
    fi
  done
  if (( sample < STABLE_SAMPLES )); then
    sleep "$SAMPLE_INTERVAL"
  fi
done

export EA_ATPO_GPUS="$GPU_LIST"
export EA_ATPO_MIN_FREE_MIB="$MIN_FREE_MIB"
export EA_ATPO_MAX_GPU_UTIL_PCT=100
export EA_ATPO_RUN_DIR="$RUN_DIR"
exec "$ROOT/run_ea_atpo_v1_smoke.sh"
