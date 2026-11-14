#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${EA_ATPO_RUN_DIR:-$ROOT/experiments/ea_atpo_v1_smoke/ea_atpo_v1_attempt09_stability}"
RUN_DIR="$(realpath -m "$RUN_DIR")"
GPU_LIST="${EA_ATPO_GPUS:-4,6}"

if [[ "$GPU_LIST" != "4,6" ]]; then
  echo "The validated stability run is pinned to GPUs 4 and 6; got '$GPU_LIST'." >&2
  exit 2
fi
if [[ -e "$RUN_DIR" ]]; then
  echo "Refusing to reuse existing run directory: $RUN_DIR" >&2
  echo "Set EA_ATPO_RUN_DIR to a new path; prior attempts are immutable." >&2
  exit 2
fi

# Ten PPO updates over 20 distinct official training cases. Four independently
# sampled Actor trajectories per state provide on-policy coverage; no action is
# forced. The dynamic legal-action context is part of the Actor prompt, so its
# generated log-probabilities remain the behavior-policy probabilities.
export EA_ATPO_RUN_DIR="$RUN_DIR"
export EA_ATPO_GPUS="$GPU_LIST"
export EA_ATPO_MIN_FREE_MIB="${EA_ATPO_MIN_FREE_MIB:-105000}"
export EA_ATPO_STABLE_GPU_SAMPLES="${EA_ATPO_STABLE_GPU_SAMPLES:-5}"
export EA_ATPO_GPU_SAMPLE_INTERVAL_SEC="${EA_ATPO_GPU_SAMPLE_INTERVAL_SEC:-15}"
export EA_ATPO_MAX_GPU_UTIL_PCT=100
export EA_ATPO_TRAIN_FRACTION="${EA_ATPO_TRAIN_FRACTION:-0.0014}"
export EA_ATPO_TEST_FRACTION="${EA_ATPO_TEST_FRACTION:-0.016}"
export EA_ATPO_TRAIN_BATCH_SIZE="${EA_ATPO_TRAIN_BATCH_SIZE:-2}"
export EA_ATPO_M_TRAJECTORIES="${EA_ATPO_M_TRAJECTORIES:-4}"
export EA_ATPO_N_CANDIDATES="${EA_ATPO_N_CANDIDATES:-4}"
export EA_ATPO_SAVE_FREQ="${EA_ATPO_SAVE_FREQ:-1}"
export EA_ATPO_TOTAL_EPOCHS="${EA_ATPO_TOTAL_EPOCHS:-1}"
export EA_ATPO_TEST_FREQ="${EA_ATPO_TEST_FREQ:-10}"
export EA_ATPO_MAX_ACTOR_ACTIONS="${EA_ATPO_MAX_ACTOR_ACTIONS:-3}"
export EA_ATPO_MAX_ASKS="${EA_ATPO_MAX_ASKS:-1}"
export EA_ATPO_MAX_CONSULTS="${EA_ATPO_MAX_CONSULTS:-1}"
export EA_ATPO_ASK_COST="${EA_ATPO_ASK_COST:-0.05}"
export EA_ATPO_CONSULT_COST="${EA_ATPO_CONSULT_COST:-0.20}"
export EA_ATPO_DEFER_COST="${EA_ATPO_DEFER_COST:-0.60}"
export EA_ATPO_CORRECTNESS_REWARD="${EA_ATPO_CORRECTNESS_REWARD:-1.0}"

mkdir -p "$RUN_DIR/validation"
"$ROOT/.venv/bin/python" - "$RUN_DIR/stability_plan.json" "$GPU_LIST" "$ROOT" <<'PY'
import hashlib, json, os, pathlib, sys
from datetime import datetime, timezone

out, gpu_list, root_arg = sys.argv[1:]
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

root=pathlib.Path(root_arg)
plan={
  "created_at_utc": datetime.now(timezone.utc).isoformat(),
  "status": "preflight_pending",
  "gpu_indices": [int(i) for i in gpu_list.split(',')],
  "base_model": "Qwen/Qwen3-8B",
  "official_train_source_sha256": sha(root/'ATPO-official/dataset/merged_train_dataset.jsonl'),
  "official_validation_source_sha256": sha(root/'ATPO-official/dataset/medqa_test_dataset.jsonl'),
  "planned": {
    "train_fraction": float(os.environ['EA_ATPO_TRAIN_FRACTION']), "expected_train_cases": 20,
    "validation_fraction": float(os.environ['EA_ATPO_TEST_FRACTION']), "expected_validation_cases": 20,
    "train_batch_size": int(os.environ['EA_ATPO_TRAIN_BATCH_SIZE']),
    "ppo_updates": int(20 / int(os.environ['EA_ATPO_TRAIN_BATCH_SIZE'])),
    "checkpoint_every_steps": int(os.environ['EA_ATPO_SAVE_FREQ']),
    "M_trajectories": int(os.environ['EA_ATPO_M_TRAJECTORIES']),
    "N_candidates": int(os.environ['EA_ATPO_N_CANDIDATES']),
    "ask_cost": float(os.environ['EA_ATPO_ASK_COST']),
    "consult_cost": float(os.environ['EA_ATPO_CONSULT_COST']),
    "defer_cost": float(os.environ['EA_ATPO_DEFER_COST']),
    "forced_actions_in_training": False,
    "memory_gate_mib_per_gpu": int(__import__('os').environ['EA_ATPO_MIN_FREE_MIB']),
    "attempt07_peak_reserved_gib_per_gpu": 88.19921875
  }
}
pathlib.Path(out).write_text(json.dumps(plan,indent=2)+'\n')
PY

exec "$ROOT/run_ea_atpo_v1_validated_smoke.sh"
