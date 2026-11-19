#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXP="$ROOT/experiments/expert_sft"
PYTHON="${EA_ATPO_SFT_PYTHON:-$ROOT/.venv/bin/python}"
GPU="${EA_ATPO_EVAL_GPU:-5}"
MIN_FREE_MIB="150000"
MAX_UTIL="20"
OUT="$EXP/evaluations/fresh_test_once"
POLICY="$EXP/evaluations/router/router_policy_lock.json"

export HF_HOME="$EXP/cache/hf_home"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$EXP/cache/datasets"
export XDG_CACHE_HOME="$EXP/cache/xdg"
export TMPDIR="$EXP/cache/tmp"
export TRITON_CACHE_DIR="$EXP/cache/triton"
export TORCH_HOME="$EXP/cache/torch"
export CUDA_VISIBLE_DEVICES="$GPU"
mkdir -p "$EXP/logs" "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCH_HOME"

[[ -f "$POLICY" ]] || { echo "Router train/validation policy must be frozen before test inference." >&2; exit 2; }
[[ -f "$EXP/checkpoints/qwen3_32b_domain_sft/selected_checkpoint_lock.json" ]] || { echo "SFT Expert checkpoint must be frozen before test inference." >&2; exit 2; }
[[ ! -e "$EXP/evaluations/router/test_opened_once.json" ]] || { echo "The one-shot fresh test cohort has already been opened." >&2; exit 2; }
[[ ! -e "$OUT" ]] || { echo "Fresh-test output directory already exists; refusing rerun." >&2; exit 2; }

gpu_gate() {
  "$PYTHON" - "$GPU" "$MIN_FREE_MIB" "$MAX_UTIL" <<'PY'
import subprocess,sys
idx,minimum,maxutil=map(int,sys.argv[1:])
line=subprocess.check_output(["nvidia-smi",f"--id={idx}","--query-gpu=index,memory.free,utilization.gpu","--format=csv,noheader,nounits"],text=True).strip()
gpu,free,util=map(lambda x:int(x.strip()),line.split(","))
if gpu!=idx or free<minimum or util>maxutil: raise SystemExit(f"GPU gate failed: gpu={gpu},free_mib={free},util={util}")
pids=subprocess.check_output(["nvidia-smi",f"--id={idx}","--query-compute-apps=pid","--format=csv,noheader,nounits"],text=True).strip()
if pids and pids.upper()!="N/A": raise SystemExit(f"GPU {idx} already has compute processes; refusing to use it: {pids}")
print(f"GPU {gpu} safe: free_mib={free}, utilization={util}, no compute processes")
PY
}

# This is post-lock materialization: the selected cases were frozen without Gold,
# and the checkpoint/router are already frozen before labels enter the evaluator.
if [[ ! -f "$EXP/data/fresh_test_sft.jsonl" ]]; then
  "$PYTHON" "$EXP/scripts/build_fresh_test_data.py" 2>&1 | tee "$EXP/logs/materialize_fresh_test.log"
fi
mkdir -p "$OUT"
BASE="$OUT/base_test/predictions.jsonl"
EXPERT_DIR="$OUT/expert_test"
EXPERT="$EXPERT_DIR/predictions.jsonl"
gpu_gate
"$PYTHON" "$EXP/scripts/evaluate_base_paired.py" \
  --data "$EXP/data/fresh_test_sft.jsonl" --split test --output "$BASE" 2>&1 | tee "$EXP/logs/base_fresh_test.log"
gpu_gate
"$PYTHON" "$EXP/scripts/evaluate_sft_checkpoint.py" \
  --adapter "$EXP/checkpoints/qwen3_32b_domain_sft/selected" \
  --data "$EXP/data/fresh_test_sft.jsonl" --split test --output-dir "$EXPERT_DIR" 2>&1 | tee "$EXP/logs/expert_fresh_test.log"
"$PYTHON" "$EXP/scripts/assemble_paired_results.py" \
  --base "$BASE" --expert "$EXPERT" --split test \
  --output "$OUT/paired_test.jsonl"
"$PYTHON" "$EXP/scripts/evaluate_frozen_router.py" \
  --test "$OUT/paired_test.jsonl" --policy-lock "$POLICY" \
  --output-dir "$OUT/router_evaluation" 2>&1 | tee "$EXP/logs/router_fresh_test.log"
