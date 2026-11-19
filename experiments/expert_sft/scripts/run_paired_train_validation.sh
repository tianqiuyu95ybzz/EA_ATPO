#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXP="$ROOT/experiments/expert_sft"
PYTHON="${EA_ATPO_SFT_PYTHON:-$ROOT/.venv/bin/python}"
GPU="${EA_ATPO_EVAL_GPU:-5}"
MIN_FREE_MIB="150000"
MAX_UTIL="20"
OUT="$EXP/evaluations/base_expert_pair"

export HF_HOME="$EXP/cache/hf_home"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$EXP/cache/datasets"
export XDG_CACHE_HOME="$EXP/cache/xdg"
export TMPDIR="$EXP/cache/tmp"
export TRITON_CACHE_DIR="$EXP/cache/triton"
export TORCH_HOME="$EXP/cache/torch"
export CUDA_VISIBLE_DEVICES="$GPU"
mkdir -p "$EXP/logs" "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCH_HOME"

[[ -f "$EXP/checkpoints/qwen3_32b_domain_sft/selected_checkpoint_lock.json" ]] || { echo "SFT Expert checkpoint is not frozen." >&2; exit 2; }
[[ -f "$EXP/assets/models/Qwen3-8B/model.safetensors.index.json" ]] || { echo "Pinned Qwen3-8B Base model is incomplete." >&2; exit 2; }
[[ -f "$EXP/data/fresh_test_selection.json" ]] || { echo "Fresh case-level test selection is missing." >&2; exit 2; }
[[ ! -f "$EXP/evaluations/router/router_policy_lock.json" ]] || { echo "Router policy is already frozen; refusing to repeat train/validation fitting." >&2; exit 2; }

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

for SPLIT in train validation; do
  DATA="$EXP/data/${SPLIT}_sft.jsonl"
  BASE="$OUT/base_${SPLIT}/predictions.jsonl"
  EXPERT_DIR="$OUT/expert_${SPLIT}"
  EXPERT="$EXPERT_DIR/predictions.jsonl"
  mkdir -p "$OUT"
  [[ ! -e "$BASE" && ! -e "$EXPERT" ]] || { echo "Existing paired inference output found for $SPLIT; refusing to overwrite." >&2; exit 2; }
  gpu_gate
  "$PYTHON" "$EXP/scripts/evaluate_base_paired.py" --data "$DATA" --split "$SPLIT" --output "$BASE" 2>&1 | tee "$EXP/logs/base_${SPLIT}_paired.log"
  gpu_gate
  "$PYTHON" "$EXP/scripts/evaluate_sft_checkpoint.py" \
    --adapter "$EXP/checkpoints/qwen3_32b_domain_sft/selected" \
    --data "$DATA" --split "$SPLIT" --output-dir "$EXPERT_DIR" 2>&1 | tee "$EXP/logs/expert_${SPLIT}_paired.log"
  "$PYTHON" "$EXP/scripts/assemble_paired_results.py" \
    --base "$BASE" --expert "$EXPERT" --split "$SPLIT" \
    --output "$OUT/paired_${SPLIT}.jsonl"
done

"$PYTHON" "$EXP/scripts/fit_cost_sensitive_router.py" \
  --train "$OUT/paired_train.jsonl" --validation "$OUT/paired_validation.jsonl" \
  --output-dir "$EXP/evaluations/router" 2>&1 | tee "$EXP/logs/router_fit.log"
