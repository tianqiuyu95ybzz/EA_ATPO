#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXP="$ROOT/experiments/expert_sft"
PYTHON="${EA_ATPO_SFT_PYTHON:-$ROOT/.venv/bin/python}"
GPU="5"
MIN_FREE_MIB="150000"
MAX_UTIL="20"

export HF_HOME="$EXP/cache/hf_home"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$EXP/cache/datasets"
export XDG_CACHE_HOME="$EXP/cache/xdg"
export TMPDIR="$EXP/cache/tmp"
export TRITON_CACHE_DIR="$EXP/cache/triton"
export TORCH_HOME="$EXP/cache/torch"
export CUDA_VISIBLE_DEVICES="$GPU"
export EA_ATPO_SFT_PHYSICAL_GPU="$GPU"
mkdir -p "$EXP/logs" "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCH_HOME"

python3 - "$GPU" "$MIN_FREE_MIB" "$MAX_UTIL" <<'PY'
import subprocess, sys
idx, min_free, max_util = map(int, sys.argv[1:])
line = subprocess.check_output(["nvidia-smi", f"--id={idx}", "--query-gpu=index,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True).strip()
gpu, free, util = [int(x.strip()) for x in line.split(",")]
if gpu != idx or free < min_free or util > max_util:
    raise SystemExit(f"GPU {idx} fails safety gate: free_mib={free}, utilization_pct={util}")
pids = subprocess.check_output(["nvidia-smi", f"--id={idx}", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip()
if pids and pids.upper() != "N/A":
    raise SystemExit(f"GPU {idx} has an existing compute process; refusing to start: {pids}")
print(f"GPU {gpu} passed safety gate: free_mib={free}, utilization_pct={util}, compute_pids=none")
PY

"$PYTHON" "$EXP/scripts/verify_model_assets.py" --model "$EXP/assets/models/Qwen3-32B" --write-manifest "$EXP/data/model_asset_manifest.json"
"$PYTHON" "$EXP/scripts/tokenization_smoke.py"
"$PYTHON" "$EXP/scripts/train_lora_sft.py" --smoke-steps 10 2>&1 | tee "$EXP/logs/sft_smoke_console.log"
"$PYTHON" "$EXP/scripts/evaluate_sft_checkpoint.py" \
  --adapter "$EXP/checkpoints/qwen3_32b_domain_sft/smoke/final_adapter" \
  --max-cases 5 --output-dir "$EXP/evaluations/smoke" 2>&1 | tee "$EXP/logs/sft_smoke_eval_console.log"
