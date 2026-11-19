#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXP="$ROOT/experiments/expert_sft"
PYTHON="${EA_ATPO_SFT_PYTHON:-$ROOT/.venv/bin/python}"
HF_CLI="$ROOT/.venv/bin/hf"
REPO="Qwen/Qwen3-32B"
REVISION="9216db5781bf21249d130ec9da846c4624c16137"
MODEL_DIR="$EXP/assets/models/Qwen3-32B"

export HF_HOME="$EXP/cache/hf_home"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$EXP/cache/datasets"
export XDG_CACHE_HOME="$EXP/cache/xdg"
export TMPDIR="$EXP/cache/tmp"
export TRITON_CACHE_DIR="$EXP/cache/triton"
export TORCH_HOME="$EXP/cache/torch"
export HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "$MODEL_DIR" "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME" "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCH_HOME"
[[ -x "$HF_CLI" ]] || { echo "Missing workspace HF CLI: $HF_CLI" >&2; exit 2; }

# huggingface_hub stores incomplete downloads in the workspace HF cache and
# resumes them on rerun. Do not use a system-wide cache path.
"$HF_CLI" download "$REPO" --repo-type model --revision "$REVISION" --local-dir "$MODEL_DIR"
"$PYTHON" "$EXP/scripts/verify_model_assets.py" --model "$MODEL_DIR" --write-manifest "$EXP/data/model_asset_manifest.json"
