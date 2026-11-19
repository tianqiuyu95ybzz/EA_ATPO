#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
MULTI="$ROOT/experiments/expert_sft/multiturn_dialogue"
ENV_DIR="$MULTI/.venv_awq"
PIP_CACHE="$MULTI/.cache/pip"
TMP_DIR="$MULTI/.cache/tmp"
LOCK="$MULTI/requirements-awq-training.in"

if [[ -e "$ENV_DIR" ]]; then
  echo "Refusing to replace existing AWQ environment: $ENV_DIR" >&2
  exit 2
fi
mkdir -p "$PIP_CACHE" "$TMP_DIR" "$MULTI/.cache/hf_home"
export PIP_CACHE_DIR="$PIP_CACHE"
export TMPDIR="$TMP_DIR"
export HF_HOME="$MULTI/.cache/hf_home"
export HF_HUB_CACHE="$HF_HOME/hub"
export XDG_CACHE_HOME="$MULTI/.cache/xdg"
export TORCH_HOME="$MULTI/.cache/torch"

PYTHON_BIN="${EA_ATPO_AWQ_PYTHON_BIN:-python3}"
"$PYTHON_BIN" -m venv "$ENV_DIR"
PY="$ENV_DIR/bin/python"
"$PY" -m pip install --upgrade pip
# GPTQModel 7.12.0 requires Torch >=2.15 and Transformers >=5.22.2 on this
# Python version. This host advertises CUDA 13.2 and the official cu132 index
# provides torch 2.15.0+cu132. The resolved versions are frozen below.
"$PY" -m pip install --index-url https://download.pytorch.org/whl/cu132 "torch==2.15.0"
"$PY" -m pip install --index-url https://pypi.org/simple -r "$LOCK"
"$PY" -m pip check
"$PY" -m pip freeze > "$MULTI/environment.lock.txt"
"$PY" - <<'PY'
import importlib.metadata as m, json, torch
names = ("torch", "transformers", "peft", "accelerate", "gptqmodel", "optimum", "safetensors")
result = {name: m.version(name) for name in names}
result["torch_cuda"] = torch.version.cuda
result["cuda_available"] = torch.cuda.is_available()
print(json.dumps(result, indent=2))
PY
echo "AWQ LoRA environment installed under $ENV_DIR. No model/GPU test is implied by installation."
