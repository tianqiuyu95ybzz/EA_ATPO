#!/usr/bin/env bash
set -euo pipefail

EA_ATPO_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EA_ATPO_PILOT_DIR="$EA_ATPO_PROJECT_DIR/experiments/real_case_pilot_20261103"
EA_ATPO_MODEL_DIR="$EA_ATPO_PILOT_DIR/assets/models/Qwen3-8B"
EA_ATPO_OUTPUT_DIR="$EA_ATPO_PILOT_DIR/outputs"
EA_ATPO_GPU_INDEX="1"
EA_ATPO_PORT="30000"
EA_ATPO_MODEL_ALIAS="qwen3-8b"
EA_ATPO_MIN_FREE_MIB="40000"
EA_ATPO_HF_HOME="$EA_ATPO_PILOT_DIR/cache/huggingface"
EA_ATPO_PYTHON="$EA_ATPO_PROJECT_DIR/.venv-real-pilot/bin/python"
EA_ATPO_VENV_BIN="$EA_ATPO_PROJECT_DIR/.venv-real-pilot/bin"
EA_ATPO_CUDA_TOOLKIT="$EA_ATPO_PROJECT_DIR/.venv-real-pilot/lib/python3.12/site-packages/nvidia/cu13"
EA_ATPO_FLASHINFER_BASE="$EA_ATPO_PILOT_DIR/cache/flashinfer_workspace"

if [[ ! -f "$EA_ATPO_MODEL_DIR/config.json" ]]; then
  echo "Model files missing; run experiments/real_case_pilot_20261103/download_model.sh first." >&2
  exit 2
fi
if [[ -f "$EA_ATPO_OUTPUT_DIR/model_server.pid" ]] && kill -0 "$(cat "$EA_ATPO_OUTPUT_DIR/model_server.pid")" 2>/dev/null; then
  echo "A model service is already running with PID $(cat "$EA_ATPO_OUTPUT_DIR/model_server.pid")." >&2
  exit 2
fi
EA_ATPO_FREE_MIB="$(nvidia-smi --id="$EA_ATPO_GPU_INDEX" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
if (( EA_ATPO_FREE_MIB < EA_ATPO_MIN_FREE_MIB )); then
  echo "GPU $EA_ATPO_GPU_INDEX has ${EA_ATPO_FREE_MIB} MiB free; need at least ${EA_ATPO_MIN_FREE_MIB} MiB. No service was started." >&2
  exit 3
fi

mkdir -p "$EA_ATPO_OUTPUT_DIR" "$EA_ATPO_HF_HOME" "$EA_ATPO_PROJECT_DIR/.tmp" \
  "$EA_ATPO_PILOT_DIR/cache/torch" "$EA_ATPO_PILOT_DIR/cache/triton" "$EA_ATPO_PILOT_DIR/cache/cuda" \
  "$EA_ATPO_PILOT_DIR/cache/datasets" "$EA_ATPO_PILOT_DIR/cache/xdg"
mkdir -p "$EA_ATPO_FLASHINFER_BASE"
export CUDA_VISIBLE_DEVICES="$EA_ATPO_GPU_INDEX"
export HF_HOME="$EA_ATPO_HF_HOME"
export HF_HUB_CACHE="$EA_ATPO_HF_HOME/hub"
export HF_DATASETS_CACHE="$EA_ATPO_PILOT_DIR/cache/datasets"
export XDG_CACHE_HOME="$EA_ATPO_PILOT_DIR/cache/xdg"
export HF_HUB_DISABLE_TELEMETRY=1
export TORCH_HOME="$EA_ATPO_PILOT_DIR/cache/torch"
export TRITON_CACHE_DIR="$EA_ATPO_PILOT_DIR/cache/triton"
export CUDA_CACHE_PATH="$EA_ATPO_PILOT_DIR/cache/cuda"
export VLLM_CACHE_ROOT="$EA_ATPO_PILOT_DIR/cache/vllm"
export TMPDIR="$EA_ATPO_PROJECT_DIR/.tmp"
# vLLM/FlashInfer compile CUDA kernels at startup. The isolated runtime contains
# ninja and the CUDA compiler, so make both discoverable without system installs.
export PATH="$EA_ATPO_VENV_BIN:$EA_ATPO_CUDA_TOOLKIT/bin:$PATH"
export CUDA_HOME="$EA_ATPO_CUDA_TOOLKIT"
export FLASHINFER_WORKSPACE_BASE="$EA_ATPO_FLASHINFER_BASE"
# The bundled FlashInfer JIT headers and CUDA 13.4 compiler are incompatible on
# this host. Keep inference on vLLM's Triton attention and native sampler paths.
export VLLM_USE_FLASHINFER_SAMPLER=0

nohup setsid "$EA_ATPO_PROJECT_DIR/.venv-real-pilot/bin/vllm" serve "$EA_ATPO_MODEL_DIR" \
  --host 127.0.0.1 --port "$EA_ATPO_PORT" \
  --served-model-name "$EA_ATPO_MODEL_ALIAS" \
  --dtype bfloat16 --max-model-len 8192 --max-num-seqs 4 \
  --attention-backend TRITON_ATTN \
  --gpu-memory-utilization 0.20 \
  >"$EA_ATPO_OUTPUT_DIR/model_server.log" 2>&1 &
echo "$!" > "$EA_ATPO_OUTPUT_DIR/model_server.pid"

for _ in $(seq 1 360); do
  if curl --silent --fail "http://127.0.0.1:$EA_ATPO_PORT/v1/models" >/dev/null; then
    "$EA_ATPO_PYTHON" - "$EA_ATPO_PILOT_DIR" "$EA_ATPO_GPU_INDEX" "$EA_ATPO_PORT" "$EA_ATPO_FREE_MIB" <<'PY'
import importlib.metadata
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

pilot_dir = Path(sys.argv[1])
record = {
    "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "framework": "vllm",
    "vllm_version": importlib.metadata.version("vllm"),
    "model_repo_id": "Qwen/Qwen3-8B",
    "model_revision": "b968826d9c46dd6066d109eabc6255188de91218",
    "local_path": str(pilot_dir / "assets/models/Qwen3-8B"),
    "served_model_name": "qwen3-8b",
    "gpu_physical_index": int(sys.argv[2]),
    "gpu_free_mib_before_start": int(sys.argv[4]),
    "port": int(sys.argv[3]),
    "dtype": "bfloat16",
    "max_model_len": 8192,
    "max_num_seqs": 4,
    "gpu_memory_utilization": 0.20,
    "attention_backend": "TRITON_ATTN",
    "flashinfer_sampler_enabled": False,
    "flashinfer_workspace_base": str(pilot_dir / "cache/flashinfer_workspace"),
}
(pilot_dir / "outputs/model_service_manifest.json").write_text(
    json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(record, indent=2))
PY
    exit 0
  fi
  if ! kill -0 "$(cat "$EA_ATPO_OUTPUT_DIR/model_server.pid")" 2>/dev/null; then
    echo "vLLM exited during startup; see $EA_ATPO_OUTPUT_DIR/model_server.log" >&2
    tail -n 100 "$EA_ATPO_OUTPUT_DIR/model_server.log" >&2
    exit 4
  fi
  sleep 2
done

echo "vLLM did not become ready within 12 minutes; see $EA_ATPO_OUTPUT_DIR/model_server.log" >&2
tail -n 100 "$EA_ATPO_OUTPUT_DIR/model_server.log" >&2
exit 5
