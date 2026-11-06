#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PILOT_DIR="$PROJECT_DIR/experiments/real_case_pilot_20261103"
MODEL_DIR="$PILOT_DIR/assets/models/Qwen3-8B"
MODEL_ALIAS="qwen3-8b-router"
PORT="${EA_ATPO_ROUTER_PORT:-30131}"
GPU="${EA_ATPO_ROUTER_GPU:-2}"
OUT_DIR="$SCRIPT_DIR/outputs"
VENV="$PROJECT_DIR/.venv-real-pilot"
PYTHON="$VENV/bin/python"
VLLM="$VENV/bin/vllm"
CUDA_TOOLKIT="$VENV/lib/python3.12/site-packages/nvidia/cu13"
FLASHINFER_BASE="$PILOT_DIR/cache/flashinfer_workspace"

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "Pinned Qwen3-8B model files are unavailable: $MODEL_DIR" >&2
  exit 2
fi
if curl --silent --fail "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
  echo "Port $PORT is already serving; reusing the existing API." >&2
  exit 0
fi
free_mib="$(nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
if (( free_mib < 35000 )); then
  echo "GPU $GPU currently has $free_mib MiB free; need at least 35000 MiB." >&2
  exit 3
fi
mkdir -p "$OUT_DIR" "$SCRIPT_DIR/cache/xdg" "$SCRIPT_DIR/cache/triton" "$SCRIPT_DIR/cache/torch" "$SCRIPT_DIR/cache/cuda" "$SCRIPT_DIR/cache/vllm" "$FLASHINFER_BASE" "$PROJECT_DIR/.tmp"
export HF_HOME="$PILOT_DIR/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_DISABLE_TELEMETRY=1
export XDG_CACHE_HOME="$SCRIPT_DIR/cache/xdg"
export TORCH_HOME="$SCRIPT_DIR/cache/torch"
export TRITON_CACHE_DIR="$SCRIPT_DIR/cache/triton"
export CUDA_CACHE_PATH="$SCRIPT_DIR/cache/cuda"
export VLLM_CACHE_ROOT="$SCRIPT_DIR/cache/vllm"
export TMPDIR="$PROJECT_DIR/.tmp"
export PATH="$VENV/bin:$CUDA_TOOLKIT/bin:$PATH"
export CUDA_HOME="$CUDA_TOOLKIT"
export FLASHINFER_WORKSPACE_BASE="$FLASHINFER_BASE"
export VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_VISIBLE_DEVICES="$GPU"

nohup setsid "$VLLM" serve "$MODEL_DIR" \
  --host 127.0.0.1 --port "$PORT" --served-model-name "$MODEL_ALIAS" \
  --dtype bfloat16 --max-model-len 8192 --max-num-seqs 2 \
  --attention-backend TRITON_ATTN --gpu-memory-utilization 0.20 \
  >"$OUT_DIR/model_service.log" 2>&1 &
echo "$!" > "$OUT_DIR/model_service.pid"

for _ in $(seq 1 360); do
  if curl --silent --fail "http://127.0.0.1:$PORT/v1/models" >/dev/null; then
    "$PYTHON" - "$SCRIPT_DIR" "$GPU" "$free_mib" "$PORT" <<'PY'
import importlib.metadata, json, sys
from datetime import datetime, timezone
from pathlib import Path
root=Path(sys.argv[1])
record={"started_at_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),"framework":"vllm","vllm_version":importlib.metadata.version("vllm"),"repo_id":"Qwen/Qwen3-8B","revision":"b968826d9c46dd6066d109eabc6255188de91218","model_path":str(root.parent/"real_case_pilot_20261103/assets/models/Qwen3-8B"),"served_model_name":"qwen3-8b-router","gpu_physical_index":int(sys.argv[2]),"gpu_free_mib_before_start":int(sys.argv[3]),"port":int(sys.argv[4]),"dtype":"bfloat16","max_model_len":8192,"max_num_seqs":2,"gpu_memory_utilization":0.20,"attention_backend":"TRITON_ATTN","enable_thinking":False}
(root/"outputs/model_service_manifest.json").write_text(json.dumps(record,indent=2),encoding="utf-8")
print(json.dumps(record,indent=2))
PY
    exit 0
  fi
  if ! kill -0 "$(cat "$OUT_DIR/model_service.pid")" 2>/dev/null; then
    echo "vLLM exited during startup; see $OUT_DIR/model_service.log" >&2
    tail -n 120 "$OUT_DIR/model_service.log" >&2
    exit 4
  fi
  sleep 2
done
echo "vLLM did not become ready within 12 minutes; see $OUT_DIR/model_service.log" >&2
tail -n 120 "$OUT_DIR/model_service.log" >&2
exit 5
