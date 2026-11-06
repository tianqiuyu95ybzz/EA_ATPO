#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PILOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_DIR="$(cd "$PILOT_DIR/../.." && pwd)"
OUT_DIR="$SCRIPT_DIR/outputs"
PYTHON="$PROJECT_DIR/.venv-real-pilot/bin/python"
VLLM="$PROJECT_DIR/.venv-real-pilot/bin/vllm"
CUDA_TOOLKIT="$PROJECT_DIR/.venv-real-pilot/lib/python3.12/site-packages/nvidia/cu13"
FLASHINFER_BASE="$SCRIPT_DIR/cache/flashinfer_workspace"

BASE_MODEL="$PILOT_DIR/assets/models/Qwen3-8B"
EXPERT_MODEL="$PILOT_DIR/assets/models/Qwen3-32B"
BASE_GPU="${BASE_GPU:-1}"
EXPERT_GPU="${EXPERT_GPU:-6}"
BASE_PORT=30131
EXPERT_PORT=30132
mkdir -p "$OUT_DIR" "$SCRIPT_DIR/cache/xdg" "$SCRIPT_DIR/cache/triton" "$SCRIPT_DIR/cache/torch" "$FLASHINFER_BASE" "$PROJECT_DIR/.tmp"
if [[ ! -f "$BASE_MODEL/config.json" || ! -f "$EXPERT_MODEL/config.json" ]]; then
  echo "Both pinned model directories must exist before starting audit services." >&2
  exit 2
fi
for port in "$BASE_PORT" "$EXPERT_PORT"; do
  if curl --silent --fail "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
    echo "Port $port is already serving; refusing to overwrite another process." >&2
    exit 3
  fi
done

base_free="$(nvidia-smi --id="$BASE_GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
expert_free="$(nvidia-smi --id="$EXPERT_GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
if (( base_free < 35000 || expert_free < 80000 )); then
  echo "Insufficient current free memory: GPU $BASE_GPU=${base_free} MiB (need 35000), GPU $EXPERT_GPU=${expert_free} MiB (need 80000)." >&2
  exit 4
fi

export HF_HOME="$PILOT_DIR/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_DISABLE_TELEMETRY=1
export XDG_CACHE_HOME="$SCRIPT_DIR/cache/xdg"
export TORCH_HOME="$SCRIPT_DIR/cache/torch"
export TRITON_CACHE_DIR="$SCRIPT_DIR/cache/triton"
export CUDA_CACHE_PATH="$SCRIPT_DIR/cache/cuda"
export VLLM_CACHE_ROOT="$SCRIPT_DIR/cache/vllm"
export TMPDIR="$PROJECT_DIR/.tmp"
export PATH="$PROJECT_DIR/.venv-real-pilot/bin:$CUDA_TOOLKIT/bin:$PATH"
export CUDA_HOME="$CUDA_TOOLKIT"
export FLASHINFER_WORKSPACE_BASE="$FLASHINFER_BASE"
export VLLM_USE_FLASHINFER_SAMPLER=0

CUDA_VISIBLE_DEVICES="$BASE_GPU" nohup setsid "$VLLM" serve "$BASE_MODEL" \
  --host 127.0.0.1 --port "$BASE_PORT" --served-model-name qwen3-8b-audit \
  --dtype bfloat16 --max-model-len 8192 --max-num-seqs 1 \
  --attention-backend TRITON_ATTN --gpu-memory-utilization 0.20 \
  >"$OUT_DIR/base_service.log" 2>&1 &
echo "$!" > "$OUT_DIR/base_service.pid"

CUDA_VISIBLE_DEVICES="$EXPERT_GPU" nohup setsid "$VLLM" serve "$EXPERT_MODEL" \
  --host 127.0.0.1 --port "$EXPERT_PORT" --served-model-name qwen3-32b-candidate \
  --dtype bfloat16 --max-model-len 8192 --max-num-seqs 1 \
  --attention-backend TRITON_ATTN --gpu-memory-utilization 0.45 \
  --disable-prefix-caching \
  >"$OUT_DIR/candidate_service.log" 2>&1 &
echo "$!" > "$OUT_DIR/candidate_service.pid"

for _ in $(seq 1 900); do
  base_ok=false
  expert_ok=false
  curl --silent --fail "http://127.0.0.1:$BASE_PORT/v1/models" >/dev/null && base_ok=true || true
  curl --silent --fail "http://127.0.0.1:$EXPERT_PORT/v1/models" >/dev/null && expert_ok=true || true
  if [[ "$base_ok" == true && "$expert_ok" == true ]]; then
    "$PYTHON" - "$SCRIPT_DIR" "$base_free" "$expert_free" <<'PY'
import importlib.metadata, json, sys
from datetime import datetime, timezone
from pathlib import Path
out=Path(sys.argv[1]) / "outputs"
record={
  "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
  "framework": "vllm", "vllm_version": importlib.metadata.version("vllm"),
  "models": [
    {"role":"base","repo_id":"Qwen/Qwen3-8B","revision":"b968826d9c46dd6066d109eabc6255188de91218","alias":"qwen3-8b-audit","gpu_physical_index":1,"free_mib_before_start":int(sys.argv[2]),"port":30131,"dtype":"bfloat16","max_model_len":8192,"max_num_seqs":1,"gpu_memory_utilization":0.20,"attention_backend":"TRITON_ATTN"},
    {"role":"candidate_expert","repo_id":"Qwen/Qwen3-32B","revision":"9216db5781bf21249d130ec9da846c4624c16137","alias":"qwen3-32b-candidate","gpu_physical_index":2,"free_mib_before_start":int(sys.argv[3]),"port":30132,"dtype":"bfloat16","max_model_len":8192,"max_num_seqs":1,"gpu_memory_utilization":0.45,"attention_backend":"TRITON_ATTN"}
  ],
  "purpose":"Independent held-out direct-answer accuracy comparison; no Pilot artifacts overwritten."
}
(out/"service_manifest.json").write_text(json.dumps(record,indent=2),encoding="utf-8")
print(json.dumps(record,indent=2))
PY
    exit 0
  fi
  sleep 2
done
echo "Timed out waiting for audit model services; inspect audit/outputs/*_service.log." >&2
exit 5
