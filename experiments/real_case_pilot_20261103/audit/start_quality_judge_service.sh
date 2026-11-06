#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PILOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_DIR="$(cd "$PILOT_DIR/../.." && pwd)"
OUT_DIR="$SCRIPT_DIR/outputs"
MODEL_DIR="$PILOT_DIR/assets/models/Qwen3-8B"
PYTHON="$PROJECT_DIR/.venv-real-pilot/bin/python"
VLLM="$PROJECT_DIR/.venv-real-pilot/bin/vllm"
GPU=6
PORT=30131
ALIAS=qwen3-8b-audit-judge
FREE_MIB="$(nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
if (( FREE_MIB < 35000 )); then
  echo "GPU $GPU has only ${FREE_MIB} MiB free; need 35000 MiB." >&2
  exit 2
fi
if curl --silent --fail "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
  echo "Port $PORT is already serving; refusing to overwrite another process." >&2
  exit 3
fi

CUDA_TOOLKIT="$PROJECT_DIR/.venv-real-pilot/lib/python3.12/site-packages/nvidia/cu13"
mkdir -p "$OUT_DIR" "$SCRIPT_DIR/cache/flashinfer_workspace" "$SCRIPT_DIR/cache/xdg" \
  "$SCRIPT_DIR/cache/triton" "$SCRIPT_DIR/cache/torch" "$PROJECT_DIR/.tmp"
export HF_HOME="$PILOT_DIR/cache/huggingface" HF_HUB_CACHE="$PILOT_DIR/cache/huggingface/hub"
export HF_HUB_DISABLE_TELEMETRY=1 XDG_CACHE_HOME="$SCRIPT_DIR/cache/xdg"
export TORCH_HOME="$SCRIPT_DIR/cache/torch" TRITON_CACHE_DIR="$SCRIPT_DIR/cache/triton"
export CUDA_CACHE_PATH="$SCRIPT_DIR/cache/cuda" VLLM_CACHE_ROOT="$SCRIPT_DIR/cache/vllm"
export TMPDIR="$PROJECT_DIR/.tmp" PATH="$PROJECT_DIR/.venv-real-pilot/bin:$CUDA_TOOLKIT/bin:$PATH"
export CUDA_HOME="$CUDA_TOOLKIT" FLASHINFER_WORKSPACE_BASE="$SCRIPT_DIR/cache/flashinfer_workspace"
export VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_VISIBLE_DEVICES="$GPU"

nohup setsid "$VLLM" serve "$MODEL_DIR" --host 127.0.0.1 --port "$PORT" \
  --served-model-name "$ALIAS" --dtype bfloat16 --max-model-len 8192 --max-num-seqs 1 \
  --attention-backend TRITON_ATTN --gpu-memory-utilization 0.30 \
  >"$OUT_DIR/consult_quality_judge_service.log" 2>&1 &
echo "$!" > "$OUT_DIR/consult_quality_judge_service.pid"

for _ in $(seq 1 360); do
  if curl --silent --fail "http://127.0.0.1:$PORT/v1/models" >/dev/null; then
    "$PYTHON" - "$SCRIPT_DIR" "$FREE_MIB" <<'PY'
import importlib.metadata, json, sys
from datetime import datetime, timezone
from pathlib import Path
audit=Path(sys.argv[1]); out=audit/"outputs"
m={"started_at_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),"purpose":"Post-hoc exploratory CONSULT quality annotations; not physician adjudication.","framework":"vllm","vllm_version":importlib.metadata.version("vllm"),"model_repo_id":"Qwen/Qwen3-8B","model_revision":"b968826d9c46dd6066d109eabc6255188de91218","served_model_name":"qwen3-8b-audit-judge","gpu_physical_index":6,"gpu_free_mib_before_start":int(sys.argv[2]),"port":30131,"dtype":"bfloat16","max_model_len":8192,"max_num_seqs":1,"gpu_memory_utilization":0.30,"attention_backend":"TRITON_ATTN"}
(out/"consult_quality_judge_service_manifest.json").write_text(json.dumps(m,indent=2),encoding="utf-8")
print(json.dumps(m,indent=2))
PY
    exit 0
  fi
  sleep 2
done
echo "Timed out waiting for quality-judge service; inspect consult_quality_judge_service.log." >&2
exit 4
