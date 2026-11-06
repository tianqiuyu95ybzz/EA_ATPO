#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PILOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_DIR="$(cd "$PILOT_DIR/../.." && pwd)"
PYTHON="$PROJECT_DIR/.venv-real-pilot/bin/python"
VLLM="$PROJECT_DIR/.venv-real-pilot/bin/vllm"
CUDA_TOOLKIT="$PROJECT_DIR/.venv-real-pilot/lib/python3.12/site-packages/nvidia/cu13"
MODEL_DIR="$PILOT_DIR/assets/models/Qwen3-8B"
OUT_DIR="$SCRIPT_DIR/outputs/advice_utilization"
GPU=1
PORT=30133
ALIAS="qwen3-8b-advice-base"

mkdir -p "$OUT_DIR" "$OUT_DIR/cache/xdg" "$OUT_DIR/cache/triton" "$OUT_DIR/cache/torch" "$OUT_DIR/cache/flashinfer"
if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "Pinned Qwen3-8B model directory is missing: $MODEL_DIR" >&2
  exit 2
fi
if curl --silent --fail "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
  echo "Port $PORT is already serving; refusing to overwrite another process." >&2
  exit 3
fi
free_mib="$(nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
if (( free_mib < 35000 )); then
  echo "GPU $GPU has only ${free_mib} MiB free; need at least 35000 MiB." >&2
  exit 4
fi

export HF_HOME="$PILOT_DIR/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_DISABLE_TELEMETRY=1
export XDG_CACHE_HOME="$OUT_DIR/cache/xdg"
export TORCH_HOME="$OUT_DIR/cache/torch"
export TRITON_CACHE_DIR="$OUT_DIR/cache/triton"
export CUDA_CACHE_PATH="$OUT_DIR/cache/cuda"
export VLLM_CACHE_ROOT="$OUT_DIR/cache/vllm"
export TMPDIR="$PROJECT_DIR/.tmp"
export PATH="$PROJECT_DIR/.venv-real-pilot/bin:$CUDA_TOOLKIT/bin:$PATH"
export CUDA_HOME="$CUDA_TOOLKIT"
export FLASHINFER_WORKSPACE_BASE="$OUT_DIR/cache/flashinfer"
export VLLM_USE_FLASHINFER_SAMPLER=0

CUDA_VISIBLE_DEVICES="$GPU" nohup setsid "$VLLM" serve "$MODEL_DIR" \
  --host 127.0.0.1 --port "$PORT" --served-model-name "$ALIAS" \
  --dtype bfloat16 --max-model-len 8192 --max-num-seqs 1 \
  --attention-backend TRITON_ATTN --gpu-memory-utilization 0.20 \
  --disable-prefix-caching \
  >"$OUT_DIR/base_service.log" 2>&1 &
echo "$!" > "$OUT_DIR/base_service.pid"

for _ in $(seq 1 300); do
  if curl --silent --fail "http://127.0.0.1:$PORT/v1/models" >/dev/null; then
    "$PYTHON" - "$OUT_DIR" "$GPU" "$free_mib" "$PORT" <<'PY'
import importlib.metadata, json, sys
from datetime import datetime, timezone
from pathlib import Path
out=Path(sys.argv[1])
manifest={
  "started_at_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),
  "framework":"vllm", "vllm_version":importlib.metadata.version("vllm"),
  "model_repo_id":"Qwen/Qwen3-8B", "model_revision":"b968826d9c46dd6066d109eabc6255188de91218",
  "local_path":"assets/models/Qwen3-8B", "served_model_name":"qwen3-8b-advice-base",
  "gpu_physical_index":int(sys.argv[2]), "gpu_free_mib_before_start":int(sys.argv[3]),
  "port":int(sys.argv[4]), "dtype":"bfloat16", "max_model_len":8192, "max_num_seqs":1,
  "gpu_memory_utilization":0.20, "attention_backend":"TRITON_ATTN", "prefix_caching":False,
  "purpose":"CONSULT post-advice decision comparison only; no training; separate output directory."
}
(out/"service_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
print(json.dumps(manifest,indent=2))
PY
    exit 0
  fi
  sleep 2
done
echo "Timed out waiting for $ALIAS on port $PORT; inspect $OUT_DIR/base_service.log." >&2
exit 5
