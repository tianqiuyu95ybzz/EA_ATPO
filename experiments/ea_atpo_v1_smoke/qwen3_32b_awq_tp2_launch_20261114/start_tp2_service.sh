#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
EA_ATPO_ROOT="${EA_ATPO_ROOT:-$(cd "$ROOT/../EA_ATPO" && pwd)}"
MODEL_DIR="$EA_ATPO_ROOT/experiments/real_case_pilot_20261103/assets/models/Qwen3-32B-AWQ"
VENV="$EA_ATPO_ROOT/.venv-real-pilot"
VLLM="$VENV/bin/vllm"
CUDA_TOOLKIT="$VENV/lib/python3.12/site-packages/nvidia/cu13"
GPU_LIST="${EA_ATPO_TP2_GPUS:-4,6}"
PORT="${EA_ATPO_TP2_PORT:-30139}"
MIN_FREE_MIB="${EA_ATPO_TP2_MIN_FREE_MIB:-65000}"
GPU_MEMORY_UTIL="${EA_ATPO_TP2_GPU_MEMORY_UTILIZATION:-0.20}"
MODEL_ALIAS="qwen3-32b-awq-tp2-smoke"

if [[ ! "$GPU_LIST" =~ ^[0-7],[0-7]$ ]]; then
  echo "Set exactly two physical GPU indices, e.g. 4,6." >&2
  exit 2
fi
IFS=',' read -r GPU_A GPU_B <<< "$GPU_LIST"
if [[ "$GPU_A" == "$GPU_B" || "$GPU_A" == 1 || "$GPU_A" == 7 || "$GPU_B" == 1 || "$GPU_B" == 7 ]]; then
  echo "Use two distinct GPUs; GPU 1 and GPU 7 are excluded." >&2
  exit 2
fi
[[ -x "$VLLM" ]] || { echo "Missing vLLM environment: $VLLM" >&2; exit 3; }
[[ -f "$MODEL_DIR/config.json" && -f "$MODEL_DIR/model.safetensors.index.json" ]] || {
  echo "Missing pinned model files: $MODEL_DIR" >&2; exit 3;
}
"$VENV/bin/python" - "$MODEL_DIR/range_download_status.json" "$MODEL_DIR/model.safetensors.index.json" <<'PY'
import json, sys
from pathlib import Path
status = json.loads(Path(sys.argv[1]).read_text())
index = json.loads(Path(sys.argv[2]).read_text())
required = set(index["weight_map"].values())
if status.get("status") != "all_shards_verified":
    raise SystemExit("Pinned AWQ shard verification is incomplete")
if status.get("revision") != "0499c3ac83fdef8810b907a23894ba91e95eddd8":
    raise SystemExit("Pinned AWQ revision does not match the verified source")
if not required.issubset(set(status.get("completed_shards", []))):
    raise SystemExit("One or more indexed AWQ shards are not verified")
if not all((Path(sys.argv[1]).parent / name).is_file() for name in required):
    raise SystemExit("One or more indexed AWQ shard files are missing")
PY
if curl --silent --fail --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
  echo "Port $PORT is already serving; refusing to overwrite it." >&2
  exit 4
fi
for GPU in "$GPU_A" "$GPU_B"; do
  FREE_MIB="$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
  if (( FREE_MIB < MIN_FREE_MIB )); then
    echo "GPU $GPU has only ${FREE_MIB} MiB free; need ${MIN_FREE_MIB} MiB." >&2
    exit 5
  fi
done

mkdir -p "$SCRIPT_DIR/cache/xdg" "$SCRIPT_DIR/cache/torch" "$SCRIPT_DIR/cache/triton" \
  "$SCRIPT_DIR/cache/cuda" "$SCRIPT_DIR/cache/vllm" /tmp/eaatpo32b
export HF_HOME="$EA_ATPO_ROOT/experiments/real_case_pilot_20261103/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_DISABLE_TELEMETRY=1
export XDG_CACHE_HOME="$SCRIPT_DIR/cache/xdg"
export TORCH_HOME="$SCRIPT_DIR/cache/torch"
export TRITON_CACHE_DIR="$SCRIPT_DIR/cache/triton"
export CUDA_CACHE_PATH="$SCRIPT_DIR/cache/cuda"
export VLLM_CACHE_ROOT="$SCRIPT_DIR/cache/vllm"
export TMPDIR=/tmp/eaatpo32b
export PATH="$VENV/bin:$CUDA_TOOLKIT/bin:$PATH"
export CUDA_HOME="$CUDA_TOOLKIT"
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ALLREDUCE_USE_FLASHINFER=0
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export CUDA_VISIBLE_DEVICES="$GPU_LIST"

LOG_FILE="$SCRIPT_DIR/model_service_repro.log"
nohup setsid "$VLLM" serve "$MODEL_DIR" \
  --host 127.0.0.1 --port "$PORT" --served-model-name "$MODEL_ALIAS" \
  --dtype auto --quantization awq --tensor-parallel-size 2 \
  --max-model-len 4096 --max-num-seqs 1 --attention-backend TRITON_ATTN \
  --gpu-memory-utilization "$GPU_MEMORY_UTIL" --disable-custom-all-reduce \
  --compilation-config '{"pass_config":{"fuse_allreduce_rms":false}}' \
  >"$LOG_FILE" 2>&1 </dev/null &
SERVER_PID=$!
printf '%s\n' "$SERVER_PID" > "$SCRIPT_DIR/model_service.pid"

for _ in $(seq 1 600); do
  if curl --silent --fail --max-time 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    "$VENV/bin/python" - "$SCRIPT_DIR/model_service_manifest.json" "$GPU_LIST" "$PORT" "$GPU_MEMORY_UTIL" <<'PY'
import importlib.metadata, json, sys
from datetime import datetime, timezone
from pathlib import Path
out, gpus, port, utilization = sys.argv[1:]
record = {
    "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "framework": "vLLM",
    "framework_version": importlib.metadata.version("vllm"),
    "model": "Qwen/Qwen3-32B-AWQ",
    "revision": "0499c3ac83fdef8810b907a23894ba91e95eddd8",
    "precision": "AWQ",
    "visible_physical_gpus": [int(x) for x in gpus.split(",")],
    "tensor_parallel_size": 2,
    "port": int(port),
    "gpu_memory_utilization": float(utilization),
    "max_model_len": 4096,
    "max_num_seqs": 1,
    "attention_backend": "TRITON_ATTN",
    "flashinfer_allreduce": False,
    "fuse_allreduce_rms": False,
    "service_pid": int(Path(out).with_name("model_service.pid").read_text()),
}
Path(out).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print(json.dumps(record, indent=2))
PY
    exit 0
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "vLLM exited before API readiness; see $LOG_FILE" >&2
    tail -n 100 "$LOG_FILE" >&2
    exit 6
  fi
  sleep 2
done
echo "Timed out waiting for API readiness; see $LOG_FILE" >&2
tail -n 100 "$LOG_FILE" >&2
exit 7
