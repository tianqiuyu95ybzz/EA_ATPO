#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
MODEL_DIR="/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/experiments/real_case_pilot_20261103/assets/models/Qwen3-32B-AWQ"
VENV="/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/.venv-real-pilot"
CUDA_TOOLKIT="$VENV/lib/python3.12/site-packages/nvidia/cu13"
GPU="${EA_ATPO_L2D_GPU:-6}"
PORT="${EA_ATPO_L2D_PORT:-30145}"
OUT_DIR="$SCRIPT_DIR/outputs/l2d_expert_screen_20261118_gpu6"
SHORT_TMP_LINK="/tmp/eaatpo_l2d_${GPU}_${PORT}"

[[ -x "$VENV/bin/vllm" ]] || { echo "Missing pinned vLLM executable: $VENV/bin/vllm" >&2; exit 2; }
[[ -f "$MODEL_DIR/config.json" && -f "$MODEL_DIR/model.safetensors.index.json" ]] || { echo "Missing local AWQ model files." >&2; exit 3; }
"$VENV/bin/python" - "$MODEL_DIR/range_download_status.json" "$MODEL_DIR/model.safetensors.index.json" <<'PY'
import json, sys
from pathlib import Path
status_path, index_path = map(Path, sys.argv[1:])
status = json.loads(status_path.read_text())
index = json.loads(index_path.read_text())
required = set(index["weight_map"].values())
if status.get("status") != "all_shards_verified" or status.get("revision") != "0499c3ac83fdef8810b907a23894ba91e95eddd8":
    raise SystemExit("Pinned Qwen3-32B-AWQ snapshot is not fully verified.")
if not required.issubset(set(status.get("completed_shards", []))):
    raise SystemExit("One or more indexed weight shards are not marked verified.")
if not all((index_path.parent / name).is_file() for name in required):
    raise SystemExit("One or more indexed weight shards are missing.")
PY
if curl --silent --fail --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
  echo "Port $PORT already serves a model; refusing to replace it." >&2
  exit 4
fi
FREE_MIB="$(nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
UTIL="$(nvidia-smi --id="$GPU" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')"
if (( FREE_MIB < 45000 || UTIL > 30 )); then
  echo "GPU $GPU fails start gate: ${FREE_MIB} MiB free, ${UTIL}% busy." >&2
  exit 5
fi

mkdir -p "$OUT_DIR"/{xdg,torch,triton,cuda,vllm,tmp,flashinfer}
if [[ -L "$SHORT_TMP_LINK" ]]; then
  [[ "$(readlink -f "$SHORT_TMP_LINK")" == "$(readlink -f "$OUT_DIR/tmp")" ]] || { echo "Short TMPDIR link points elsewhere: $SHORT_TMP_LINK" >&2; exit 8; }
elif [[ -e "$SHORT_TMP_LINK" ]]; then
  echo "Short TMPDIR path already exists and is not our symlink: $SHORT_TMP_LINK" >&2
  exit 8
else
  ln -s "$OUT_DIR/tmp" "$SHORT_TMP_LINK"
fi
export HF_HOME="$OUT_DIR/hf_home"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_DISABLE_TELEMETRY=1
export XDG_CACHE_HOME="$OUT_DIR/xdg"
export TORCH_HOME="$OUT_DIR/torch"
export TRITON_CACHE_DIR="$OUT_DIR/triton"
export CUDA_CACHE_PATH="$OUT_DIR/cuda"
export VLLM_CACHE_ROOT="$OUT_DIR/vllm"
export TMPDIR="$SHORT_TMP_LINK"
export FLASHINFER_WORKSPACE_BASE="$OUT_DIR/flashinfer"
export PATH="$VENV/bin:$CUDA_TOOLKIT/bin:$PATH"
export CUDA_HOME="$CUDA_TOOLKIT"
export VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_VISIBLE_DEVICES="$GPU"

"$VENV/bin/vllm" serve "$MODEL_DIR" \
  --host 127.0.0.1 --port "$PORT" --served-model-name qwen3-32b-awq-l2d \
  --dtype auto --quantization awq --max-model-len 4096 --max-num-seqs 1 \
  --attention-backend TRITON_ATTN --gpu-memory-utilization 0.25 \
  >"$OUT_DIR/model_service.log" 2>&1 </dev/null &
SERVICE_PID=$!
printf '%s\n' "$SERVICE_PID" > "$OUT_DIR/model_service.pid"

for _ in $(seq 1 600); do
  if curl --silent --fail --max-time 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    "$VENV/bin/python" - "$OUT_DIR/model_service_manifest.json" "$GPU" "$PORT" "$FREE_MIB" <<'PY'
import importlib.metadata, json, sys
from datetime import datetime, timezone
from pathlib import Path
out, gpu, port, free = sys.argv[1:]
record = {
    "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "framework": "vLLM", "framework_version": importlib.metadata.version("vllm"),
    "model": "Qwen/Qwen3-32B-AWQ", "revision": "0499c3ac83fdef8810b907a23894ba91e95eddd8",
    "quantization": "AWQ", "physical_gpu": int(gpu), "port": int(port),
    "free_mib_before_start": int(free), "max_model_len": 4096, "max_num_seqs": 1,
    "gpu_memory_utilization": 0.25, "attention_backend": "TRITON_ATTN",
}
Path(out).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print(json.dumps(record, indent=2))
PY
    exit 0
  fi
  if ! kill -0 "$SERVICE_PID" 2>/dev/null; then
    echo "vLLM exited during startup; see $OUT_DIR/model_service.log" >&2
    tail -n 100 "$OUT_DIR/model_service.log" >&2
    exit 6
  fi
  sleep 2
done
echo "Timed out waiting for vLLM; see $OUT_DIR/model_service.log" >&2
tail -n 100 "$OUT_DIR/model_service.log" >&2
exit 7
