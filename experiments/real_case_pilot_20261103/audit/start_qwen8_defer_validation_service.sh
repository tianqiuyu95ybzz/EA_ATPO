#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$SCRIPT_DIR/outputs/qwen32_defer_validation_20261119_attempt01/services/attempt03/base"
VLLM_CACHE_DIR="$SCRIPT_DIR/outputs/qwen32_defer_validation_20261119_attempt01/services/attempt02/base/vllm"
MODEL_DIR="/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/experiments/real_case_pilot_20261103/assets/models/Qwen3-8B"
VENV="/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/.venv-real-pilot"
CUDA_TOOLKIT="$VENV/lib/python3.12/site-packages/nvidia/cu13"
GPU=2 PORT=30157 ALIAS="qwen3-8b-defer-validation-base"
SHORT_TMP="/pfs/pfs-spMCeu/image_team/workspace/zjr/.ea_atpo_defer_base_tmp03"
if [[ -e "$OUT_DIR/service.log" || -e "$OUT_DIR/service_manifest.json" || -e "$OUT_DIR/service.pid" ]]; then
  echo "Existing Base service artifacts found; refusing to overwrite: $OUT_DIR" >&2
  exit 8
fi
[[ -x "$VENV/bin/vllm" && -f "$MODEL_DIR/config.json" ]] || { echo "Pinned vLLM or Qwen3-8B files missing." >&2; exit 2; }
if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then echo "Port $PORT is occupied; refusing to replace." >&2; exit 3; fi
FREE_MIB="$(nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
UTIL="$(nvidia-smi --id="$GPU" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')"
if (( FREE_MIB < 31000 || UTIL > 35 )); then echo "GPU $GPU fails gate: ${FREE_MIB} MiB free, ${UTIL}% busy." >&2; exit 4; fi
mkdir -p "$OUT_DIR"/{xdg,torch,triton,cuda,vllm,tmp,flashinfer}
if [[ -e "$SHORT_TMP" || -L "$SHORT_TMP" ]]; then
  [[ -L "$SHORT_TMP" && "$(readlink "$SHORT_TMP")" == "$OUT_DIR/tmp" ]] || { echo "Short TMP path is already owned by another target: $SHORT_TMP" >&2; exit 7; }
else
  ln -s "$OUT_DIR/tmp" "$SHORT_TMP"
fi
export HF_HOME="$OUT_DIR/hf_home" HF_HUB_CACHE="$OUT_DIR/hf_home/hub" HF_HUB_DISABLE_TELEMETRY=1
export XDG_CACHE_HOME="$OUT_DIR/xdg" TORCH_HOME="$OUT_DIR/torch" TRITON_CACHE_DIR="$SCRIPT_DIR/outputs/qwen32_defer_validation_20261119_attempt01/services/attempt02/base/triton" CUDA_CACHE_PATH="$OUT_DIR/cuda"
export VLLM_CACHE_ROOT="$VLLM_CACHE_DIR" TMPDIR="$SHORT_TMP" FLASHINFER_WORKSPACE_BASE="$OUT_DIR/flashinfer" CUDA_VISIBLE_DEVICES="$GPU" VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_HOME="$CUDA_TOOLKIT" PATH="$VENV/bin:$CUDA_TOOLKIT/bin:$PATH"
nohup "$VENV/bin/vllm" serve "$MODEL_DIR" --host 127.0.0.1 --port "$PORT" --served-model-name "$ALIAS" --dtype bfloat16 --max-model-len 4096 --max-num-seqs 1 --attention-backend TRITON_ATTN --disable-flashinfer-autotune --gpu-memory-utilization 0.12 >"$OUT_DIR/service.log" 2>&1 </dev/null &
PID=$!; printf '%s\n' "$PID" > "$OUT_DIR/service.pid"
for _ in $(seq 1 300); do
  if curl -fsS --max-time 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    "$VENV/bin/python" - "$OUT_DIR/service_manifest.json" "$GPU" "$PORT" "$FREE_MIB" "$ALIAS" <<'PY'
import importlib.metadata,json,sys
from datetime import datetime,timezone
from pathlib import Path
out,gpu,port,free,alias=sys.argv[1:]
obj={'started_at_utc':datetime.now(timezone.utc).isoformat(timespec='seconds'),'framework':'vLLM','framework_version':importlib.metadata.version('vllm'),'model':'Qwen/Qwen3-8B','revision':'b968826d9c46dd6066d109eabc6255188de91218','physical_gpu':int(gpu),'port':int(port),'served_model_name':alias,'free_mib_before_start':int(free),'max_model_len':4096,'max_num_seqs':1,'gpu_memory_utilization':0.12}
Path(out).write_text(json.dumps(obj,indent=2)+'\n'); print(json.dumps(obj,indent=2))
PY
    exit 0
  fi
  if ! kill -0 "$PID" 2>/dev/null; then echo "vLLM exited; see $OUT_DIR/service.log" >&2; tail -n 80 "$OUT_DIR/service.log" >&2; exit 5; fi
  sleep 2
done
echo "vLLM startup timed out; see $OUT_DIR/service.log" >&2
tail -n 80 "$OUT_DIR/service.log" >&2
exit 6
