#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$SCRIPT_DIR/outputs/qwen32_awq_complementarity_20261117_gpu7"
MODEL_DIR="/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/experiments/real_case_pilot_20261103/assets/models/Qwen3-8B"
VENV="/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/.venv-real-pilot"
CUDA_TOOLKIT="$VENV/lib/python3.12/site-packages/nvidia/cu13"
GPU=3 PORT=30147 ALIAS="qwen3-8b-complementarity-base"
[[ -x "$VENV/bin/vllm" && -f "$MODEL_DIR/config.json" ]] || { echo "Pinned vLLM or Qwen3-8B files missing." >&2; exit 2; }
if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then echo "Port $PORT is occupied; refusing to replace." >&2; exit 3; fi
FREE_MIB="$(nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
UTIL="$(nvidia-smi --id="$GPU" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')"
if (( FREE_MIB < 31000 || UTIL > 35 )); then echo "GPU $GPU fails gate: ${FREE_MIB} MiB free, ${UTIL}% busy." >&2; exit 4; fi
mkdir -p "$OUT_DIR"/{base_xdg,base_torch,base_triton,base_cuda,base_vllm,base_tmp}
export HF_HOME="$OUT_DIR/base_hf_home" HF_HUB_CACHE="$OUT_DIR/base_hf_home/hub" HF_HUB_DISABLE_TELEMETRY=1
export XDG_CACHE_HOME="$OUT_DIR/base_xdg" TORCH_HOME="$OUT_DIR/base_torch" TRITON_CACHE_DIR="$OUT_DIR/base_triton" CUDA_CACHE_PATH="$OUT_DIR/base_cuda"
export VLLM_CACHE_ROOT="$OUT_DIR/base_vllm" TMPDIR="$OUT_DIR/base_tmp" CUDA_VISIBLE_DEVICES="$GPU" VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_HOME="$CUDA_TOOLKIT" PATH="$VENV/bin:$CUDA_TOOLKIT/bin:$PATH"
"$VENV/bin/vllm" serve "$MODEL_DIR" --host 127.0.0.1 --port "$PORT" --served-model-name "$ALIAS" --dtype bfloat16 --max-model-len 4096 --max-num-seqs 1 --gpu-memory-utilization 0.14 >"$OUT_DIR/base_service.log" 2>&1 </dev/null &
PID=$!; printf '%s\n' "$PID" > "$OUT_DIR/base_service.pid"
for _ in $(seq 1 300); do
  if curl -fsS --max-time 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    "$VENV/bin/python" - "$OUT_DIR/base_service_manifest.json" "$GPU" "$PORT" "$FREE_MIB" "$ALIAS" <<'PY'
import importlib.metadata,json,sys
from datetime import datetime,timezone
from pathlib import Path
out,gpu,port,free,alias=sys.argv[1:]
obj={'started_at_utc':datetime.now(timezone.utc).isoformat(timespec='seconds'),'framework':'vLLM','framework_version':importlib.metadata.version('vllm'),'model':'Qwen/Qwen3-8B','revision':'b968826d9c46dd6066d109eabc6255188de91218','physical_gpu':int(gpu),'port':int(port),'served_model_name':alias,'free_mib_before_start':int(free),'max_model_len':4096,'max_num_seqs':1,'gpu_memory_utilization':0.14}
Path(out).write_text(json.dumps(obj,indent=2)+'\n'); print(json.dumps(obj,indent=2))
PY
    exit 0
  fi
  if ! kill -0 "$PID" 2>/dev/null; then echo "vLLM exited; see $OUT_DIR/base_service.log" >&2; tail -n 80 "$OUT_DIR/base_service.log" >&2; exit 5; fi
  sleep 2
done
echo "vLLM startup timed out; see $OUT_DIR/base_service.log" >&2
tail -n 80 "$OUT_DIR/base_service.log" >&2
exit 6
