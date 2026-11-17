#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$SCRIPT_DIR/outputs/qwen32_awq_complementarity_20261117_gpu7"
MODEL_DIR="/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/experiments/real_case_pilot_20261103/assets/models/Qwen3-32B-AWQ"
VENV="/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/.venv-real-pilot"
CUDA_TOOLKIT="$VENV/lib/python3.12/site-packages/nvidia/cu13"
GPU=7 PORT=30146 ALIAS="qwen3-32b-awq-complementarity"
[[ -x "$VENV/bin/vllm" ]] || { echo "Pinned vLLM executable missing." >&2; exit 2; }
[[ -f "$MODEL_DIR/model_source_lock.json" && -f "$MODEL_DIR/range_download_status.json" ]] || { echo "Pinned model lock/status missing." >&2; exit 3; }
"$VENV/bin/python" - "$MODEL_DIR" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); lock=json.loads((p/'model_source_lock.json').read_text()); status=json.loads((p/'range_download_status.json').read_text()); idx=json.loads((p/'model.safetensors.index.json').read_text()); required=set(idx['weight_map'].values())
assert lock.get('huggingface_revision')=='0499c3ac83fdef8810b907a23894ba91e95eddd8'
assert status.get('status')=='all_shards_verified' and status.get('revision')=='0499c3ac83fdef8810b907a23894ba91e95eddd8'
assert required.issubset(set(status.get('completed_shards',[]))) and all((p/n).is_file() for n in required)
print(f'Pinned AWQ snapshot verified: {len(required)} shards')
PY
if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then echo "Port $PORT is occupied; refusing to replace." >&2; exit 4; fi
FREE_MIB="$(nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
UTIL="$(nvidia-smi --id="$GPU" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')"
if (( FREE_MIB < 70000 || UTIL > 30 )); then echo "GPU $GPU fails gate: ${FREE_MIB} MiB free, ${UTIL}% busy." >&2; exit 5; fi
mkdir -p "$OUT_DIR"/{xdg,torch,triton,cuda,vllm,tmp,flashinfer}
export HF_HOME="$OUT_DIR/hf_home" HF_HUB_CACHE="$OUT_DIR/hf_home/hub" HF_HUB_DISABLE_TELEMETRY=1
export XDG_CACHE_HOME="$OUT_DIR/xdg" TORCH_HOME="$OUT_DIR/torch" TRITON_CACHE_DIR="$OUT_DIR/triton" CUDA_CACHE_PATH="$OUT_DIR/cuda"
export VLLM_CACHE_ROOT="$OUT_DIR/vllm" TMPDIR="$OUT_DIR/tmp" FLASHINFER_WORKSPACE_BASE="$OUT_DIR/flashinfer" CUDA_VISIBLE_DEVICES="$GPU" VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_HOME="$CUDA_TOOLKIT" PATH="$VENV/bin:$CUDA_TOOLKIT/bin:$PATH"
"$VENV/bin/vllm" serve "$MODEL_DIR" --host 127.0.0.1 --port "$PORT" --served-model-name "$ALIAS" --dtype auto --quantization awq --max-model-len 4096 --max-num-seqs 1 --attention-backend TRITON_ATTN --gpu-memory-utilization 0.25 >"$OUT_DIR/model_service.log" 2>&1 </dev/null &
PID=$!; printf '%s\n' "$PID" > "$OUT_DIR/model_service.pid"
for _ in $(seq 1 600); do
  if curl -fsS --max-time 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    "$VENV/bin/python" - "$OUT_DIR/model_service_manifest.json" "$GPU" "$PORT" "$FREE_MIB" "$ALIAS" <<'PY'
import importlib.metadata,json,sys
from datetime import datetime,timezone
from pathlib import Path
out,gpu,port,free,alias=sys.argv[1:]
obj={'started_at_utc':datetime.now(timezone.utc).isoformat(timespec='seconds'),'framework':'vLLM','framework_version':importlib.metadata.version('vllm'),'model':'Qwen/Qwen3-32B-AWQ','revision':'0499c3ac83fdef8810b907a23894ba91e95eddd8','quantization':'AWQ','physical_gpu':int(gpu),'port':int(port),'served_model_name':alias,'free_mib_before_start':int(free),'max_model_len':4096,'max_num_seqs':1,'gpu_memory_utilization':0.25,'attention_backend':'TRITON_ATTN'}
Path(out).write_text(json.dumps(obj,indent=2)+'\n'); print(json.dumps(obj,indent=2))
PY
    exit 0
  fi
  if ! kill -0 "$PID" 2>/dev/null; then echo "vLLM exited; see $OUT_DIR/model_service.log" >&2; tail -n 80 "$OUT_DIR/model_service.log" >&2; exit 6; fi
  sleep 2
done
echo "vLLM startup timed out; see $OUT_DIR/model_service.log" >&2
tail -n 80 "$OUT_DIR/model_service.log" >&2
exit 7
