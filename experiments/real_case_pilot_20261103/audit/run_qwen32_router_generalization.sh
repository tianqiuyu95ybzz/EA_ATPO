#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$SCRIPT_DIR/outputs/qwen32_router_generalization_20261118_attempt04"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VENV="/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/.venv-real-pilot"
CUDA_TOOLKIT="$VENV/lib/python3.12/site-packages/nvidia/cu13"
BASE_MODEL="/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/experiments/real_case_pilot_20261103/assets/models/Qwen3-8B"
EXPERT_MODEL="/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/experiments/real_case_pilot_20261103/assets/models/Qwen3-32B-AWQ"
BASE_PORT=30257 EXPERT_PORT=30256
BASE_ALIAS="qwen3-8b-generalization-base" EXPERT_ALIAS="qwen3-32b-awq-generalization-expert"
EXPERT_GPU="" BASE_GPU="" EXPERT_PID="" BASE_PID="" MON_PID="" RESUME=0 SERVICE_ATTEMPT=""

usage() { echo "Usage: $0 --expert-gpu {1|2|5} --base-gpu {1|2|5} [--resume]" >&2; }
while (($#)); do
  case "$1" in
    --expert-gpu) EXPERT_GPU="$2"; shift 2 ;;
    --base-gpu) BASE_GPU="$2"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    *) usage; exit 2 ;;
  esac
done
[[ -n "$EXPERT_GPU" && -n "$BASE_GPU" && "$EXPERT_GPU" != "$BASE_GPU" ]] || { usage; exit 2; }
for gpu in "$EXPERT_GPU" "$BASE_GPU"; do
  case "$gpu" in 1|2|5) ;; *) echo "GPU $gpu is outside the currently approved pool {1,2,5}." >&2; exit 2 ;; esac
done
[[ -x "$VENV/bin/vllm" ]] || { echo "Pinned vLLM is missing: $VENV/bin/vllm" >&2; exit 3; }
[[ -f "$RUN_DIR/case_selection.json" ]] || { echo "Frozen Attempt04 case selection is missing; no service started." >&2; exit 3; }
for n in $(seq -w 1 99); do
  candidate="attempt$n"
  if [[ ! -e "$RUN_DIR/services/$candidate/base/service.log" && ! -e "$RUN_DIR/services/$candidate/expert/service.log" && ! -e "$RUN_DIR/gpu_usage_${candidate}.csv" ]]; then
    SERVICE_ATTEMPT="$candidate"
    break
  fi
done
[[ -n "$SERVICE_ATTEMPT" ]] || { echo "No unused service attempt directory remains." >&2; exit 3; }
[[ ! -e "$RUN_DIR/analysis/analysis_manifest.json" ]] || { echo "Attempt04 final test analysis already exists; refusing a second test run." >&2; exit 4; }
if (( ! RESUME )); then
  for marker in "$RUN_DIR/smoke_20/run_manifest.json" "$RUN_DIR/paired_rollouts/run_manifest.json" "$RUN_DIR/paired_rollouts/base_answer_trajectories.jsonl" "$RUN_DIR/paired_rollouts/qwen32_defer_trajectories.jsonl"; do
    [[ ! -e "$marker" ]] || { echo "Attempt04 partial or complete rollout exists; use --resume after auditing it: $marker" >&2; exit 4; }
  done
fi

verify_expert_snapshot() {
  "$VENV/bin/python" - "$EXPERT_MODEL" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); lock=json.loads((p/'model_source_lock.json').read_text()); status=json.loads((p/'range_download_status.json').read_text()); idx=json.loads((p/'model.safetensors.index.json').read_text()); required=set(idx['weight_map'].values())
expected='0499c3ac83fdef8810b907a23894ba91e95eddd8'
assert lock.get('huggingface_revision')==expected and status.get('revision')==expected
assert status.get('status')=='all_shards_verified' and required.issubset(set(status.get('completed_shards',[])))
assert all((p/name).is_file() for name in required)
print('Pinned Qwen3-32B-AWQ revision and all indexed shards verified.')
PY
}
verify_expert_snapshot
[[ -f "$BASE_MODEL/config.json" && -f "$BASE_MODEL/model.safetensors.index.json" ]] || { echo "Qwen3-8B local files are incomplete." >&2; exit 3; }
for port in "$BASE_PORT" "$EXPERT_PORT"; do
  if curl -fsS --max-time 2 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then echo "Port $port already has a model service; refusing to attach or replace it." >&2; exit 5; fi
done

resource_gate() {
  local gpu="$1" role="$2" minimum="$3" free util
  free="$(nvidia-smi --id="$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
  util="$(nvidia-smi --id="$gpu" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')"
  if (( free < minimum || util > 25 )); then
    echo "GPU $gpu fails $role safety gate: ${free} MiB free, utilization ${util}%; requires >=${minimum} MiB and <=25%. No service started." >&2
    exit 6
  fi
  printf '%s %s %s\n' "$free" "$util" "$gpu"
}

# Gate both devices before either model claims any GPU memory.
resource_gate "$EXPERT_GPU" expert 81920 >/dev/null
resource_gate "$BASE_GPU" base 46080 >/dev/null

start_service() {
  local role="$1" gpu="$2" min_free="$3" port="$4" alias="$5" model="$6" memory_fraction="$7" dtype="$8" quant="$9"
  local out="$RUN_DIR/services/$SERVICE_ATTEMPT/$role" free util pid gate_output short_tmp short_role
  if ! gate_output="$(resource_gate "$gpu" "$role" "$min_free")"; then return 6; fi
  read -r free util _ <<< "$gate_output"
  [[ ! -e "$out/service.log" && ! -e "$out/service.pid" && ! -e "$out/service_manifest.json" ]] || { echo "Service artifact exists; refusing overwrite: $out" >&2; return 7; }
  mkdir -p "$out"/{xdg,torch,triton,cuda,vllm,tmp,flashinfer}
  short_role="E"
  [[ "$role" == "base" ]] && short_role="B"
  short_tmp="/pfs/pfs-spMCeu/image_team/workspace/zjr/.q32_${short_role}_${SERVICE_ATTEMPT}"
  if [[ -e "$short_tmp" || -L "$short_tmp" ]]; then
    [[ -L "$short_tmp" && "$(readlink "$short_tmp")" == "$out/tmp" ]] || { echo "Short TMP path belongs to another target: $short_tmp" >&2; return 7; }
  else
    ln -s "$out/tmp" "$short_tmp"
  fi
  local args=(serve "$model" --host 127.0.0.1 --port "$port" --served-model-name "$alias" --max-model-len 4096 --max-num-seqs 1 --attention-backend TRITON_ATTN --gpu-memory-utilization "$memory_fraction")
  if [[ "$role" == "expert" ]]; then args+=(--dtype auto --quantization awq); else args+=(--dtype bfloat16 --disable-flashinfer-autotune); fi
  nohup env \
    CUDA_VISIBLE_DEVICES="$gpu" CUDA_HOME="$CUDA_TOOLKIT" PATH="$VENV/bin:$CUDA_TOOLKIT/bin:$PATH" \
    HF_HOME="$out/hf_home" HF_HUB_CACHE="$out/hf_home/hub" HF_HUB_DISABLE_TELEMETRY=1 \
    XDG_CACHE_HOME="$out/xdg" TORCH_HOME="$out/torch" TRITON_CACHE_DIR="$out/triton" CUDA_CACHE_PATH="$out/cuda" \
    VLLM_CACHE_ROOT="$out/vllm" TMPDIR="$short_tmp" FLASHINFER_WORKSPACE_BASE="$out/flashinfer" VLLM_USE_FLASHINFER_SAMPLER=0 \
    "$VENV/bin/vllm" "${args[@]}" >"$out/service.log" 2>&1 </dev/null &
  pid=$!; printf '%s\n' "$pid" > "$out/service.pid"
  if [[ "$role" == "expert" ]]; then EXPERT_PID="$pid"; else BASE_PID="$pid"; fi
  for _ in $(seq 1 600); do
    health_json="$(curl -fsS --max-time 3 "http://127.0.0.1:$port/v1/models" 2>/dev/null || true)"
    if [[ -n "$health_json" ]] && printf '%s' "$health_json" | "$VENV/bin/python" -c 'import json,sys; alias=sys.argv[1]; data=json.load(sys.stdin); sys.exit(0 if alias in [x.get("id") for x in data.get("data",[])] else 1)' "$alias" 2>/dev/null; then
      "$VENV/bin/python" - "$out/service_manifest.json" "$gpu" "$port" "$free" "$util" "$alias" "$role" <<'PY'
import importlib.metadata,json,sys
from datetime import datetime,timezone
from pathlib import Path
path,gpu,port,free,util,alias,role=sys.argv[1:]
models={'expert':('Qwen/Qwen3-32B-AWQ','0499c3ac83fdef8810b907a23894ba91e95eddd8','AWQ',0.25),'base':('Qwen/Qwen3-8B','b968826d9c46dd6066d109eabc6255188de91218',None,0.12)}
repo,rev,quant,mem=models[role]
obj={'started_at_utc':datetime.now(timezone.utc).isoformat(timespec='seconds'),'framework':'vLLM','framework_version':importlib.metadata.version('vllm'),'model':repo,'revision':rev,'quantization':quant,'physical_gpu':int(gpu),'port':int(port),'served_model_name':alias,'free_mib_before_start':int(free),'utilization_pct_before_start':int(util),'max_model_len':4096,'max_num_seqs':1,'gpu_memory_utilization':mem,'attention_backend':'TRITON_ATTN','temperature':0.0,'top_p':1.0,'enable_thinking':False}
Path(path).write_text(json.dumps(obj,indent=2)+'\n'); print(json.dumps(obj,indent=2))
PY
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then echo "$role vLLM exited; see $out/service.log" >&2; tail -n 80 "$out/service.log" >&2; return 8; fi
    sleep 2
  done
  echo "$role vLLM startup timed out; see $out/service.log" >&2; tail -n 80 "$out/service.log" >&2; return 9
}

stop_one() {
  local pid="$1" model="$2" port="$3" args
  [[ -n "$pid" ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    args="$(ps -p "$pid" -o args= 2>/dev/null || true)"
    if [[ "$args" == *"vllm"*"serve"*"$model"*"--port $port"* ]]; then kill -TERM "$pid" 2>/dev/null || true; fi
  fi
}
cleanup() {
  [[ -n "$MON_PID" ]] && kill "$MON_PID" 2>/dev/null || true
  stop_one "$BASE_PID" "$BASE_MODEL" "$BASE_PORT"
  stop_one "$EXPERT_PID" "$EXPERT_MODEL" "$EXPERT_PORT"
}
trap cleanup EXIT INT TERM

start_service expert "$EXPERT_GPU" 81920 "$EXPERT_PORT" "$EXPERT_ALIAS" "$EXPERT_MODEL" 0.25 auto awq
start_service base "$BASE_GPU" 46080 "$BASE_PORT" "$BASE_ALIAS" "$BASE_MODEL" 0.12 bfloat16 none

mkdir -p "$RUN_DIR"
GPU_LOG="$RUN_DIR/gpu_usage_${SERVICE_ATTEMPT}.csv"
[[ ! -e "$GPU_LOG" ]] || { echo "GPU telemetry already exists; refusing overwrite: $GPU_LOG" >&2; exit 4; }
printf 'timestamp_utc,gpu,memory_used_mib,memory_free_mib,utilization_pct\n' > "$GPU_LOG"
(
  while true; do
    ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    for gpu in "$EXPERT_GPU" "$BASE_GPU"; do
      vals="$(nvidia-smi --id="$gpu" --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')"
      [[ -n "$vals" ]] && printf '%s,%s,%s\n' "$ts" "$gpu" "$vals" >> "$GPU_LOG"
    done
    sleep 5
  done
) & MON_PID=$!

export EA_ATPO_SERVICE_ATTEMPT="$SERVICE_ATTEMPT"
SMOKE_MANIFEST="$RUN_DIR/smoke_20/run_manifest.json"
if [[ ! -e "$SMOKE_MANIFEST" || "$("$VENV/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status",""))' "$SMOKE_MANIFEST")" != "complete" ]]; then
  SMOKE_ARGS=(--smoke)
  if [[ -e "$RUN_DIR/smoke_20/base_answer_trajectories.jsonl" || -e "$RUN_DIR/smoke_20/qwen32_defer_trajectories.jsonl" ]]; then SMOKE_ARGS+=(--resume); fi
  "$VENV/bin/python" "$SCRIPT_DIR/run_qwen32_router_generalization.py" "${SMOKE_ARGS[@]}"
fi
"$VENV/bin/python" - "$RUN_DIR/smoke_20/run_manifest.json" <<'PY'
import json,sys
from pathlib import Path
m=json.loads(Path(sys.argv[1]).read_text())
assert m.get('status')=='complete' and m.get('paired_rows')==20, m
assert m.get('invalid_expert')==0 and m.get('invalid_base')==0, m
print('20-case protocol smoke complete; Base/Expert parser failures: 0.')
PY
if [[ -e "$RUN_DIR/paired_rollouts/base_answer_trajectories.jsonl" || -e "$RUN_DIR/paired_rollouts/qwen32_defer_trajectories.jsonl" || -e "$RUN_DIR/paired_rollouts/run_manifest.json" ]]; then
  "$VENV/bin/python" "$SCRIPT_DIR/run_qwen32_router_generalization.py" --resume
else
  "$VENV/bin/python" "$SCRIPT_DIR/run_qwen32_router_generalization.py"
fi
kill "$MON_PID" 2>/dev/null || true; MON_PID=""
stop_one "$BASE_PID" "$BASE_MODEL" "$BASE_PORT"; BASE_PID=""
stop_one "$EXPERT_PID" "$EXPERT_MODEL" "$EXPERT_PORT"; EXPERT_PID=""
if [[ ! -e "$RUN_DIR/router_generalization/policy_lock.json" ]]; then
  "$VENV/bin/python" "$SCRIPT_DIR/fit_qwen32_router_generalization.py"
fi
if [[ ! -e "$RUN_DIR/analysis/analysis_manifest.json" ]]; then
  "$VENV/bin/python" "$SCRIPT_DIR/analyze_qwen32_router_generalization.py"
fi
echo "Attempt04 data collection and one-time test analysis complete: $RUN_DIR"
