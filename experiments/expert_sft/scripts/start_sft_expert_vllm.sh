#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXP="$ROOT/experiments/expert_sft"
CFG="$EXP/configs/model_services_sft_expert.yaml"
PYTHON="/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/.venv-real-pilot/bin/python"
VLLM="/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/.venv-real-pilot/bin/vllm"
GPU="${EA_ATPO_SFT_EXPERT_GPU:-5}"
PORT="30483"
MIN_FREE_MIB="150000"
MAX_UTIL="20"
MODEL="$EXP/assets/models/Qwen3-32B"
ADAPTER="$EXP/checkpoints/qwen3_32b_domain_sft/selected"
PID_FILE="$EXP/logs/sft_expert_vllm.pid"
LOG="$EXP/logs/sft_expert_vllm.log"
MANIFEST="$EXP/logs/sft_expert_service_manifest.json"

export HF_HOME="$EXP/cache/hf_home"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$EXP/cache/datasets"
export XDG_CACHE_HOME="$EXP/cache/xdg"
export TMPDIR="$EXP/cache/tmp"
export TRITON_CACHE_DIR="$EXP/cache/triton"
export TORCH_HOME="$EXP/cache/torch"
export VLLM_CACHE_ROOT="$EXP/cache/vllm_runtime"
export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "$EXP/logs" "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCH_HOME" "$VLLM_CACHE_ROOT"

[[ -x "$PYTHON" && -x "$VLLM" ]] || { echo "Pinned VI inference environment is unavailable." >&2; exit 2; }
[[ -f "$MODEL/config.json" && -f "$MODEL/model.safetensors.index.json" ]] || { echo "Pinned BF16 Qwen3-32B model is incomplete." >&2; exit 2; }
[[ -f "$ADAPTER/adapter_config.json" && -f "$ADAPTER/adapter_model.safetensors" ]] || { echo "Validation-selected SFT adapter is missing." >&2; exit 2; }
[[ -f "$EXP/checkpoints/qwen3_32b_domain_sft/selected_checkpoint_lock.json" ]] || { echo "Selected adapter lock is missing." >&2; exit 2; }
[[ -f "$CFG" ]] || { echo "SFT Expert service config is missing: $CFG" >&2; exit 2; }

if [[ -f "$PID_FILE" ]]; then
  EXISTING="$(cat "$PID_FILE")"
  if kill -0 "$EXISTING" 2>/dev/null; then
    echo "Recorded SFT Expert service PID $EXISTING is still running; refusing to start another." >&2
    exit 2
  fi
  rm -f "$PID_FILE"
fi

"$PYTHON" - "$GPU" "$MIN_FREE_MIB" "$MAX_UTIL" <<'PY'
import subprocess, sys
idx, min_free, max_util = map(int, sys.argv[1:])
line = subprocess.check_output(["nvidia-smi", f"--id={idx}", "--query-gpu=index,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True).strip()
gpu, free, util = [int(x.strip()) for x in line.split(",")]
if gpu != idx or free < min_free or util > max_util:
    raise SystemExit(f"GPU {idx} fails SFT Expert gate: free_mib={free}, utilization_pct={util}; no service started")
pids = subprocess.check_output(["nvidia-smi", f"--id={idx}", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip()
if pids and pids.upper() != "N/A":
    raise SystemExit(f"GPU {idx} has existing compute processes; refusing to start: {pids}")
print(f"GPU {gpu} passed SFT Expert gate: free_mib={free}, utilization_pct={util}, no compute processes")
PY

"$VLLM" --version
nohup "$VLLM" serve "$MODEL" \
  --host 127.0.0.1 --port "$PORT" \
  --served-model-name vi-qwen3-32b-domain-sft-expert \
  --dtype bfloat16 --max-model-len 4096 --max-num-seqs 1 \
  --tensor-parallel-size 1 --gpu-memory-utilization 0.60 \
  --enable-lora --lora-modules "vi-qwen3-32b-domain-sft-expert=$ADAPTER" \
  --max-lora-rank 16 --max-loras 1 \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --disable-log-requests >"$LOG" 2>&1 &
PID=$!
echo "$PID" > "$PID_FILE"

if ! "$PYTHON" - "$PID" "$PORT" "$ADAPTER" "$MODEL" "$MANIFEST" "$GPU" <<'PY'
import hashlib, json, os, subprocess, sys, time, urllib.request
pid, port, adapter, model, manifest, gpu = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6])
url = f"http://127.0.0.1:{port}/v1/models"
deadline = time.time() + 1200
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            body = json.loads(response.read())
        ids = [x.get("id") for x in body.get("data", [])]
        if "vi-qwen3-32b-domain-sft-expert" in ids:
            break
    except Exception:
        if subprocess.call(["kill", "-0", str(pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0:
            raise SystemExit("SFT Expert vLLM process exited during startup; inspect the project-local log")
        time.sleep(5)
else:
    raise SystemExit("SFT Expert did not become healthy within 20 minutes; inspect the project-local log")
gpu_rows = subprocess.check_output(["nvidia-smi", f"--id={gpu}", "--query-gpu=index,name,uuid,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True).strip().split(",")
names = ["index", "name", "uuid", "total_mib", "used_mib", "free_mib", "utilization_pct"]
gpu_info = {k: (int(v.strip()) if k in {"index", "total_mib", "used_mib", "free_mib", "utilization_pct"} else v.strip()) for k, v in zip(names, gpu_rows)}
if gpu_info["free_mib"] < 32768:
    raise SystemExit(f"SFT Expert started but left less than the required free-memory reserve: {gpu_info['free_mib']} MiB")
def sha_tree(path):
    h=hashlib.sha256()
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        p=os.path.join(path,name)
        h.update(name.encode())
        with open(p,"rb") as f:
            for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
entry={"status":"healthy","pid":pid,"base_model":"Qwen/Qwen3-32B","base_revision":"9216db5781bf21249d130ec9da846c4624c16137","adapter_path":adapter,"adapter_sha256":sha_tree(adapter),"model_path":model,"served_model":"vi-qwen3-32b-domain-sft-expert","endpoint":url,"vllm_version":subprocess.check_output(["/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/.venv-real-pilot/bin/vllm","--version"],text=True).strip(),"physical_gpu":gpu_info,"thinking":False,"request_body_logging":False,"chat_template":"model-native","started_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
with open(manifest,"w",encoding="utf-8") as f: json.dump(entry,f,ensure_ascii=False,indent=2); f.write("\n")
print(json.dumps({"status":"healthy","pid":pid,"endpoint":url,"gpu_free_mib":gpu_info["free_mib"]}))
PY
then
  # This PID was created by this script; never signal another process.
  kill -TERM "$PID" 2>/dev/null || true
  rm -f "$PID_FILE"
  exit 1
fi
