#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PILOT_DIR="$PROJECT_DIR/experiments/real_case_pilot_20261103"
VENV="$PROJECT_DIR/.venv-real-pilot"
VLLM="$VENV/bin/vllm"
PYTHON="$VENV/bin/python"
CUDA_TOOLKIT="$VENV/lib/python3.12/site-packages/nvidia/cu13"
VARIANT="${EA_ATPO_EXPERT_VARIANT:-AWQ}"
GPU="${EA_ATPO_EXPERT_GPU:-2}"
PORT="${EA_ATPO_EXPERT_PORT:-30132}"
OUT_DIR="$SCRIPT_DIR/outputs/strong_expert_${VARIANT,,}"
if [[ "$VARIANT" == "AWQ" ]]; then
  MODEL_DIR="$PILOT_DIR/assets/models/Qwen3-32B-AWQ"
  MODEL_ALIAS="qwen3-32b-awq-expert"
  REPO_ID="Qwen/Qwen3-32B-AWQ"
  REVISION="0499c3ac83fdef8810b907a23894ba91e95eddd8"
  QUANTIZATION="awq"
elif [[ "$VARIANT" == "FP8" ]]; then
  MODEL_DIR="$PILOT_DIR/assets/models/Qwen3-32B-FP8"
  MODEL_ALIAS="qwen3-32b-fp8-expert"
  REPO_ID="Qwen/Qwen3-32B-FP8"
  REVISION="aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df"
  QUANTIZATION="fp8"
else
  echo "EA_ATPO_EXPERT_VARIANT must be AWQ or FP8." >&2
  exit 2
fi

if [[ "$VARIANT" == "AWQ" ]]; then
  "$PYTHON" - "$MODEL_DIR/range_download_status.json" "$MODEL_DIR/model.safetensors.index.json" "$REVISION" <<'PY'
import json,sys
from pathlib import Path
status_path,index_path,revision=sys.argv[1:]
status=json.loads(Path(status_path).read_text())
index=json.loads(Path(index_path).read_text())
required=set(index["weight_map"].values())
if status.get("status") != "all_shards_verified" or status.get("revision") != revision:
    raise SystemExit("AWQ download manifest does not show all shards verified at the pinned revision")
if not required.issubset(set(status.get("completed_shards", []))):
    raise SystemExit("AWQ download manifest is missing a verified indexed weight shard")
PY
fi

for file in config.json model.safetensors.index.json tokenizer.json tokenizer_config.json; do
  [[ -f "$MODEL_DIR/$file" ]] || { echo "Missing model file: $MODEL_DIR/$file" >&2; exit 3; }
done
shard_count="$("$PYTHON" - "$MODEL_DIR/model.safetensors.index.json" <<'PY'
import json,sys
from pathlib import Path
index=json.loads(Path(sys.argv[1]).read_text())
print(len(set(index["weight_map"].values())))
PY
)"
for shard in $("$PYTHON" - "$MODEL_DIR/model.safetensors.index.json" <<'PY'
import json,sys
from pathlib import Path
index=json.loads(Path(sys.argv[1]).read_text())
print("\n".join(sorted(set(index["weight_map"].values()))))
PY
); do
  [[ -f "$MODEL_DIR/$shard" ]] || { echo "Model weight shard missing: $MODEL_DIR/$shard" >&2; exit 4; }
done
if find "$MODEL_DIR/.cache/huggingface/download" -type f -name '*.incomplete' -print -quit 2>/dev/null | rg -q .; then
  echo "Incomplete Hub download fragments remain; refusing to start an incomplete model." >&2
  exit 5
fi
if curl --silent --fail "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
  echo "Port $PORT is already serving; refusing to overwrite another service." >&2
  exit 6
fi
free_mib="$(nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
if (( free_mib < 45000 )); then
  echo "GPU $GPU currently has $free_mib MiB free; quantized 32B needs at least 45000 MiB free." >&2
  exit 7
fi
mkdir -p "$OUT_DIR" "$SCRIPT_DIR/cache/$VARIANT/xdg" "$SCRIPT_DIR/cache/$VARIANT/triton" "$SCRIPT_DIR/cache/$VARIANT/torch" "$SCRIPT_DIR/cache/$VARIANT/cuda" "$SCRIPT_DIR/cache/$VARIANT/vllm" "$PILOT_DIR/cache/flashinfer_workspace" "$PROJECT_DIR/.tmp"
export HF_HOME="$PILOT_DIR/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_DISABLE_TELEMETRY=1
export XDG_CACHE_HOME="$SCRIPT_DIR/cache/$VARIANT/xdg"
export TORCH_HOME="$SCRIPT_DIR/cache/$VARIANT/torch"
export TRITON_CACHE_DIR="$SCRIPT_DIR/cache/$VARIANT/triton"
export CUDA_CACHE_PATH="$SCRIPT_DIR/cache/$VARIANT/cuda"
export VLLM_CACHE_ROOT="$SCRIPT_DIR/cache/$VARIANT/vllm"
export TMPDIR="$PROJECT_DIR/.tmp"
export PATH="$VENV/bin:$CUDA_TOOLKIT/bin:$PATH"
export CUDA_HOME="$CUDA_TOOLKIT"
export FLASHINFER_WORKSPACE_BASE="$PILOT_DIR/cache/flashinfer_workspace"
export VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_VISIBLE_DEVICES="$GPU"

nohup setsid "$VLLM" serve "$MODEL_DIR" \
  --host 127.0.0.1 --port "$PORT" --served-model-name "$MODEL_ALIAS" \
  --dtype auto --quantization "$QUANTIZATION" --max-model-len 8192 --max-num-seqs 1 \
  --attention-backend TRITON_ATTN --gpu-memory-utilization 0.30 \
  >"$OUT_DIR/model_service.log" 2>&1 &
echo "$!" > "$OUT_DIR/model_service.pid"

for _ in $(seq 1 600); do
  if curl --silent --fail "http://127.0.0.1:$PORT/v1/models" >/dev/null; then
    "$PYTHON" - "$OUT_DIR" "$REPO_ID" "$REVISION" "$VARIANT" "$MODEL_ALIAS" "$GPU" "$free_mib" "$PORT" "$shard_count" <<'PY'
import importlib.metadata,json,sys
from datetime import datetime,timezone
from pathlib import Path
out=Path(sys.argv[1])
record={"started_at_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),"framework":"vllm","vllm_version":importlib.metadata.version("vllm"),"repo_id":sys.argv[2],"revision":sys.argv[3],"precision":sys.argv[4],"served_model_name":sys.argv[5],"gpu_physical_index":int(sys.argv[6]),"gpu_free_mib_before_start":int(sys.argv[7]),"port":int(sys.argv[8]),"dtype":"auto","quantization":sys.argv[4].lower(),"max_model_len":8192,"max_num_seqs":1,"gpu_memory_utilization":0.30,"attention_backend":"TRITON_ATTN","shard_count":int(sys.argv[9]),"model_files_verified_present":True}
(out/"model_service_manifest.json").write_text(json.dumps(record,indent=2),encoding="utf-8")
print(json.dumps(record,indent=2))
PY
    exit 0
  fi
  if ! kill -0 "$(cat "$OUT_DIR/model_service.pid")" 2>/dev/null; then
    echo "Quantized vLLM exited during startup; see $OUT_DIR/model_service.log" >&2
    tail -n 120 "$OUT_DIR/model_service.log" >&2
    exit 8
  fi
  sleep 2
done
echo "Timed out waiting for quantized model service; see $OUT_DIR/model_service.log" >&2
tail -n 120 "$OUT_DIR/model_service.log" >&2
exit 9
