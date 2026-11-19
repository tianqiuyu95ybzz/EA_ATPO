#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
MULTI="$ROOT/experiments/expert_sft/multiturn_dialogue"
PY="${EA_ATPO_AWQ_PYTHON:-$MULTI/.venv_awq/bin/python}"
GPU="${EA_ATPO_AWQ_GPU:-3}"
OUT="$MULTI/checkpoints/awq_lora/smoke_$(date -u +%Y%m%dT%H%M%SZ)"
[[ -x "$PY" ]] || { echo "Missing isolated AWQ Python: $PY (run scripts/setup_awq_env.sh first)" >&2; exit 2; }
[[ "$GPU" != "1" && "$GPU" != "7" ]] || { echo "GPU $GPU is excluded." >&2; exit 2; }
export CUDA_VISIBLE_DEVICES="$GPU"
export EA_ATPO_AWQ_PHYSICAL_GPU="$GPU"
export HF_HOME="$MULTI/.cache/hf_home"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$MULTI/.cache/datasets"
export XDG_CACHE_HOME="$MULTI/.cache/xdg"
export TMPDIR="$MULTI/.cache/tmp"
export TRITON_CACHE_DIR="$MULTI/.cache/triton"
export TORCH_HOME="$MULTI/.cache/torch"
mkdir -p "$MULTI/logs" "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCH_HOME"

"$PY" "$MULTI/scripts/gpu_safety_gate.py" --gpu "$GPU" --min-free-mib 150000 --max-util 20 --manifest "$MULTI/logs/gpu_preflight_awq_smoke_$(basename "$OUT").json"
"$PY" "$MULTI/scripts/train_multiturn_sft.py" \
  --mode awq --gpu "$GPU" --smoke-steps 10 --output-dir "$OUT" \
  2>&1 | tee "$MULTI/logs/awq_lora_smoke_$(basename "$OUT").log"
"$PY" - "$OUT" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
m = json.loads((p / "run_manifest.json").read_text())
g = json.loads((p / "single_batch_gradient_update_preflight.json").read_text())
checks = {
  "ten_optimizer_steps": m.get("optimizer_steps") == 10,
  "forward_backward_and_update": g.get("status") == "passed" and g.get("lora_parameter_update_l2", 0) > 0,
  "adapter_roundtrip": g.get("adapter_save_reload_roundtrip") == "passed",
  "no_gold_or_hidden_input": g.get("gold_or_hidden_atomic_facts_sent_as_model_input") is False,
}
print(json.dumps(checks, indent=2))
if not all(checks.values()):
    raise SystemExit("AWQ LoRA smoke gate failed")
PY
