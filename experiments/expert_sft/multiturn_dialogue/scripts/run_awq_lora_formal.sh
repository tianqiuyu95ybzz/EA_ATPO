#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
MULTI="$ROOT/experiments/expert_sft/multiturn_dialogue"
PY="${EA_ATPO_AWQ_PYTHON:-$MULTI/.venv_awq/bin/python}"
GPU="${EA_ATPO_AWQ_GPU:-3}"
OUT="$MULTI/checkpoints/awq_lora/formal_$(date -u +%Y%m%dT%H%M%SZ)"
[[ -x "$PY" ]] || { echo "Missing isolated AWQ Python: $PY" >&2; exit 2; }
[[ "$GPU" != "1" && "$GPU" != "7" ]] || { echo "GPU $GPU is excluded." >&2; exit 2; }
[[ -f "$MULTI/environment.lock.txt" ]] || { echo "AWQ environment lock is missing." >&2; exit 2; }

export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HOME="$MULTI/.cache/hf_home"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$MULTI/.cache/datasets"
export XDG_CACHE_HOME="$MULTI/.cache/xdg"
export TMPDIR="$MULTI/.cache/tmp"
export TRITON_CACHE_DIR="$MULTI/.cache/triton"
export TORCH_HOME="$MULTI/.cache/torch"
mkdir -p "$MULTI/logs" "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCH_HOME"

"$PY" - "$MULTI" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
smokes = sorted((root / "checkpoints/awq_lora").glob("smoke_*/run_manifest.json"))
passed = False
for p in reversed(smokes):
    m = json.loads(p.read_text())
    g = json.loads(p.parent / "single_batch_gradient_update_preflight.json")
    if m.get("status") == "completed" and m.get("mode") == "awq" and m.get("optimizer_steps") == 10 and g.get("adapter_save_reload_roundtrip") == "passed" and g.get("lora_gradient_global_norm", 0) > 0 and g.get("lora_parameter_update_l2", 0) > 0:
        passed = True
        break
if not passed:
    raise SystemExit("Formal AWQ SFT blocked: no passing 10-step AWQ load/gradient/update/save/reload smoke exists")
PY
"$PY" "$MULTI/scripts/gpu_safety_gate.py" --gpu "$GPU" --min-free-mib 150000 --max-util 20 --manifest "$MULTI/logs/gpu_preflight_awq_formal_$(basename "$OUT").json"
"$PY" "$MULTI/scripts/train_multiturn_sft.py" \
  --mode awq --gpu "$GPU" --epochs 3 --output-dir "$OUT" \
  2>&1 | tee "$MULTI/logs/awq_lora_formal_$(basename "$OUT").log"
