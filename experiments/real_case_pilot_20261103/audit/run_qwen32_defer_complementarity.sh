#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$SCRIPT_DIR/outputs/qwen32_awq_complementarity_20261117_gpu7"
VENV="/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/.venv-real-pilot"
RESUME_ARG=""
if [[ "${1:-}" == "--resume" ]]; then RESUME_ARG="--resume"; elif [[ $# -gt 0 ]]; then echo "Usage: $0 [--resume]" >&2; exit 2; fi
started32=0 started8=0
cleanup() {
  if (( started8 )); then
    if [[ -f "$OUT_DIR/base_service.pid" ]]; then kill -TERM "$(cat "$OUT_DIR/base_service.pid")" 2>/dev/null || true; fi
  fi
  if (( started32 )); then
    if [[ -f "$OUT_DIR/model_service.pid" ]]; then kill -TERM "$(cat "$OUT_DIR/model_service.pid")" 2>/dev/null || true; fi
  fi
}
trap cleanup EXIT INT TERM
bash "$SCRIPT_DIR/start_qwen32_awq_complementarity_service.sh"
started32=1
bash "$SCRIPT_DIR/start_qwen8_base_complementarity_service.sh"
started8=1
if [[ -n "$RESUME_ARG" ]]; then
  "$VENV/bin/python" "$SCRIPT_DIR/run_qwen32_defer_complementarity.py" --resume
else
  "$VENV/bin/python" "$SCRIPT_DIR/run_qwen32_defer_complementarity.py"
fi
python3 "$SCRIPT_DIR/fit_public_feature_router.py"
