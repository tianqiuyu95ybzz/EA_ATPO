#!/usr/bin/env bash
set -euo pipefail

EA_ATPO_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EA_ATPO_PILOT_DIR="$EA_ATPO_PROJECT_DIR/experiments/real_case_pilot_20261103"
EA_ATPO_OUTPUT_DIR="$EA_ATPO_PILOT_DIR/outputs/full"
EA_ATPO_PYTHON="$EA_ATPO_PROJECT_DIR/.venv-real-pilot/bin/python"
EA_ATPO_SMOKE_MANIFEST="$EA_ATPO_PILOT_DIR/outputs/smoke/run_manifest.json"
export HF_HOME="$EA_ATPO_PILOT_DIR/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$EA_ATPO_PILOT_DIR/cache/datasets"
export XDG_CACHE_HOME="$EA_ATPO_PILOT_DIR/cache/xdg"
export TMPDIR="$EA_ATPO_PROJECT_DIR/.tmp"
export PYTHONPATH="$EA_ATPO_PROJECT_DIR/ea_atpo_stage${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$EA_ATPO_OUTPUT_DIR" "$EA_ATPO_PROJECT_DIR/.tmp" "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME"

UV_CACHE_DIR="$EA_ATPO_PROJECT_DIR/.cache/uv" TMPDIR="$EA_ATPO_PROJECT_DIR/.tmp" \
uv pip install --require-hashes \
  --default-index https://pypi.tuna.tsinghua.edu.cn/simple \
  --python "$EA_ATPO_PYTHON" \
  -r "$EA_ATPO_PILOT_DIR/PILOT_ANALYSIS_REQUIREMENTS.lock.txt"

"$EA_ATPO_PYTHON" - "$EA_ATPO_SMOKE_MANIFEST" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit("Run the 10-case smoke test before the full pilot.")
manifest = json.loads(path.read_text(encoding="utf-8"))
if manifest.get("status") != "completed" or manifest.get("valid_trajectory_count", 0) == 0:
    raise SystemExit("Smoke test is not sufficiently complete; inspect its report before running 100 cases.")
if manifest.get("failed_trajectory_count", 0) > 0:
    raise SystemExit("Smoke test contains failed rollouts; resolve protocol failures before the full pilot.")
PY

"$EA_ATPO_PYTHON" "$EA_ATPO_PILOT_DIR/run_pilot.py" \
  --data "$EA_ATPO_PILOT_DIR/assets/train.pilot.parquet" \
  --output-dir "$EA_ATPO_OUTPUT_DIR" \
  --cases 100 --seed 20261103 --max-states-per-case 3 \
  --rollouts-per-action 3 --resume 2>&1 | tee -a "$EA_ATPO_OUTPUT_DIR/run.log"

"$EA_ATPO_PYTHON" "$EA_ATPO_PILOT_DIR/plot_pilot_tradeoffs.py" \
  --summary "$EA_ATPO_OUTPUT_DIR/summary.csv" --output-dir "$EA_ATPO_OUTPUT_DIR"
"$EA_ATPO_PYTHON" "$EA_ATPO_PILOT_DIR/make_pilot_report.py" \
  --output-dir "$EA_ATPO_OUTPUT_DIR"
