#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"
"$PROJECT_DIR/.venv-real-pilot/bin/python" "$SCRIPT_DIR/train_router.py" \
  --pilot "$PROJECT_DIR/experiments/real_case_pilot_20261103/outputs/full/pilot_trajectories.jsonl" \
  --output-dir "$SCRIPT_DIR/artifacts" --seed 20261105
