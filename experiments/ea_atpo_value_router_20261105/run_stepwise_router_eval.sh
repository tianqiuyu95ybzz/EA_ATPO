#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"
"$SCRIPT_DIR/start_router_model_service.sh"
"$PROJECT_DIR/.venv-real-pilot/bin/python" "$SCRIPT_DIR/run_stepwise_router_eval.py" \
  --data "$PROJECT_DIR/experiments/real_case_pilot_20261103/assets/test.pilot.parquet" \
  --artifacts "$SCRIPT_DIR/artifacts" --output-dir "$SCRIPT_DIR/outputs/stepwise_test_v2" \
  --cases 200 --seed 20261106 --max-turns 10 --temperature 0.2 --resume
"$PROJECT_DIR/.venv-real-pilot/bin/python" "$SCRIPT_DIR/summarize_stepwise_router_eval.py" \
  --output-dir "$SCRIPT_DIR/outputs/stepwise_test_v2"
