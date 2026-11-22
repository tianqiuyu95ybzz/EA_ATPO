#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="$PROJECT_DIR/.venv-real-pilot/bin/python"
EXPERIMENT="$SCRIPT_DIR/run_relative_value_experiment.py"
OUT="$SCRIPT_DIR/outputs/relative_value_20261106"

if [[ ! -f "$OUT/validation_policy_config.json" ]]; then
  echo "Validation config is not frozen; refusing to open test set." >&2
  exit 2
fi
"$SCRIPT_DIR/start_relative_model_service.sh"
"$PYTHON" "$EXPERIMENT" --phase collect-test
"$PYTHON" "$EXPERIMENT" --phase repair-test-protocol
"$PYTHON" "$EXPERIMENT" --phase evaluate-test
