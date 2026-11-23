#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="$PROJECT_DIR/.venv-real-pilot/bin/python"
EXPERIMENT="$SCRIPT_DIR/run_relative_value_experiment.py"

"$PYTHON" -m py_compile "$SCRIPT_DIR/relative_value_router.py" "$EXPERIMENT"
if [[ ! -f "$SCRIPT_DIR/outputs/relative_value_20261106/checkpoint_relative_initial_only/relative_value_checkpoint.npz" ]]; then
  "$PYTHON" "$EXPERIMENT" --phase train
fi
"$PYTHON" "$EXPERIMENT" --phase prepare-samples
"$SCRIPT_DIR/start_relative_model_service.sh"
"$PYTHON" "$EXPERIMENT" --phase collect-validation
if [[ ! -f "$SCRIPT_DIR/outputs/relative_value_20261106/validation_policy_config.json" ]]; then
  "$PYTHON" "$EXPERIMENT" --phase evaluate-validation
fi
echo "Validation thresholds are frozen in outputs/relative_value_20261106/validation_policy_config.json"
