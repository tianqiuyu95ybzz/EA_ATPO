#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"
EXPERT_VARIANT="${EA_ATPO_EXPERT_VARIANT:-AWQ}"
bash "$SCRIPT_DIR/start_quantized_expert_service.sh"
if [[ "$EXPERT_VARIANT" == "AWQ" ]]; then
  REPO_ID="Qwen/Qwen3-32B-AWQ"
  REVISION="0499c3ac83fdef8810b907a23894ba91e95eddd8"
  ALIAS="qwen3-32b-awq-expert"
else
  REPO_ID="Qwen/Qwen3-32B-FP8"
  REVISION="aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df"
  ALIAS="qwen3-32b-fp8-expert"
fi
"$PROJECT_DIR/.venv-real-pilot/bin/python" "$SCRIPT_DIR/run_quantized_expert_eval.py" \
  --repo-id "$REPO_ID" --revision "$REVISION" --precision "$EXPERT_VARIANT" \
  --alias "$ALIAS" --output-dir "$SCRIPT_DIR/outputs/strong_expert_${EXPERT_VARIANT,,}" --resume
