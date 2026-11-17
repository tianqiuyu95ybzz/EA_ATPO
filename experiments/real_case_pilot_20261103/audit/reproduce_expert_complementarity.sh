#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
MODE="${1:-offline}"
case "$MODE" in
  offline)
    python3 "$SCRIPT_DIR/analyze_existing_expert_complementarity.py"
    ;;
  --run-qwen32)
    python3 "$SCRIPT_DIR/analyze_existing_expert_complementarity.py"
    bash "$SCRIPT_DIR/run_qwen32_defer_complementarity.sh"
    ;;
  --resume-qwen32)
    bash "$SCRIPT_DIR/run_qwen32_defer_complementarity.sh" --resume
    ;;
  *) echo "Usage: $0 [offline|--run-qwen32|--resume-qwen32]" >&2; exit 2 ;;
esac
