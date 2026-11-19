#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$(cd "$SCRIPT_DIR/.." && pwd)/configs/model_services.yaml"
PYTHON="/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/.venv-real-pilot/bin/python"
[[ -x "$PYTHON" ]] || { echo "Pinned inference environment missing: $PYTHON" >&2; exit 2; }
exec "$PYTHON" "$SCRIPT_DIR/manage_vi_services.py" start --config "$CONFIG"
