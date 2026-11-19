#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"

if [[ ! -f "${SCRIPT_DIR}/pilot_cases.csv" || ! -f "${SCRIPT_DIR}/selection_manifest.json" ]]; then
  "${PYTHON}" "${SCRIPT_DIR}/select_pilot_cases.py"
fi
"${PYTHON}" "${SCRIPT_DIR}/run_information_pilot.py" --config "${SCRIPT_DIR}/pilot_config.json" "$@"
