#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/model_service.pid"
if [[ ! -s "$PID_FILE" ]]; then
  echo "No recorded model-service PID."
  exit 0
fi
PID="$(cat "$PID_FILE")"
if kill -0 "$PID" 2>/dev/null; then
  kill -TERM -- "-$PID" 2>/dev/null || true
  for _ in $(seq 1 15); do
    if ! kill -0 -- "-$PID" 2>/dev/null; then break; fi
    sleep 1
  done
  if kill -0 -- "-$PID" 2>/dev/null; then
    kill -9 -- "-$PID" 2>/dev/null || true
  fi
fi
: > "$PID_FILE"
echo "Stopped service process group $PID."
