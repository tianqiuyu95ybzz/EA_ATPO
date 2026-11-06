#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$SCRIPT_DIR/outputs/advice_utilization"
PID_FILE="$OUT_DIR/base_service.pid"
MANIFEST="$OUT_DIR/service_manifest.json"
if [[ ! -s "$PID_FILE" ]]; then
  echo "No saved service PID at $PID_FILE"
  exit 0
fi
pid="$(cat "$PID_FILE")"
if [[ ! -r "/proc/$pid/cmdline" ]]; then
  echo "Service PID $pid is already stopped."
  exit 0
fi
cmdline="$(tr '\0' ' ' <"/proc/$pid/cmdline")"
if [[ "$cmdline" != *"vllm serve"* || "$cmdline" != *"--port 30133"* || "$cmdline" != *"Qwen3-8B"* ]]; then
  echo "Refusing to stop unexpected PID $pid: $cmdline" >&2
  exit 6
fi
process_group="$(ps -o pgid= -p "$pid" | tr -d ' ')"
if [[ "$process_group" != "$pid" ]]; then
  echo "Refusing to stop PID $pid with unexpected process group $process_group" >&2
  exit 7
fi
kill -TERM -- "-$process_group"
for _ in $(seq 1 30); do
  if ! kill -0 "$pid" 2>/dev/null; then break; fi
  sleep 1
done
if kill -0 "$pid" 2>/dev/null; then
  echo "Service process group $process_group did not stop after SIGTERM." >&2
  exit 8
fi
"$(dirname "$SCRIPT_DIR")/../../.venv-real-pilot/bin/python" - "$MANIFEST" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
p=Path(sys.argv[1])
if p.exists():
    data=json.loads(p.read_text(encoding="utf-8"))
    data["stopped_at_utc"]=datetime.now(timezone.utc).isoformat(timespec="seconds")
    data["stop_signal"]="SIGTERM after decision experiment"
    p.write_text(json.dumps(data,indent=2)+"\n",encoding="utf-8")
PY
echo "Stopped advice-utilization service process group $process_group."
