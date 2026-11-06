#!/usr/bin/env bash
set -euo pipefail

EA_ATPO_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EA_ATPO_PILOT_DIR="$EA_ATPO_PROJECT_DIR/experiments/real_case_pilot_20261103"
EA_ATPO_OUTPUT_DIR="$EA_ATPO_PILOT_DIR/outputs/smoke"
EA_ATPO_PYTHON="$EA_ATPO_PROJECT_DIR/.venv-real-pilot/bin/python"
export HF_HOME="$EA_ATPO_PILOT_DIR/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$EA_ATPO_PILOT_DIR/cache/datasets"
export XDG_CACHE_HOME="$EA_ATPO_PILOT_DIR/cache/xdg"
export TMPDIR="$EA_ATPO_PROJECT_DIR/.tmp"
export PYTHONPATH="$EA_ATPO_PROJECT_DIR/ea_atpo_stage${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$EA_ATPO_OUTPUT_DIR" "$EA_ATPO_PROJECT_DIR/.tmp" "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME"

"$EA_ATPO_PYTHON" "$EA_ATPO_PILOT_DIR/run_service_role_smoke.py" \
  --config "$EA_ATPO_PILOT_DIR/pilot_config.yaml" \
  --data "$EA_ATPO_PILOT_DIR/assets/train.pilot.parquet" \
  --output "$EA_ATPO_PILOT_DIR/outputs/service_role_smoke.jsonl" \
  2>&1 | tee "$EA_ATPO_PILOT_DIR/outputs/service_role_smoke.log"

"$EA_ATPO_PYTHON" "$EA_ATPO_PILOT_DIR/run_pilot.py" \
  --data "$EA_ATPO_PILOT_DIR/assets/train.pilot.parquet" \
  --output-dir "$EA_ATPO_OUTPUT_DIR" \
  --cases 10 --seed 20261103 --max-states-per-case 3 \
  --rollouts-per-action 1 --resume 2>&1 | tee -a "$EA_ATPO_OUTPUT_DIR/run.log"

"$EA_ATPO_PYTHON" - "$EA_ATPO_OUTPUT_DIR/run_manifest.json" <<'PY'
import json
import sys
from collections import defaultdict
from pathlib import Path
manifest_path = Path(sys.argv[1])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("status") != "completed":
    raise SystemExit(f"Smoke test did not complete: {manifest.get('status')}")
if manifest.get("valid_trajectory_count", 0) == 0:
    raise SystemExit("Smoke test has no valid model trajectories; full pilot is withheld.")
if manifest.get("failed_trajectory_count", 0):
    raise SystemExit("Smoke test contains failed rollouts; full pilot is withheld.")
records = [json.loads(line) for line in manifest_path.with_name("pilot_trajectories.jsonl").read_text(encoding="utf-8").splitlines() if line]
trajectories = [record for record in records if record.get("record_type") == "trajectory"]
expected = {"ANSWER", "ASK", "CONSULT", "DEFER"}
seen = {record.get("action") for record in trajectories}
if seen != expected:
    raise SystemExit(f"Smoke test action coverage is incomplete: {sorted(seen)}")
paired = defaultdict(set)
for record in trajectories:
    paired[(record["case_id"], record["state_id"], record["rollout_index"])].add(record["action"])
incomplete = [key for key, actions in paired.items() if actions != expected]
if incomplete:
    raise SystemExit(f"Smoke test has {len(incomplete)} incomplete paired states.")
print(json.dumps({k: manifest.get(k) for k in ("status", "selected_cases", "trajectory_count", "valid_trajectory_count", "failed_trajectory_count")}, indent=2))
PY
