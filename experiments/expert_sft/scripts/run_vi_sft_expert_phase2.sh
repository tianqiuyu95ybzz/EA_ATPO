#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXP="$ROOT/experiments/expert_sft"
PHASE2="$ROOT/experiments/vi_ea_atpo/phase2"
PYTHON="/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/.venv-real-pilot/bin/python"
MODE="${1:-}"
SMOKE_OUT="$EXP/evaluations/vi_phase2_sft_expert_smoke"
PILOT_OUT="$EXP/evaluations/vi_phase2_sft_expert_full"
CONFIG="$EXP/configs/vi_ea_atpo_sft_expert_pilot.json"
SELECTION="$PHASE2/pilot_cases.csv"
DATA="$EXP/data/reconstructed_compat/train.pilot.parquet"
SERVICE_PID="$EXP/logs/sft_expert_vllm.pid"
SERVICE_MANIFEST="$EXP/logs/sft_expert_service_manifest.json"

export HF_HOME="$EXP/cache/hf_home"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$EXP/cache/datasets"
export XDG_CACHE_HOME="$EXP/cache/xdg"
export TMPDIR="$EXP/cache/tmp"
export TRITON_CACHE_DIR="$EXP/cache/triton"
export TORCH_HOME="$EXP/cache/torch"
export VLLM_CACHE_ROOT="$EXP/cache/vllm_runtime"

[[ "$MODE" == "smoke" || "$MODE" == "pilot" ]] || { echo "Usage: $0 smoke|pilot" >&2; exit 2; }
[[ -x "$PYTHON" ]] || { echo "Pinned Phase-2 API environment is unavailable: $PYTHON" >&2; exit 2; }
[[ -f "$SELECTION" && -f "$DATA" ]] || { echo "Frozen Phase-2 selection/source data are missing; do not rebuild the queue here." >&2; exit 2; }
[[ -f "$EXP/checkpoints/qwen3_32b_domain_sft/selected_checkpoint_lock.json" ]] || { echo "Validation-selected SFT Expert is not frozen." >&2; exit 2; }
[[ -f "$SERVICE_MANIFEST" && -f "$SERVICE_PID" ]] || { echo "New SFT Expert service has not been started." >&2; exit 2; }
kill -0 "$(cat "$SERVICE_PID")" 2>/dev/null || { echo "SFT Expert PID is not alive." >&2; exit 2; }

if [[ "$MODE" == "pilot" ]]; then
  "$PYTHON" - "$SMOKE_OUT" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
m=json.loads((p/'execution_manifest.json').read_text(encoding='utf-8'))
if m.get('status')!='completed' or m.get('selected_run_cases')!=2:
    raise SystemExit('150-case Pilot blocked: the two-case real smoke did not complete')
rows=[json.loads(x) for x in (p/'intervention_rollouts.jsonl').open(encoding='utf-8') if x.strip()]
required={'DIRECT_ANSWER','DIRECT_DEFER','ASK_TASK_ANSWER','ASK_DECISION','CONSULT_TASK_ANSWER','CONSULT_DECISION'}
cases={}
for r in rows: cases.setdefault(r['case_id'],set()).add(r['intervention'])
if len(cases)!=2 or any(v!=required for v in cases.values()):
    raise SystemExit('150-case Pilot blocked: smoke lacks a paired action/continuation branch')
if any(r.get('error') for r in rows):
    raise SystemExit('150-case Pilot blocked: smoke includes a protocol/API error')
review=p/'smoke_quality_review.json'
if not review.is_file():
    raise SystemExit('150-case Pilot blocked: review Patient fact recall, CONSULT non-answer semantics, information boundaries, and cost accounting; write smoke_quality_review.json only if all pass')
q=json.loads(review.read_text(encoding='utf-8'))
required_checks=('patient_fact_recall_passed','consult_did_not_leak_terminal_answer','gold_hidden_fact_isolation_passed','cost_accounting_passed')
if q.get('status')!='passed' or any(q.get(k) is not True for k in required_checks):
    raise SystemExit('150-case Pilot blocked: recorded smoke quality review did not pass every required check')
print('Two-case model smoke and recorded protocol/quality review passed.')
PY
  OUT="$PILOT_OUT"
  [[ ! -e "$OUT/intervention_rollouts.jsonl" ]] || { echo "Full Pilot output already exists; refusing overwrite." >&2; exit 2; }
  "$PYTHON" "$PHASE2/run_information_pilot.py" \
    --config "$CONFIG" --selection "$SELECTION" --data "$DATA" \
    --run-pilot --output-dir "$OUT" 2>&1 | tee "$EXP/logs/vi_phase2_sft_expert_full.log"
else
  OUT="$SMOKE_OUT"
  [[ ! -e "$OUT/intervention_rollouts.jsonl" ]] || { echo "Smoke output already exists; refusing overwrite." >&2; exit 2; }
  "$PYTHON" "$PHASE2/run_information_pilot.py" \
    --config "$CONFIG" --selection "$SELECTION" --data "$DATA" \
    --smoke-cases 2 --output-dir "$OUT" 2>&1 | tee "$EXP/logs/vi_phase2_sft_expert_smoke.log"
fi
