# EA-ATPO real-case pilot

Inference-only feasibility experiment. No Router training or ATPO RL training is
performed. Base Agent, Patient Simulator, and Expert Simulator share one
`Qwen/Qwen3-8B` vLLM service; role-specific prompts and context control access.
Patient and Expert are simulated LLM roles, not real doctors.

## Pinned inputs

- ATPO data commit: `25af83688c06917f94245a4db06be7d0924c090c`
- HF model: `Qwen/Qwen3-8B`
- HF model revision: `b968826d9c46dd6066d109eabc6255188de91218`
- Data conversion: the pinned upstream `mix_data_jsonl.py`; the separate
  `adapt_atpo_parquet.py` validates fields and writes Pilot-ready copies.
- Model service: one local vLLM OpenAI-compatible endpoint on `127.0.0.1:30000`.

Raw JSONL, converted and adapted Parquet, model weights, HF cache, Python/uv
cache, CUDA/Triton cache, logs, and results are stored on the project PFS volume.
Source JSONL is preserved. Downloads support restart/resume.

## Setup and run

```bash
cd /pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO
mkdir -p .cache/uv .tmp
export UV_CACHE_DIR="$PWD/.cache/uv"
export TMPDIR="$PWD/.tmp"
uv venv --python python3.12 .venv-real-pilot
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}pypi.tuna.tsinghua.edu.cn"
export no_proxy="$NO_PROXY"
UV_HTTP_TIMEOUT=180 UV_HTTP_RETRIES=10 UV_CONCURRENT_DOWNLOADS=50 \
uv pip install --require-hashes \
  --default-index https://pypi.tuna.tsinghua.edu.cn/simple \
  --python .venv-real-pilot/bin/python \
  -r experiments/real_case_pilot_20261103/runtime-requirements.lock.txt
UV_CACHE_DIR="$PWD/.cache/uv" TMPDIR="$PWD/.tmp" \
uv pip install --require-hashes \
  --default-index https://pypi.tuna.tsinghua.edu.cn/simple \
  --python .venv-real-pilot/bin/python \
  -r experiments/real_case_pilot_20261103/PILOT_ANALYSIS_REQUIREMENTS.lock.txt
EA_ATPO_PILOT_PYTHON="$PWD/.venv-real-pilot/bin/python" \
  bash experiments/real_case_pilot_20261103/restore_public_atpo_data.sh
bash experiments/real_case_pilot_20261103/download_model.sh
bash experiments/real_case_pilot_20261103/start_model_service.sh
bash experiments/real_case_pilot_20261103/run_smoke_test.sh
bash experiments/real_case_pilot_20261103/run_full_pilot.sh
```

The smoke script first exercises all roles against one authentic training row,
then runs 10 cases with one rollout per action and checks paired action coverage
and failures. Action strings use vLLM regex-constrained structured output; this
constrains the requested action format while leaving the model to generate the
question, consultation focus, or answer. The last Base turn requires a terminal
answer. A rollout allows at most one CONSULT, matching the pre-registered
protocol. The full script starts only if smoke completed with all rollouts
valid. It runs 100 cases, seed `20261103`, at most three public states per case,
and three rollouts per action. It resumes from the persisted public-state and
case/state/action JSONL records after interruption.

The runtime lock contains SHA-256 hashes generated from PyPI metadata. The
Tsinghua index above is used only to transport Python wheels in this environment;
`--require-hashes` verifies every selected artifact. Actual package versions,
lock hash, platform, and validation results are recorded in
`runtime_install_manifest.json` and `runtime-environment.freeze.txt`.

## Data contract

`train.parquet` and `test.parquet` are the output of the upstream converter.
The adapter writes `train.pilot.parquet` and `test.pilot.parquet`, and records
row counts, source/output hashes, required fields, choice labels, gold format,
and `atomic_facts` presence in `assets/adapter_manifest.json`. Missing train
facts or malformed labels stop the run; no cases or facts are synthesized.

The Base sees only the public prompt and prior public dialogue. Patient requests
contain the source row's `atomic_facts`; those facts are never sent to Base or
Expert. CONSULT questions are generated from the visible state. Expert receives
only that state and the concrete consultation question, and may return advice
but not a final option. DEFER gives Expert a terminal decision based on the same
public state. Base generation uses Qwen3 with thinking disabled so the visible
action is not lost to an unfinished private-reasoning budget. Gold is read only
after each branch for scoring.

## Outputs

- `outputs/service_role_smoke.jsonl` and manifest/log: four role checks on one
  real case.
- `outputs/smoke/`: 10-case engineering smoke, complete action traces, report,
  summary, and manifest.
- `outputs/full/REAL_PILOT_REPORT.md`
- `outputs/full/pilot_trajectories.jsonl`
- `outputs/full/summary.csv`
- `outputs/full/action_value_analysis.csv`
- `outputs/full/run_manifest.json`
- `outputs/full/accuracy_cost_curve.png` and `.svg`
- `outputs/full/accuracy_deferral_curve.png` and `.svg`
- `outputs/full/action_cost_accuracy_points.csv` and `tradeoff_curve_points.csv`
- `outputs/full/analysis_manifest.json`
- `model_service_config.yaml` and `model_service_manifest.json` in this directory and `outputs/` respectively

The figures use a separate hash-locked Matplotlib analysis dependency set.
The accuracy-cost figure compares the four fixed actions using measured total
tokens. The accuracy-deferral figure uses the post-hoc action-outcome oracle
from the cost-sensitivity grid and is explicitly not a deployable Router result.

Action costs are measured in role token use, patient turns, and latency. The
cost-sensitivity grid is expressed in utility penalty units, not USD; this
single local inference service has no assigned monetary price. No real doctor
participates in these runs.
