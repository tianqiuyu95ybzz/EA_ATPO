# EA-ATPO / VI-EA-ATPO Progress and Strategy — Latest

**Updated:** 2026-11-19 (Asia/Shanghai)
**Repository:** `/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO_V1`
**Branch / base HEAD:** `ea-atpo-v1` / `2aebfa8ce33db872b9b4c204377a4a11cfb09c97`

## Current research route

The active route is **Domain SFT Expert → paired Base/Expert evaluation → cost-sensitive Router → ASK/CONSULT pilot → VOI estimator → gated tree-search experiment → gated ATPO RL → fresh case generalization**.

The present turn implements the data and code path through SFT, paired evaluation, Router, VI service adaptation, and VOI estimator preparation. It does not yet have a completed SFT checkpoint or new-model rollout. No original ATPO training/search code was changed and no RL was started.

## Prior evidence retained

- Historical Base/Expert pairing: 1,700 cases, split 1,000/300/400. The 400-case historical holdout has been opened in earlier analyses and is **not** a clean test for this route.
- Historical Qwen3-32B-AWQ vs Qwen3-8B: Base 48.50%, Expert 57.75%; 72 helpful, 35 harmful, 159 both correct, 134 both wrong. Uncertainty→Helpful AUROC 0.710. At cost 0.60 the old learned policy deferred 0%.
- The old Expert is pinned to Qwen3-32B-AWQ revision `0499c3ac83fdef8810b907a23894ba91e95eddd8`; it is a historical inference baseline, not an SFT source checkpoint.
- The learned Router did not generalize reliably in prior work. Neither those old labels nor the historical holdout are reused as new Expert supervision or as the new test.
- VI Phase 2 already has a frozen 100 train / 50 validation cohort and a real rollout runner. Its prior two-case engineering smoke found a Patient fact-recall miss, so the 150-case pilot was marked not ready. The cohort is retained; do not rebuild it. Recheck Patient fidelity in the new Expert smoke.
- Phase 2 VOI results, VOI search benefit, and RL benefit are not established. No result in this file should be read as evidence that SFT or VOI works.

## New SFT data and frozen case roles

Official ATPO data at commit `25af83688c06917f94245a4db06be7d0924c090c` was restored under `experiments/expert_sft/data/`. Its adapted train Parquet SHA-256 is `c586d8653460e8bef102f7c6231a1356dbfa88b5b35c3538219724efd4c1a5c5` (14,256 rows). The SFT builder matched the pinned prior selection by case, prompt, and source-row hashes.

- SFT Train: 1,000 cases; JSONL SHA-256 `8390c9c383610c2b0358f393d8f0fb109fa76b0d53ffbabbc751d9d97e879834`.
- SFT Validation: 300 cases; JSONL SHA-256 `e2d406f744afce1b7ba9984053492c5fd39ae8085dc2092ccd86c83839e4042a`.
- Both splits contain only system/user input plus one assistant answer target. The scan found no prior assistant response, no hidden-fact fields, and no answer-like content before the target. Targets are only the official A–D answer letter; no Gold reasoning was invented. Label integrity was checked, not clinically re-adjudicated.
- A new 400-case case-level cohort was frozen from unused ATPO train rows with seed `20261120`, before SFT checkpoint selection. It excludes historical IDs/hashes/prompts, all Attempt04 cases, frozen VI Phase 2 cases, and SFT Train/Validation. A second pass reading only the Parquet `prompt` column reproduced the exact selection-file SHA-256 `13ed6251d05a77f016746d44a3a13d1a48271c7eeeb9395caecca35469c7a005`; no Gold/atomic-fact fields are used to select it. Its case-hash-list SHA-256 is `fbb88ee4116b8a269bfd795b74fd9cc2b68e84ac3fd9a2c9ead65184036940f8`.
- That cohort is a new **within-dataset ATPO-train holdout**, not an external clinical cohort and not the official ATPO test split. Gold has not been materialized for it. Gold may be materialized only after the SFT checkpoint and train/validation Router are frozen.
- Router training on the prescribed SFT Train outputs is in-sample for the fitted Expert, and the SFT Validation cases serve both checkpoint selection and threshold calibration. Treat those development metrics as potentially optimistic; use only the untouched fresh 400-case evaluation as confirmatory. Cross-fitting is a possible later correction if the first controlled run shows this bias limits learning.

## Model and environment lock

- Intended training Expert: unquantized official `Qwen/Qwen3-32B` BF16, revision `9216db5781bf21249d130ec9da846c4624c16137`, with LoRA r=16, alpha=32, dropout 0.05. The existing AWQ inference checkpoint is not used for QLoRA or SFT.
- Base: fixed official `Qwen/Qwen3-8B`, revision `b968826d9c46dd6066d109eabc6255188de91218`; local project copy is not downloaded yet.
- HF model page describes Qwen3-32B as a native chat model and Qwen3 architecture; the actual local tokenizer/template and full weight layout remain unverified until the pinned download completes. See [official Qwen3-32B model card](https://huggingface.co/Qwen/Qwen3-32B).
- Current project `.venv`: Python 3.12, torch 2.7.1+cu128, transformers 4.54.1, PEFT 0.21.2, accelerate 1.15.0, datasets 4.0.0, pandas 2.3.3, pyarrow 19.0.1, huggingface_hub 0.36.2. TRL and bitsandbytes are absent; the planned BF16 LoRA path uses HF Trainer/PEFT and does not require either.
- All new model downloads, HF/datasets/XDG/Triton/Torch caches, temporary files, logs, checkpoints, and evaluation outputs are under this project’s `experiments/expert_sft/` tree. No new file is directed to the system disk.
- A final code review fixed a Trainer placement issue: the SFT script now explicitly moves the LoRA model to the sole CUDA device exposed by the launcher before the gradient preflight. Safe launchers permit an explicit physical GPU override but exclude GPUs 1 and 7; no GPU work was started during this code correction.
- At the latest audit, the Qwen3-32B download was still resumable and incomplete: 8 partial files, about 8.8 GB downloaded, no complete weight shards/index. The downloader command is `bash experiments/expert_sft/scripts/download_qwen3_32b.sh` after the current downloader exits; do not launch a concurrent copy.
- Latest GPU snapshot: all eight GPUs had at least one existing compute process; free memory ranged about 43–116 GiB. None met the reserved 150,000 MiB, <=20% utilization, no-compute-process SFT gate. GPU processes were not stopped or altered. BF16 capability has not yet been verified on a free card.

## Implemented under `experiments/expert_sft/`

- Pinned configs for the SFT Expert, original BF16 model, and separate VI SFT Expert service.
- Official-data restore and hash-checked answer-only 1k/300 SFT data builder.
- Local tokenizer/assistant-mask smoke, one-batch gradient preflight, 10-step LoRA smoke, 3-epoch formal SFT runner, validation-only checkpoint selector, and generation evaluator.
- Workspace-local 32B and 8B download scripts with pinned revisions and resumable HF cache.
- Case-level paired Base evaluation with A–D probability features, strict pairing/identity join, Train/Validation Router fitter, and one-shot fresh-test evaluator.
- Router comparisons: uncertainty threshold, logistic relative gain, Ridge relative gain, small MLP relative gain, frozen historical Learned Router, Always Answer, Always DEFER, budget-matched Random DEFER, and test-only Oracle upper bound. Costs are fixed at 0/.05/.10/.20/.40/.60. The test evaluator locks on the frozen policy, checks case/hash isolation, records a one-time test-open marker, and computes case bootstrap intervals. Router input features are public prompt features and Base-only pre-decision outputs; Gold and Expert outputs are labels only.
- New vLLM LoRA service config/launcher on a separate port for the selected Domain-SFT Expert, plus a separate Phase 2 pilot config. It does not stop or reconfigure the existing Base/Patient/AWQ services. The fixed Phase 2 queue remains in use.
- The answer-only SFT target is terminal-answer compatible but does not teach separate CONSULT-hint behavior. Treat CONSULT non-answer leakage as a hard smoke gate; no synthetic hint reasoning is added.
- A gated train/validation VOI-estimator script based on the fixed `ASK_DECISION` / `CONSULT_DECISION` continuation. It will refuse incomplete/error rollouts and never opens test. It has not been run.
- Environment audit script/report. `bash -n` and Python AST syntax checks passed for scripts present at the time; no CUDA/model smoke ran.

## Actual run state

| Stage | State | Evidence / next gate |
|---|---|---|
| ATPO data restore | Complete | `data/reconstructed_compat/train.pilot.parquet`, SHA above |
| SFT data build/audit | Complete | `data/train_sft.jsonl`, `data/validation_sft.jsonl`, `reports/SFT_DATA_AUDIT.md` |
| Fresh 400 selection | Frozen | `data/fresh_test_selection.json`; no Gold materialized |
| Qwen3-32B download | Running/resumable | `assets/models/Qwen3-32B`; incomplete weights |
| Qwen3-8B download | Not started | Start after 32B transfer to avoid competing downloads |
| Tokenization smoke | Not run | Requires complete 32B tokenizer files |
| One-batch/backward and 10-step SFT smoke | Not run | Requires complete model + safe, process-free GPU |
| Formal LoRA SFT and checkpoint selection | Not run | Only start after smoke passes and GPU gate passes |
| Base/Expert paired Train/Validation | Not run | Run `run_paired_train_validation.sh` after Base and selected Expert exist |
| New Expert Router fit | Not run | Frozen only from new SFT paired Train/Validation; test remains unopened |
| Fresh holdout evaluation | Not run | One run only after checkpoint and Router policy locks |
| SFT Expert Phase 2 ASK/CONSULT | Not run | Start new service; 2-case smoke first; old Patient fact miss must be reviewed |
| VOI estimator / search | Not run | Fit only after acceptable real Phase 2 rollout; tree integration remains gated |
| PPO/ATPO RL | Not started | No RL until positive predictive VOI and search parity gates pass |

## Next commands (in order)

1. Let the current Qwen3-32B downloader finish; if interrupted, resume with:
   `bash experiments/expert_sft/scripts/download_qwen3_32b.sh`
2. Verify the completed checkpoint and local tokenizer:
   `./.venv/bin/python experiments/expert_sft/scripts/verify_model_assets.py --model experiments/expert_sft/assets/models/Qwen3-32B --write-manifest experiments/expert_sft/data/model_asset_manifest.json`
3. When a GPU passes the gate, run:
   `EA_ATPO_SFT_PHYSICAL_GPU=<safe-physical-GPU> bash experiments/expert_sft/scripts/run_sft_smoke.sh`
   It checks BF16/tokenization, a finite one-batch LoRA gradient, 10 optimizer steps, and five engineering-format generations. If this passes, formal training is authorized by the task:
   `EA_ATPO_SFT_PHYSICAL_GPU=<safe-physical-GPU> bash experiments/expert_sft/scripts/run_formal_sft.sh`
4. Download the pinned Base model inside the workspace:
   `bash experiments/expert_sft/scripts/download_qwen3_8b.sh`
5. Run only new paired Train/Validation and freeze the Router:
   `bash experiments/expert_sft/scripts/run_paired_train_validation.sh`
6. Only after checkpoint and policy locks exist, run the one-shot 400-case evaluation:
   `bash experiments/expert_sft/scripts/run_fresh_test_once.sh`
7. For VI Phase 2, wait for a free GPU and verify existing Base/Patient endpoints; then run `bash experiments/expert_sft/scripts/start_sft_expert_vllm.sh`, `bash experiments/expert_sft/scripts/run_vi_sft_expert_phase2.sh smoke`. Review patient fact fidelity, CONSULT answer leakage, information isolation, and costs in `smoke_quality_review.json`. Only if all pass, run `bash experiments/expert_sft/scripts/run_vi_sft_expert_phase2.sh pilot`.
8. If the real 150-case rollout completes, fit development-only VOI estimates:
   `./.venv/bin/python experiments/expert_sft/scripts/fit_voi_estimator.py --rollouts experiments/expert_sft/evaluations/vi_phase2_sft_expert_full/intervention_rollouts.jsonl --manifest experiments/expert_sft/evaluations/vi_phase2_sft_expert_full/execution_manifest.json --output-dir experiments/expert_sft/evaluations/voi_validation`
   No search integration or RL follows unless validation shows positive, predictable utility and `lambda_voi=0` parity is validated.

## Context recovery prompt

> Continue the EA-ATPO Domain-SFT task from `EA_ATPO_VI_PROGRESS_AND_STRATEGY_LATEST.md`. Keep all artifacts under `/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO_V1`. Preserve the frozen 1,000/300 SFT split and fresh 400-case holdout; never open the fresh test before the validation-selected SFT checkpoint and Router lock are written. Do not stop existing GPU jobs or services. Resume the pinned Qwen3-32B BF16 revision download, recheck the process-free GPU gate, run the requested SFT smoke, and start formal SFT only after the smoke passes. No PPO/ATPO RL until the new Expert, paired Router, Patient/CONSULT smoke, and VOI validation gates pass.
