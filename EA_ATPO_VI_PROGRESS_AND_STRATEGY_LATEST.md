# EA-ATPO / VI-EA-ATPO Progress and Strategy — Latest

**Updated:** 2026-11-19 (Asia/Shanghai)
**Repository:** `/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO_V1`
**Branch / starting commit for this update:** `ea-atpo-v1` / `26a6748`

> The dated history below retains earlier plans and results. The last dated execution section is the current source of truth for the multi-turn AWQ SFT run; it supersedes earlier BF16-first and 1,000/300 data plans.

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
- At the final code review, the Qwen3-32B download was still resumable and incomplete: about 14 GB downloaded, with only two partial weight shards visible and no complete index/tokenizer. The downloader is already running; do not launch a concurrent copy. Resume later with `bash experiments/expert_sft/scripts/download_qwen3_32b.sh` if it exits.
- Latest spot-check of eligible GPUs (0, 2, 3, 4, 5, 6; GPUs 1 and 7 remain excluded): GPU0 had 136,099 MiB free at 39% utilization; GPU2 had 63,893 MiB free; GPU3 had 99,355 MiB free at 93%; GPUs4/5/6 had 113,999/118,663/118,661 MiB free at 100/93/93% utilization. None met the reserved 150,000 MiB and <=20% utilization gate. Existing work was not stopped or altered. BF16 capability has not yet been verified on a free card.

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

> Continue from the latest dated section in `EA_ATPO_VI_PROGRESS_AND_STRATEGY_LATEST.md` and the current execution report under `experiments/expert_sft/multiturn_dialogue/outputs/data_audit_20261119Tfinal/`. Preserve the frozen 70/15/15 case split and unopened 400-case holdout. Use the existing Qwen3-32B-AWQ LoRA environment and run its real 10-step gradient/save/reload smoke only after an allowed card passes the memory and no-other-process gate. Do not stop existing GPU work. Start formal SFT only after that smoke passes; do not open the holdout or start Router/PPO/ATPO RL prematurely.

## 2026-11-19 steering update: multi-turn states and AWQ LoRA first

This update supersedes the earlier SFT model order. The Expert's main SFT data now comes from existing multi-turn ATPO public-state snapshots, not the earlier 1,000/300 one-turn answer-only corpus. The earlier corpus remains intact as a separate experiment.

- New frozen case-level split: 70 Train / 15 Validation / 15 Internal Test cases. From 100 historical Pilot cases the builder creates 205 / 42 / 44 current-public-state samples; one example per state. Gold appears only as the assistant answer target. Hidden facts, future replies, expert advice and Base final answers are excluded from input.
- First candidate is the existing locked `Qwen/Qwen3-32B-AWQ`, tested only through a new isolated LoRA-training environment with GPTQModel `torch_awq`. The BF16 Qwen3-32B path remains the fallback only if a real AWQ gradient/save/reload smoke demonstrates a model/backend incompatibility. A busy GPU is not an AWQ incompatibility.
- Added `experiments/expert_sft/multiturn_dialogue/`: data builder, AWQ/BF16 configs, isolated environment setup, 10-step gradient/save/reload smoke, training script, validation loss checkpoint lock, same-prompt evaluator, case-cluster paired analysis, clean-holdout gate, source-lock and recovery manifests.
- CPU tokenizer/chat-template check passed for all 291 samples; max full token lengths were 346 (train), 322 (validation), and 299 (internal test), below the 4096 limit. Case IDs are disjoint. No AWQ model load, LoRA injection, backward pass, optimizer update, adapter round-trip, SFT, or paired model evaluation has run.
- At the last resource check, all eligible GPUs 0,2,3,4,5,6 had other compute processes and failed the no-process/150,000-MiB-free/20%-utilization gate. GPU1 and GPU7 remain excluded. No active process/service was touched.
- Safe cleanup replaced only exact SHA-256 duplicate Parquets/metadata outputs with relative links while preserving original paths. Unique checkpoints and active service compile caches remain untouched. Generated medical JSONL/Parquet remains local and ignored; the source inputs and regeneration metadata are already in the repository.
- The AWQ compatibility report is `experiments/expert_sft/multiturn_dialogue/reports/MULTITURN_SFT_AND_DATA_RECOVERY_20261119.md`. GitHub upload should contain code, configs, source/model hash locks, split counts and recovery docs, not weights, case text, Gold-bearing SFT JSONL, caches or optimizer states.

Next: use `bash experiments/expert_sft/multiturn_dialogue/scripts/setup_awq_env.sh`, then wait for a safe card and run `EA_ATPO_AWQ_GPU=<eligible-card> bash experiments/expert_sft/multiturn_dialogue/scripts/run_awq_lora_smoke.sh`. Only a passing real 10-step AWQ adapter round-trip smoke unlocks the AWQ formal SFT runner. Do not materialize the fresh 400-case holdout until the formal validation checkpoint lock exists. No Router/ATPO RL is authorized by this update.

## 2026-11-19 cleanup and GitHub recovery bundle

- Exact-SHA deduplication is now repeatable with `experiments/expert_sft/multiturn_dialogue/scripts/deduplicate_project_artifacts.py`. It preserves all original paths as relative symlinks and records hashes. It covers the 22 repeated Attempt Parquet paths, duplicate selection/metric files, the duplicate ModelScope package-cache body, and identical tokenizer/config assets in the Attempt09 recovery tree. Estimated duplicate payload shared: 54,332,055 bytes. Two stale Python bytecode files (104,507 bytes) and the task-created 15 MB dependency/tokenizer cache were removed.
- Unique Attempt06/07/09 model and optimizer checkpoints, clinical rollout records, logs, and potentially active CUDA/Triton caches remain intact. GPU1 and GPU7 remain excluded. `.task_cache` and service runtime/pid paths are now ignored by Git.
- The frozen source state/trajectory JSONL and case split are already present on the GitHub branch, so the 291-row generated SFT JSONLs can be reconstructed from the committed builder and local hashes; the case text and Gold-bearing derivatives are not duplicated into the upload.
- The AWQ/LoRA compatibility test, 10-step SFT smoke, formal SFT, and SFT model evaluation remain unrun. The upload is recovery code/config/manifest/report material only; no model or optimizer checkpoint is being pushed.
- GitHub recovery bundle successfully pushed to `ea-atpo-v1` at commit `a3f4c4f` (75 files, 1,681,580 bytes). The remote branch now contains the SFT recovery path, historical router/VOI summaries, and safe aggregate result tables; raw generated case text, per-case prompts, model/checkpoint weights, and runtime caches remain outside Git.

## 2026-11-19 execution update: restored-source audit and AWQ training preflight

- The official source audit read the 14,256-row `merged_train_dataset.jsonl` and all four restored Parquet variants. The original ATPO rows are single-turn (`system,user`); the available multi-turn examples are the existing 100-case Base/Patient simulator Pilot, not human transcripts. All Parquet rows map back to source index/prompt/label. The full source has 17 exact duplicate prompt texts; none occur among the selected 100 Pilot cases.
- Revalidated 291 public states and the case-isolated SFT files: Train 205 examples/70 cases, Validation 42/15, Internal Test 44/15. All current user inputs exactly reconstruct from their public-state prefix, all answer-only A–D targets match source labels, split overlap is zero, and the 400-case fresh holdout remains unopened.
- The deepest archived public state averages 1.91 patient-reply turns per case (using the deepest snapshot to avoid counting prior replies again); each case has 2.91 snapshots on average.
- Installed the isolated AWQ environment at `experiments/expert_sft/multiturn_dialogue/.venv_awq`; package resolution and `pip check` pass. Config/tokenizer-only preflight under Transformers 5.22.2 loads `model_type=qwen3`, `Qwen3ForCausalLM`, AWQ 4-bit config, and accepts the `torch_awq` backend. CPU tokenization/chat-template prefix checks pass for all 291 samples (maximum 346 tokens). This is not a model load or LoRA compatibility result.
- Added and ran the repeatable CPU-only `scripts/preflight_awq_software.py`; it verifies the local source-lock inventory/small asset hashes and all three tokenized splits without loading weights or touching CUDA.
- A fresh GPU snapshot still has no eligible card: every allowed GPU (0,2,3,4,5,6) has another compute process; GPU6 has 166,451 MiB free and 0% utilization but is occupied by a process, while other allowed cards have less than 150,000 MiB free or high utilization. GPUs1/7 remain excluded. No process was touched and no model weights were loaded.
- AWQ LoRA gradient/optimizer/save/reload smoke, formal SFT, paired model evaluation, and holdout evaluation are still **not run**. No checkpoint or SFT metric exists. BF16 fallback remains incomplete (3/17 shards; missing full index/tokenizer), and is not being downloaded because AWQ has not failed a real model-level compatibility check.
- The previous user-owned uncommitted `collect_counterfactuals.py` change remains untouched. Final safe aggregate evidence is under `experiments/expert_sft/multiturn_dialogue/outputs/data_audit_20261119Tfinal/`; generated medical JSONL, weights, caches and checkpoints are excluded from Git.

Next action: wait until an allowed GPU has at least 150,000 MiB free, utilization no higher than 20%, and no other compute process. Then run the already-prepared 10-step AWQ smoke. Formal SFT remains gated on a fully passing model-level adapter round-trip; do not open the fresh holdout or start Router/RL before that.
