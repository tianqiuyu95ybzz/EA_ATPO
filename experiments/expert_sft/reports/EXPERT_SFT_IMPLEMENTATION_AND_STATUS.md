# EA-ATPO Domain-SFT Expert — Implementation and Current Status

**Date:** 2026-11-19
**Project root:** `/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO_V1`
**Current phase:** Data and pipeline implementation complete; Qwen3-32B download and safe-GPU wait in progress.

## Result in one paragraph

The official ATPO train source was restored and verified. The frozen 1,000/300 Train/Validation cases are converted to answer-only Expert SFT examples, and a new 400-case within-ATPO-train holdout has been frozen without using Gold or hidden facts for selection. LoRA SFT, validation checkpoint selection, paired Base/Expert evaluation, train/validation Router fitting, a one-shot test evaluator, VI SFT Expert service configuration, and a gated VOI estimator are implemented. **No tokenizer/model smoke, training update, SFT checkpoint, paired model evaluation, Router metric, Phase 2 rollout, VOI result, or RL result exists yet.** The BF16 model is still downloading, and no GPU currently passes the no-other-process SFT gate. No existing GPU process or model service was stopped.

## Data and leakage controls

- Source: official ATPO train data, commit `25af83688c06917f94245a4db06be7d0924c090c`.
- Adapted Parquet SHA-256: `c586d8653460e8bef102f7c6231a1356dbfa88b5b35c3538219724efd4c1a5c5`.
- Train: 1,000 cases; SHA-256 `8390c9c383610c2b0358f393d8f0fb109fa76b0d53ffbabbc751d9d97e879834`.
- Validation: 300 cases; SHA-256 `e2d406f744afce1b7ba9984053492c5fd39ae8085dc2092ccd86c83839e4042a`.
- Both JSONL splits were scanned: 1 system + 1 user + 1 target assistant message per case; no prior assistant output and no hidden-fact field in the constructed message. Targets are only source Gold letters in `Expert Final Answer: X` format. Gold label integrity was checked; labels were not clinically re-adjudicated.
- Fresh test selection: 400 unused ATPO-train rows, seed `20261120`, selection SHA-256 `13ed6251d05a77f016746d44a3a13d1a48271c7eeeb9395caecca35469c7a005`. It excludes prior model-selection and analysis cases plus the frozen VI Phase 2 queue. Reproducing the same SHA after reading only `prompt` verifies that the selection does not need Gold or atomic facts. It is a within-dataset holdout, not external clinical data. Gold has not been materialized for it.
- The historical 400-case holdout remains excluded from new training/checkpoint selection and is not treated as an independent test.
- The specified SFT `Train` predictions are in-sample for the fitted Expert, and the 300-case SFT `Validation` split is used for checkpoint selection and later Router threshold calibration. These development results may be optimistic. The frozen fresh 400-case evaluation is the only confirmatory Router result in this implementation; a positive result must be based there. Cross-fitted Expert predictions could reduce second-stage training-label optimism, but are not added as another SFT method in this scoped run.
- The answer-only SFT target is designed for terminal Expert answers. It does not teach separate consultation-hint behavior. The Phase 2 smoke must verify that the action-specific CONSULT prompt still yields a non-terminal, non-answer-leaking suggestion. No synthetic hint/reasoning target is being added.

## Model lock and environment

- SFT model: `Qwen/Qwen3-32B`, BF16, revision `9216db5781bf21249d130ec9da846c4624c16137`; original unquantized weights. AWQ weights are not passed to the SFT trainer.
- Base: `Qwen/Qwen3-8B`, revision `b968826d9c46dd6066d109eabc6255188de91218`; the workspace-local Base download is not yet complete.
- Historical AWQ Expert (`0499c3ac83fdef8810b907a23894ba91e95eddd8`) is kept only as historical context.
- Training environment package metadata: Python 3.12; torch 2.7.1+cu128; transformers 4.54.1; PEFT 0.21.2; accelerate 1.15.0; datasets 4.0.0; pandas 2.3.3; pyarrow 19.0.1; huggingface_hub 0.36.2. `trl` and `bitsandbytes` are absent and not needed for the chosen full-BF16-base LoRA path.
- HF/model, dataset, XDG, Triton, Torch, vLLM and temporary caches are configured under `experiments/expert_sft/`. The workspace reports over 42 TiB free.
- Download is resumable but incomplete. Latest environment audit found 8 partial files totaling about 8.8 GB; the official model shard index/tokenizer completeness check has not passed.
- GPU audit: all eight B20Z cards had one or more existing compute processes and fell below at least one configured gate (150,000 MiB free, <=20% utilization, zero existing compute processes). A torch import for read-only inventory was interrupted because import/path scanning was very slow; the audit now uses package metadata and `nvidia-smi`. Actual BF16 support will be checked by smoke on a genuinely free GPU.

## Implemented files

All listed files are under `experiments/expert_sft/`.

- `configs/domain_sft.json`: pinned model/data, answer-only target, LoRA and resource settings.
- `scripts/restore_atpo_train_data.sh`, `scripts/build_sft_dataset.py`: official source restore and checked 1,000/300 SFT construction.
- `scripts/download_qwen3_32b.sh`, `scripts/download_qwen3_8b.sh`, `scripts/verify_model_assets.py`: pinned resumable downloads and non-AWQ asset verification.
- `scripts/tokenization_smoke.py`, `scripts/train_lora_sft.py`, `scripts/run_sft_smoke.sh`: local chat-template/mask check, one-batch gradient check, 10 optimizer steps, 5-case protocol check, peak-memory and model/config manifests.
- `scripts/run_formal_sft.sh`, `scripts/evaluate_sft_checkpoint.py`, `scripts/select_validation_checkpoint.py`: three-epoch run, validation generation, and a fixed validation-only checkpoint lock. A post-load GPU reserve/PID check is performed before optimizer updates.
- `scripts/evaluate_base_paired.py`, `scripts/assemble_paired_results.py`: fixed Base inference, Base option-probability features, strict case/hash/gold pairing, no silent mismatches.
- `scripts/fit_cost_sensitive_router.py`: uncertainty, logistic, Ridge, MLP, and frozen historical Learned Router; validation-only cost thresholds for `[0,.05,.10,.20,.40,.60]`; no test read.
- `scripts/select_fresh_holdout.py`, `scripts/build_fresh_test_data.py`, `scripts/evaluate_frozen_router.py`, `scripts/run_paired_train_validation.sh`, `scripts/run_fresh_test_once.sh`: fresh cohort isolation, policy lock, case-level bootstrap and Oracle-only upper bound. Test evaluator refuses reuse after the first opened test run.
- `configs/model_services_sft_expert.yaml`, `configs/vi_ea_atpo_sft_expert_pilot.json`, `scripts/start_sft_expert_vllm.sh`, `scripts/run_vi_sft_expert_phase2.sh`: separate Qwen3-32B+LoRA Expert service and a copy of the frozen VI queue config; existing model services are not modified or stopped.
- `scripts/fit_voi_estimator.py`: train/validation-only, case-grouped, cost-inclusive VOI fitting; no Oracle input, no test access, no search integration.
- `scripts/audit_environment.py` and `reports/environment_audit_20261119.{json,md}`.
- Root progress/strategy files: `EA_ATPO_VI_PROGRESS_AND_STRATEGY_LATEST.md` and `EA_ATPO_VI_PROGRESS_AND_STRATEGY_20261119.md`.

No ATPO Actor, Critic, PPO loss, Reward, tree-search, or policy code was modified. No network was trained.

## Commands actually run

```bash
bash experiments/expert_sft/scripts/restore_atpo_train_data.sh
EA_ATPO_RECON_DIR=experiments/expert_sft/data/reconstructed_compat EA_ATPO_RECON_PYTHON=/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/.venv-real-pilot/bin/python bash experiments/expert_sft/scripts/restore_atpo_train_data.sh
./.venv/bin/python experiments/expert_sft/scripts/build_sft_dataset.py
./.venv/bin/python experiments/expert_sft/scripts/select_fresh_holdout.py
./.venv/bin/python experiments/expert_sft/scripts/select_fresh_holdout.py --output experiments/expert_sft/data/fresh_test_selection_prompt_only_verify.json
./.venv/bin/python experiments/expert_sft/scripts/audit_environment.py
bash -n experiments/expert_sft/scripts/*.sh
```

The first Parquet serialization produced environment-dependent bytes. Rebuilding with the already-used project-adjacent ATPO runtime reproduced the frozen hash exactly. A one-character config hash typo was corrected before SFT data generation; the builder then completed and verified all 1,300 rows. The latest prompt-only test-selection reproduction returned the same frozen selection SHA.

The active downloader command is:

```bash
bash experiments/expert_sft/scripts/download_qwen3_32b.sh
```

It is already running resumably under the project tree. Do not start another concurrent download; the same command resumes after that process exits.

## Not run; no performance claims

- Qwen3-32B tokenizer/chat-template local verification, tokenization smoke, one-batch forward/backward, 10-step training, formal three-epoch SFT, and checkpoint evaluation: **Not Run** (weights incomplete/GPU safety gate closed).
- Workspace Qwen3-8B download: **Not Run**.
- Base/Expert Train/Validation pairing and Router metrics: **Not Run**.
- New Expert 2-case ASK/CONSULT smoke and 150-case Phase 2: **Not Run**. The previously observed Patient fact-recall miss is an explicit quality gate. A recorded review must confirm patient fact fidelity, CONSULT does not reveal a final answer, information boundaries, protocol, and cost accounting before 150 cases.
- VOI fitting, ATPO tree integration, PPO/ATPO RL, and final generalization: **Not Run**.

## Next steps

1. Allow/resume the current 32B download; verify its pinned revision, full safetensor shard set, tokenizer and chat template.
2. When a process-free GPU passes the gate, run `bash experiments/expert_sft/scripts/run_sft_smoke.sh`; only after it passes, run `bash experiments/expert_sft/scripts/run_formal_sft.sh`.
3. Download Qwen3-8B to this project; pair Train/Validation and freeze Router thresholds; then run the one-shot 400-case within-dataset holdout exactly once.
4. Start the separate SFT Expert service and retain the existing Base/Patient services. Run the frozen two-case VI smoke, review it, and only if it passes run the 150-case queue.
5. Fit VOI on train and validate on validation. Do not change original tree search or start RL until predictive and utility gates are supported.

For a complete restart handoff, read `EA_ATPO_VI_PROGRESS_AND_STRATEGY_LATEST.md`.

## 2026-11-19 final code review addendum

- The first implementation review found that `from_pretrained()` left the SFT model on CPU while the gradient smoke batch was placed on CUDA. The trainer now explicitly moves the LoRA-wrapped model to the one CUDA device exposed by `CUDA_VISIBLE_DEVICES` before the one-batch forward/backward check. No GPU training was run to validate this correction because the model download and resource gate are still incomplete.
- SFT, paired-evaluation, and Expert-service launchers now allow a safe physical GPU override while explicitly rejecting GPUs 1 and 7 per the standing resource instruction. They continue to require 150,000 MiB free, utilization at most 20%, and no compute process before launch; the training script rechecks the physical GPU reserve after model load and gradient preflight.
- Static syntax checks were rerun after this correction. Full CUDA smoke remains **Not Run**.
