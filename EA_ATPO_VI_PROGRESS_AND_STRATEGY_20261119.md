# EA-ATPO / VI-EA-ATPO research progress archive — 2026-11-19

This date archive records the latest continuation of `EA_ATPO_VI_PROGRESS_AND_STRATEGY_LATEST.md`. The earlier progress history remains in that file; this archive captures only the new execution evidence and current gate.

## Current route and state

The active route remains Domain-SFT Expert → paired Base/Expert evaluation → Router → ASK/CONSULT → VOI/search → gated ATPO RL. This run completed a data audit and an AWQ software/tokenizer preflight. It did **not** load model weights or train. No new model performance or Router result exists.

## Data audit

- Official ATPO source: 14,256 indexed rows, all single-turn `system,user`; all labels parse A–D. Four recovered Parquet files each have 14,256 unique indexed rows and match the source prompt and label row-for-row. The source has 17 repeated full prompt texts; none is in the selected 100-case Pilot set.
- Available multi-turn material: 100 Base/Patient simulator Pilot cases, 291 valid public states, 191 states with patient replies and 283 reply-turn appearances across snapshots. These are simulated conversations, not human transcripts.
- SFT state samples: Train 205/70 cases, Validation 42/15, Internal Test 44/15. The split is by `case_id`, with no overlap. All 291 user inputs match the corresponding public state exactly; all assistant targets are answer-only and match the official labels. The fresh 400-case within-dataset holdout is disjoint and remains unopened.
- Pilot rollout quality: 3,492 action rollouts; 3,465 marked valid; 27 invalid (ANSWER 0, ASK 5, CONSULT 8, DEFER 14). All trajectory Gold labels match the source; final option parse is 99.23%.

## Model and environment

- Candidate: existing `Qwen/Qwen3-32B-AWQ`, HF revision `0499c3ac83fdef8810b907a23894ba91e95eddd8`; ModelScope revision `c14568581d436715d61d40f471dee3cac433d1d9`. The existing model files remain in place and unchanged.
- Installed isolated training environment: Torch 2.15.0+cu132, Transformers 5.22.2, PEFT 0.21.3, Accelerate 1.15.0, GPTQModel 7.12.0, Optimum 2.3.0, Safetensors 0.8.0. `pip check` passed. Local config loads as Qwen3/AWQ and the script's `torch_awq` config mutation succeeds.
- Reusable CPU-only preflight `experiments/expert_sft/multiturn_dialogue/scripts/preflight_awq_software.py` passed and records sanitized evidence; it checks the local AWQ asset inventory/small-file hashes and all 291 tokenizer examples without loading model weights or touching CUDA.
- CPU-only chat-template/tokenization check passed for all 291 SFT examples: all prompt sequences are prefixes of full sequences, every target is valid, maximum sequence length is 346 tokens. This verifies data/template handling only.
- BF16 fallback is incomplete (3 of 17 shards; missing index/tokenizer) and has not been downloaded.

## GPU decision and training gate

GPU1 and GPU7 remain excluded. On the last snapshot all eligible GPUs 0,2,3,4,5,6 had existing compute processes. Free memory/utilization: GPU0 128,355 MiB / 87%; GPU2 127,675 MiB / 88%; GPU3 107,389 MiB / 16%; GPU4 51,203 MiB / 0%; GPU5 49,611 MiB / 100%; GPU6 166,451 MiB / 0%. The runner requires at least 150,000 MiB free, at most 20% utilization, and no other compute process; GPU6 therefore fails despite its high nominal free memory. No existing process was stopped or modified.

Consequently, AWQ weights were not loaded; LoRA injection, backward, gradient checks, optimizer update, adapter save/reload, 10-step smoke and formal SFT are **not run**. No checkpoint, accuracy, helpful/harmful rate or expert complementarity result may be reported from this stage. The environment and all its caches are under project PFS; no large artifact was written to the system disk.

## Recovery evidence

- Aggregate data audit: `experiments/expert_sft/multiturn_dialogue/outputs/data_audit_20261119Tfinal/AUDIT_REPORT.md`
- Integrity/split details: adjacent `data_audit.json`, `split_summary.csv`, and `rollout_validity_by_action.csv`
- AWQ tokenizer-only test: adjacent `tokenizer_cpu_smoke.json`
- Transformers/GPTQModel local config preflight: adjacent `awq_software_preflight.json`
- Full execution record: `experiments/expert_sft/multiturn_dialogue/outputs/data_audit_20261119Tfinal/EXECUTION_STATUS.md`
- Resolved environment: `experiments/expert_sft/multiturn_dialogue/environment.lock.txt`

## Next authorized operation

When a permitted card passes the existing memory/process gate, run:

```bash
EA_ATPO_AWQ_GPU=<eligible-card> \
  bash experiments/expert_sft/multiturn_dialogue/scripts/run_awq_lora_smoke.sh
```

Only a successful real 10-step smoke, including nonzero LoRA gradients, changed parameters, adapter save/reload equivalence and finite loss, unlocks formal SFT. Select a checkpoint using validation only, then run the paired Base/original-AWQ/SFT evaluation. Keep the 400-case holdout unopened until the validation checkpoint is locked. Do not start Router or PPO/ATPO RL from software installation or tokenizer results.

## Repository hygiene

Preserve the local user edit to `ea_atpo_stage/recipe/ea_atpo/collect_counterfactuals.py`. Do not commit `.venv_awq`, caches, generated case JSONL/Parquet, model weights, adapters, optimizer states, credentials or per-case/holdout predictions. Only code, lockfiles, hashes and aggregate reports are eligible for GitHub.
