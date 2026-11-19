# Multi-turn Expert SFT and Data Recovery Status

**Date:** 2026-11-19  
**Status:** Multi-turn data build and reproducibility scripts prepared; AWQ LoRA compatibility test and all training/evaluation remain unrun.

## Research adjustment

This track replaces the earlier plan to train the DEFER Expert primarily from a small set of one-turn answers. It uses existing ATPO public multi-turn states and trains only the final-answer target with assistant-only causal cross-entropy. No CONSULT-hint target, Router loss, ATPO reward, Actor/Critic update, or RL is included.

## Frozen data and leakage boundary

| Split | Cases | Public state samples | Use |
|---|---:|---:|---|
| Train | 70 | 205 | Expert SFT; in-sample fit reported separately |
| Validation | 15 | 42 | Select checkpoint by assistant-token validation cross-entropy |
| Internal test | 15 | 44 | Development-only diagnostic; not the clean final test |
| Clean holdout | 400 | 400 selected cases | Not materialized until the validation checkpoint lock is written |

The source Pilot contains 100 cases, 291 unique public states and 3,492 action rollouts. Each state has three repetitions of each four-action rollout. Public states contain 0, 2 or 4 dialogue messages (100, 99 and 92 states respectively). The builder checks that the visible `assistant` turn maps to the Base question and the visible `user` turn maps to the Patient reply; only the current state prefix is included. It excludes the original Base system prompt, hidden facts, Expert advice, Base final answers and future replies. Gold is read only as the supervised Assistant target.

Generated files and SHA-256 hashes are in `data/dataset_manifest.json`. The source state/trajectory JSONL and frozen split are already tracked in this repository, so the training JSONL can be rebuilt without uploading a second copy. The upstream ATPO page describes datasets adapted from public benchmarks but does not list an explicit dataset redistribution license in its repository file listing. To keep this research repository portable without republishing question text, the generated medical JSONL/Parquet remains local and ignored; GitHub gets the builder, hashes, split metadata, model lock and commands.

## Model audit and AWQ backend plan

The existing AWQ directory was inspected read-only. It is the complete four-shard `Qwen/Qwen3-32B-AWQ` model (`Qwen3ForCausalLM`, `qwen3`, AWQ 4-bit, group size 128, GEMM, FP16) with tokenizer/chat template. Its pinned Hugging Face revision is `0499c3ac83fdef8810b907a23894ba91e95eddd8`; ModelScope revision is `c14568581d436715d61d40f471dee3cac433d1d9`. The copied `data/qwen3_32b_awq_source_lock.json` includes SHA-256 for all four shards and tokenizer/config files. No AWQ weight was copied, changed, or re-downloaded.

The project `.venv` is not a supported AWQ training environment (`torch 2.7.1+cu128`, Transformers 4.54.1, PEFT 0.21.2; no GPTQModel). GPTQModel's official LoRA guide says AWQ training must set the trainable `torch_awq` backend and import GPTQModel before Transformers loads the model; ordinary inference kernels do not guarantee gradients. Its current 7.12.0 package metadata requires Torch >=2.15.0 and Transformers >=5.22.2 on Python >=3.11. The host driver advertises CUDA 13.2, and the official PyTorch cu132 index exposes Torch 2.15.0. The new isolated-environment installer pins GPTQModel 7.12.0, Torch 2.15.0+cu132, Transformers 5.22.2, PEFT 0.21.3, Optimum 2.3.0 and Accelerate 1.15.0, then records the resolved `pip freeze`. Sources: [GPTQModel LoRA guide](https://github.com/ModelCloud/GPTQModel/blob/main/docs/peft.md), [GPTQModel package requirements](https://github.com/ModelCloud/GPTQModel/blob/main/requirements.txt), [official PyTorch CUDA 13.2 wheel index](https://download.pytorch.org/whl/cu132).

This makes AWQ LoRA a viable candidate to test; it does **not** establish that this model can be trained. The smoke script must complete every requested model-level check: load, LoRA injection, finite forward loss, backward pass, finite nonzero LoRA gradients, changed parameters after an optimizer step, adapter save, adapter reload, and exact adapter tensor round-trip. If that test fails due to a model/backend incompatibility, the separate pinned BF16 fallback is prepared. GPU unavailability alone does not count as an AWQ failure.

## Resource gate at audit time

GPU 1 and GPU 7 remain excluded. At the 2026-11-19 audit, every eligible GPU had an existing compute process and failed the required free-memory/utilization/no-process gate: GPU0 had about 92 GiB free at 81% utilization; GPU2 about 27 GiB at 73%; GPU3 about 8 GiB at 100%; GPU4 about 50 GiB at 93%; GPU5 about 70 GiB at 100%; GPU6 about 32 GiB at 97%. No model load or training was attempted. Existing services/processes were not stopped or modified.

## Work performed and actual status

| Check or run | Result |
|---|---|
| Frozen 70/15/15 case split, 291 state examples and 3,492 source rollouts | Passed data-builder checks |
| Public current-state boundary; Gold target-only; no future reply/hidden-fact input | Passed static data checks and manifest assertions |
| Existing AWQ config, full shard set and source hash lock | Inspected and verified from existing files |
| AWQ tokenizer/chat template and target/prompt prefix | CPU validation passed for all 291 examples; maximum full sequence was 346 tokens |
| AWQ model load through trainable backend | Not run; isolated environment not installed and GPU gate closed |
| LoRA injection / forward / backward / nonzero gradient / optimizer step | Not run |
| Adapter save / reload | Not run |
| 10-step SFT smoke or formal SFT | Not run |
| Base8B / old AWQ / SFT paired predictions | Not run |
| Clean 400-case holdout | Not materialized; checkpoint selection lock does not exist |
| Router or ATPO RL | Not started |

## Exact-duplicate cleanup

Only byte-identical copies were deduplicated. Four `train.parquet`/`test.parquet` hash groups (22 paths) now point through relative symlinks to one project-local copy per hash; historical attempt paths still resolve. Exact duplicate selection/metric files and repeated tokenizer assets in the Attempt09 recovery tree were treated the same way. The cleanup script is idempotent and records hashes, canonical paths, preserved paths, and estimated bytes saved in `data/dedup_manifest.json`: 54,332,055 bytes of duplicate payload are now shared. Two stale Python bytecode files (104,507 bytes) were removed. A temporary dependency/tokenizer cache created during inspection (about 15 MB) was also removed; the setup script will recreate its project-local cache when needed.

No unique model/checkpoint, clinical trajectory, run log or active CUDA/Triton cache was removed. A sampled comparison of the first 8 MiB of Attempt06, Attempt07 and Attempt09 optimizer shards showed different bytes, so those checkpoints were not classified as duplicate files. GPU services may still use their local compile caches, so cache paths were left untouched.

## Recovery commands

```bash
# Regenerate SFT data without overwriting the frozen local copy
python3 experiments/expert_sft/multiturn_dialogue/scripts/build_multiturn_sft_dataset.py \
  --output-dir experiments/expert_sft/multiturn_dialogue/data/rebuilt_YYYYMMDD

# Build isolated training environment (all caches remain under the project)
bash experiments/expert_sft/multiturn_dialogue/scripts/setup_awq_env.sh

# When an eligible GPU passes the gate; do not use GPU 1 or 7
EA_ATPO_AWQ_GPU=3 bash experiments/expert_sft/multiturn_dialogue/scripts/run_awq_lora_smoke.sh

# Formal SFT is gated on a successful 10-step adapter round-trip smoke
EA_ATPO_AWQ_GPU=3 bash experiments/expert_sft/multiturn_dialogue/scripts/run_awq_lora_formal.sh
```

The clean holdout builder requires `selected_checkpoint_lock.json` from a completed validation-only training run. The final paired evaluation uses identical prompts and greedy decoding for Base8B, original AWQ, and the selected SFT Expert. It reports state-level paired outcomes and case-cluster bootstrap intervals; multiple dialogue states from one case are never treated as independent bootstrap units.
