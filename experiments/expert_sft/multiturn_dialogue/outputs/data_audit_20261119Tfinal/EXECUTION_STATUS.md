# Multi-turn Expert SFT execution status — 2026-11-19

## Result in brief

The restored official ATPO training files and the already-collected EA-ATPO public-state Pilot are readable and internally consistent. The original ATPO training examples are single-turn. The available multi-turn material comes from 100 prior model-simulated Pilot cases. I audited those states, validated the three case-isolated SFT files and exercised the pinned Qwen3-32B-AWQ tokenizer and chat template on CPU.

The isolated AWQ training environment installed cleanly, and its local AWQ configuration loads as a Qwen3 AWQ config with `backend=torch_awq`. This is still only a software/configuration preflight. No GPU passed the configured safe-use gate, so I did not load the 32B weights, run LoRA forward/backward, save/reload an adapter, start SFT, or run a model comparison. There is no SFT checkpoint or SFT performance result from this execution.

## Source data audit

- The restored raw ATPO `merged_train_dataset.jsonl` has **14,256 rows**. Its stable `index` is unique and sequential from 0 to 14,255. All prompts have only `system,user` roles. Thus, the raw official train file itself contains **no multi-turn dialogues**.
- All 14,256 source labels parse as A–D. There are 17 duplicate full prompt texts in the complete source file. None of the selected 100 Pilot cases has duplicate source prompt text, and the audit found no duplicate source prompt split across the selected case partitions.
- All four reconstructed Parquet variants contain 14,256 rows with a unique stable index. Every row maps back to its source row, its public prompt matches, its label parses, and its label matches the source (`14,256/14,256` for each check). No Parquet was changed.
- The existing public-state source is a 100-case simulator Pilot: **291** valid unique states, with 0/1/2 patient replies in 100/99/92 states respectively. **191** state samples include a patient reply; those snapshots contain 283 reply-turn appearances in total. These are model-simulated conversations, not human patient transcripts.
- The associated trajectory file contains 3,492 action rollouts: 873 per action. Of these, 3,465 are marked valid and 27 invalid (ANSWER 0, ASK 5, CONSULT 8, DEFER 14 invalid). Strict final-option parse rate is 99.23%. Every rollout label maps back to the official source answer (`3,492/3,492`). Invalid rollouts remain represented and are not reported as successful answers.

## SFT samples, split and leakage checks

The already-built answer-only SFT set contains **291** public-state samples, all reconstructed from the frozen current state rather than future turns:

| Partition | Cases | Samples | Samples with patient reply | Zero-reply anchor samples | Exact public-state input match | Target parses and matches source |
|---|---:|---:|---:|---:|---:|---:|
| Train | 70 | 205 | 135 | 70 | 205/205 | 205/205 |
| Validation | 15 | 42 | 27 | 15 | 42/42 | 42/42 |
| Internal test | 15 | 44 | 29 | 15 | 44/44 | 44/44 |

The split unit is `case_id` (seed `20261105`); case overlap is zero. All 291 SFT user messages exactly match the allowed public case plus the dialogue prefix for their state. All targets are exactly `Expert Final Answer: [A-D]` and match the source answer. No duplicate SFT prompt hash or selected duplicate source prompt split was found. The 400-case fresh holdout is disjoint from the Pilot/SFT cases and remains unmaterialized; it has not been used for training, checkpoint selection, or this audit.

These counts are consistent with the frozen case split and source artifacts. They do **not** turn the simulator Pilot into a real patient dialogue corpus, and the 291 examples should not be described as 291 independent cases.

## Model and environment checks actually run

- Existing model directory: `Qwen/Qwen3-32B-AWQ`, pinned Hugging Face revision `0499c3ac83fdef8810b907a23894ba91e95eddd8`, ModelScope revision `c14568581d436715d61d40f471dee3cac433d1d9`. The source lock lists all four AWQ shards and tokenizer/config files with hashes. The local directory has all four shards, index, config and tokenizer assets; the weights were not copied or modified.
- The model config resolves to `model_type=qwen3`, `Qwen3ForCausalLM`, AWQ 4-bit GEMM, group size 128, zero-point enabled. In the isolated environment, importing GPTQModel before Transformers, loading the small local config, and setting `quantization_config.backend="torch_awq"` succeeded. This does not prove model-weight loading or gradient support.
- Isolated environment installation and `pip check` passed. Locked versions: Python 3.12; Torch `2.15.0+cu132`; Transformers `5.22.2`; PEFT `0.21.3`; Accelerate `1.15.0`; GPTQModel `7.12.0`; Optimum `2.3.0`; Safetensors `0.8.0`; CUDA build `13.2`. Resolved packages are in `environment.lock.txt`.
- CPU tokenizer test passed on all 291 examples: prompt-token sequence is a prefix of the full supervised sequence, all answer targets are valid, chat template exists, and maximum full sequence lengths are 346 train / 322 validation / 299 internal-test tokens (the configured maximum is 4,096). This verifies data tokenization only, not model inference or training.
- The repeatable CPU-only preflight is `scripts/preflight_awq_software.py`; it checks the source-lock file inventory/small-file hashes, local Qwen3/AWQ config, `torch_awq` config assignment, tokenizer/chat template, and split tokenization without loading weights or touching CUDA. Its result is `awq_software_preflight.json`.
- Reproduction commands completed in this run:

  ```bash
  ./.venv/bin/python experiments/expert_sft/multiturn_dialogue/scripts/audit_multiturn_sources.py \
    --output-dir experiments/expert_sft/multiturn_dialogue/outputs/data_audit_20261119Tfinal
  bash experiments/expert_sft/multiturn_dialogue/scripts/setup_awq_env.sh
  experiments/expert_sft/multiturn_dialogue/.venv_awq/bin/pip check
  experiments/expert_sft/multiturn_dialogue/.venv_awq/bin/python \
    experiments/expert_sft/multiturn_dialogue/scripts/preflight_awq_software.py \
    --output experiments/expert_sft/multiturn_dialogue/outputs/data_audit_20261119Tfinal/awq_software_preflight.json
  ```

- GPTQModel's official LoRA guide documents `torch_awq` as the trainable AWQ backend and warns that default inference kernels do not propagate gradients. That establishes a supported software path in general, not compatibility of this particular Qwen3-32B checkpoint; only the blocked model-level smoke can establish that. See [GPTQModel PEFT LoRA guide](https://github.com/ModelCloud/GPTQModel/blob/main/docs/peft.md) and [GPTQModel 7.12.0 release](https://pypi.org/project/GPTQModel/7.12.0/).
- BF16 fallback is incomplete: the local directory has only 3 of 17 weight shards and lacks a complete index/tokenizer set. No BF16 download was started because AWQ has not failed a model-level compatibility test; GPU availability is not evidence of AWQ incompatibility.

## GPU safety decision

At the latest single snapshot, eligible cards were GPUs 0, 2, 3, 4, 5, and 6. GPUs 1 and 7 remain excluded. Every eligible card had at least one existing compute process. Observed free memory/utilization were: GPU0 128,355 MiB / 87%; GPU2 127,675 MiB / 88%; GPU3 107,389 MiB / 16%; GPU4 51,203 MiB / 0%; GPU5 49,611 MiB / 100%; GPU6 166,451 MiB / 0%.

The existing pre-load gate requires at least 150,000 MiB free, utilization at most 20%, and no other compute process; the post-load smoke must leave at least 32 GiB free. GPU3 fails memory and process checks. GPU6 has enough nominal free memory but fails the no-other-process condition. The remaining eligible cards also fail memory, utilization, or both. I did not stop, modify, or share memory with any existing process and did not try to load the model.

Project artifacts and caches remain on the project PFS. At the audit, PFS had 86 TB free; the system filesystem had about 4.4 GB free. The large Torch/GPTQModel environment and package cache were written only below `experiments/expert_sft/multiturn_dialogue/`.

## Training and evaluation state

| Stage | Status |
|---|---|
| Restored source and Parquet audit | Complete; see `data_audit.json` |
| Current public-state and SFT split audit | Complete; no raw clinical text or case IDs copied into the report |
| AWQ environment and local config/tokenizer preflight | Complete; no weights loaded |
| AWQ LoRA injection, forward/backward, gradient, optimizer update | **Not run**; no safe GPU |
| Adapter save/reload equivalence | **Not run** |
| 10-step training smoke | **Not run** |
| Formal Expert SFT/checkpoint selection | **Not started** |
| Base / original AWQ / SFT paired evaluation | **Not run** |
| Holdout predictions | Not materialized |
| Router or PPO/ATPO RL | Not started |

Therefore the current answers for SFT-before/after accuracy, helpful/harmful cases, paired confidence intervals, independent expert complementarity, and Router readiness are **not available**. No performance conclusion can be drawn from installing the training stack or loading the small config.

## Reproduction after a safe card is available

The 10-step smoke is the first authorized model operation. Use only an eligible card that meets the configured gate; do not bypass it. For example:

```bash
EA_ATPO_AWQ_GPU=<eligible-card> \
  bash experiments/expert_sft/multiturn_dialogue/scripts/run_awq_lora_smoke.sh
```

Start formal SFT only after that real model smoke reports finite loss and gradients, a nonzero LoRA update, and an exact saved/reloaded adapter round-trip:

```bash
EA_ATPO_AWQ_GPU=<eligible-card> \
  bash experiments/expert_sft/multiturn_dialogue/scripts/run_awq_lora_formal.sh
```

Do not materialize or evaluate the 400-case holdout until a checkpoint is selected solely on validation and a checkpoint lock is saved. Keep the pre-existing collector modification, historical experiments and checkpoints unchanged.
