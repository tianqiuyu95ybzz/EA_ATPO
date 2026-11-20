# Recovery and training protocol

The current queue is not reviewed. An actual human reviewer must work on a **new private copy** of `data/private/gold_review_queue_with_base.csv` and complete all 291 decisions under `GOLD_REVIEW_PROTOCOL.md`. `gold_review_priority.csv` merely prioritizes disagreements with historical labels; it is not an accuracy or action-label file. Do not edit the initial queue or put either file in Git.

After review, define paths from the worktree root. `V3_REVIEW` below must name the new completed review copy, not the still-pending initial file:

```bash
V3=experiments/ea_atpo_v3/expert_sft
V3_SOURCE=/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO_V1
V3_BASE="$V3/data/private/base_generation_20261120_attempt04"
V3_REVIEW="$V3/data/private/gold_review_queue_REVIEWED.csv"
V3_PAIRED="$V3/data/private/paired_REVIEWED_NEW"
V3_MODEL=/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/experiments/real_case_pilot_20261103/assets/models/Qwen3-32B-AWQ
V3_PY=/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO_V1/experiments/expert_sft/multiturn_dialogue/.venv_awq/bin/python

python3 "$V3/scripts/check_gold_review.py" --queue "$V3_REVIEW" --source-root "$V3_SOURCE" \
  --audit-manifest "$V3/data/data_manifest.json" --base-answers "$V3_BASE/base_answers.jsonl" \
  --base-manifest "$V3_BASE/base_generation_manifest.json"
python3 "$V3/scripts/build_paired_dataset.py" --queue "$V3_REVIEW" --source-root "$V3_SOURCE" \
  --audit-manifest "$V3/data/data_manifest.json" --base-answers "$V3_BASE/base_answers.jsonl" \
  --base-manifest "$V3_BASE/base_generation_manifest.json" --output-dir "$V3_PAIRED"
```

Only after both commands pass, a fresh synthetic A/B loss integration smoke passes as described in `AWQ_LORA_SMOKE_STATUS.md`, and an allowed physical GPU is currently safe, substitute its number for both `GPU` placeholders. The runner repeats the human-review, data, 10-step smoke, loss-integration, model fingerprint and GPU gates. It does not read Internal Test for training or Dev checkpoint selection.

```bash
CUDA_VISIBLE_DEVICES=GPU "$V3_PY" "$V3/scripts/train_experts.py" --expert A --gpu GPU \
  --model-dir "$V3_MODEL" --paired-dir "$V3_PAIRED" --review-queue "$V3_REVIEW" \
  --source-root "$V3_SOURCE" --audit-manifest "$V3/data/data_manifest.json" \
  --base-answers "$V3_BASE/base_answers.jsonl" --base-manifest "$V3_BASE/base_generation_manifest.json" \
  --smoke-report "$V3/data/awq_smoke_manifest.json" \
  --smoke-adapter "$V3/outputs/awq_smoke_20261120_attempt01/adapter/adapter_model.safetensors" \
  --loss-smoke-report "$V3/outputs/trainer_loss_smoke_SUCCESS/trainer_loss_smoke_report.json" \
  --config "$V3/configs/expert_a_base_aware.json" --output-dir "$V3/outputs/expert_a/run_NEW"
```

Run B separately, after its own fresh GPU check, with `--expert B`, the copied `--model-dir "$V3/models/expert_b_awq_init"`, `--expert-b-copy-manifest "$V3/data/expert_b_initialization_manifest.json"`, `--config "$V3/configs/expert_b_independent.json"`, and a new `--output-dir "$V3/outputs/expert_b/run_NEW"`; the other arguments are identical. Do not share A's adapter, optimizer or output path. Neither command uses PPO/ATPO RL or a BF16/model fallback.

Each epoch writes an adapter, optimizer checkpoint, checksums, per-class loss sums/counts and action metrics. A uses restricted first-token logits to implement the requested three-class action cross-entropy without modifying model architecture. Correction loss is computed only for REVISE, with both weights fixed to 1.0; B uses assistant-only cross-entropy. KEEP/ABSTAIN answer bytes are copied from Base by the inference wrapper. Tokenization beyond 4,096 tokens fails rather than silently truncating a conversation.

The trained run leaves checkpoint selection pending. For each candidate checkpoint, use `predict_experts.py` on Dev only, then `prepare_evaluation_records.py` for post-generation correctness adjudication, `calibrate_threshold.py` for non-Oracle Dev threshold selection, and `freeze_expert_checkpoint.py` to lock the chosen checkpoint/config/threshold before one Internal Test generation and evaluation. These steps have synthetic boundary/metric tests but have not yet been run end to end on a trained checkpoint. `evaluate_experts.py --help` lists the required private artifacts; keep all patient-bearing predictions and scored files on PFS. The frozen 400-case holdout is not touched.
