# EA-ATPO V1 Joint-Training Smoke

## Outcome

Attempt 07 passed all deterministic protocol tests, then completed two normal-policy PPO update steps on GPUs 4 and 6. The launcher exited 0; both Actor/Critic checkpoints and optimizer shards were saved. No OOM occurred. This validates the engineering path for real model sampling, ASK/CONSULT continuation, tree reward/return/Advantage propagation, PPO updates, and checkpoint writes.

It does **not** show that the model learned a good policy. Four rollouts were generated from two training rows: two wrong final answers after ASK → CONSULT → ANSWER, and two episodes that exhausted the one-ASK budget. No correct final answer was produced and the real policy did not sample DEFER. Validation metrics were disabled (`Final validation metrics: None`).

## Run record

- Model: Qwen3-8B Actor/Critic/Reference.
- Physical GPUs: 4 and 6; Actor/Critic FSDP world size 2; SGLang tensor parallel 2. GPUs 1 and 7 were excluded.
- Startup gate: at least 105,000 MiB free on each selected card across five samples 15 seconds apart. This exceeds Attempt 06's measured peak reservation of 88.20 GiB by about 14 GiB. The gate checks free memory directly; utilization is recorded but is not used as a VRAM proxy.
- Data: two training rows and one validation row. Both Parquet SHA256 hashes are identical to the original ATPO Attempt 20 inputs.
- Training: seed 1, batch size 1, two optimizer steps, two tree trajectories/candidates, maximum three Actor turns, one ASK and one CONSULT, same learning rates and model as the baseline smoke.
- Reward: correctness +1, ASK cost 0.05, CONSULT cost 0.20, DEFER cost 0.60, each subtracted once.
- Memory: maximum trainer-reported allocation/reservation 76.30/88.20 GiB.

The original ATPO baseline used a 256-token response cap; EA uses the previously validated 512-token cap to accommodate multi-turn dialogues. Attempt 07's measured response lengths were 212.5 and 117 tokens per sample, with zero clipping. The two runs match on model, data, cards, batch, step count, and tree candidate budget, but this is not a strict matched-cap performance comparison. No accuracy comparison is claimed.

## Reproduction and artifacts

From the project root:

```bash
EA_ATPO_RUN_DIR="$PWD/experiments/ea_atpo_v1_smoke/ea_atpo_v1_attempt07" \
EA_ATPO_GPUS=4,6 ./run_ea_atpo_v1_validated_smoke.sh
```

The wrapper runs the three deterministic tests, requires five stable per-GPU memory samples, then launches the unchanged EA trainer with the Attempt 06 FSDP/SGLang setup and a higher free-memory threshold. Full values are in `experiments/ea_atpo_v1_smoke/ea_atpo_v1_attempt07/ea_atpo_v1_config.yaml` and `run_manifest.json`.

- Training traces: `traces/train.jsonl`
- Rollout dumps: `rollouts/1.jsonl`, `rollouts/2.jsonl`
- Test and GPU-gate logs: `validation/`
- Raw training log: `logs/ea_atpo_v1_smoke_20261114T034613Z.log`
- Latest checkpoint iteration: `checkpoints/latest_checkpointed_iteration.txt` (2)
- Checkpoints: `checkpoints/global_step_1/` and `checkpoints/global_step_2/`

Each checkpoint contains 32 files. Together they occupy approximately 352.3 GiB and remain on PFS; they are not uploaded to GitHub. The source, configuration, reports, traces, rollout dumps, small validation logs, and raw training log are committed to the `ea-atpo-v1` branch.
