# EA-ATPO V1 Attempt 06 Results

## Outcome

The real-model engineering smoke completed both configured PPO updates and exited with code 0. It ran Qwen3-8B Actor/Critic FSDP and one SGLang tensor-parallel-2 service across physical GPUs 4 and 6. No OOM occurred. The run saved complete Actor/Critic model and optimizer state at `global_step_1` and `global_step_2`.

At launch, GPU 4 had 111,699 MiB free and GPU 6 had 111,311 MiB free (approximately 109.1 and 108.7 GiB). The launcher used a 100,000 MiB minimum-free-memory gate. The maximum trainer-reported memory was 76.30 GiB allocated and 88.20 GiB reserved. GPU 1 and GPU 7 were not used.

## Training steps

| Step | Valid / sampled | Correct | Actor PG loss | Actor grad norm | Critic VF loss | Critic grad norm |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 2 / 2 | 0 / 2 | -0.3576 | 29.89 | 1.4331 | 432.99 |
| 2 | 0 / 2 | 0 / 2 | -0.5312 | 65.32 | 16.0407 | 1994.58 |

Both steps logged nonzero Actor and Critic gradient norms and completed their update/checkpoint paths. The unusually large second-step Critic loss and gradient, plus only two training rows, make this unsuitable as evidence of stable learning.

## Generated action and episode outcomes

The two training rows each generated two trajectories, for four trajectories total.

| Measure | Count |
|---|---:|
| ASK action tags | 6 |
| CONSULT action tags | 2 |
| ANSWER action tags | 2 |
| DEFER action tags | 0 |
| Patient calls | 4 |
| Expert calls | 2 |
| Final answers | 2 |
| Correct final answers | 0 |
| `ask_budget_exhausted` failures | 2 |

Two trajectories followed ASK → CONSULT → ANSWER and returned option D where the post-hoc gold was C. The other two attempted ASK twice; the second ASK exhausted the one-ASK budget, so those episodes ended without a final answer. They were counted as invalid and incorrect by the post-rollout logger. The action trace records the ASK attempt, while the Patient was called only once per episode.

The real model sampled ASK, CONSULT, and ANSWER, but not DEFER. DEFER and all four transitions remain covered by the deterministic protocol smoke, not this real-model rollout. Therefore this run does not demonstrate real-model coverage of all four actions.

## Interpretation and limitations

This run establishes that the two-GPU memory layout works with about 109 GiB free on each card, the real Qwen3-8B/SGLang service can generate actions, Actor/Critic updates execute, post-rollout metrics no longer crash, and both checkpoints are written. It does **not** establish model quality, successful learning, action-value improvement, or a meaningful accuracy estimate. There were only four correlated trajectories from two training rows; no validation evaluation was produced (`Final validation metrics: None`).

The launcher reached 2/2 steps and returned exit code 0. Ray emitted a worker-shutdown warning after the progress bar reached 100%; the final step metrics and checkpoint writes preceded it.

## Artifacts

- Full rollout traces: `traces/train.jsonl`
- ATPO rollout dumps: `rollouts/1.jsonl`, `rollouts/2.jsonl`
- Raw launcher log: `logs/ea_atpo_v1_smoke_20261113T200605Z.log`
- Environment/data/model manifest: `run_manifest.json`
- Checkpoints: `checkpoints/global_step_1/`, `checkpoints/global_step_2/`

Each checkpoint directory contains 32 files. The two checkpoints total approximately 352.3 GiB and remain on PFS; they are intentionally not committed to GitHub. The small rollout traces, manifests, logs, and this report are committed for review and reproducibility.
