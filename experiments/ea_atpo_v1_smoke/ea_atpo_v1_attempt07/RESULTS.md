# EA-ATPO V1 Attempt 07 Results

## Summary

The validated wrapper passed all three deterministic protocol/RewardManager tests and a five-sample free-memory gate. It then ran two real Qwen3-8B PPO updates across GPUs 4 and 6. The launcher exited 0, both Actor/Critic checkpoints were saved, and no OOM occurred. The forced-action fixtures were not part of PPO training.

This is an engineering smoke, not a policy-quality experiment. The normal model generated ASK, CONSULT, and ANSWER, but never DEFER. Two trajectories answered incorrectly after ASK → CONSULT → ANSWER; two ended after a second ASK exceeded the budget. No terminal answer was correct. Step 2 had 0/2 valid rollouts and a larger Critic loss/gradient, even though the optimizer step and checkpoint completed.

## Real optimizer metrics

| Step | Valid / sampled | Correct | Actor PG loss | Actor grad norm | Critic VF loss | Critic grad norm | Max allocated / reserved |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 / 2 | 0 / 2 | 0.0185 | 24.016 | 2.1411 | 624.508 | 76.30 / 88.20 GiB |
| 2 | 0 / 2 | 0 / 2 | 1.0968 | 34.061 | 12.7497 | 1676.140 | 76.30 / 88.20 GiB |

## Action and reward outcomes

| Action | Sampled | Valid | Invalid | Actor tokens | Gradient-eligible segments/tokens |
|---|---:|---:|---:|---:|---:|
| ANSWER | 2 | 2 | 0 | 10 | 2 / 10 |
| ASK | 6 | 4 | 2 | 128 | 6 / 128 |
| CONSULT | 2 | 2 | 0 | 50 | 2 / 50 |
| DEFER | 0 | 0 | 0 | 0 | 0 / 0 |

The eligibility count requires generated Actor tokens and nonzero Advantage. The trainer provides aggregate Actor gradient norms; it does not measure action-specific parameter-gradient norms. See `action_gradient_coverage.csv` for event-level returns, Advantages, and the proxy definition.

Episode rewards were -0.25, -0.25, -0.05, and -0.05; aggregate resource cost was 0.60 and correctness reward was 0. No validation evaluation ran.

## Reproduction artifacts

- Configuration: `ea_atpo_v1_config.yaml`
- Run manifest and hashes: `run_manifest.json`
- Forced-test output: `validation/four_action_protocol_tests.log`
- GPU free-memory gate: `validation/gpu_memory_gate.log`
- Normal-policy traces: `traces/train.jsonl`
- Rollout dumps: `rollouts/1.jsonl`, `rollouts/2.jsonl`
- Raw trainer log: `logs/ea_atpo_v1_smoke_20261114T034613Z.log`
- Checkpoints: `checkpoints/global_step_1/`, `checkpoints/global_step_2/` (`latest_checkpointed_iteration.txt` is 2)

Each checkpoint contains 32 files; together they use approximately 352.3 GiB and remain on PFS. They are excluded from GitHub. Full audits are in the project-root `EA_ATPO_FOUR_ACTION_PROTOCOL_REPORT.md`, `EA_ATPO_REWARD_RETURN_ADVANTAGE_AUDIT.md`, `EA_ATPO_PPO_GRADIENT_COVERAGE.md`, and `EA_ATPO_V1_JOINT_TRAINING_SMOKE.md`.
