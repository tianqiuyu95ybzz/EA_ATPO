# EA-ATPO V1 Initial Results

## Engineering smoke

The first complete real-model EA-ATPO V1 training smoke is Attempt 06. It used Qwen3-8B as the Actor/Critic base, with the two roles sharded over physical GPUs 4 and 6 and SGLang tensor parallel size 2. Approximately 109 GiB was free on each card at launch. Training completed two PPO updates without OOM, logged nonzero Actor and Critic gradient norms, and saved both step checkpoints.

The four generated trajectories came from two training rows with two trajectories per row. The model generated ASK, CONSULT, and ANSWER, but not DEFER. Two episodes reached an incorrect final answer, and two terminated because ASK exceeded its one-call budget. Step 1 reported 2/2 valid and 0/2 correct; step 2 reported 0/2 valid and 0/2 correct. These results are an engineering check only and must not be interpreted as model performance.

## What this establishes

- Qwen3-8B can be deployed with this Actor/Critic + rollout layout across GPUs 4 and 6 at the observed memory headroom.
- The modified loop generated real actions and made Patient and Expert calls.
- Actor/Critic gradient and update paths ran for two PPO steps.
- Post-rollout validity logging and checkpoint saving completed.

## What it does not establish

- It does not show that EA-ATPO improves accuracy or learns a useful collaboration policy.
- It does not show a reliable estimate of accuracy from four trajectories.
- It does not show real-model execution of DEFER; that action is covered only by the deterministic protocol test so far.
- It does not verify generalization or stable optimization. Step 2 had a large Critic loss/gradient and no valid answers.

## Reproducibility record

See [`Attempt 06 results`](experiments/ea_atpo_v1_smoke/ea_atpo_v1_attempt06/RESULTS.md) and its `run_manifest.json`, JSONL traces, ATPO rollout dumps, and raw log. Actor/Critic checkpoints are retained on PFS at `experiments/ea_atpo_v1_smoke/ea_atpo_v1_attempt06/checkpoints/`; they total approximately 352.3 GiB and are not stored in Git.

No Router or large-scale joint ATPO RL training was launched; Attempt 07 is limited to a two-step engineering smoke.

## Four-action validation and Attempt 07

Four independent deterministic fixtures now force ANSWER, ASK, CONSULT, and DEFER to verify transitions, terminal behavior, one-time costs, tree returns/Advantages, and Actor/Critic response masks. Three protocol/RewardManager tests pass. These forced fixtures are excluded from PPO.

Attempt 07 then completed two real PPO update steps on the same Qwen3-8B model, two training rows, and exact Parquet hashes used by the original ATPO Attempt 20. The normal policy sampled ASK, CONSULT, and ANSWER but not DEFER. All sampled Actor segments had nonzero Advantage and appeared in the policy-loss mask; aggregate Actor/Critic gradient norms were nonzero on both steps. Step 2 had no valid terminal rollout, and all four final outcomes across both steps were incorrect or missing. This verifies the training pipeline and cost/return wiring only. Full reports and checkpoint locations are in `EA_ATPO_V1_JOINT_TRAINING_SMOKE.md` and the Attempt 07 directory.
