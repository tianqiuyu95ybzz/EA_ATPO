# EA-ATPO V1 Implementation Record

## Current status

The original ATPO baseline and deterministic four-action protocol smoke passed. EA Attempt 06 completed two Qwen3-8B PPO updates across GPUs 4 and 6, logged post-rollout metrics, and saved complete step-1 and step-2 Actor/Critic checkpoints without OOM. The adapter derives validity/correctness from the completed response and gold only after rollout; this is logging metadata and does not alter reward targets or model input. Real model rollouts included ASK, CONSULT, and ANSWER, but not DEFER; two of four trajectories exhausted the ASK budget. This is engineering evidence, not a performance result. GPUs 1 and 7 remain excluded.

Official ATPO source base: `25af83688c06917f94245a4db06be7d0924c090c`; no file under `ATPO-official/codes/` is modified.

## Policy interface

The Actor emits one action tag and its content in a single assistant response:

- `ANSWER`: final answer; terminate.
- `ASK`: patient question; append the patient reply and continue with the same Actor.
- `CONSULT`: specific expert question; append bounded expert advice and continue with the same Actor.
- `DEFER`: transfer the decision to the Expert, which independently returns a final answer; terminate.

Tags are ordinary sampled text, so no new vocabulary rows or special-token embeddings are required. The generated response mask covers each tag and its content. Patient and Expert observations have mask zero. The loop reuses ATPO's `convert_to_agent_output` so action-level advantages and returns attach to the Actor-generated segments.

## Reward and value targets

Each action edge receives one EA net reward:

`r_t = terminal_correctness_reward_t - lambda_ask * ask_cost * I(ASK) - lambda_consult * consult_cost * I(CONSULT) - lambda_defer * defer_cost * I(DEFER)`

The correctness term is nonzero only when an episode terminates with an answer. Costs are charged once at the edge. The EA runner supplies these edge rewards to ATPO's tree backup; returns train the existing Critic and the resulting advantages train the Actor. Because the loop returns precomputed advantages and returns, the trainer skips a second GAE calculation. The EA reward adapter copies the tree's episode total into the trainer's score field for reporting; it does not add another training reward.

The original clipped PPO Actor loss and clipped Critic MSE remain unchanged. V1 adds no information-gain reward, clinical-harm reward, new Critic head, Router loss, or joint RL objective.

## Information boundaries

- The Patient simulator receives the hidden atomic facts and the Actor's ASK question. Its reply alone enters the public dialogue.
- The Expert receives the public case/dialogue and the Actor's specific CONSULT question. It receives neither hidden facts nor gold.
- DEFER uses the same public-information boundary; the Expert independently selects the final answer.
- Gold is read for terminal reward and post-hoc evaluation only.
- ASK and CONSULT return control to the same Actor. DEFER and ANSWER terminate.

## Implementation files

EA code lives in `ea_atpo_stage/recipe/ea_atpo/` and uses ATPO's `agent_loop_config_path`:

- `ea_agent_loop.py`: Actor log-probability handling, role-isolated calls, four-action tree expansion, terminal/failure handling, net rewards, and trace capture.
- `protocol.py`: public-state tracking, action parsing, and Expert information-boundary checks.
- `ea_reward.py`: standalone score utility for reading stored action text.
- `ea_reward_manager.py` and `mt_main_ea.py`: EA episode-score adapter and entrypoint. They leave ATPO source unchanged and require the loop's trace-derived `reward_score`.
- `agent.yaml`: maps the official dataset's `user_assistant_interaction` agent name to the EA loop.
- `test_ea_agent_loop.py`: deterministic transition, role-isolation, cost, and token-mask checks.
- `run_ea_atpo_v1_smoke.sh`: two-GPU Qwen3-8B Actor/Critic smoke runner with isolated outputs and data/model hashes.

## Verification status

The deterministic protocol smoke executes ANSWER, ASK, CONSULT, and DEFER; confirms ASK/CONSULT continuation through the same Actor; verifies action tokens have mask 1 and external observations mask 0; checks the Expert prompt excludes hidden facts and gold; and checks each configured action cost appears once. It does not measure model quality.

Attempt 06 confirms model-generated Patient/Expert calls, nonzero Actor and Critic gradient norms, two PPO update steps, checkpoint writes, and successful post-update metrics. Its four rollouts are insufficient to validate reward-learning quality or cost accounting statistically, and DEFER was not sampled by the real model. No accuracy claim is made from this smoke run; see the attempt-specific results record.

## Real-run debugging note

The first real EA training attempt successfully distributed the 8B Actor/Critic and SGLang rollout across physical GPUs 4 and 6. It exited before generation because the loop did not pass `correctness_reward` into the search runner. This was a code/configuration error rather than a memory limit; the failed manifest and raw log are preserved under `experiments/ea_atpo_v1_smoke/ea_atpo_v1_attempt01/`. The field is now included, and the 21 protocol/data-collection tests pass. No GPU 1 or 7 was used.

## Real-run debugging note: post-update metrics

Attempt 2 demonstrated the distributed setup is feasible on GPUs 4 and 6: the two-rank SGLang service initialized, two real episodes completed, and one PPO update wrote Actor/Critic model and optimizer state. It exited afterward because the legacy ATPO accuracy logger required `is_valid` and other interaction fields absent from EA reward metadata. The EA adapter now derives answer validity and correctness from the completed trace and fills protocol-inapplicable legacy counters with zero. Attempt 2's local manifest, raw rollout records, log, and partial checkpoint are preserved; the partial two-case accuracy is not a model-quality result.


## Post-hoc legacy metrics

The EA Reward Manager reads the generated response and the dataset gold only after the trajectory is complete. It derives `is_valid` and `is_correct` for ATPO logging and marks unrelated human-interaction counters as not applicable. These values do not participate in Actor/Critic targets.
