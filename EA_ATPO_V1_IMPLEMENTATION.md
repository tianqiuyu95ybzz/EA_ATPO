# EA-ATPO V1 Implementation Record

## Current status

The original ATPO training baseline completed two PPO updates on physical GPUs 4 and 6. The four-action EA loop is now implemented outside the read-only ATPO checkout. Protocol smoke tests pass for all four actions, ASK/CONSULT continuation, DEFER termination, Actor/observation masks, information isolation, and one-time action costs. Those deterministic tests use a stand-in generation client and are not model-performance evidence. A real Qwen3-8B EA Actor/Critic training smoke is being prepared. GPUs 1 and 7 remain excluded.

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

The correctness term is nonzero only when an episode terminates with an answer. Costs are charged once at the edge. The EA runner supplies these edge rewards to ATPO's tree backup; returns train the existing Critic and the resulting advantages train the Actor. Because the loop returns precomputed advantages and returns, the trainer skips a second GAE calculation. The scalar reward function is for outcome reporting and does not add another training reward.

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
- `ea_reward.py`: reporting-only scalar score.
- `agent.yaml`: maps the official dataset's `user_assistant_interaction` agent name to the EA loop.
- `test_ea_agent_loop.py`: deterministic transition, role-isolation, cost, and token-mask checks.
- `run_ea_atpo_v1_smoke.sh`: two-GPU Qwen3-8B Actor/Critic smoke runner with isolated outputs and data/model hashes.

## Verification status

The deterministic protocol smoke executes ANSWER, ASK, CONSULT, and DEFER; confirms ASK/CONSULT continuation through the same Actor; verifies action tokens have mask 1 and external observations mask 0; checks the Expert prompt excludes hidden facts and gold; and checks each configured action cost appears once. It does not measure model quality.

The real training run must still confirm model-generated actions, Patient/Expert calls, nonzero Actor and Critic updates, valid checkpoints, and no cost/reward duplication. Any accuracy from this tiny engineering run will not be presented as a study result.
