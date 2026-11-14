# EA-ATPO Four-Action Protocol and Environment Tests

## Result

**Pass for deterministic engineering coverage.** Three protocol/RewardManager tests passed. A new test uses four independent public-state fixtures and a test-only deterministic client to force one target action in each state. The forced fixtures were not sent to the PPO trainer and produced no optimizer updates.

Command:

```bash
PYTHONPATH="$PWD/ea_atpo_stage:$PWD/ATPO-official/codes" \
  .venv/bin/python -m unittest recipe.ea_atpo.test_ea_agent_loop -v
```

Result: 3 tests passed. The captured output is `experiments/ea_atpo_v1_smoke/ea_atpo_v1_attempt07/validation/four_action_protocol_tests.log`.

## Forced-state outcomes

Each row below is an independent deterministic test fixture, not a real patient result. For tree contrast only, a second test candidate returns an incorrect direct answer; neither candidate is passed to PPO.

| Forced action | State transition | Terminal result | Cost | Net reward |
|---|---|---|---:|---:|
| ANSWER | Base emits final option | `base_answer` | 0.00 | 1.00 |
| ASK | Patient sees hidden fixture fact; Base resumes and answers | `base_answer` | 0.05 | 0.95 |
| CONSULT | Expert sees public state and specific question; Base resumes and answers | `base_answer` | 0.20 | 0.80 |
| DEFER | Expert sees public state and returns the terminal option | `expert_deferred_answer` | 0.60 | 0.40 |

The target fixture answer is B in each case. Costs are charged once on the action edge; correctness is added once on terminal evaluation. ASK and CONSULT are nonterminal and continue through the same Actor. ANSWER and DEFER terminate.

## Token and information-boundary checks

- Every generated Actor token is marked `response_mask=1`, including the action label and ASK/CONSULT question text.
- Patient replies, Expert advice, and the DEFER Expert final answer are appended as observations with mask 0; Expert output does not become an Actor token target.
- The Critic `value_response_mask` marks the first five tokens of each Actor action segment and no observation tokens.
- Patient fixtures receive the hidden atomic fact. Both Expert prompts exclude the hidden fact and serialized gold.
- Each forced action has a recorded nonzero tree Advantage in the controlled test. This validates calculation/mapping only; it is not evidence that the policy learned that action.

Implementation lives in `ea_atpo_stage/recipe/ea_atpo/ea_agent_loop.py`, `protocol.py`, and `test_ea_agent_loop.py`. The original ATPO checkout was not edited.

## Real-policy distinction

Attempt 07 separately ran normal Qwen3-8B sampling. It sampled ASK, CONSULT, and ANSWER, but not DEFER. Therefore this report's DEFER result is protocol-path coverage only. Real-policy action coverage and PPO updates are reported in `EA_ATPO_PPO_GRADIENT_COVERAGE.md` and `EA_ATPO_V1_JOINT_TRAINING_SMOKE.md`.
