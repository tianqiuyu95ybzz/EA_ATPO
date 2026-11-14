# EA-ATPO PPO Gradient Coverage

## What counts as coverage here

An action event is called **policy-gradient eligible** when it has generated Actor tokens, those tokens are placed in the PPO response mask, and its precomputed Advantage is nonzero. This is a coverage indicator, not an action-specific parameter-gradient norm. The existing trainer reports one aggregate Actor gradient norm per update; this run did not add hooks to split the model's parameter gradient by action.

The forced-state test checks token-level mask alignment and nonzero Advantages for all four code paths. Those test trajectories never enter the trainer. Real coverage below is derived only from Attempt 07 model-generated traces.

## Real-policy coverage

| Action | Sampled events | Valid / invalid | Actor tokens | Eligible events / tokens | Critic value-mask slots* | Mean edge reward | Mean Advantage |
|---|---:|---:|---:|---:|---:|---:|---:|
| ANSWER | 2 | 2 / 0 | 10 | 2 / 10 | 10 | 0.000 | 1.933 |
| ASK | 6 | 4 / 2 | 128 | 6 / 128 | 30 | -0.033 | -0.387 |
| CONSULT | 2 | 2 / 0 | 50 | 2 / 50 | 10 | -0.200 | -0.961 |
| DEFER | 0 | 0 / 0 | 0 | 0 / 0 | 0 | — | — |

\*The Critic uses a five-token value mask per Actor turn; these slots are derived from action token counts and the tested mask mapping. The event-specific CSV is `experiments/ea_atpo_v1_smoke/ea_atpo_v1_attempt07/action_gradient_coverage.csv`.

All ten sampled action segments had nonzero Advantage and generated tokens, for 188 policy-gradient-eligible Actor tokens. ASK/CONSULT question text and ANSWER text are Actor-generated segments. Patient and Expert outputs are observations with response mask zero; they are excluded from Actor loss and Critic value response masks. The deterministic test verifies this mapping for ANSWER, ASK, CONSULT, and DEFER.

## Optimizer evidence

| PPO step | Actor PG loss | Actor grad norm | Critic value loss | Critic grad norm | Valid / sampled | Correct |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.0185 | 24.016 | 2.1411 | 624.508 | 2 / 2 | 0 / 2 |
| 2 | 1.0968 | 34.061 | 12.7497 | 1676.140 | 0 / 2 | 0 / 2 |

Both steps had nonzero aggregate Actor and Critic gradient norms and saved model/optimizer state. Step 2's two rollouts terminated on out-of-budget ASK attempts, so its update is technically nonzero but is not evidence of useful collaboration behavior. Critic loss/gradient grew substantially on that step.

## Coverage limits

- No normal-policy DEFER was sampled. DEFER has **zero real PPO gradient coverage** in this run; only the deterministic protocol fixture verifies its mask, terminal transition, reward, and Advantage path.
- No per-action parameter-gradient norm was measured. The report records event/token eligibility and aggregate optimizer gradients separately.
- Four trajectories from two training rows are insufficient to estimate learning quality or stable action-use frequencies.
