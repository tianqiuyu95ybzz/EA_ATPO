# EA-ATPO Reward, Return, and Advantage Audit

## Definitions used in Attempt 07

For an Actor action edge, the EA reward is:

```text
r_t = terminal_correctness_reward_t
      - lambda_ask * ask_cost * I(ASK)
      - lambda_consult * consult_cost * I(CONSULT)
      - lambda_defer * defer_cost * I(DEFER)
```

Attempt 07 uses `gamma=1`, correctness reward 1.0, and costs ASK 0.05, CONSULT 0.20, DEFER 0.60. ANSWER has no action cost. A nonterminal ASK/CONSULT edge receives only its action cost; the correctness reward is added at the final Base answer. DEFER receives its cost and the Expert's terminal correctness on the same terminal edge.

The inherited ATPO tree backup is:

```text
V_mdp(s) = mean over explored children a of [r(s,a) + gamma * V_mdp(s')]
A(s,a)   = [r(s,a) + gamma * V(s') - V(s)] / max(explore_num, 1)
```

The Critic supplies the state values used by ATPO's Advantage calculation. Tree-derived `returns` and `advantages` are attached to Actor turns by the upstream converter. The trainer log states “Found pre-computed advantages in batch. Using them directly,” so it does not run another GAE pass over these targets.

## Forced protocol arithmetic

The four independent deterministic fixtures each contrast the forced target action with an incorrect direct-answer candidate. The test verifies the target tree root values and action Advantages:

| Target action | Target net episode reward | Root tree value | Target action Advantage(s) |
|---|---:|---:|---|
| ANSWER | 1.00 | 0.500 | 0.800 |
| ASK → ANSWER | 0.95 | 0.475 | -0.050, 0.800 |
| CONSULT → ANSWER | 0.80 | 0.400 | -0.200, 0.800 |
| DEFER | 0.40 | 0.200 | 0.200 |

These are deterministic unit-test values under a fixed Critic stub, not experimental performance results. Forced trajectories never enter PPO.

## Real Attempt 07 accounting

| Quantity | Result |
|---|---:|
| ANSWER events | 2, reward 0.00 each |
| Valid ASK calls | 4, reward -0.05 each |
| Out-of-budget ASK attempts | 2, reward 0.00 each; no Patient call |
| CONSULT calls | 2, reward -0.20 each |
| DEFER calls | 0 |
| Correct terminal rewards | 0 |
| Total external action cost | 0.60 |
| Total episode net reward | -0.60 across four trajectories |

The two ASK → CONSULT → ANSWER episodes each have net reward -0.25. The two ASK → invalid ASK episodes each have net reward -0.05; the invalid second ASK makes no external call and has zero incremental cost. This matches the event ledger and episode totals.

The RewardManager test verifies that it returns the incoming precomputed `rm_scores` unchanged (including a positive reward and a negative net reward). It adds post-rollout validity/correctness metadata but does not deduct costs again. The training loop logs precomputed Advantages/Returns, preventing a duplicate GAE computation.

## Audit conclusion

The deterministic tests and actual traces agree with the configured reward accounting: costs are represented once in the tree edge rewards, tree returns/Advantages use those rewards, and the reporting RewardManager does not add a second cost. No new Critic head or alternate loss was introduced. The real smoke is too small and has no correct answers, so it verifies plumbing rather than reward quality or policy improvement.
