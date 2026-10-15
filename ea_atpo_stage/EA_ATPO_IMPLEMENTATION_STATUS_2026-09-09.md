# EA-ATPO implementation status — 2026-09-09

## Completed baseline evidence

- Official Qwen3-4B SFT and corrected ATPO training chain completed through step 20.
- Full deterministic test set: SFT 43.45% accuracy, ATPO 46.53% (+3.08 percentage
  points); average turns decreased from about 4.05 to 3.39.
- The critic value loss decreased substantially, but online batch accuracy was not
  monotonic. The gain is preliminary and has not yet been shown statistically
  significant.

## Frozen method decisions

1. The paper remains learn-to-defer, implemented as sequential collaborative
   deferral rather than a hand-written uncertainty rule.
2. The router learns pre-action counterfactual outcome values for `ANSWER`, `ASK`,
   `CONSULT`, and `DEFER`.
3. The expert sees only the public dialogue. Gold labels and hidden patient facts
   are forbidden.
4. `CONSULT` is low-workload advice followed by the base agent; `DEFER` is a
   high-workload terminal expert takeover.
5. Outcome value and action cost are stored separately to prevent double counting.

## Current stage

Stage 2 feasibility infrastructure is being built as an isolated `recipe/ea_atpo`
package. The first unit provides the public-state schema, leakage guards, expert
protocols, four-action parsing, and cost/harm-aware action selection.

## Execution queue

1. Unit-test the protocol and leakage boundary.
2. Implement endpoint-backed paired counterfactual collection.
3. Deploy base policy, patient simulator, and independent expert on GPUs 1/2/3.
4. Smoke-test 10--20 states and inspect every expert exchange for leakage.
5. Collect 500 states with repeated stochastic rollouts per action.
6. Plot optimal-action proportions and value gaps; only proceed if collaborative
   actions occupy meaningful, non-degenerate regions.
7. Train the four-head action-value router and evaluate cost/budget curves.
