# Change log

This log records research-repository changes and major experiment milestones.
It does not replace the dated run reports in each experiment directory.

## 2026-11-20

- Added a unified V1/V2/V3 project entry point, experiment index, versioning
  rules, and PFS artifact locator on branch `codex/repo-management-20261120`.
- Added a point-in-time audit of local branches, worktrees, remote refs, dirty
  worktrees, A/B/C results, and legacy public-data risks.
- Published the missing branch refs `codex/ea-atpo-v2-task-b` and
  `codex/ea-atpo-v2-relative-router` without changing their commits.
- No model training, GPU work, checkpoint changes, other-window file edits, or
  history rewrites were performed.
- No cache or experiment data was deleted. The audit found no byte-identical
  README copies that could safely be removed; specialized experiment READMEs
  remain in place.

## 2026-11-20 — V2 Router and independent audit milestones

- V2 Q Regression and Q-guided Policy Distillation were run on 50 training
  cases (146 states; 438 valid action rollouts), with a 35/15 case-level
  train/dev split and three seeds.
- A case-cluster bootstrap audit corrected the uncertainty calculation. The
  best Q-guided Distillation result remains development-only; the corrected
  interval versus Always ANSWER includes zero.
- The remaining 20 official training cases were found in historical V1 Router
  fit/target data; they are not an independent confirmation cohort. The 400
  case holdout remains sealed.

## 2026-11-20 — V3 Expert-SFT gate

- V3 Expert-SFT preparation and AWQ-LoRA compatibility work were recorded on
  the dedicated V3 branch. A 10-step compatibility smoke is not formal Expert
  SFT; final Expert evaluation, Router training, and RL remain incomplete.

## Earlier milestones

- 2026-11-20: V2 three-action counterfactual pilot recovery bundle was
  published with 50 train cases, 146 states, and 438 valid trajectories.
- 2026-11-20: V1 Attempt09 completed a 10-step PPO engineering smoke. It did
  not establish policy improvement; see its dated report.
- 2026-11-03 onward: real-case pilot, expert reliability, CONSULT, and action
  value analyses are indexed in `PUBLISHED_RESULTS_INDEX.md`.
