# EA-ATPO Research Repository

EA-ATPO studies sequential medical decision making with active information
acquisition and selective expert handoff. This repository contains research
code, protocols, reports, and a limited set of historical experiment artifacts.
The official ATPO implementation is an external dependency; see
[`PROJECT_LAYOUT.md`](PROJECT_LAYOUT.md).

> **Data handling:** Do not commit patient-level source data, Gold answers,
> private predictions, model weights, adapters, credentials, or service logs.
> Some legacy case-level JSONL and Gold-bearing records already exist in the
> public Git history. They were not added by the repository-management changes
> on branch `codex/repo-management-20261120`; see the
> [local/remote audit](docs/repo_management/LOCAL_REMOTE_AUDIT_20261120.md).
> This branch does not rewrite history.

## Start here

- [Research versions and current status](EXPERIMENT_INDEX.md)
- [Documentation index](DOCUMENTATION_INDEX.md)
- [Published and historical result index](PUBLISHED_RESULTS_INDEX.md)
- [Versioning and artifact rules](VERSIONING.md)
- [PFS artifact locator](PFS_ARTIFACT_INDEX.md)
- [Change log](CHANGELOG.md)
- [Repository layout and upstream ATPO pin](PROJECT_LAYOUT.md)

## Research status at 2026-11-20

| Line | Main question | Current evidence | Status |
|---|---|---|---|
| V1 | Can four actions be trained in an ATPO Actor-Critic loop? | Attempt09 completed 10 PPO steps; 79/80 rollouts entered PPO, but validation was 4/20 and Critic explained variance was negative for all 10 steps. | Engineering smoke only; no demonstrated policy gain. |
| V2 | Can counterfactual ASK/ANSWER/DEFER values support a light Router? | 50 training cases, 146 states, 438 action trajectories. Best 15-case Dev result: Accuracy 0.507, Utility 0.491; the case-clustered interval versus Always ANSWER includes zero. | Development evidence only; no uncontaminated confirmation set is available among the remaining 20 cases. The 400-case holdout remains sealed. |
| V3 | Does a domain SFT Expert improve conditional complementarity? | The gated AWQ-LoRA compatibility smoke performed 10 optimizer steps and saved/reloaded an adapter. Formal Expert SFT, final Expert pairing, Router training, and RL are not complete. | Preparation and compatibility evidence only. |

These versions are research stages, not claims of clinical effectiveness. Read
the experiment-specific reports before reusing any metric. The A/B/C window
mapping and exact source branches are in the [experiment index](EXPERIMENT_INDEX.md).

## Code and data layout

- `ea_atpo_stage/`: V1 multi-action protocol and experiment runner.
- `experiments/real_case_pilot_20261103/`: real-model pilot, audits, and
  historical expert comparisons.
- `experiments/ea_atpo_v1_smoke/`: V1 training attempts and diagnostics.
- `experiments/ea_atpo_v2/`: V2 three-action protocol, counterfactual runner,
  and Q/Router framework.
- `experiments/expert_sft/`: earlier SFT data and recovery audits.
- `experiments/vi_ea_atpo/`: VI-EA-ATPO planning and phase work.
- V2 Router training and V3 Expert SFT are maintained on their own branches;
  do not assume those branch trees have been merged into `ea-atpo-v1`.

Large/private PFS artifacts are indexed by path and manifest in
[`PFS_ARTIFACT_INDEX.md`](PFS_ARTIFACT_INDEX.md). Git-ignored files are not
automatically backed up; verify their manifest before moving or cleaning a
workspace.

## Repository workflow

1. Use a dedicated branch and worktree for each active research window.
2. Keep each run immutable: new seed, cost, prompt, model revision, or policy
   gets a new config/output identifier.
3. Commit code, safe configs, aggregate reports, and manifests. Keep private
   records, Gold, embeddings, predictions, and weights in controlled PFS
   storage.
4. Before pushing, inspect staged paths and confirm the remote branch SHA after
   every push. Never force-push or rewrite published history.
5. Record negative and failed runs as part of the experiment history.

See [`VERSIONING.md`](VERSIONING.md) for the full policy.
