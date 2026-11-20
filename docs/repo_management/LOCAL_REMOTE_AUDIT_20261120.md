# Local–remote repository audit — 2026-11-20

## Scope and method

This is a point-in-time audit of the EA-ATPO local clones, shared Git worktrees,
branch refs, GitHub heads, and the A/B/C research windows. It was performed
from a fresh documentation-only worktree based on `ea-atpo-v1`. Other windows'
working trees were inspected read-only. No GPU was used. No model training,
checkpoint mutation, cache deletion, raw-data copy, force-push, merge, or history
rewrite was performed.

Remote: `tianqiuyu95ybzz/EA_ATPO` (`origin`). SHA values below are full commit
IDs captured during the audit. Local worktree status counts use
`git status --porcelain --untracked-files=all`; ignored files are not included
in those counts.

## Local clones and worktrees

The workspace contains one older independent clone and ten linked worktrees
after adding this repository-management worktree.

| Path | Branch / HEAD | Tracked dirty | Untracked | Handling |
|---|---|---:|---:|---|
| `/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO` | `main` / `ad9d0e31660b973a3e5ddfcadce6701f767e0f62` | 1 | 961 | Audit only. One tracked Router evaluator is modified; untracked content includes an audit note and roughly 960 files under the legacy value-router experiment. Includes checkpoint/labels/metrics/caches. Nothing was staged or changed. |
| `/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO_V1` | `ea-atpo-v1` / `4c3a0516a16f326dd3ccdc386f611faa28e762e4` | 1 | 1,937 | Audit only. `collect_counterfactuals.py` is modified; untracked content is dominated by the old `experiments/ea_atpo_v2/parallel_cache` tree plus Attempt09 recovery and VI-EA-ATPO outputs. Nothing was staged or changed. |
| `EA_ATPO_V1_expert_sft_20261120` | `codex/ea-atpo-v3-expert-sft-20261120` / `80834ad4890a4200549238a4fc28a970c8eea9f6` | 0 | 0 | Tracked worktree clean. PFS-only V3 assets are ignored and were left untouched. |
| `EA_ATPO_V1_expert_sft_followup_20261120` | `codex/ea-atpo-expert-sft-20261120` / `436ae23871721db6f82d4adf5472976babcff627` | 0 | 0 | Tracked worktree clean. Private PFS review data is ignored and was left untouched. |
| `EA_ATPO_V1_final_eval` | `codex/ea-atpo-v2-final-eval-20261120` / `b11121c18a5ecab7f9a17ee2c90d273dd32707ce` | 0 | 0 | Clean. Frozen cohort design exists; no test predictions/outcomes. |
| `EA_ATPO_V1_router_audit` | `codex/router-results-audit-20261120` / `9b3f978ec9d0d8fb2d7dcc550fd2295743329777` | 0 | 0 | Clean. |
| `EA_ATPO_V1_router_training` | `codex/router-training-20261120` / `ba698a8d8176ae2dfe34e8f654adc12ca6208530` | 0 | 0 | Clean. PFS outputs are ignored; the six Router checkpoints were not copied or modified. |
| `EA_ATPO_V1_taskB` | `codex/ea-atpo-v2-task-b` / `66e476d94fd6d4a944018d67bc7565b613432122` | 0 | 0 | Clean. |
| `EA_ATPO_V1_windowB` | `codex/ea-atpo-v2-window-b` / `d375f892d8019cc2b034dc173cd07c44ec5c40ab` | 0 | 0 | Clean. |
| `EA_ATPO_V1_windowB_router` | `codex/ea-atpo-v2-relative-router` / `8336c37c125382dbf01ef2b6e9f312376e73b8e5` | 0 | 0 | Clean. |
| `EA_ATPO_V1_repo_management_20261120` | `codex/repo-management-20261120` / based on `4c3a0516a16f326dd3ccdc386f611faa28e762e4` | 0 before documentation edits | 0 | New isolated documentation-only worktree. Changes from this task are restricted here. |

The separate `EA_ATPO` clone reports approximately 179,702 ignored files and
an earlier inventory estimated about 57.95 GB of ignored content. The V3
worktree has about 19.96 GB of ignored model assets. These numbers are
inventory estimates, not a deletion list. No `git clean` or cache cleanup was
run. The dirty `parallel_cache` was not assumed valid or invalid and was not
removed.

No exact duplicate README contents were identified across the inspected V1,
V2, and V3 trees. Specialized experiment READMEs are distinct and remain in
place.

## Local branch refs and GitHub heads

At the start of this task, all pre-existing local commit objects were already
reachable from a remote branch, but two local branch names did not yet have
same-name GitHub refs. Those branch refs were published without modifying
their commits. The full `origin` head list observed afterward was:

| GitHub branch | Local / remote SHA | Result |
|---|---|---|
| `main` | `ad9d0e31660b973a3e5ddfcadce6701f767e0f62` | Exact match |
| `ea-atpo-v1` | `4c3a0516a16f326dd3ccdc386f611faa28e762e4` | Exact match |
| `codex/router-training-20261120` | `ba698a8d8176ae2dfe34e8f654adc12ca6208530` | Exact match |
| `codex/router-results-audit-20261120` | `9b3f978ec9d0d8fb2d7dcc550fd2295743329777` | Exact match |
| `codex/ea-atpo-v2-window-b` | `d375f892d8019cc2b034dc173cd07c44ec5c40ab` | Exact match |
| `codex/ea-atpo-v2-final-eval-20261120` | `b11121c18a5ecab7f9a17ee2c90d273dd32707ce` | Exact match |
| `codex/ea-atpo-v3-expert-sft-20261120` | `80834ad4890a4200549238a4fc28a970c8eea9f6` | Exact match |
| `codex/ea-atpo-expert-sft-20261120` | `436ae23871721db6f82d4adf5472976babcff627` | Exact match |
| `codex/ea-atpo-v2-task-b` | `66e476d94fd6d4a944018d67bc7565b613432122` | New branch ref; local SHA verified immediately after push |
| `codex/ea-atpo-v2-relative-router` | `8336c37c125382dbf01ef2b6e9f312376e73b8e5` | New branch ref; local SHA verified immediately after push |
| `codex/repo-management-20261120` | `0a422442f5c99983180aed22d9d48e484da04247` | Phase 1 documentation push; local SHA equals `ls-remote` |

The legacy local tracking ref `ea-atpo/codex/router-training-20261120` at
`7a9bb2a` is an ancestor of the published `origin/codex/router-training` head;
it is not a lost commit. No other-window branch was merged.

## A/B/C research-window audit

### A — Router

Window A trained absolute/relative value and Q-guided distillation Routers from
the 50-case V2 counterfactual set. The selected Q-guided model and Relative
Advantage comparator each have three frozen seeds. The Dev result is
model-selection evidence only: 15 Dev cases, Accuracy 0.507, Utility 0.491,
and case-cluster paired Utility 95% CI versus Always ANSWER `[-0.014, 0.297]`.
The remaining 20 official Train cases were found in historical V1 Router fit
or target data, so no clean confirmation cohort remained. The 400-case Frozen
Holdout remains sealed. See `codex/router-training-20261120` and
`codex/router-results-audit-20261120`.

### B — V2 cost and learnability analysis

The V2 counterfactual runner and 50-case data, cost/relative-router analyses,
and final-cohort design are on separate branches. The new 200-case final
cohort manifest has overlap audits, but no final predictions or outcome
metrics. No Router or PPO model was trained as part of the final-cohort design.
See `codex/ea-atpo-v2-task-b`, `codex/ea-atpo-v2-window-b`,
`codex/ea-atpo-v2-relative-router`, and
`codex/ea-atpo-v2-final-eval-20261120`.

### C — V3 Expert SFT

V3 remains at a gated preparation stage. A 10-step AWQ-LoRA compatibility
smoke performed parameter updates and adapter save/reload; this was not a
formal Expert-SFT run. The paired Expert evaluation and formal Expert A/B/C
training are not complete; the Gold review is pending, and `med_qwen` has no
reported inference result. No Router/RL result is claimed. See
`codex/ea-atpo-v3-expert-sft-20261120` and its follow-up audit branch.

## Existing GitHub data and artifact exposure

A schema-only scan of `origin/main` identified nine tracked JSONL files with
Gold/gold-text fields (9,106 rows counted across files; file overlap is not
deduplicated). The scan did not print or copy any record values. The affected
paths are:

- `experiments/ea_atpo_value_router_20261105/outputs/action_rollouts.jsonl`
- `experiments/ea_atpo_value_router_20261105/outputs/policy_episodes.jsonl`
- `experiments/ea_atpo_value_router_20261105/outputs/stepwise_test_v2/policy_episodes.jsonl`
- `experiments/ea_atpo_value_router_20261105/outputs/strong_expert_awq/qwen3_32b_direct_answers.jsonl`
- `experiments/real_case_pilot_20261103/audit/outputs/advice_utilization/advice_utilization_trajectories.jsonl`
- `experiments/real_case_pilot_20261103/audit/outputs/advice_utilization/frozen_consult_inputs.jsonl`
- `experiments/real_case_pilot_20261103/audit/outputs/consult_quality_sample.jsonl`
- `experiments/real_case_pilot_20261103/audit/outputs/direct_eval/direct_reliability_trajectories.jsonl`
- `experiments/real_case_pilot_20261103/outputs/full/pilot_trajectories.jsonl`

Two small Router `.npz` checkpoints are also already tracked on the public
`main` history. This repository-management change adds no case-level records,
Gold content, checkpoint, model weight, adapter, or secret. Because the user
prohibited history rewrite and asked to preserve historical experiments, no
existing file or commit was removed. The sensitivity and publication status
of the legacy records should receive a separate data-governance decision; the
presence of public benchmark data does not by itself establish that every
derived record is safe to publish.

## Cleanup and protection decisions

- No README deletion: no exact duplicates were found.
- No ignored/untracked file deletion: files in dirty worktrees remain owned by
  those windows; other ignored data may be unique or active.
- No `.gitignore` broadening: patterns such as `*.npz` can hide legitimate
  experiment artifacts, and current branch-specific ignore rules differ.
- No model or data artifact was copied into this management branch.
- Root indexes now document the status and PFS locators; specialized reports
  remain the source of truth for detailed results.

## Repository-management pushes

The documentation was staged and published in two non-force-pushed phases.
Phase 1 (`0a422442f5c99983180aed22d9d48e484da04247`) added the unified README,
documentation index, changelog, experiment index, version policy, and PFS
artifact locator. Immediately after pushing, local `HEAD` and
`git ls-remote origin refs/heads/codex/repo-management-20261120` both returned
that SHA. Phase 2 adds this audit; its final SHA is verified after pushing and
reported in the task completion message. No other branch is merged.
