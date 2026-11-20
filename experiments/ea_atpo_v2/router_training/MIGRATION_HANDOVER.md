# EA-ATPO Window A Migration Handover — 2026-11-20

## Scope and frozen state

This handover covers only Window A's EA-ATPO V2 Router study. The Git source
snapshot is branch `codex/router-training-20261120`, initially at
`54075815f2a7580765c493d7f303d9ad2cd7eaf3`; that commit was present on the
GitHub remote when this archive was prepared. The study is a development-only
counterfactual Router experiment, not an independent confirmation result.

The primary frozen candidate is the three-seed frozen-Qwen3-8B embedding
Q-guided Distillation Router (seeds 17, 42, 2026). The three-seed Embedding
Relative Advantage MLP is retained as a comparator. Exact checkpoint paths,
configuration, architecture, feature dimension, training parameters, and
checkpoint SHA256 values are recorded in
`confirmation_20261120/configs/frozen_candidate_20261120.json` and the private
checkpoint archive manifest.

## Data and conclusions

- The V2 source cohort contains 50 Train cases, 146 public pre-action states,
  and 438 valid counterfactual ASK/ANSWER/DEFER trajectories (one rollout per
  state/action). The Router split is by `case_id`: 35 Train cases / 103 states
  and 15 Dev cases / 43 states.
- Registered costs are ANSWER 0.00, ASK 0.01, DEFER 0.05. The Q targets already
  include these costs exactly once; inference must not subtract them again.
- On the 15-case development set, Q-guided Distillation had Accuracy 0.507 and
  Utility 0.491. Its case-clustered paired Utility 95% CI versus Always ANSWER
  was [-0.014, 0.297]. This is development/model-selection evidence only.
- For the 43 paired Base/Expert Dev states, the four cells are 11 / 6 / 9 / 17
  (both correct / Base correct-Expert wrong / Base wrong-Expert correct / both
  wrong). Expert corrected 9/26 Base errors and spoiled 6/17 Base-correct
  states. These are state-level Dev counts, not independent-test evidence.
- The nominal remaining 20 cases in the official 70-case Train pool had
  already appeared in historical V1 Router fit/target data. There are zero
  eligible unseen cases for confirmation. No new case-level generalization
  evidence was produced.
- The 400-case Frozen Holdout remains sealed: it was not read, trained on,
  tuned against, or included in this archive.

## What is in the workspace-copy archive

The sibling archive root is
`/pfs/pfs-spMCeu/image_team/workspace-copy/EA-ATPO_Window_A/`. It contains
separate code/report, private case-level data, and checkpoint archives. All
files keep their project-relative paths so they can be extracted back into the
EA_ATPO_V1 project root. See that directory's `ARCHIVE_README.md` and
`MANIFEST.sha256` for exact contents, checksums, verification status, and
restore commands. The private archives are not to be committed to GitHub.

## Restore and reproduce

From the project root after retrieving the archive directory through approved
private storage:

```bash
zstd -dc /path/to/EA-ATPO_Window_A/archives/code_and_reports.tar.zst | tar -xf -
zstd -dc /path/to/EA-ATPO_Window_A/archives/private_experiment_data.tar.zst | tar -xf -
zstd -dc /path/to/EA-ATPO_Window_A/archives/router_checkpoints.tar.zst | tar -xf -
```

Install the pinned dependencies listed in `requirements-ea-atpo-v1-resolved.txt`
or the CUDA-specific `requirements-ea-atpo-v1-cu128.txt`, as appropriate. The
frozen embedding base is `Qwen/Qwen3-8B` revision
`b968826d9c46dd6066d109eabc6255188de91218`; model weights are not bundled and
must be downloaded at that revision. Re-run commands in
`reports/RUN_COMMANDS_20261120.md` from the worktree root. The private data
bundle is needed for the commands that consume case-level states, labels, and
features.

The Router `.pt` files are inference checkpoints. They include model and
feature state, but do not contain optimizer, scheduler, epoch continuation, or
RNG state; they are not exact mid-training resume checkpoints. Full retraining
is reproducible from the archived private inputs/configuration and pinned
software/model revisions, subject to hardware/library numerical differences.

## GitHub and privacy note

The remote repository `tianqiuyu95ybzz/EA_ATPO` is public. This archive task
does not add case-level data, predictions, embeddings, or checkpoints to
GitHub. The current public Git history already contains some historical
case-level JSONL paths; those pre-existing commits were not rewritten here.
Treat that as an existing exposure risk and review repository access/history
separately before placing any newly collected private clinical data there.

## Safe migration

The archive is a copy; source files and the original worktree were not deleted,
moved, or overwritten. After the archive passes checksum, extraction, checkpoint
load, and inference-smoke verification, the GPU allocation can be released.
Do not clean the source worktree until the restored copy has been checked in its
destination environment and any repository-history privacy decision has been
made.
