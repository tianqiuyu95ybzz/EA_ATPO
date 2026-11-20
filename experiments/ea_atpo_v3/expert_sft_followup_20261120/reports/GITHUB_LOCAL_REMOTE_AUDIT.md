# GitHub and local audit

Snapshot 2026-11-20: origin fetched; remote repository https://github.com/tianqiuyu95ybzz/EA_ATPO.git. Baseline 80834ad4890a4200549238a4fc28a970c8eea9f6; this branch codex/ea-atpo-expert-sft-20261120. Remote containment below is commit reachability, not a claim that every matching branch name exists remotely.

| Local branch | HEAD | Remote refs containing commit |
|---|---|---|
| codex/ea-atpo-expert-sft-20261120 | 80834ad4890a4200549238a4fc28a970c8eea9f6 | origin/codex/ea-atpo-v3-expert-sft-20261120 |
| codex/ea-atpo-v2-final-eval-20261120 | b11121c18a5ecab7f9a17ee2c90d273dd32707ce | origin/codex/ea-atpo-v2-final-eval-20261120 |
| codex/ea-atpo-v2-relative-router | 8336c37c125382dbf01ef2b6e9f312376e73b8e5 | origin/codex/ea-atpo-v2-final-eval-20261120 |
| codex/ea-atpo-v2-task-b | 66e476d94fd6d4a944018d67bc7565b613432122 | ea-atpo/codex/router-training-20261120, ea-atpo/ea-atpo-v1, origin/codex/ea-atpo-v2-final-eval-20261120, origin/codex/ea-atpo-v2-window-b, origin/codex/ea-atpo-v3-expert-sft-20261120, origin/codex/router-results-audit-20261120, origin/codex/router-training-20261120, origin/ea-atpo-v1 |
| codex/ea-atpo-v2-window-b | d375f892d8019cc2b034dc173cd07c44ec5c40ab | origin/codex/ea-atpo-v2-final-eval-20261120, origin/codex/ea-atpo-v2-window-b |
| codex/ea-atpo-v3-expert-sft-20261120 | 80834ad4890a4200549238a4fc28a970c8eea9f6 | origin/codex/ea-atpo-v3-expert-sft-20261120 |
| codex/router-results-audit-20261120 | 9b3f978ec9d0d8fb2d7dcc550fd2295743329777 | origin/codex/router-results-audit-20261120 |
| codex/router-training-20261120 | ba698a8d8176ae2dfe34e8f654adc12ca6208530 | origin/codex/router-results-audit-20261120, origin/codex/router-training-20261120 |
| ea-atpo-v1 | 4c3a0516a16f326dd3ccdc386f611faa28e762e4 | ea-atpo/codex/router-training-20261120, ea-atpo/ea-atpo-v1, origin/codex/ea-atpo-v2-final-eval-20261120, origin/codex/ea-atpo-v2-window-b, origin/codex/ea-atpo-v3-expert-sft-20261120, origin/codex/router-results-audit-20261120, origin/codex/router-training-20261120, origin/ea-atpo-v1 |

All current local branch HEADs are reachable from at least one remote ref; several local branch names do not have their own remote branch. The main EA_ATPO_V1 worktree has a pre-existing modified collector and untracked recovery/cache/VI artifacts. Other worktrees were inspected read-only and their changes were not imported. Untracked material is not automatically safe to publish; file-level review is still required.

| Material | Local | Remote | Treatment |
|---|---|---|---|
| V3 code/configs/audit/aggregate manifests | committed | origin/codex/ea-atpo-v3-expert-sft-20261120 | retained and extended on this branch |
| Other worktree changes and untracked runtime files | preserved in original worktrees | not established file-by-file | do not copy or publish without review |
| Base answers, review queue, B weights, smoke adapter | private PFS | intentionally excluded from new commits | retained; inherited PFS inventory |
| Historical tracked JSONL trajectories | present in Git history | inherited in published baseline | audit risk; no history rewrite or deletion |

Important: the tracked historical pilot_trajectories.jsonl contains simulated public dialogue and a gold field. This is a protected-answer/redistribution risk, not evidence that real patient records were exposed. No secret-scanning conclusion is claimed. No destructive cleanup, history rewrite, force push or foreign-worktree modification was performed.
