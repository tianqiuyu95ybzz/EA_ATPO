# Changelog

## 2026-11-22 — server migration and recovery audit

- Added a current experiment index, versioning rules, PFS artifact index, and
  old-server/local-versus-GitHub migration report.
- Added a reproducible GitHub restore guide and a public-file SHA-256 manifest.
- Preserved audited, non-sensitive legacy Router source and aggregate metrics
  from the older clone. Case-level targets, case splits, checkpoints, raw
  medical text, service logs, and weights were not copied to GitHub.
- Recorded the uncommitted V1 runner change as a patch artifact without
  modifying the source worktree.
- Confirmed the V3 pilot used original Qwen3-8B, not SFT+ATPO; 20 pairs / 40
  responses remain unscored, and DEFER remains N/A.
- No model training or inference was started. New-server inspection is pending
  interactive JumpServer authentication.

## Earlier history

See the immutable Git commit history and `PUBLISHED_RESULTS_INDEX.md`. Historical
failures and development-only results remain in their original experiment
directories.
