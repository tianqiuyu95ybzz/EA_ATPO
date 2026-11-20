# Versioning and artifact policy

Updated: 2026-11-20.

## Research versions

V1, V2, and V3 name method-development stages, not stable releases or claims of
clinical readiness. Each run remains anchored by its Git commit, dated config,
data/model manifest, and report. A new cost, prompt, model revision, split,
continuation policy, or seed is a new run; do not silently replace old outputs.

Current branch responsibilities:

- `ea-atpo-v1`: V1 baseline and shared historical records.
- `codex/router-training-20261120`: Window A Router training and migration
  records.
- `codex/router-results-audit-20261120`: independent result audit.
- `codex/ea-atpo-v2-task-b`, `codex/ea-atpo-v2-window-b`,
  `codex/ea-atpo-v2-relative-router`, and
  `codex/ea-atpo-v2-final-eval-20261120`: separate V2 workstreams.
- `codex/ea-atpo-v3-expert-sft-20261120` and
  `codex/ea-atpo-expert-sft-20261120`: V3 preparation and follow-up audit.
- `codex/repo-management-20261120`: cross-version documentation and repository
  audit only.

Do not merge another window's branch as part of routine documentation work.
Use a separate reviewed integration change if branches later need to be
combined.

## Branch and worktree rules

1. Use a dedicated worktree per active window. Record its branch, base commit,
   and output directory in the run manifest.
2. Do not edit, reset, clean, or delete another worktree's dirty/untracked files.
   Treat ignored PFS outputs as owned by their experiment until explicitly
   released by that window.
3. Do not force-push, amend published commits, or rewrite history.
4. After each push, compare the local branch HEAD with the exact remote ref SHA.
5. Keep failed runs and negative results. Mark superseded configurations; do
   not remove the historical result.

## GitHub content rules

Suitable for GitHub:

- source code, tests, non-secret configs, documentation, aggregate metrics,
  sanitized logs, and manifests that contain hashes and non-sensitive locators;
- small model artifacts only when explicitly approved and proven non-sensitive.

Do not newly commit:

- patient-level records, public-state or dialogue text, Gold answers/rationales,
  hidden facts, private predictions, embeddings, raw rollout JSONL, or
  case-level identifiers;
- foundation-model weights, adapters, optimizer state, or large checkpoints;
- PATs, `.env` files, credentials, private service endpoints, or raw server logs.

The public repository already contains legacy case-level/Gold-bearing files in
its Git history, plus two small tracked Router `.npz` checkpoints. The
2026-11-20 audit records the paths and counts without exposing record values.
This management work does not erase that history. Do not interpret the current
ignore rules as proof that every historical file is safe to publish.

## PFS artifact policy

Keep large weights and private run data in PFS. In Git, preserve a locator with
relative path, owning branch, expected manifest/hash file, and restore command.
Before migration or cleanup, verify that the PFS copy exists and passes its
manifest. A path in this document is a locator, not proof that the artifact is
currently mounted or intact.
