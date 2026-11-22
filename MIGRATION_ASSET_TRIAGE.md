# EA-ATPO V3 migration asset triage

Audit date: 2026-11-22. This is an aggregate inventory; it contains no case
IDs, patient dialogue, Gold, model weights, or checkpoint digests.

## Old-server worktree inventory

The audit recorded 2,899 untracked paths across three worktrees, plus two
tracked source modifications. The original worktrees remain unchanged.

| Source | Count | Classification and disposition |
|---|---:|---|
| EA_ATPO_V1 primary worktree | 1,937 | 1,883 runtime/cache files; 43 data/result/manifest files; four checkpoint-scoped chat templates; four tokenizer merge assets; three logs. The tracked collector change is preserved as a reviewable patch. Data, logs, and checkpoint-scoped files were not added to GitHub. |
| Legacy EA_ATPO clone | 961 | 939 runtime/cache files; 11 Router/analysis scripts; one config; six data/result files; three Router checkpoint artifacts; one method report. Reviewed source, config, and aggregate-only metrics were copied to this migration branch; case-level data and checkpoints were excluded. |
| Repository-management worktree | 1 | One audit script, copied to this migration branch. |

The four checkpoint chat templates are byte-identical. They remain with the
private Attempt09/20 recovery assets; the current V3 pilot uses the native
tokenizer template for the pinned raw Qwen3-8B model.

## Checkpoint inventory

- The Attempt09 Actor, Attempt09 Critic, and Attempt20 Actor Hugging Face
  inference exports contain 12 safetensors shards across three indexed model
  directories. All index-referenced shards exist; each index and shard header
  passed a CPU structural check. SHA-256 was calculated for all file targets.
- A separate `global_step_10` training checkpoint contains 32 files totaling
  189,142,648,842 bytes, including Actor/Critic model shards and optimizer
  shards. All 32 files were hashed. This establishes file integrity, not a
  successful training-resume test; no model was loaded and no GPU was used.
- The recovery tree includes symlinks into that training checkpoint. The
  private manifest records link targets and verified target hashes.
- The current V3 pilot does not use these Attempt09/20 weights. It uses the
  pinned original Qwen3-8B base model. No verified SFT+ATPO Qwen3-8B checkpoint
  was located.

Detailed shard manifests and exact hashes stay in restricted PFS storage and
are intentionally excluded from this public repository.

## Migration state

| Asset | GitHub | Old server / PFS | New server |
|---|---|---|---|
| Migration code, docs, configs, reviewed aggregate results | Pushed on `codex/ea-atpo-v3-migration-20261122` | Preserved | Not cloned yet |
| Private dialogue, case mapping, review text, and per-case outputs | Not included in this migration commit | Preserved in restricted storage | Not transferred |
| Attempt09/20 checkpoints and exact SHA manifests | Not included | Present and hashed as described above | Not transferred |
| Original Qwen3-8B base weights | Not included; public model revision is pinned in the manifest | Available on old PFS | Not restored |

SSH authentication to the new server has not completed. Therefore its compute
node, GPU, driver/CUDA, Python/PyTorch, disk, GitHub connectivity, and restore
smoke status remain unverified. No SSH retries, training, or GPU jobs were run
as part of this audit.

## Existing public-history risk

The GitHub repository was confirmed public. Earlier commits already contain
case-level and Gold-bearing Router artifacts. This migration did not add or
rewrite those historical files. Restrict repository access while reviewing
data rights and privacy; history rewriting requires separate authorization.
