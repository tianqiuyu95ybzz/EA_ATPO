# EA-ATPO V3 local-to-GitHub migration audit

Audit date: 2026-11-22 (Asia/Shanghai). Scope: EA-ATPO Git worktrees, public
GitHub branches, V3 current pilot artifacts, the supplied PFS recovery area,
and preparation for the user-specified JumpServer. No old file was deleted,
moved, or cleaned. No model was trained or called on GPU.

## 1. GitHub baseline and worktrees

Repository: `https://github.com/tianqiuyu95ybzz/EA_ATPO.git`.

After `git fetch ea-atpo`, the latest V3 audit branch was
`codex/ea-atpo-gold-oracle-audit-20261121` at
`8fcbd488987da4b7d5a2db9b9a063a11b382102f`. The task's remembered value
`8fcdb488...` has the `b` and `d` transposed; the fetched remote value is
`8fcbd488...`. Remote `main` was `ad9d0e31660b973a3e5ddfcadce6701f767e0f62`,
and `ea-atpo-v1` was `4c3a0516a16f326dd3ccdc386f611faa28e762e4`.

The EA_ATPO_V1 repository has 15 registered worktrees. At audit time, all
except the following had no tracked/untracked changes:

| Worktree | Branch / HEAD | Local difference | Disposition |
|---|---|---|---|
| `EA_ATPO_V1` | `ea-atpo-v1` / `4c3a0516...` | One tracked edit to `collect_counterfactuals.py`; 1,937 untracked paths. | Left untouched. The tracked diff is copied as a patch in this branch. |
| `EA_ATPO` legacy clone | `main` / `ad9d0e316...` | One tracked edit to `run_stepwise_router_eval.py`; 961 untracked paths. | Left untouched. Reviewed code/config/report updates and aggregate metrics were copied into the legacy experiment archive; case data and checkpoints were excluded. |
| `EA_ATPO_V1_repo_management_20261120` | `codex/repo-management-20261120` / `5c034c7...` | One untracked `audit_local_assets.py`. | Source was copied to this branch; original worktree was not changed. |

Other V1 worktrees, including the V3 Gold/Oracle audit worktree at
`8fcbd488...`, were clean in the read-only status scan. The existing V3 audit
branch is preserved as a base; this migration uses a separate branch and does
not merge or rewrite another window's history.

## 2. Itemized old-server asset classification

`git status --porcelain -z --untracked-files=all` found:

| Source | Untracked paths | Classification |
|---|---:|---|
| Legacy `EA_ATPO` clone | 961 | 939 cache/runtime files; 3 small Router checkpoint files; 11 source files; 1 config; 6 data/result files; 1 report. |
| `EA_ATPO_V1` primary worktree | 1,937 | 1,883 V2 parallel-cache files; 43 data/manifest files; 4 Attempt09/20 checkpoint-scoped `chat_template.jinja` copies; 3 logs. The four templates are byte-identical (SHA-256 `a55ee1b1…1974d8`). |
| Repository-management worktree | 1 | One inventory script. |
| **Total** | **2,899** | Individually recorded in the restricted PFS inventory, including relative path, size, category, file hash where scanned, and archive-match status. |

The prior 2026-11-20 private audit independently hashed all 2,898 legacy/V1
paths, found no exact match against the supplied Window A archive inventory,
and scanned structured keys and direct-identifier patterns. It flagged 2
sensitive candidates and 74 private/unknown files; 2,822 were regenerable
cache/runtime artifacts. This scan is heuristic and does not prove that other
files contain no sensitive text. The extra management script is item 2,899 in
the new private inventory. No per-case file or checkpoint was added to GitHub.

The full path-level inventory is intentionally private because filenames,
case mappings, hashes, and local paths can reveal experiment and case metadata.
Its location is listed in [PFS_ARTIFACT_INDEX.md](PFS_ARTIFACT_INDEX.md).

## 3. Important unpushed work and publication decision

- The V1 patient-simulator/counterfactual collector edit changes semantic
  matching and adds `top_p`. It was preserved as a patch, not silently applied
  to the current V3 implementation.
- The legacy clone's relative-value Router implementation, minimal-fix
  diagnostics, configs, run scripts, method audit, and aggregate precheck
  metrics were absent from the V3 branch at matching blob hashes. Reviewed
  non-sensitive source and aggregate results are copied under
  `experiments/ea_atpo_value_router_20261105/`. They are explicitly marked as
  historical and are not the V3 Router or an independent test result.
- The untracked `case_split.json`, case-level state/action Q targets, Router
  NPZ/JSON checkpoints, V2 parallel cache, service logs, raw patient outputs,
  and review mappings remain local/private and are not committed.
- The management inventory script was copied to
  `scripts/repo_management/audit_local_assets.py`. It writes sensitive
  file-path/hash inventories and must be directed to a restricted output path.
- The four V1 `chat_template.jinja` files are identical and belong to the
  private Attempt09/Attempt20 recovery tree. They were not copied as general
  project code: the current V3 pilot uses the raw Qwen3-8B model's native
  tokenizer template, while the associated Actor/Critic weights remain
  private and are not part of this restore.

## 4. V3 current pilot integrity and evidence status

The current pilot records 50 distinct ReMeDi dialogue IDs and 100 completed
ANSWER/ASK-Replay generations. The selected human-review subset is 20 paired
cases / 40 outputs. The source output, source review packets, restricted review
packets, and case mapping all match their existing private SHA-256 manifests.
The review status remains 0/40 scored responses and 0/20 scored case pairs;
external review is not authorized. DEFER has no comparable initial-state
result and remains N/A. No Router was trained for this V3 pilot.

The model manifest pins `Qwen/Qwen3-8B` revision
`b968826d9c46dd6066d109eabc6255188de91218`, with five original model shards and
no adapter. This is the raw pretrained Base, **not** an SFT-ATPO model. The
checkpoint-lineage audit did not locate a verifiable SFT+ATPO Qwen3-8B final
checkpoint. Attempt20 Actor weights are ATPO-only from the raw base and are
not a substitute.

Git-tracked V3 summary CSVs contain hashed case references and completion
flags, not answer text. The full transcripts, original outputs, restricted
review packets, and mappings remain mode 0600 in private PFS storage. The
published SHA list for the V3 pilot passes; the restricted review packet and
case-mapping hashes also match the manifests. SHA checks do not grant rights
to distribute the data.

## 5. Checkpoints, model files, environment, and backups

- Original Qwen3-8B base weights exist on old-host PFS at the path in
  `PFS_ARTIFACT_INDEX.md`; the five shard files total about 16.4 GB. Their
  revision/index/tokenizer identity is recorded. They are not on GitHub.
- Attempt09 Actor/Critic and Attempt20 Actor recovery exports exist under the
  old V1 worktree, totaling about 47.9 GB. They were not used for the current
  V3 pilot. The Attempt09 shard set has no verified per-shard manifest in its
  recovery directory and no matching file in the checked Window A archive
  inventory.
- The previously verified Window A archive contains V2 Router checkpoints,
  not V3 model weights or the Attempt09 recovery tree. Its manifest had zero
  exact hash matches among the 2,898 audited legacy/V1 untracked files.
- PFS Oracle runs 07 and 08 have valid `SHA256SUMS`; they are CPU-only Dev
  outputs and their state-level JSONL remains private.
- Old host: `b200-world-2`, system Python 3.12.3, Git 2.43.0, PFS mount has
  94 TB free at audit. System Python has no Torch/Transformers/PEFT/TRL. The
  V3 run manifest records a separate execution runtime of Transformers 5.20.2
  and Torch 2.14.0+cu130, CPU-only; Python executable/version for that runtime
  was not recorded.

## 6. New-server connection and restore status

The user-provided SSH address was reached and the host key was stored in a
PFS-local `known_hosts` file. The session is waiting at the JumpServer password
prompt. No password was requested or entered; no node was guessed or selected.
Therefore hostname, GPU model/count/memory, driver/CUDA, Python/PyTorch/Git,
disk, and GitHub connectivity on the new server are **not yet verified**.

No clone or GPU smoke test has been run there. After the user completes
interactive authentication (and selects a node if prompted), follow
`MIGRATION_RESTORE.md`. Inspect `nvidia-smi` and existing processes before any
GPU smoke. Run a short synthetic-prompt smoke only if a GPU is clearly idle;
do not run training or a 50-case inference job as part of this migration.

## 7. GitHub publication boundary

This migration branch includes documentation, reviewed code/configuration,
aggregate reports, and SHA manifests. It excludes patient text, Gold, case
mappings, per-case outcomes, output/review text, credentials, model weights,
and checkpoints. No force-push or history rewrite is used. The initial
migration payload commit `398128b00b36d45a5fd24318e054f348299bd6ab` was pushed
to `codex/ea-atpo-v3-migration-20261122`; a subsequent fetch returned the same
SHA. Its parent is the audited V3 base `8fcbd488987da4b7d5a2db9b9a063a11b382102f`.
The remote tree contains all 34 manifest entries plus the manifest and its
sidecar; the local/remote trees matched at verification time.

### Previously published files on the migration base

The selected base already contains the legacy Router artifacts from the
previously pushed `ad9d0e3` mainline commit. A metadata-only audit of the
tracked files confirmed that `case_split.json` and `state_action_targets.csv`
contain case/state identifiers, while the tracked `action_rollouts.jsonl`,
`policy_episodes.jsonl`, stepwise episodes, and direct-answer JSONL have fields
for `gold`, model outputs, patient/expert text, and/or public states. The two
Router NPZ checkpoints are also already tracked. None of these files is part
of this migration's new staged payload or SHA manifest, and this work does not
rewrite or delete their history. Since they are already reachable on the
public remote, this migration cannot certify the existing repository history
as free of case-level or Gold-bearing artifacts. Repository owners should
review the prior publication and applicable data rights separately; any
history remediation would require a separately authorized operation.
