# PFS artifact index

Updated: 2026-11-20. This is a locator registry, not a copy of any private
artifact. It contains no patient records, Gold text, weights, adapter contents,
credentials, or case identifiers. Verify the named manifest before recovery.

| Owner / version | PFS location | Required locator or integrity record | Contents / status |
|---|---|---|---|
| V1 Attempt09 | `/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO_V1/experiments/ea_atpo_v1_smoke/ea_atpo_v1_attempt09_stability/` | `run_manifest.json`; `paired_evaluation/run_manifest.json`; `checkpoints/global_step_10/` | Actor/Critic/Tokenizer checkpoint and evaluation artifacts. The public report says checkpoint weights and Parquet cases are PFS-only. Confirm the run manifest and hashes before restore. |
| V2 Window A Router | `/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO_V1_router_training/experiments/ea_atpo_v2/router_training/outputs/real_q_router_20261120/` | Branch-tracked `confirmation_20261120/configs/frozen_candidate_20261120.json`; private checkpoint archive manifest; workspace-copy `EA-ATPO_Window_A/MANIFEST.sha256` | Six frozen inference checkpoints: three Q-guided Distillation seeds and three Relative Advantage MLP comparator seeds; embeddings, predictions, costed labels, and case-level run data. Checkpoints do not include optimizer/scheduler/RNG resume state. |
| Window A verified archive | `/pfs/pfs-spMCeu/image_team/workspace-copy/EA-ATPO_Window_A/` | `MANIFEST.sha256`, `ARCHIVE_README.md`; `archives/code_and_reports.tar.zst`, `archives/private_experiment_data.tar.zst`, `archives/router_checkpoints.tar.zst` | Verified copy of code/reports, private experiment data, and checkpoints. Private archives are not for public GitHub. Archive extraction and checkpoint loading were reported as verified in the handover. |
| V2 frozen final-evaluation cohort | `/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO_V1_final_eval/experiments/ea_atpo_v2/final_evaluation/manifests/final_test_20261120_v1/` | `case_manifest.json`, `overlap_audit.json`, `overlap_audit_detailed.json` | Frozen cohort design and overlap audit only. No final model predictions/outcomes. Keep the cohort sealed until the registered evaluation is run. |
| V3 Expert SFT | `/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO_V1_expert_sft_20261120/experiments/ea_atpo_v3/expert_sft/` | branch-tracked `SHA256SUMS`; `data/data_manifest.json`; `data/expert_b_initialization_manifest.json`; `data/awq_smoke_manifest.json`; local ignored `data/pfs_artifact_inventory_v2.json` | PFS-only model shards, AWQ-LoRA smoke adapter, private Base outputs/review data, and run logs. Formal Expert SFT was not reported complete; the smoke adapter is not a production Expert. |

## Recovery procedure

1. Check out the owning branch from `EXPERIMENT_INDEX.md`.
2. Resolve the PFS mount and verify the expected manifest before using files.
3. Restore only the required artifact subset to the documented relative path.
4. Re-run the owning experiment's integrity or smoke command from its README.
5. Keep the source PFS copy until the recovered copy passes hash and load checks.

Do not run `git clean`, delete caches, or remove PFS folders based only on this
index. Worktree ownership and ignored files are recorded in the
[repository audit](docs/repo_management/LOCAL_REMOTE_AUDIT_20261120.md).
