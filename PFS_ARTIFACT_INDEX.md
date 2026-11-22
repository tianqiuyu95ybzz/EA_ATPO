# PFS artifact index

This index points to restricted assets; it does not make them public or grant
access. Paths below are relative to locally configured roots. The owner-specific
PFS prefixes are intentionally omitted from this public repository. On a host
that has the old PFS mounted, resolve them with local variables such as
`$EA_ATPO_OLD_WORKSPACE` and `$EA_ATPO_WORKSPACE_COPY`; do not assume those roots
exist on the new server.

| Asset group | Old-host location | Status / use | GitHub policy |
|---|---|---|---|
| V3 50-dialogue pilot inputs and 100 raw outputs | `$EA_ATPO_OLD_WORKSPACE/EA_ATPO_V1_gold_oracle_audit_20261121/experiments/ea_atpo_v3/chinese_pilot_20261121/private/` | Files are mode 0600 in a restricted directory. Output SHA matches the committed run manifest. | Never upload source dialogue or outputs. |
| V3 20-case / 40-response review package and case mapping | `$EA_ATPO_OLD_WORKSPACE/EA_ATPO_V1_gold_oracle_audit_20261121/experiments/ea_atpo_v3/chinese_pilot_20261121/{quality_audit_20261121,review_privacy_20261122,human_review_readiness_20261122}/private/` | Restricted review packet and case-mapping hashes match manifests. 0/40 response ratings and 0/20 paired scores; external review is not authorized. | Never upload review text, mappings, or score forms. |
| Original Qwen3-8B base weights | `$EA_ATPO_OLD_WORKSPACE/EA_ATPO/experiments/real_case_pilot_20261103/assets/models/Qwen3-8B/` | 5 original shards (~16.4 GB); revision is `b968826d9c46dd6066d109eabc6255188de91218`. Index/tokenizer hashes are pinned in the V3 model manifest. This is the model used by the V3 CPU pilot. | No weights in GitHub. |
| Attempt09 / Attempt20 Actor-Critic recovery | `$EA_ATPO_OLD_WORKSPACE/EA_ATPO_V1/experiments/ea_atpo_v1_smoke/ea_atpo_v1_attempt09_stability/recovery/` | Local recovery tree is ~47.9 GB. Attempt20 Actor is not an SFT+ATPO checkpoint; Attempt09 Actor/Critic were not used by the V3 pilot. Four checkpoint-scoped `chat_template.jinja` copies are byte-identical (SHA-256 `a55ee1b1…1974d8`). No matching entry was found in the checked Window A archive inventory, and the Attempt09 weight shards lack a verified checksum manifest. | Keep private; do not substitute for the V3 base model. |
| Window A V2 Router archive | `$EA_ATPO_WORKSPACE_COPY/EA-ATPO_Window_A/` | Archive integrity and restore were previously verified. Contains 146 states / 438 counterfactual trajectories, private inputs and Router checkpoints. | Keep the private archives restricted; publish only code and aggregate report. |
| V3 Oracle Dev runs 07/08 | `$EA_ATPO_WORKSPACE_COPY/EA-ATPO_Oracle_20261121/run_07/` and `run_08/` | SHA manifests verify. CPU-only, Dev-only, no Frozen Holdout. Per-state result JSONL is private. | Do not upload per-state files. |
| Full untracked-path audit | `$EA_ATPO_PRIVATE_AUDIT/old_server_untracked_assets_PRIVATE.csv` | Itemized metadata for the 961 legacy-clone paths, 1,937 V1 paths, and the additional management script. The CSV contains local paths and hashes; retain privately. | Do not commit the inventory. |
| Previous detailed privacy scan | `$EA_ATPO_PRIVATE_AUDIT/EA-ATPO_Repository_Audit_20261121/private_assets/` | Detailed inventory for the 2,898 legacy/V1 paths as of 2026-11-20. It flagged 2 sensitive candidates and 74 private/unknown assets; none matched the supplied Window A archive manifest. | Restricted inventory; do not publish. |

No SFT+ATPO final Qwen3-8B checkpoint was found in the audited old-host assets.
The V3 pilot's current evidence must therefore remain labeled as original-base
Qwen3-8B. The new server's PFS visibility is not yet confirmed.
