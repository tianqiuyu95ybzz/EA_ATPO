# Gold provenance and independent Oracle audit

Start with `reports/GOLD_SOURCE_AUDIT.md`, `reports/INDEPENDENT_CASES_AND_BLOCKERS.md`, and `reports/ORACLE_REPLAY_AND_FINAL_DECISION.md`. Private source archives and answer-bearing record mappings remain under ignored `private/` on PFS. The committed CSV contains only official IDs and hashes, not question or answer text.

Reproduction uses requests for author downloads and the existing project venv for CPU replay. Fresh run required: scripts refuse existing raw download/mapping/replay outputs; preserve prior private files and run in a new worktree/experiment copy.

```sh
python3 experiments/ea_atpo_v3/gold_oracle_audit_20261121/scripts/download_official.py
python3 experiments/ea_atpo_v3/gold_oracle_audit_20261121/scripts/audit_gold_provenance.py
python3 experiments/ea_atpo_v3/gold_oracle_audit_20261121/scripts/prepare_independent_candidates.py
CUDA_VISIBLE_DEVICES= /pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO_V1/.venv/bin/python experiments/ea_atpo_v3/gold_oracle_audit_20261121/scripts/replay_provenance_subset.py
```

The audit and replay explicitly depend on existing local ATPO source, V2 trajectories and Oracle run_08 private state records (the latter include frozen Router actions). Exact paths are in the scripts and source files are read-only. Input SHA values are recorded in the aggregate manifests. New public candidate IDs are unverified and do not grant permission to train or evaluate until provenance, history and protocol gates are closed. No 400-case Frozen Holdout files are read.
