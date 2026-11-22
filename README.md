# EA-ATPO research project

This repository contains research additions and experiment plans for extending
ATPO multi-turn medical dialogue with sequential patient and expert
collaboration. It does not contain the official ATPO checkout, private data,
model weights, checkpoints, or previous server logs.

## Start here

- [Documentation index and current sources of truth](DOCUMENTATION_INDEX.md)
- [Experiment index](EXPERIMENT_INDEX.md)
- [Latest migration audit](LOCAL_REMOTE_MIGRATION_AUDIT.md)
- [Migration asset triage and recovery status](MIGRATION_ASSET_TRIAGE.md)
- [PFS artifact index](PFS_ARTIFACT_INDEX.md)
- [Restore and continue on a new server](MIGRATION_RESTORE.md)
- [Versioning rules](VERSIONING.md)
- [Change log](CHANGELOG.md)
- [Migration SHA-256 manifest](MIGRATION_SHA256SUMS)
- [Project layout and upstream dependency](PROJECT_LAYOUT.md)
- [Research review and literature positioning](EA_ATPO_RESEARCH_REVIEW_AND_NEW_METHOD_2026-11-02.md)
- [Current roadmap and innovation status](EA_ATPO_ROADMAP_2026-11-02.md)
- [Real-data experiment preregistration and annotation protocol](EA_ATPO_EXPERIMENT_PREREGISTRATION_2026-11-03.md)
- [EA-ATPO V1 implementation record](EA_ATPO_V1_IMPLEMENTATION.md)
- [EA-ATPO V1 latest results and stability report](EA_ATPO_V1_SMALL_SCALE_RESULTS.md)
- [Published experiment results index](PUBLISHED_RESULTS_INDEX.md)
- [Real-case feasibility pilot: preflight, runner, and data recovery](experiments/real_case_pilot_20261103/README.md)
- [Reliability-conditioned router synthetic pilot](experiments/reliability_conditioned_router_20261102/README.md)
- [V3 Chinese dialogue pilot and current review status](experiments/ea_atpo_v3/chinese_pilot_20261121/README.md)
- [V3 checkpoint lineage audit](experiments/ea_atpo_v3/atpo_checkpoint_audit_20261121/REPORT.md)

The synthetic pilot is only a routing-logic feasibility check. It does not
replace the required paired rollout and held-out evaluation with the project's
medical benchmark and model checkpoints.

## Current V3 status (2026-11-22)

The current 50-dialogue ReMeDi pilot used the pinned original `Qwen/Qwen3-8B`
base revision. It did not use an SFT+ATPO checkpoint. Twenty ANSWER/ASK pairs
(40 response records) are prepared for human review; no human scores exist,
and same-state DEFER remains unavailable. See the experiment index and audit
before using any result as a model-quality claim. Private dialogues, review
packets, case mappings, and all weights remain outside GitHub.
