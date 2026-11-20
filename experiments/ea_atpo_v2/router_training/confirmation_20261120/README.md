# Frozen Router confirmation readiness audit

This Window A directory freezes the best V2 Router family and audits whether
the nominal remaining official Train cases can serve as new-case confirmation.
The audit found no untouched cases, so no confirmation inference or episode was
started. The 400-case Frozen Holdout was not opened.

## Frozen candidate

The primary candidate is the three-seed **Embedding Q-guided Distillation**
family selected on the existing 15-case Dev split. All three checkpoint files
are pinned by SHA-256 in `configs/frozen_candidate_20261120.json`. The requested
Embedding Relative Advantage MLP comparator is frozen there as well. The seeds
are evaluated separately and averaged only as a predeclared replicate summary;
no seed is selected from confirmation data and no new ensemble is created.

The embeddings and Router checkpoints remain in the existing ignored PFS
`outputs/real_q_router_20261120/` directory. This folder contains no patient
text, case-level predictions, checkpoint binaries, or case identifiers.

## Reproduce the audits

From the `EA_ATPO_V1_router_training` worktree root:

```bash
python3 -m experiments.ea_atpo_v2.router_training.confirmation_20261120.scripts.audit_confirmation_cohort \
  --root . \
  --output experiments/ea_atpo_v2/router_training/confirmation_20261120/outputs/cohort_audit_aggregate.json

python3 -m experiments.ea_atpo_v2.router_training.confirmation_20261120.scripts.recheck_prior_dev \
  --root . \
  --output experiments/ea_atpo_v2/router_training/confirmation_20261120/outputs/prior_dev_metric_recheck.json
```

The first command reads identifiers, split/checkpoint metadata, and hashes of
historical public initial messages; it does not print case IDs or message text.
The second command reconciles already-used Dev state metrics against saved
per-case metrics. Neither command opens the Frozen Holdout.

See `reports/CONFIRMATION_READINESS_AND_CONTAMINATION_AUDIT_20261120.md` for the
result and decision.
