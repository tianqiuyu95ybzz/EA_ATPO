# EA-ATPO Window A Result Registry — 2026-11-20

| ID | Experiment | Data scope | Status | Key result | Interpretation |
|---|---|---|---|---|---|
| WA-V2-RTR-01 | Counterfactual Q Regression + lightweight Router | 50 official Train cases; 146 states; 438 valid trajectories; 35/15 case split | Complete | Primary embedding Q-guided Distillation: Dev Accuracy 0.507, Utility 0.491; paired case-clustered Utility 95% CI vs Always ANSWER [-0.014, 0.297] | Dev set was also used for checkpoint selection; development-only, not independent confirmation |
| WA-V2-RTR-02 | Embedding Relative Advantage MLP comparator | Same Train/Dev partition and costs | Complete | Dev Accuracy 0.493, Utility 0.469 | Development-only comparator; not independent confirmation |
| WA-V2-EXP-PAIR | Base/Expert paired Dev states | 43 Dev states from 15 Dev cases | Complete | Four cells 11 / 6 / 9 / 17; Expert correction 9/26 Base-wrong; spoilage 6/17 Base-correct | Paired state-level Dev analysis; no holdout inference |
| WA-V2-CONFIRM-AUDIT | Proposed remaining-Train confirmation cohort | Nominal 20 cases | Blocked by contamination | All 20 were found in historical V1 Router fit/target data; eligible unseen cases: 0 | No new case-level generalization evidence |
| WA-V2-HOLDOUT | Frozen final holdout | 400 cases | Sealed | Not read, trained on, tuned against, or evaluated in this work | Must remain sealed until a separately registered final protocol |

## Frozen candidate artifacts

The selected candidate family is Embedding Q-guided Distillation. Its three
seed checkpoints and the three Embedding Relative Advantage MLP comparator
checkpoints are frozen in
`confirmation_20261120/configs/frozen_candidate_20261120.json`, including
checkpoint SHA256, best epoch, validation loss, model revision, prompt/template
hash, 4096-dimensional feature definition, cost schedule, and training setup.
That JSON is the canonical model-selection record; the data archive contains
the corresponding predictions, logs, feature cache, and training outputs.

## Reproducibility and privacy

The complete source and aggregate reports are registered under this directory.
Case-level public states, trajectories, Q labels, embeddings, split mapping,
per-state predictions, and diagnostics are stored only in the private
workspace-copy archive, not added to GitHub by this update. The GitHub remote is
public, and its existing history includes some previously committed
case-level JSONL files; this update does not rewrite that history.

See `MIGRATION_HANDOVER.md` and the external archive's
`ARCHIVE_README.md` / `MANIFEST.sha256` for restoration instructions and
artifact verification.
