# Route change: pause Expert SFT; evaluate an answer-key Oracle

**Date:** 2026-11-21 (Asia/Shanghai)

**Scope:** EA-ATPO V3 only; historical V1/V2 experiments remain unchanged.

## Decision

Pause the planned Expert A/B dual-model SFT route. The current target is an
offline Oracle Expert upper-bound analysis: use a source benchmark answer key
only after the Router's decision has been fixed, preserve a correct Base answer,
and return the source answer key when the Base answer is wrong.

## Reason

The V3 291-row queue is not verified supervision: all 291 rows are still
`PENDING`. The 169 differences from the historical labels are only review
priorities and cannot be treated as 169 Base errors. Training a new Expert on
those unreviewed labels would risk converting label disagreement into false
supervision. The immediate question—whether a Router can direct a perfect
answer-key Expert toward Base errors—can be tested without training an Expert.

The Oracle is an exam-benchmark idealization, not a clinician, a clinical
adjudication, or a deployable Expert. The selected source answer keys are
provenance-checked and exactly reconcile with the existing 129 Dev rollout
outcomes, but have not received independent clinician review.

## Phase 0 actions and preservation

- Recorded branch, HEAD, status, worktrees and GPU/process status before work.
- No formal Expert SFT/LoRA training process was running; no SFT process was
  stopped. No unrelated or other-session process was touched.
- No GPU or model service was used for the Oracle run.
- Preserved the existing 10-step AWQ LoRA compatibility smoke, model/config
  locks, data audits, training scripts, all SFT reports and checkpoints.
- Did not open the 400-case Frozen Holdout or the V3 Internal Test split.
- The Oracle branch uses a separate worktree and writes case-level output to
  a private PFS directory. Public Git contains only code, configuration,
  aggregate metrics, and reports.

## Route gates

1. Treat the current Oracle results as development-only because the frozen
   Router was selected using this same 15-case Dev set.
2. Do not claim independent Router generalization or use the Oracle as a
   production model.
3. Any later Router confirmation requires untouched case IDs, a predeclared
   answer-key source, locked costs, and a single frozen policy evaluation.
4. Resume Expert SFT only under a new explicit research decision with an
   approved supervision source; do not repurpose the pending 291 labels.
