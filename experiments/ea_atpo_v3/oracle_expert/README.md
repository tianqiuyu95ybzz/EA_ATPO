# EA-ATPO V3 Oracle Expert experiment

This experiment replaces the planned Expert A/B SFT work with an offline
answer-key Oracle upper-bound evaluation. No Expert model is trained or
called. The Oracle is a scoring-time idealization over an existing benchmark
answer key; it is not a deployable model or clinically adjudicated truth.

## Data and split

The run reuses the already-used V2 50-case Train-only counterfactual sample:
35 cases used for Router fitting and 15 Dev cases used for early stopping and
model selection. The evaluation has 43 public states and 129 action rollouts.
All states remain grouped by `case_id`. This is development-only evidence,
not an independent test. The 400-case Frozen Holdout is not opened. The V3
291-row Gold review queue remains pending and is not used as a Gold source.

The scored labels are the original ATPO source benchmark `ground_truth.answer`
answer keys. The run verifies the source hash, source-row mapping, exact match
between the source prompt and the recorded initial public prompt, final-answer
parsing, and equality of all three rollout correctness labels to source-key
scoring. The answer key can support exam-benchmark Oracle analysis; it does
not establish clinical correctness and has no clinician adjudication here.

## Oracle and policies

Utility is `terminal_correctness - 0.01 * cumulative_ASK - c_DEFER * cumulative_DEFER`.
ASK cost remains the frozen Q-target cost; it is not subtracted again from an
already-net Q value. The Router's fixed decisions do not change over the
DEFER-cost grid `{0.05, 0.10, 0.20, 0.60}`.

The `Oracle Conditional` strategy retains a correct Base answer at zero
expert calls and calls the answer-key Oracle only for a Base error. For
`Learned Router + Oracle`, the frozen Router's proposed DEFER is passed through
the same ideal safety gate: a Base-correct answer is retained without a call;
a Base error is corrected. This is an explicitly optimistic upper bound.
`Always DEFER (direct answer-key Oracle)` is reported separately and charges
one call for every state. The actual Qwen3-32B-AWQ DEFER policy is also included
to distinguish Oracle potential from model performance.

## Reproduction

Run on CPU with the project venv; no model service or GPU is used:

```bash
CUDA_VISIBLE_DEVICES= /pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO_V1/.venv/bin/python \
  experiments/ea_atpo_v3/oracle_expert/scripts/evaluate_oracle_router.py \
  --output-dir /pfs/pfs-spMCeu/image_team/workspace-copy/EA-ATPO_Oracle_20261121/run_01
```

The runner refuses to overwrite an existing output directory. Aggregate
metrics and the per-state file are created with private permissions. The
per-state output contains case/state identifiers and correctness-derived
results; do not add it to public GitHub. Only sanitized aggregate metrics,
code, configuration, reports, and checksums belong in the repository.

Run the unit tests with:

```bash
CUDA_VISIBLE_DEVICES= /pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO_V1/.venv/bin/python -m unittest discover \
  -s experiments/ea_atpo_v3/oracle_expert/tests -v
```

## Interpretation boundary

This reanalysis tests whether the frozen Router's prior Dev choices overlap
with Base errors that a perfect answer-key expert could correct. The same
15 cases were used to select the Router, so positive findings cannot establish
generalization. A future confirmatory result needs newly collected, untouched
cases with answer-key provenance frozen before model selection.
