# EA-ATPO V2 Final Case-Level Evaluation

This directory freezes a case-ID cohort and provides one post-hoc evaluation
interface for the three fixed baselines and three learned routers. It contains
no patient prompt text, Gold answer, model output, model weight, or real test
outcome.

## Frozen cohort

`manifests/final_test_20261120_v1/case_manifest.json` freezes 200 unique
`case_id`s. The list SHA-256 is
`193419fc13517ede123856afb273786766f05aa3ac0e8337a04ad750ca2814e0`.
The manifest stores IDs, source row indices, and prompt fingerprints only.
It does not contain public prompt text or labels. The original source is the
official `merged_train_dataset.jsonl`; its 14,256 records are single-turn
source cases, not multi-turn trajectories. The selected records are reserved
from future training. This is a repository-level case-disjoint cohort, not an
externally sourced benchmark; overlap with model pretraining data cannot be
verified from this repository.

`manifests/final_test_20261120_v1/overlap_audit.json` records the original
cohort audit. `overlap_audit_detailed.json` adds a source-by-source audit after
deduplicating repeated references to the same historical file. The selected
200 cases have zero exact case-ID, case-ID-hash, and public-prompt-hash overlap
with the audited historical sources. The 400-case Frozen Holdout and Window A
confirmation outcomes were not opened.

The manifest generator reads only record identity and public prompt fields for
selection. It never uses labels or hidden facts. It refuses to overwrite an
existing output directory. Keep the source dataset private and do not commit it.

## Evaluation inputs

The evaluator joins two local JSONL files against the frozen manifest. Keep
both files private; neither is included here.

`outcomes.jsonl`: one row per frozen case with post-hoc correctness and raw
call counts:

```json
{
  "case_id": "...",
  "base_correct": false,
  "expert_correct": true,
  "action_outcomes": {
    "ASK": {"final_correct": true, "valid": true, "ask_count": 1, "defer_count": 0},
    "ANSWER": {"final_correct": false, "valid": true, "ask_count": 0, "defer_count": 0},
    "DEFER": {"final_correct": true, "valid": true, "ask_count": 0, "defer_count": 1}
  }
}
```

The example above documents the schema only; it is not an experimental result.
Each case must have all three counterfactual outcomes. `final_correct` must
already apply the project's answer-scoring rule, including its handling of
invalid outputs. Utility is recomputed from `final_correct` and raw call counts,
so action costs must not already be subtracted from correctness labels.

`predictions.jsonl`: prediction-only output from each learned policy, one row
per policy and frozen case. The three policy names are
`q_guided_distillation`, `relative_advantage`, and `learned_router`.

```json
{
  "case_id": "...",
  "policy": "learned_router",
  "selected_action": "ANSWER",
  "action_scores": {"ASK": 0.2, "ANSWER": 0.7, "DEFER": 0.1},
  "model_revision": "record-the-real-revision"
}
```

Rows may instead provide `defer_score` and `non_defer_action` explicitly when
action scores are unavailable. These prediction fields are required for the
risk-coverage sweep. Gold, correctness, outcome, or hidden-fact fields in the
prediction file are rejected. The evaluator also rejects an Oracle prediction.

## Metrics and decision boundaries

Action order is `ASK / ANSWER / DEFER`. Registered primary costs are
`ASK=0.01`, `ANSWER=0.00`, and `DEFER=0.05`. Utility is
`final_correct - ASK_cost * ask_count - DEFER_cost * defer_count`; cost is
recomputed once from raw counts.

The evaluator reports accuracy, utility, ASK/DEFER call rates, mean cost,
expert correction rate, and correct-answer degradation rate. Expert correction
rate is measured among selected DEFER cases where Base was wrong. Correct-answer
degradation rate is measured among selected non-ANSWER cases where Base was
correct. Rates with an empty denominator are reported as null.

The fixed baselines are Always Answer, Always ASK, and Always DEFER. Learned
policies are supplied through the prediction-only file. The retrospective
Oracle chooses the highest-utility action per case from post-hoc action
correctness and costs; it is emitted as an aggregate upper bound only. It is
not accepted as a prediction, training target, or selectable policy.

95% percentile intervals and paired differences against Always Answer use
case-ID bootstrap resampling (10,000 replicates by default, fixed seed
20261120). Risk-coverage rows report the fraction not deferred and error rate
among those covered. The sweep varies the prediction-only defer score threshold;
it does not use Gold to choose a deployed threshold.

## Reproduce the frozen case list

Run from the repository root with the original source and audit files available
locally. The output directory must be new:

```bash
python -m experiments.ea_atpo_v2.final_evaluation.case_manifest \
  --source-jsonl /path/to/EA_ATPO_V1/experiments/expert_sft/data/source/dataset/merged_train_dataset.jsonl \
  --historical-case-split /path/to/EA_ATPO_V1/experiments/ea_atpo_value_router_20261105/artifacts/case_split.json \
  --current-router-split /path/to/EA_ATPO_V1_router_training/experiments/ea_atpo_v2/router_training/outputs/real_q_router_20261120/case_split_35_train_15_dev.json \
  --v2-recovery-bundle /path/to/EA_ATPO_V1/experiments/ea_atpo_v2/recovery_bundle_20261120 \
  --historical-direct-results /path/to/EA_ATPO_V1/experiments/real_case_pilot_20261103/audit/outputs/direct_eval/direct_reliability_trajectories.jsonl \
  --historical-output-roots /path/to/EA_ATPO_V1/experiments/real_case_pilot_20261103/audit/outputs \
    /path/to/EA_ATPO_V1/experiments/real_case_pilot_20261103/audit/results \
    /path/to/EA_ATPO_V1/experiments/real_case_pilot_20261103/audit/relative_gain_predictability_20261118 \
    /path/to/EA_ATPO_V1/experiments/ea_atpo_value_router_20261105/artifacts \
    /path/to/EA_ATPO_V1/experiments/real_case_pilot_20261103/outputs/full \
  --sample-size 200 \
  --output-dir /path/to/new/final_test_manifest
```

Do not regenerate the frozen manifest in place. A regeneration is a verification
run; compare its case-ID list hash before treating it as the same cohort.

## Run evaluation on a future real result

Run from the repository root. The output directory must not already exist.

```bash
python -m experiments.ea_atpo_v2.final_evaluation.evaluate \
  --case-manifest experiments/ea_atpo_v2/final_evaluation/manifests/final_test_20261120_v1/case_manifest.json \
  --outcomes /private/path/final_test_outcomes.jsonl \
  --predictions /private/path/final_test_predictions.jsonl \
  --output-dir /private/path/new_final_eval_20261120 \
  --ask-cost 0.01 --defer-cost 0.05 \
  --bootstrap-replicates 10000 --bootstrap-seed 20261120
```

Outputs are `metrics.csv`, `paired_deltas_vs_always_answer.csv`,
`risk_coverage.csv`, and `evaluation_manifest.json`. The manifest hashes all
inputs and records costs, policy order, bootstrap settings, and the Oracle
boundary. No real model was run and no real test metrics exist yet.

## Synthetic code checks

The test suite uses only tiny synthetic fixtures and makes no claim about
medical performance:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python -m unittest discover \
  -s experiments/ea_atpo_v2/final_evaluation/tests -v
```

The captured result is `reports/SYNTHETIC_TEST_LOG.txt`.
