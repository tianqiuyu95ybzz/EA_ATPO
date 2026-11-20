# EA-ATPO V2 Window B diagnostics

This directory contains an independent, read-only analysis of the 50-case
counterfactual training dataset. It does not import Window A checkpoints or
modify its code, configuration, outputs, or data producer.

## Reproduce

From the repository root, with Python 3.12, NumPy, and Matplotlib installed:

```bash
python3 -m experiments.ea_atpo_v2.window_b_diagnostics.analyze_window_b \
  --data-dir ../EA_ATPO_V1/experiments/ea_atpo_v2/outputs/ask_unknown_repair_50case_20261120/merged \
  --case-split experiments/ea_atpo_value_router_20261105/artifacts/case_split.json \
  --output-dir experiments/ea_atpo_v2/window_b_diagnostics/results_reproduced
```

The input Q dataset, trajectories, merged manifest, batch manifests/configs,
and official case split are hash-checked before analysis. The output directory
must be new or disposable because the script writes aggregate tables, JSON
summaries, and SVG figures there. It does not copy raw trajectories into the
output directory or call model services.

The checked-in results are under
[`results_20261120_rerun2/`](results_20261120_rerun2/). The exact audited data
hashes, model revisions, inference settings, and split hash are recorded in
[`data_version.json`](results_20261120_rerun2/data_version.json); the run
environment and isolation checks are recorded in
[`analysis_run_manifest.json`](results_20261120_rerun2/analysis_run_manifest.json).

## Main outputs

- [`REPORT.md`](REPORT.md): findings, limits, and interpretation.
- [`cost_sensitivity.csv`](results_20261120_rerun2/cost_sensitivity.csv): all
  24 cost pairs × four policies, including Accuracy, Utility, action rates,
  and case-clustered bootstrap intervals.
- [`oracle_action_distribution.csv`](results_20261120_rerun2/oracle_action_distribution.csv):
  retrospective best-action mix at each cost pair, with ties split evenly.
- [`action_complementarity.csv`](results_20261120_rerun2/action_complementarity.csv):
  Base/Expert and ASK/ANSWER correctness transitions.
- [`ask_information.csv`](results_20261120_rerun2/ask_information.csv):
  ASK outcomes grouped by Unknown and a lexical novelty proxy.
- [`q_advantage_distribution.csv`](results_20261120_rerun2/q_advantage_distribution.csv)
  and [`advantage_by_cost.csv`](results_20261120_rerun2/advantage_by_cost.csv):
  gross/stored-Q advantages and cost-adjusted positive-label counts.
- [`feature_advantage_diagnostic.json`](results_20261120_rerun2/feature_advantage_diagnostic.json):
  fixed, case-grouped Ridge diagnostic using public-state text only.
- [`repeat_sampling_estimate.json`](results_20261120_rerun2/repeat_sampling_estimate.json):
  K=3 resampling plans and estimated inference calls; no sampling was run.
- [`accuracy_utility_curve.svg`](results_20261120_rerun2/accuracy_utility_curve.svg)
  and [`gross_advantage_distribution.svg`](results_20261120_rerun2/gross_advantage_distribution.svg):
  aggregate figures.

All confidence intervals resample `case_id` clusters. The 146 states are not
treated as independent cases. The retrospective Oracle uses observed outcomes
and is an in-sample upper bound, never a deployable Router estimate.
