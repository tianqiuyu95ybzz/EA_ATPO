# EA-ATPO V2 recovery bundle — 2026-11-20

This bundle preserves the key generated artifacts from the 50-case, Train-only real counterfactual pilot. It is intended to let later work resume Q-target analysis without rerunning all 438 model calls.

## Contents

- `pilot_trajectories.jsonl`: all 438 paired ASK/ANSWER/DEFER rollouts.
- `q_dataset.jsonl`: 146 states with three valid realized return targets each.
- `pilot_summary.json`, `action_summary.csv`, and `batch_summary.csv`: aggregate results.
- `ask_unknown_classifications.csv` and `ask_unknown_by_case.csv`: the 23-row Unknown audit and case-level counts.
- `case_batch_manifest.json` and `batch_XX_case_ids.json`: deterministic Train-only batch membership.
- `recovery_manifest.json`: model revisions, protocol, costs, and per-batch run metadata.
- `SHA256SUMS`: integrity hashes for all packaged files.

## Data boundary

The bundle contains public case states and generated model/patient-simulator responses, but no gold answer text or hidden atomic-fact fields. The Q targets and `final_accuracy` values are post-episode outcomes and therefore encode correctness; this is a Train-only experimental artifact, not a deployment-time feature set. The case IDs map to the frozen Train split. No Validation, Internal Test, or Frozen Holdout cases are included.

The underlying source cases come from the public ATPO training data. Check its upstream terms before redistributing these derivatives outside this project. No model weights, optimizer states, API credentials, or service logs are included.

## Restore / verify

From the repository root, verify the payload with:

```bash
cd experiments/ea_atpo_v2/recovery_bundle_20261120
sha256sum -c SHA256SUMS
```

To use Q targets in a later offline experiment, load `q_dataset.jsonl` directly. Reproducing the model calls requires the pinned model revisions in `recovery_manifest.json`, the committed V2 config/scripts, and a permitted GPU/service environment.
