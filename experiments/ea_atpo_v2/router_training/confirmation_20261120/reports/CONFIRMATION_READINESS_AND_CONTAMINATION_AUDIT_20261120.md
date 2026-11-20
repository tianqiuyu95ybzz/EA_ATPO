# Frozen Router confirmation readiness and cohort audit

**Status: blocked before inference.** The frozen Router is identifiable and
reproducible, but the 20 nominal cases left out of the recent 50-case V2 run
are not new cases. All 20 were included in an earlier 100-case V1 Router fit
and its counterfactual target table. The eligible untouched confirmation cohort
inside the official 70-case Train split is therefore **0 cases**. No new model
request or end-to-end episode was run. The 400-case Frozen Holdout was not
opened.

## 1. Frozen Router and action rule

The best reported development family remains Embedding Q-guided Distillation.
The three optimizer seeds from that run are frozen as separate checkpoints;
they are not combined into a new ensemble. The requested Embedding Relative
Advantage MLP comparator is also frozen with its three original seeds. The
confirmation set is not used to choose a seed, retrain a head, or change any
threshold or cost.

| Family | Seed | Best epoch | Dev loss | Checkpoint SHA-256 |
|---|---:|---:|---:|---|
| Q-guided Distillation | 17 | 10 | 1.13519 | `9821e7ca701189d37df9567fbf8ffaff9eaddb537e97448d8019803e3b8e2d32` |
| Q-guided Distillation | 42 | 8 | 1.15324 | `b461fab29c675070b3d1f02abb52b7ca84d8218c6a9c0dae4ac523c0bf0157b4` |
| Q-guided Distillation | 2026 | 12 | 1.17167 | `3405648a8284a526db22b0e211e190302705443753d788e858b0d93b7ac83aeb` |
| Relative Advantage MLP | 17 | 18 | 0.13638 | `4982446d860d0a70794fd8f606ca03d47c16985225213fd64f752333e7e6da8f` |
| Relative Advantage MLP | 42 | 18 | 0.13844 | `7ecdc640e0951772e7a027787e24de03a941843f058360157c9784f00599a2b0` |
| Relative Advantage MLP | 2026 | 20 | 0.14751 | `efc9a44ef83453aea8d31c6139f85e9903931159745786f3541d58d70dc7ec77` |

The full freeze record, relative checkpoint paths, sizes, model metadata, and
hashes of the preprocessing/config/code artifacts are in
[`frozen_candidate_20261120.json`](../configs/frozen_candidate_20261120.json).
The checkpoint binaries and embedding vectors remain in the ignored PFS
`outputs/real_q_router_20261120/` run directory; they are not copied into this
Git change.

Frozen input and action details:

- State encoder: frozen `Qwen/Qwen3-8B`, revision
  `b968826d9c46dd6066d109eabc6255188de91218`; final-layer mean pooling;
  dimension 4096; BF16; max length 2048; left truncation; no L2 normalization.
- Input serialization: only decision-time public user/assistant messages,
  native tokenizer chat template, `enable_thinking=false`, no generation
  prompt. Template SHA-256 is
  `41d5929bf73796beb66809ac700b2cf3ff81694f933e5c14d52b7fd6963c947d`.
- Router head: two linear layers with hidden size 16 and GELU. Distillation
  outputs three policy logits; Relative Advantage outputs ASK and DEFER
  margins, with ANSWER anchored at zero.
- Action order: `ASK / ANSWER / DEFER`; choose argmax. Ties follow the first
  action in that order. No extra cost is subtracted at inference because costs
  are already included in the training targets.
- Frozen costs: ANSWER `0`, ASK `0.01`, DEFER `0.05`.
- Train/dev: 35/15 cases, grouped by `case_id` (103/43 states). Early stopping
  used the same Dev set, with max 300 epochs and patience 30. Q-guided
  Distillation was selected by mean Dev Utility across the three fixed seeds.

The primary Q-guided family had prior Dev Accuracy `0.5074` and Utility
`0.4906`. The Relative Advantage MLP had Dev Accuracy `0.4926` and Utility
`0.4686`. These are development-only results, not confirmation evidence.

## 2. Remaining-case contamination audit

The official case split contains 70 Train cases. The current V2 counterfactual
cohort contains 50 of them, leaving a nominal 20. The audit cross-checked the
official split and V2 batch manifest against the old Router checkpoint,
historical state-action target table, and prior public-state bundle. It read
only IDs and canonical hashes of public initial messages, not hidden facts or
Gold values.

| Audit item | Result |
|---|---:|
| Official Train cases | 70 |
| Cases in current V2 Router split | 50 |
| Nominal remainder | 20 |
| Remainder included in historical V1 Router fit | **20/20** |
| Remainder present in historical V1 target table | **20/20** |
| Remainder present in historical public-state bundle | **20/20** |
| Exact duplicate initial-message hashes among these 70 IDs | 0 groups |
| Unseen cases eligible for confirmation | **0** |

The historical checkpoint metadata records a final fit on 100 cases and 291
states; all 20 candidate IDs occur in its saved training-state list. The
historical target table covers 100 cases/291 states as well. Thus, even though
the 20 were not in the recent V2 50-case Router fit, their prior counterfactual
outcomes and public states were used by earlier Router work. A new evaluation
on them would be a reuse/retrospective analysis, not a new-case confirmation.

The exact-message hash check found no duplicate initial messages among the 70
IDs. It does not rule out semantic duplicates, but that does not change the
decision: all 20 are already historically exposed. The aggregate audit and
source hashes are in
[`cohort_audit_aggregate.json`](../outputs/cohort_audit_aggregate.json). No
case IDs or clinical text are committed in this report.

## 3. Recheck of the existing Dev results

The 15-case Dev set has 43 paired states. Correctness was recovered from saved
real rollout outcomes, and each costed Q value reconciled with terminal
correctness minus cumulative ASK/DEFER costs. No action cost was subtracted a
second time. Recomputed per-case metrics match the stored per-case results to
maximum absolute difference below `7e-18`.

### Base/Expert paired table

Counts use the **43 paired states** as denominator; conditional rates use the
Base-correct/Base-wrong row totals.

| | Expert correct | Expert wrong | Total |
|---|---:|---:|---:|
| Base correct | 11 | 6 | 17 |
| Base wrong | 9 | 17 | 26 |
| Total | 20 | 23 | 43 |

- Expert correction: `9/26 = 34.6%` of Base-wrong states.
- Correct-answer regression: `6/17 = 35.3%` of Base-correct states.
- State-weighted Base Accuracy: `17/43 = 39.5%`; Expert Accuracy:
  `20/43 = 46.5%`.
- Equal-case-weighted Base Accuracy is `0.400`; Expert Accuracy is `0.478`.
  The difference in weighting explains why the case-weighted development
  summary differs from the pooled state counts.

### Existing offline policy results

Accuracy and Utility below are equal-case-weighted over the same 15 cases and,
for learned families, averaged over the three fixed seeds. The recorded
case-clustered intervals are from the original 5,000-replicate run; this audit
reconciled its per-case inputs but did not rerun that bootstrap implementation.

| Policy | Accuracy | Utility | ASK | ANSWER | DEFER | Paired Δ Accuracy 95% CI | Paired Δ Utility 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| Always ANSWER | 0.400 | 0.400 | 0% | 100% | 0% | reference | reference |
| Always ASK | 0.444 | 0.434 | 100% | 0% | 0% | [-0.067, 0.178] | [-0.077, 0.168] |
| Always DEFER | 0.478 | 0.428 | 0% | 0% | 100% | [-0.189, 0.344] | [-0.239, 0.294] |
| Static L2D | 0.400 | 0.400 | 0% | 100% | 0% | reference | reference |
| Relative Advantage MLP | 0.493 | 0.469 | 34.8% | 24.1% | 41.1% | [0.022, 0.322] | [0.001, 0.295] |
| Q-guided Distillation | 0.507 | 0.491 | 47.4% | 28.5% | 24.1% | [0.000, 0.322] | [-0.014, 0.297] |

Static L2D chose ANSWER because the frozen train mean DEFER-minus-ANSWER
advantage was negative. The first four action-rate columns above follow the
original equal-case macro convention: within-case state rates are averaged
across the 15 cases. For clarity, pooling all 129 state/seed decisions instead
gives Q-guided action counts `ASK 61/129`, `ANSWER 38/129`, `DEFER 30/129`
(47.3%, 29.5%, 23.3%); Relative Advantage gives `46/129`, `31/129`, `52/129`
(35.7%, 24.0%, 40.3%). Neither denominator changes the reported accuracy or
utility, but they are different action-rate summaries.

The Relative Advantage head's Dev Relative Advantage MAE is `0.392`; positive
advantage AUROC is `0.643` for ASK and `0.609` for DEFER. Q-guided Distillation
predicts policy logits rather than Q values, so Q MAE and advantage AUROC are
not defined for that head. The per-seed and denominator audit is in
[`prior_dev_metric_recheck.json`](../outputs/prior_dev_metric_recheck.json).

## 4. Confirmation experiment status and decision

| Requested component | Status |
|---|---|
| Freeze checkpoints, feature revision, action mapping, costs and training split | Complete; no retraining or re-selection |
| Audit the nominal 20 cases and ID-level exact duplicates | Complete; 20/20 historically exposed, 0 eligible |
| New real ANSWER/ASK/DEFER counterfactual data | Not run; no eligible cases |
| Frozen Router state-level evaluation on new cases | Not run |
| Full sequential Router-driven episodes | Not run |
| Always ANSWER / original ATPO questioning / Static L2D episode comparison | Not run |
| New-case paired 95% CI | Not estimable |
| Frozen Holdout | Not opened |

Therefore this work cannot answer whether the frozen model beats Always
ANSWER, the best fixed action, or the original ATPO policy on new cases. It
also cannot establish sequential effectiveness or provide a new-case CI. The
largest failure mode is cohort contamination: the apparent 20-case reserve is
already part of a prior Router fit. The existing Dev score remains suggestive
only; its utility CI includes zero.

Do not relabel the 20 cases as confirmation data. A valid next confirmation
requires a genuinely new, case-deduplicated cohort not used by the historical
V1 or current V2 experiments, with the Router/config frozen as recorded. Keep
the Frozen Holdout sealed unless it is formally designated as the final test.

## Reproduction commands

From the Window A worktree root:

```bash
python3 -m experiments.ea_atpo_v2.router_training.confirmation_20261120.scripts.audit_confirmation_cohort \
  --root . \
  --output experiments/ea_atpo_v2/router_training/confirmation_20261120/outputs/cohort_audit_aggregate.json

python3 -m experiments.ea_atpo_v2.router_training.confirmation_20261120.scripts.recheck_prior_dev \
  --root . \
  --output experiments/ea_atpo_v2/router_training/confirmation_20261120/outputs/prior_dev_metric_recheck.json
```

The first command never reads Frozen Holdout files. The second uses only prior
Train-derived V2 Q targets, recorded rollout outcomes, and saved predictions.
