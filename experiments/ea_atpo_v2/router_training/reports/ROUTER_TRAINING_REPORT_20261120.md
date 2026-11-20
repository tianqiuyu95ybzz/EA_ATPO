# EA-ATPO V2 Router Training Report — 2026-11-20

## Executive result

Completed the Train-only counterfactual Q-regression and lightweight Router experiment on the 50-case recovery cohort. All 146 states had valid ASK, ANSWER, and DEFER labels. Targets were rebuilt from terminal correctness and cumulative action counts at the requested costs. No Frozen Holdout, official Validation, or Internal Test outcomes were read.

The strongest Dev result was the frozen-Qwen3-8B-embedding Q-guided distillation head: case-weighted accuracy 0.507 and utility 0.491, versus Always ANSWER at 0.400/0.400. Its paired case-bootstrap utility interval versus Always ANSWER was [-0.014, 0.297]. The embedding Relative Advantage MLP averaged 0.493 accuracy and 0.469 utility, with a nominal utility-difference interval [0.001, 0.295]. These are **development-only results**: the same 15 cases selected checkpoints by early stopping, so neither interval is confirmatory and neither establishes that a deployable Router has learned stable conditional advantage.

Text-only Ridge severely overfit. Stronger regularization reduced the train fit but did not produce consistent Dev gains across model families. The data support further development, but do not support opening the Frozen Holdout yet. Keep all final-test data sealed until one policy and configuration are frozen.

## Data, split, and target audit

- Source: committed recovery bundle under `experiments/ea_atpo_v2/recovery_bundle_20261120/`.
- Raw cohort: 50 official Train cases, 146 public states, 438 real trajectories (146 per action); one rollout per state-action (K=1); all 438 trajectories were valid.
- Fixed split: 35 train cases / 15 Dev cases, using SHA256("20261120|" + case_id) ordering. This produced 103 train states and 43 Dev states. Every state for a case stayed in that case's partition.
- ASK labels follow the single registered continuation policy `ASK_ONCE_THEN_FORCE_BASE_ANSWER_PATIENT_REVIEW_V2`; every state/action has K=1, so they are single realized returns, not low-variance expected-value estimates.
- No source case belonged to official Validation or Internal Test. Frozen Holdout was not opened.
- Old labels remain unchanged in the recovery bundle. A new Q JSONL was constructed separately; old Q values were retained in `Q_ASK_old`, `Q_ANSWER_old`, and `Q_DEFER_old`.
- Old target formula reconciled for all 438 action rows: `correctness - 0.05 * ask_count - 0.60 * defer_count`.
- New target formula: `Q(s,a) = terminal_correctness - 0.01 * cumulative_ask_count - 0.05 * cumulative_defer_count`.
- Each old return and Q label matched its raw trajectory. Expert calls/tokens, prompt/completion tokens, and latency remained resource telemetry; no additional resource cost was added to utility. Costs were rebuilt from raw counts and applied once; evaluation did not deduct them again.
- The feature file contains only `case_id`, `state_id`, and pre-action `public_state`. Gold, hidden atomic facts, future replies, final answers, correctness, and rollout results are absent from Router inputs. Supervision labels were used only as targets and post-hoc evaluation.
- All 438 action masks remained valid. No missing label was replaced with zero.

| Partition | States | ASK mean Q | ANSWER mean Q | DEFER mean Q | ASK better than ANSWER | DEFER better than ANSWER |
|---|---:|---:|---:|---:|---:|---:|
| Train | 103 | 0.349 | 0.388 | 0.377 | 6/103 (5.8%) | 16/103 (15.5%) |
| Dev | 43 | 0.432 | 0.395 | 0.415 | 4/43 (9.3%) | 9/43 (20.9%) |

On Dev, Base ANSWER and Expert DEFER had both correct 11, Base correct/Expert wrong 6, Base wrong/Expert correct 9, both wrong 17. State-weighted accuracy was 39.5% vs 46.5%; case-weighted accuracy was 40.0% vs 47.8%. The case-clustered paired 95% CI for case-weighted accuracy difference was [-18.9, 34.4] percentage points. ASK versus ANSWER had 4 corrections, 2 regressions, and 37 unchanged outcomes. The relative labels are imbalanced: only 4/43 ASK and 9/43 DEFER states have positive net advantage at the requested costs.

## Features and training

- Text features: existing audited TF-IDF + text-structure encoder; vocabulary and statistics fit on the 35 training cases only.
- Frozen embeddings: Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`, mean-pooled final-layer hidden states, 4,096 dimensions, native chat template with `enable_thinking=false`, BF16, max length 2,048, batch size 1. All 146 embeddings were extracted from public state only; the model was frozen.
- Router training ran on CPU with CUDA hidden, deterministic PyTorch, one CPU thread, and seeds 17, 42, and 2026.
- Compared train-only Constant Mean, Ridge absolute Q, two-layer MLP absolute Q (hidden 16), Ridge relative advantage, two-layer MLP relative advantage, and masked-softmax Q-guided distillation (temperature 0.1). Regression used masked Huber; distillation used masked softmax(Q/T) and cross-entropy.
- No genuine Base confidence, log-probability, entropy, or margin was recorded, including nested traces. The Uncertainty Threshold baseline is unavailable; no text-derived uncertainty proxy was substituted.
- GPU2 was used only to extract frozen embeddings. Immediately before launch it had 171,865 MiB free, 1% utilization, and an unrelated local inference process; that process was left untouched. Embedding extraction completed with no OOM. Router fitting ran on CPU.
- All 30 learned checkpoints were reloaded. Dev predictions matched saved scores within maximum absolute difference `4.18e-7`.

## Dev policy results

Metrics are equal-case-weighted averages, averaged over three seeds for learned models. Action rates are predicted state fractions. These are **offline paired-counterfactual policy estimates**, not live Router-driven rollouts. The corrections/spoilage column reports the mean number of Base-wrong→Expert-correct and Base-correct→Expert-wrong cases among the selected DEFER states per seed. Paired case-cluster bootstrap intervals use 5,000 resamples versus Always ANSWER on the same Dev cases used for early stopping; they are descriptive and optimistic, not confirmatory.

The table below gives mean Train/Dev policy performance across seeds, followed by Train and Dev action rates. This makes the train-development gap visible; it does not turn the selected 15-case Dev partition into an independent test.

| Model / features | Train accuracy / utility | Train ASK / DEFER | Dev accuracy / utility | Dev ASK / DEFER |
|---|---:|---:|---:|---:|
| Constant Mean | 0.386 / 0.386 | 0% / 0% | 0.400 / 0.400 | 0% / 0% |
| Ridge absolute, TF-IDF | 0.581 / 0.574 | 6.2% / 13.3% | 0.378 / 0.365 | 46.7% / 16.7% |
| MLP absolute, TF-IDF | 0.568 / 0.548 | 34.0% / 33.3% | 0.437 / 0.424 | 78.5% / 10.4% |
| MLP relative, TF-IDF | 0.581 / 0.563 | 37.9% / 27.6% | 0.430 / 0.402 | 51.5% / 44.1% |
| Q-guided distillation, TF-IDF | 0.575 / 0.563 | 31.1% / 17.1% | 0.422 / 0.410 | 50.4% / 13.7% |
| Ridge absolute, embedding | 0.581 / 0.574 | 6.2% / 13.3% | 0.444 / 0.429 | 32.2% / 24.4% |
| MLP absolute, embedding | 0.581 / 0.563 | 41.7% / 27.3% | 0.500 / 0.482 | 44.8% / 26.3% |
| MLP relative, embedding | 0.581 / 0.561 | 34.3% / 32.1% | 0.493 / 0.469 | 34.8% / 41.1% |
| Q-guided distillation, embedding | 0.581 / 0.568 | 35.9% / 17.9% | 0.507 / 0.491 | 47.4% / 24.1% |

| Policy / features | Accuracy | Utility | ASK | DEFER | Corrections / spoilage | Δ Acc 95% CI | Δ Utility 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| Always ANSWER | 0.400 | 0.400 | 0% | 0% | 0 / 0 | [0.000, 0.000] | [0.000, 0.000] |
| Always ASK | 0.444 | 0.434 | 100% | 0% | 0 / 0 | [-0.067, 0.178] | [-0.077, 0.168] |
| Always DEFER | 0.478 | 0.428 | 0% | 100% | 9 / 6 | [-0.189, 0.344] | [-0.239, 0.294] |
| Static L2D (train mean) | 0.400 | 0.400 | 0% | 0% | 0 / 0 | [0.000, 0.000] | [0.000, 0.000] |
| Constant Mean Router | 0.400 | 0.400 | 0% | 0% | 0 / 0 | [0.000, 0.000] | [0.000, 0.000] |
| TF-IDF Ridge absolute / relative | 0.378 | 0.365 | 46.7% | 16.7% | 0 / 1 | [-0.089, 0.044] | [-0.106, 0.035] |
| TF-IDF MLP absolute | 0.437 | 0.424 | 78.5% | 10.4% | 0 / 0 | [-0.067, 0.111] | [-0.080, 0.102] |
| TF-IDF MLP relative | 0.430 | 0.402 | 51.5% | 44.1% | 2.67 / 2.00 | [-0.111, 0.267] | [-0.150, 0.234] |
| TF-IDF Q-guided distillation | 0.422 | 0.410 | 50.4% | 13.7% | 0.33 / 0.67 | [-0.133, 0.111] | [-0.150, 0.096] |
| Embedding Ridge absolute / relative | 0.444 | 0.429 | 32.2% | 24.4% | 2 / 0 | [-0.067, 0.178] | [-0.082, 0.161] |
| Embedding MLP absolute | 0.500 | 0.482 | 44.8% | 26.3% | 2 / 0 | [-0.022, 0.178] | [-0.041, 0.169] |
| Embedding MLP relative | 0.493 | 0.469 | 34.8% | 41.1% | 4.33 / 1.67 | [0.022, 0.322] | [0.001, 0.295] |
| Embedding Q-guided distillation | 0.507 | 0.491 | 47.4% | 24.1% | 4 / 0.33 | [0.000, 0.322] | [-0.014, 0.297] |
| Offline Oracle (not deployable) | 0.633 | 0.625 | 8.9% | 14.4% | 6 / 0 | [0.067, 0.422] | [0.064, 0.410] |

Static L2D compares train-set mean net DEFER Q against ANSWER Q. The mean `(Q_DEFER - Q_ANSWER)` was -0.0112, so Static L2D selected ANSWER. Constant Mean also selected ANSWER. Oracle chooses the best observed counterfactual Q separately at each Dev state; it is an offline optimistic bound, not a deployable policy.

## Value diagnostics and overfitting

| Model / features | Train Q MAE | Dev Q MAE | Train relative MAE | Dev relative MAE | Dev ASK advantage AUROC | Dev DEFER advantage AUROC |
|---|---:|---:|---:|---:|---:|---:|
| Constant Mean | 0.475 | 0.485 | 0.234 | 0.269 | 0.500 | 0.500 |
| Ridge, TF-IDF | 0.0003 | 0.472 | 0.0004 | 0.367 | 0.359 | 0.307 |
| MLP absolute, TF-IDF | 0.121 | 0.465 | 0.176 | 0.376 | 0.524 | 0.495 |
| MLP relative, TF-IDF | — | — | 0.107 | 0.368 | 0.573 | 0.447 |
| Ridge, embedding | 0.0016 | 0.434 | 0.0023 | 0.539 | 0.737 | 0.552 |
| MLP absolute, embedding | 0.106 | 0.404 | 0.143 | 0.402 | 0.724 | 0.536 |
| MLP relative, embedding | — | — | 0.133 | 0.392 | 0.643 | 0.609 |

MAE averages valid targets. AUROC treats positive net advantage against ANSWER as the positive class. Distillation outputs action logits, not calibrated Q values, so Q/advantage MAE and AUROC are undefined for that head.

Ridge nearly interpolated train but had large Dev errors. Absolute MLP train-to-Dev Q MAE gaps were 0.344 (TF-IDF) and 0.298 (embedding). The embedding Relative MLP relative-MAE gap was 0.259. Dev checkpoint epochs varied by seed. These are overfitting and initialization-sensitivity signals.

## Higher-regularization sensitivity

One limited follow-up used the same cases/split/seeds, Ridge alpha 100, MLP hidden size 8, and weight decay 0.1. It reduced capacity/increased regularization without adding modules.

| Model / features | Primary utility | Stronger-regularization utility | Primary Dev Q MAE | Stronger Dev Q MAE |
|---|---:|---:|---:|---:|
| Ridge, TF-IDF | 0.365 | 0.387 | 0.472 | 0.471 |
| MLP absolute, TF-IDF | 0.424 | 0.428 | 0.465 | 0.470 |
| MLP relative, TF-IDF | 0.402 | 0.417 | — | — |
| Q distillation, TF-IDF | 0.410 | 0.436 | — | — |
| Ridge, embedding | 0.429 | 0.451 | 0.434 | 0.408 |
| MLP absolute, embedding | 0.482 | 0.428 | 0.404 | 0.414 |
| MLP relative, embedding | 0.469 | 0.425 | — | — |
| Q distillation, embedding | 0.491 | 0.444 | — | — |

Stronger regularization modestly helped Ridge, but did not resolve the value-prediction gap. It reduced embedding MLP and distillation utility; no configuration dominated across feature/target groups. This used the same Dev cases and is not an independent evaluation.

## Decision

- Always DEFER observed 9 corrections and 6 regressions among 43 Dev states; conditional rates were 9/26 (34.6%) among Base-wrong states and 6/17 (35.3%) among Base-correct states.
- Oracle utility 0.625 shows potential counterfactual headroom. Learned models do not yet identify that value reliably: the best small embedding Relative MLP DEFER advantage AUROC was 0.609 on Dev and varied by seed; Ridge DEFER AUROC was weak.
- Best mean Dev utility was 0.491 for embedding Q-guided distillation, but its interval includes zero. The nominally positive interval for embedding Relative MLP is affected by checkpoint-selection reuse.
- **Decision: do not claim Router success or open the Frozen Holdout from this run.** The training/checkpoint pipeline works; generalization remains unresolved. Freeze one candidate and its configuration using Dev evidence or collect more Train-only paired cases before a final evaluation. Keep the 400-case Frozen Holdout sealed until the final plan is registered.

## Artifacts and reproduction

Run-local costed Q labels, public-only states, fixed split, embeddings, 30 checkpoints per primary/sensitivity run, per-seed Dev predictions, diagnostics, metrics, and checkpoint reload audit are on PFS under:

`experiments/ea_atpo_v2/router_training/outputs/real_q_router_20261120/`

That outputs directory is Git-ignored because it contains case-level labels/predictions and model artifacts. Repository code/config/report contain no case-level clinical text. Reproduction entry points are `scripts/prepare_costed_dataset.py`, `scripts/train_evaluate.py`, and `scripts/verify_checkpoints.py`; run configurations are in `configs/primary_20261120.json` and `configs/regularization_sensitivity_20261120.json`.

The full command list is in `reports/RUN_COMMANDS_20261120.md`, and actual completion statuses are summarized in `reports/RUN_LOG_SUMMARY_20261120.md`. Each run has 30 learned `.pt` checkpoints plus six Constant Mean JSON snapshots; the primary `.pt` checkpoints passed reload verification.

Executed commands from the worktree root:

1. `python3 -m experiments.ea_atpo_v2.router_training.scripts.prepare_costed_dataset --q-source experiments/ea_atpo_v2/recovery_bundle_20261120/q_dataset.jsonl --trajectories experiments/ea_atpo_v2/recovery_bundle_20261120/pilot_trajectories.jsonl --official-case-split experiments/ea_atpo_value_router_20261105/artifacts/case_split.json --output-dir experiments/ea_atpo_v2/router_training/outputs/real_q_router_20261120`
2. Frozen embedding extraction: `CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 ... python -m experiments.ea_atpo_v2.training.extract_state_embeddings` with local Qwen3-8B revision, public_states_only.jsonl, BF16, max length 2048, batch size 1, and seed 20261120 (full flags in the embedding manifest).
3. Primary fit: `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 ... python -m experiments.ea_atpo_v2.router_training.scripts.train_evaluate` using the cost-adjusted Q, raw paired trajectories, frozen 35/15 split, and embedding cache; output `training_final/`.
4. Sensitivity fit: same command with `--ridge-alpha 100 --hidden-dim 8 --weight-decay 0.1`; output `regularization_sensitivity/`.
5. Reload verification: `python -m experiments.ea_atpo_v2.router_training.scripts.verify_checkpoints` with the same Q, split, embedding cache, and `training_final/`.

No Router deployment, PPO, GRPO, Actor LoRA, or independent test was run.
