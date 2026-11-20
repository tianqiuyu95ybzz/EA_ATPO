# EA-ATPO V2 Router training (Window A)

This worktree contains Window A's isolated 50-case Train-only Router experiment. It is intentionally separate from the shared checkout and Window B's cost-sensitivity work.

## Evidence boundary

- Input is the committed `recovery_bundle_20261120` generated from 50 cases in the official Train split. The 400-case Frozen Holdout, official Validation, and Internal Test are not opened.
- The fixed development partition is 35 Train cases / 15 Dev cases, grouped by `case_id`. All states for a case stay in one partition.
- Each state/action currently has one rollout (K=1), so each Q target is one realized return, not a stable estimate of expected value. Treat all Dev results as model-selection evidence only; they are not independent generalization evidence.
- The original rollout and Q files are read-only. A separate Q file is rebuilt from raw terminal correctness and action counts at the requested costs; the original cost schedule and targets are preserved in the recovery bundle and an explicit recalculation audit.
- Features use only the pre-action `public_state`. The frozen Qwen3-8B embedding extractor does not inspect labels. Gold correctness is used only to construct offline targets and score selected actions.

## Primary costs

`cost_answer=0.00`, `cost_ask=0.01`, `cost_defer=0.05`.

The target is `final_accuracy - 0.01 * ask_count - 0.05 * defer_count`. It is recomputed once from raw outcomes. At evaluation time, model outputs are already net values; do not deduct costs again. Token count, latency, and expert-call/token counts are resource measurements, not additional utility penalties in this registered objective.

## Models and baselines

- Features: train-fitted custom TF-IDF plus text-structure features; frozen Qwen3-8B final-layer mean embeddings when extraction succeeds.
- Targets: absolute three-action Q regression, two-action relative advantages `ASK-ANSWER` and `DEFER-ANSWER`, and masked soft-label policy distillation.
- Heads: closed-form Ridge and two-linear-layer MLP (hidden size 16), plus a train-only Constant Mean baseline.
- Baselines: Always ANSWER/ASK/DEFER, Constant Mean Router, Ridge/MLP argmax routers, Static L2D (ASK disabled; compare predicted net Q for ANSWER vs DEFER), and Q-guided Distillation. An Uncertainty Threshold is only reported if genuine Base confidence/log-probability exists; no uncertainty proxy is fabricated.

Three fixed optimization seeds are used: `17`, `42`, `2026`. The case split is fixed across seeds. Closed-form Ridge is deterministic; repeated seed records should therefore coincide.

## Run order

From the worktree root:

1. `python -m experiments.ea_atpo_v2.router_training.scripts.prepare_costed_dataset ...`
2. Extract pre-action embeddings only after confirming an authorized GPU and a safe free-memory margin. Current checkpoint: `Qwen/Qwen3-8B`, revision `b968826d9c46dd6066d109eabc6255188de91218`.
3. `python -m experiments.ea_atpo_v2.router_training.scripts.train_evaluate ...`

The exact commands, config, audit, checkpoints, per-seed predictions, metrics, and report are stored in this experiment's unique `outputs/` run directory. No V2 collector, simulator, service config, or Window B files are modified.

## Completed run (2026-11-20)

The primary 3-seed train/dev run and one higher-regularization sensitivity run have completed. See `reports/ROUTER_TRAINING_REPORT_20261120.md` for aggregate results and limitations. Run-local Q labels, frozen embeddings, per-state predictions, logs, and checkpoints are under the ignored `outputs/real_q_router_20261120/` directory on PFS. The checkpoint reload audit matched every learned checkpoint to saved Dev scores within 4.2e-7. The 15-case Dev partition was used for early stopping, so the reported bootstrap intervals are descriptive development intervals, not confirmatory test intervals.

Reproduction entry points are `scripts/prepare_costed_dataset.py`, `scripts/train_evaluate.py`, and `scripts/verify_checkpoints.py`. The primary config is `configs/primary_20261120.json`; the follow-up regularization setting is in `configs/regularization_sensitivity_20261120.json`.
