# Router experiment run log summary — 2026-11-20

Worktree: `EA_ATPO_V1_router_training`
Branch: `codex/router-training-20261120`
Run root: `experiments/ea_atpo_v2/router_training/outputs/real_q_router_20261120/`

This file records actual command outcomes. The output directory contains the machine-readable data audit, manifests, per-seed result JSON/CSV, predictions, and checkpoints.

| Stage | Result | Evidence |
|---|---|---|
| Costed Q rebuild | Completed, exit 0 | 50 cases, 146 states, 438/438 raw old-return/Q reconciliations; all 438 new labels valid; new costs applied once. See `outputs/real_q_router_20261120/cost_recalculation_audit.json`. |
| First embedding attempt | Failed before model load | Existing extractor tried to read `raw_metadata` from its `PublicState` object. No model inference or label generation occurred. The extractor was minimally corrected to rely on its public-field-only adapter. |
| Frozen embedding extraction retry | Completed, exit 0 | Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`; 146 states, 4096 dimensions; BF16; GPU2; seed 20261120. See embedding JSONL sidecar manifest. |
| Primary Router training | Completed, exit 0 | TF-IDF and frozen-embedding groups; three seeds; 46 detailed result rows; 22 policy summaries; 30 learned checkpoints. See `training_final/training_results.json` and `training_final/metrics.csv`. |
| Stronger-regularization sensitivity | Completed, exit 0 | Same split and seeds; Ridge alpha 100, MLP hidden 8, weight decay 0.1; 46 result rows, 22 summaries, 30 checkpoints. See `regularization_sensitivity/training_results.json`. |
| Checkpoint reload | Passed, exit 0 | All 30 primary learned checkpoints reloaded; maximum absolute Dev prediction difference 4.18e-7. See `training_final/checkpoint_reload_audit.json`. |
| Frozen Holdout / official Validation / Internal Test | Not opened | Only the selected 50 official Train cases were used; 35/15 case-disjoint train/Dev split. |
| PPO / GRPO / Actor LoRA / deployment | Not run | Outside this phase. |

The standalone Router fits ran on CPU with CUDA hidden. GPU2 was only used for frozen feature extraction. A separate inference service was visible on that device and was left running; extraction completed without OOM. No other-user process, model service, rollout, or Window B output was modified.
