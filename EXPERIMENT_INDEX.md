# EA-ATPO experiment index

Updated: 2026-11-20. This index separates research stages and concurrent
windows. Branches are intentionally not merged by this index.

## Version map

| Version | Research focus | Code and primary records | Evidence and limits |
|---|---|---|---|
| V1 | Four-action ANSWER/ASK/CONSULT/DEFER protocol integrated with the original ATPO Actor-Critic/PPO path. | `ea-atpo-v1`; `ea_atpo_stage/`; `experiments/ea_atpo_v1_smoke/`. | Attempt09 completed 10 PPO steps, with 79/80 rollouts entering PPO and natural DEFER observed. Its 20-case validation answered 4/20; Critic explained variance was negative across all 10 steps. Engineering smoke only. |
| V2 Phase 1 | ASK/ANSWER/DEFER counterfactual collection and a light Router trained by Q Regression / Q-guided Policy Distillation. | `codex/ea-atpo-v2-task-b`; `experiments/ea_atpo_v2/`. | 50 training cases, 146 states, 438 valid action trajectories. This is the real counterfactual data foundation; the recovery bundle is historical and must not be overwritten. |
| V2 Window A | Costed Q Regression and lightweight Router comparisons. | `codex/router-training-20261120`; `experiments/ea_atpo_v2/router_training/`. | 35 train / 15 dev cases; three seeds. Best Q-guided Distillation Dev Accuracy 0.507, Utility 0.491; paired case-cluster Utility CI versus Always ANSWER `[-0.014, 0.297]`. Development/model-selection evidence, not confirmation. |
| V2 Window A audit | Independent recheck of predictions, grouping, and case-cluster bootstrap. | `codex/router-results-audit-20261120`; report path below. | Corrected intervals include zero for the principal Q-guided comparison. No retraining. The audit does not establish independent generalization. |
| V2 Window B | Relative Router, cost sensitivity, expert advantage learnability, and oracle analysis. | `codex/ea-atpo-v2-window-b`; `codex/ea-atpo-v2-relative-router`. | Diagnostic analysis only; no PPO/ATPO RL. Review branch reports before reusing numbers. |
| V2 final evaluation design | Freeze a new 200-case evaluation cohort and prepare a reproducible evaluator. | `codex/ea-atpo-v2-final-eval-20261120`; `experiments/ea_atpo_v2/final_evaluation/`. | Cohort design/overlap audit exists; no final cohort model predictions or outcomes are recorded. Do not describe it as a completed test. |
| V3 | Domain Expert SFT → paired evaluation → Router → active information acquisition. | `codex/ea-atpo-v3-expert-sft-20261120`; `experiments/ea_atpo_v3/expert_sft/`. | Gated preparation. AWQ-LoRA 10-step compatibility smoke only; formal Expert SFT and independent evaluation are not complete. `med_qwen` has no reported inference result. |

## Concurrent work windows

### Window A — Router

- Training branch: [`codex/router-training-20261120`](https://github.com/tianqiuyu95ybzz/EA_ATPO/tree/codex/router-training-20261120)
- Audit branch: [`codex/router-results-audit-20261120`](https://github.com/tianqiuyu95ybzz/EA_ATPO/tree/codex/router-results-audit-20261120)
- Handover and result registry: [`MIGRATION_HANDOVER.md`](https://github.com/tianqiuyu95ybzz/EA_ATPO/blob/codex/router-training-20261120/experiments/ea_atpo_v2/router_training/MIGRATION_HANDOVER.md), [`RESULT_REGISTRY.md`](https://github.com/tianqiuyu95ybzz/EA_ATPO/blob/codex/router-training-20261120/experiments/ea_atpo_v2/router_training/RESULT_REGISTRY.md)
- Corrected audit report: [`AUDIT_REPORT.md`](https://github.com/tianqiuyu95ybzz/EA_ATPO/blob/codex/router-results-audit-20261120/experiments/ea_atpo_v2/router_training/audits/20261120/AUDIT_REPORT.md)

### Window B — V2 counterfactual analysis

- Counterfactual implementation and recovery records: [`codex/ea-atpo-v2-task-b`](https://github.com/tianqiuyu95ybzz/EA_ATPO/tree/codex/ea-atpo-v2-task-b)
- Cost/learnability analysis: [`codex/ea-atpo-v2-window-b`](https://github.com/tianqiuyu95ybzz/EA_ATPO/tree/codex/ea-atpo-v2-window-b)
- Relative Router diagnostic: [`codex/ea-atpo-v2-relative-router`](https://github.com/tianqiuyu95ybzz/EA_ATPO/tree/codex/ea-atpo-v2-relative-router)
- Frozen-cohort evaluator: [`codex/ea-atpo-v2-final-eval-20261120`](https://github.com/tianqiuyu95ybzz/EA_ATPO/tree/codex/ea-atpo-v2-final-eval-20261120)

### Window C — V3 Expert-SFT preparation

- Main branch: [`codex/ea-atpo-v3-expert-sft-20261120`](https://github.com/tianqiuyu95ybzz/EA_ATPO/tree/codex/ea-atpo-v3-expert-sft-20261120)
- Follow-up audit: [`codex/ea-atpo-expert-sft-20261120`](https://github.com/tianqiuyu95ybzz/EA_ATPO/tree/codex/ea-atpo-expert-sft-20261120)
- V3 decision report: [`V3_FINAL_DECISION.md`](https://github.com/tianqiuyu95ybzz/EA_ATPO/blob/codex/ea-atpo-v3-expert-sft-20261120/experiments/ea_atpo_v3/expert_sft/reports/V3_FINAL_DECISION.md)

## Rules for interpreting results

1. Keep train, development, confirmation, and frozen-holdout evidence distinct.
2. Router states are not independent cases; use case-level grouping for splits,
   bootstrap intervals, and paired comparisons.
3. Oracle scores are upper bounds, not deployable policies.
4. A compatibility smoke or successful checkpoint reload is not a completed
   training or performance result.
5. Every report must state the model revision, dataset/split, prompt/parser,
   cost, seed, and whether the result is exploratory or confirmatory.
