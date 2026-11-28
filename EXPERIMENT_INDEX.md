# EA-ATPO experiment index

Last checked: 2026-11-28. Research version labels V1/V2/V3 identify method
stages; they are not package releases or claims of clinical validation.

| Stage | Main artifacts | Evidence status | Recovery boundary |
|---|---|---|---|
| V1 joint Actor-Critic smoke | [V1 implementation](EA_ATPO_V1_IMPLEMENTATION.md), [small-scale results](EA_ATPO_V1_SMALL_SCALE_RESULTS.md), `experiments/ea_atpo_v1_smoke/` | Engineering and protocol evidence only. The 2-step/10-step smoke runs do not establish clinical utility. | Attempt09 Actor/Critic exports remain private PFS assets; they are not the model used by the V3 Chinese pilot. |
| V2 counterfactual Router | `experiments/ea_atpo_v2/router_training/`, [published results index](PUBLISHED_RESULTS_INDEX.md), [Window A recovery record](https://github.com/tianqiuyu95ybzz/EA_ATPO/tree/codex/router-training-20261120) | 50 train cases / 15 Dev cases; best Dev Accuracy/Utility 0.507/0.491. Case-clustered Utility CI versus Always ANSWER was [-0.014, 0.297]. This is model-selection evidence, not independent generalization. | Case-level Q data, embeddings, and Router checkpoints are in the controlled PFS Window A archive; do not treat them as V3 results. |
| V3 Chinese dialogue pilot | [Pilot README](experiments/ea_atpo_v3/chinese_pilot_20261121/README.md), [pilot report](experiments/ea_atpo_v3/chinese_pilot_20261121/REPORT.md), [review status](experiments/ea_atpo_v3/chinese_pilot_20261121/review_privacy_20261122/HUMAN_EVALUATION_SUMMARY.md) | 50 ReMeDi dialogues, 100 ANSWER/ASK-Replay generations. Twenty paired cases / 40 outputs are queued for human review; 0 responses scored. ASK is retrospective historical replay, not autonomous question generation. DEFER is N/A. | Raw dialogue, model outputs, review packets, and case mapping are private. The data and output hashes are checked against local manifests; content is not in GitHub. |
| V3 checkpoint lineage | [Audit report](experiments/ea_atpo_v3/atpo_checkpoint_audit_20261121/REPORT.md), [model lineage](experiments/ea_atpo_v3/atpo_checkpoint_audit_20261121/model_lineage.json) | The pilot used original Qwen3-8B at revision `b968826d9c46dd6066d109eabc6255188de91218`, without an adapter. A verified SFT+ATPO Qwen3-8B final checkpoint was not found. | Base and Attempt09 weights remain on PFS. No model weights are uploaded. |
| V3 Oracle / benchmark admission | `experiments/ea_atpo_v3/gold_oracle_audit_20261121/`, `experiments/ea_atpo_v3/gold_dataset_admission_20261121/`, [Oracle design/report](experiments/ea_atpo_v3/oracle_expert/README.md) | CPU-only audit and Dev analyses. The PFS runs 07/08 are Dev-only; no Frozen Holdout was opened. | Per-state files remain private PFS artifacts. Aggregate summaries and scripts are in GitHub. |
| Historical V2 relative Router recovery | `experiments/ea_atpo_value_router_20261105/` | Legacy exploratory code and aggregate precheck metrics are preserved separately. Case-level targets, splits, checkpoints, and service logs are excluded from GitHub. | Detailed local file classification is in the private migration inventory indexed by [PFS artifact index](PFS_ARTIFACT_INDEX.md). |
| V3 Blackwell restore | [Restore and BF16 smoke](experiments/ea_atpo_v3/blackwell_restore_20261127/REPORT.md) | Synthetic-prompt generation, embedding extraction, BF16 kernel and CPU Oracle checks. | Detailed logs, vectors and model shards stay in controlled server storage. |
| V3 Chinese synthetic diagnosis pilot | [Admission and paired replay](experiments/ea_atpo_v3/lingxi_admission_20261128/REPORT.md) | Eight validation cases served as an engineering check for label scoring and ASK replay. Same-state DEFER remains unavailable. | Only the runner and aggregates are published. |
| V3 ReMeDi label restoration | [Official annotations and replay preparation](experiments/ea_atpo_v3/remedi_label_audit_20261128/REPORT.md) | Restored 1,517 uniquely mapped dialogues and 3,506 replay windows after historical pilot exclusion. | Semantic tags and case-level replay indices remain in controlled storage. |

## Result terminology

- **Engineering check:** format, environment, checkpoint-load, or protocol test;
  not an accuracy or clinical quality result.
- **Exploratory / development result:** used during method or checkpoint
  selection; cannot support an independent generalization claim.
- **Independent evaluation:** only use this label when the case-level split was
  frozen before method selection and never used for training or tuning.
- **N/A:** the outcome was not evaluated. It is not zero and must not be
  converted into a score.

The 400-case Frozen Holdout remains sealed. Human evaluation remains pending.
