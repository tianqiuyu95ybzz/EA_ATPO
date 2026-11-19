# Restored ATPO Multi-turn Data Audit

**Status:** aggregate audit completed from local raw files. No model was loaded and no GPU was used.

## Finding on data provenance

The official ATPO train source at the pinned project commit contains **14,256 training cases**. Every prompt has the two-role shape `system/user`; this restored source has no multi-turn dialogue fields. Its stable row key is `index`, unique and sequential over 0–14255. All 14,256 official answer labels parse to A–D. The existing reconstructed Parquets are audited separately in `data_audit.json`; none was overwritten.

The multi-turn material available locally is from the prior 100-case EA-ATPO Pilot, where Base and Patient simulators generated public dialogue states. These are **simulated model conversations, not human patient transcripts**. The 100 cases have 291 public states; 191 states include at least one Patient reply and 283 Patient replies appear across snapshots. Public state IDs map to stable ATPO source indices and their source prompt text matched for 291/291 states.

Across the 100 cases, the deepest archived state has an average of **1.91 patient-reply turns per case**. This uses the deepest state snapshot per case so earlier turns repeated in later snapshots are not double-counted. Each case contributes an average of 2.91 public-state snapshots.

## Extractable Expert SFT data

The existing answer-only SFT files contain 3 case-isolated splits:

| Split | Cases | State samples | Samples with Patient reply | Target format valid | Gold matches source |
|---|---:|---:|---:|---:|---:|
| Train | 70 | 205 | 135 | 205 | 205 |
| Validation | 15 | 42 | 27 | 42 | 42 |
| Internal test | 15 | 44 | 29 | 44 | 44 |

Strict multi-turn states with Patient replies: **191** (99 one-reply states; 92 two-reply states). The present SFT set also includes 100 zero-reply states as the direct-answer anchor. Each source case contributes up to three public states, all kept in the same case_id split.

## Rollout and label quality

The source trajectory file contains 3492 action rollouts and 2 run-status records. 3465 rollout records are marked valid; 27 are invalid. Strict final option parse rate is 99.23%; invalid branches remain recorded and are not treated as successful model responses. All rollout Gold labels match the official source answer for their mapped case: 3492/3492.

The 400-case fresh holdout remains a pre-frozen **within-ATPO-train** holdout. Its identity metadata is disjoint from the Pilot/SFT state cases and no Gold or atomic facts were stored or read. The internal 15-case split is development-only; it is not a substitute for the untouched 400-case Holdout.

## Leakage boundary

SFT model input contains the original public case prompt plus only the current state prefix. The answer is the sole Assistant target. The builder does not serialize hidden atomic facts, Gold in input, future Patient replies, Expert advice, or Base terminal answers into the input. This report stores aggregate data only; it contains no prompt text, answer labels by case, Patient text, case IDs, or raw model outputs.

Detailed file hashes, schemas, validity counts by action, case split checks, and Parquet integrity are in `data_audit.json`, `split_summary.csv`, and `rollout_validity_by_action.csv`.
