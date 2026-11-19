# EA-ATPO V2 progress archive — 2026-11-20

This archive records the V2 ASK diagnosis and 50-case counterfactual expansion. Earlier progress reports remain intact. The latest active handoff is `EA_ATPO_VI_PROGRESS_AND_STRATEGY_LATEST.md`.

## Current method and boundary

Active actions: `ASK / ANSWER / DEFER`. Current data collection uses real Qwen3-8B Base/Patient and Qwen3-32B-AWQ DEFER Expert. This stage performed rollout collection only. It did not run Expert SFT, Q Regression, Actor distillation, PPO/ATPO RL, or holdout evaluation.

## ASK diagnosis and minimal repair

- Previous 10 Train cases / 29 states had 23 Unknown replies. Manual, Gold-blind review categorized 5 as fact under-matches, 3 as already public/partly answered, 2 as irrelevant/invalid, 11 as missing requested facts, and 2 as vague/compound.
- A single Patient-only fact review runs after standalone Unknown. It never receives Gold, and accepts fact-grounded partial replies. A targeted real-model smoke on five previously missed cases produced useful partial information in two; three remained Unknown.
- In the 10-case re-run, final Unknown fell 23→21, but ASK accuracy moved 37.9%→34.5%. Re-run variation affected one Base answer and one ASK answer, so this is not causal evidence of an accuracy gain.

## 50-case training-only rollout

- Seed `20261119`; 50 unique cases selected from the frozen 70-case Train partition; 15 Validation, 15 Internal Test, and the historical 400-case holdout were not read for selection or evaluation.
- Five batches of 10 cases: 146 public states, 438 action rollouts, all valid; 438/438 Q targets are available. ASK patient review produced 111 retries and 13 accepted replies, leaving 98/146 ASK responses classified Unknown.
- Accuracy: ANSWER 39.0%, ASK 38.4%, DEFER 43.8%. ASK corrections/regressions were 10/11. DEFER corrections/regressions were 25/18; exact Base/Expert option prediction agreement was 72/146 (49.3%).
- With ASK cost 0.05 and DEFER cost 0.60 already deducted once, mean return was 0.390 ANSWER, 0.334 ASK, -0.162 DEFER. These are Train-only exploratory outcomes, not independent generalization results.
- The historical 200-case protocol used the same model revisions but different prompts, output formats, generation caps, seed handling, structured decoding, parser and cases; its old accuracy cannot be used as a controlled comparison.
- Three malformed public states remain included and flagged for future sensitivity analysis.

## Resources and artifacts

GPU 2 was safe and process-free at startup, retained >115 GiB free during rollout, and was returned to 182,621 MiB free after both services were stopped. GPU 1/7 and other users’ processes were not touched. V2 output and runtime cache are under `experiments/ea_atpo_v2/outputs/ask_unknown_repair_50case_20261120/`.

Key paths:

- Merged trajectories: `experiments/ea_atpo_v2/outputs/ask_unknown_repair_50case_20261120/merged/pilot_trajectories.jsonl`
- Q Dataset: `experiments/ea_atpo_v2/outputs/ask_unknown_repair_50case_20261120/merged/q_dataset.jsonl`
- Batch summary: `experiments/ea_atpo_v2/outputs/ask_unknown_repair_50case_20261120/reports/batch_summary/batch_summary.csv`
- ASK audit: `experiments/ea_atpo_v2/reports/ASK_UNKNOWN_CLASSIFICATION_20261120.md`
- Expert protocol audit: `experiments/ea_atpo_v2/reports/EXPERT_PROTOCOL_COMPATIBILITY_20261120.md`
- Full pilot summary: `experiments/ea_atpo_v2/reports/COUNTERFACTUAL_50_CASE_PILOT_20261120.md`

## Next gate

The current Q Dataset can support an exploratory, lightweight Q Regression prototype, but has one rollout per state/action and therefore uncertain targets. A repeated-seed variance check on Train states is recommended before fitting. If fitting is later approved, keep the current 50 cases as Train, use the existing Validation cases for model/threshold selection, and reserve Internal Test for one frozen evaluation. The old 400-case holdout remains unopened. No further training starts automatically.
