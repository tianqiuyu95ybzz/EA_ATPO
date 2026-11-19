# EA-ATPO V2 Protocol Repair and Real Counterfactual Pilot

**Date:** 2026-11-20 (Asia/Shanghai)
**Scope:** Repair the real ANSWER / ASK / DEFER smoke protocol, then run the frozen 10-case, 29-state training pilot. No Q Regression, Actor distillation, Expert SFT, PPO, or RL training was run.

## Result summary

The real-service smoke passed on three distinct frozen training cases. The full pilot then completed all 87 paired action rollouts: 29 ASK, 29 ANSWER, and 29 DEFER. All were parse-valid and all 29 states produced a complete three-action Q row. The 400-case frozen holdout was not opened.

| Action | Correct | Accuracy | Mean net return | Mean total tokens | Mean sequential latency | Mean action cost |
|---|---:|---:|---:|---:|---:|---:|
| ANSWER | 9/29 | 31.03% | 0.3103 | 244.5 | 0.0599 s | 0.00 |
| ASK → Patient → Base answer | 11/29 | 37.93% | 0.3293 | 803.0 | 0.2778 s | 0.05 |
| DEFER → Expert answer | 9/29 | 31.03% | -0.2897 | 260.6 | 0.0992 s | 0.60 |

The utility is `final_accuracy - 0.05 * ask_count - 0.60 * defer_count`. Costs were applied once in the rollout return; token usage and latency are reported separately.

ASK corrected 3 ANSWER errors and degraded 1 ANSWER-correct state. Its raw accuracy was 6.90 percentage points higher than ANSWER; after the registered ASK cost, mean utility was 0.0190 higher. This is descriptive evidence from only 10 cases, not a stable performance claim.

DEFER had 0 corrections and 0 regressions relative to ANSWER in this pilot: Base/Expert correctness cells were both-correct 9, helpful 0, harmful 0, both-wrong 20. Expert accuracy matched Base at 31.03%; at cost 0.60 its observed mean utility was 0.60 below ANSWER. The small sample does not establish that the Expert lacks complementarity generally.

ASK returned non-unknown, textually novel information in 6/29 states (20.69%); the Patient simulator returned UNKNOWN in 23/29 (79.31%). The 6/29 figure is an operational proxy, not a clinical semantic assessment. All 29 ASK branches passed structural state-transition checks, appended the question and Patient reply, and continued to a parsed Base answer.

## Failure diagnosis and repair

The previous one-case smoke exposed two distinct issues:

1. The public question asked which artery had ruptured while its displayed A–D options were parasite names. Based only on the public text, this is a source-content mismatch. Base generated a “none of these” explanation that hit the 24-token limit; Expert returned a similar explanation and stopped normally. The strict A–D parser correctly rejected both. The Patient simulator was asked for a blood-pressure detail absent from the supplied facts and correctly returned UNKNOWN. This was not a reason to invent a patient response.
2. That earlier single case did not establish general parser or service failure; it combined a malformed public item with a poorly grounded ASK question.

The repair keeps strict answer parsing and adds vLLM constrained decoding using `^Final Answer: [A-D]$` for Base and Expert terminal answers, and a question-shape regex for ASK. The Base and Expert contracts now use the same final-answer format and require one listed option. The ASK prompt explicitly grounds the question in the visible symptoms/history and prohibits unrelated measurements. Patient matching retains the V1 semantic/paraphrase rule and continues to return UNKNOWN when the facts do not answer the question. No parser relaxation, Gold-based prompt correction, hidden-fact injection into Base/Expert, or patient-response fabrication was used.

The repaired 3-case smoke passed:

- Base ANSWER parsed: 3/3.
- Expert DEFER parsed: 3/3.
- ASK question, one Patient reply, legal state update, and post-ASK Base answer: 3/3.
- At least one non-UNKNOWN, textually new Patient reply: 1/3. The other two ASK replies remained valid UNKNOWN responses and were preserved.
- Gold was read only after the model calls for each case, for correctness/return calculation. Neither the Gold value nor the hidden atomic-fact list was serialized.

In the full pilot all 145 completion requests ended with `finish_reason=stop`; there were no parser failures, API failures, invalid trajectories, repeated ASK questions, or failed state transitions. The strict formats therefore repaired the observed output-protocol failure under the frozen generation settings. Constrained decoding ensures syntax, not clinical correctness; the public source mismatch remains visible in the scored data.

## Cost and inference accounting

There were 145 model requests: 87 Base, 29 Patient, and 29 Expert. The ASK branch makes two Base calls and one Patient call; ANSWER makes one Base call; DEFER makes one Expert call.

| Action | Prompt tokens total | Completion tokens total | Total tokens | Sequential model latency total | Total action cost |
|---|---:|---:|---:|---:|---:|
| ANSWER | 6,945 | 145 | 7,090 | 1.736 s | 0.00 |
| ASK | 22,118 | 1,170 | 23,288 | 8.056 s | 1.45 |
| DEFER | 7,411 | 145 | 7,556 | 2.876 s | 17.40 |
| **Total** | **36,474** | **1,460** | **37,934** | **12.669 s** | **18.85** |

Latency is the sum of sequential API-call latencies recorded within each rollout, not a service-startup time or a wall-clock throughput benchmark. Costs are utility units, not monetary inference prices.

## Data and Q-target audit

- Frozen training split only: 10 cases, 29 public states; ANSWER / ASK / DEFER share the same state digest for each state.
- Each state has one rollout per action under `ASK_ONCE_THEN_FORCE_BASE_ANSWER_V1`.
- Valid Q targets: **87/87**, aggregated into **29** state rows. Cost is already included in each target; no downstream cost was subtracted here.
- Each state-action has only one sample, so `return_std` is null. These labels do not estimate rollout variance.
- Serialized trajectories contain public states, visible Patient replies, model outputs, scores, costs, and telemetry. They contain no `gold`, `ground_truth`, `atomic_facts`, or `answer_idx` fields and do not store the Patient prompt or hidden-fact list. Patient replies are recorded because they become visible dialogue in the ASK branch.
- One public source case appears in three frozen states with an artery/parasite option mismatch. This finding is based only on the public prompt. Those states remain in the pilot and were not edited or dropped.

The Q interface is now exercised with real, paired returns. However, **this is not enough data to train a generalizing Q Regressor**: there are only 10 case clusters, 29 nested states, one sample per state-action, high ASK UNKNOWN frequency, and no observed DEFER correction in this slice. No Q Regression was started. More training cases and repeated action samples are needed before fitting/validating a value model.

## Models, environment, and reproducibility

- Base: `Qwen/Qwen3-8B`, revision `b968826d9c46dd6066d109eabc6255188de91218`.
- Expert: `Qwen/Qwen3-32B-AWQ`, revision `0499c3ac83fdef8810b907a23894ba91e95eddd8`.
- Patient: the same Qwen3-8B service with a separate Patient role prompt and access to only the current source case's atomic facts.
- vLLM 0.33.0; PyTorch 2.14.0; Transformers 5.20.2; BF16 Base and AWQ Expert; thinking disabled; temperature 0; top-p 1.0.
- GPU2 only. Preflight: 182,621 MiB free, 0% utilization, no compute PIDs. GPU6/7 had other users' workloads; neither was used. Services were stopped by the run's manager after the pilot; GPU2 returned to 0 MiB used and ports 30481/30482 closed. No other process was interrupted.
- All logs, caches, trajectories, smoke responses, and case-level summaries were saved under `experiments/ea_atpo_v2/outputs/phase1_gpu2_protocol_repair_20261119T173133Z/`. No system-disk cache/output was used. The transient `/dev/shm` path was a symlink into that project-local runtime directory and was removed after service shutdown.
- Unit regression suite: **12/12 passed**; Python compile checks passed. The run used base commit `6785727c9fbda3107b1e669c44c6077675c7d380` with the V2 protocol/runner edits present in the worktree. Both run manifests record the execution source SHA256 values so the uncommitted execution code is identifiable.
- The runner and protocol hashes recorded in the manifest are `03c87903880b1c42e9638ac896eb4174e3c0977ef3b1a7d500ff5f4afbab17d8` and `970616e563e661b22067d9675b708847c7b2cab42581df2c5ab8b4d9d143d409`, respectively. The inference configuration SHA256 is `c9d3519989e35f91cb14222f484428a525b60ab9ac5bec27ecf4da6f168b2ecd`.

Reproduction commands used the established vLLM runtime and a run-local copy of the GPU2 service configuration. Set `V2_PROJECT_ROOT` to the repository clone and `V2_PYTHON` to the configured serving environment's Python executable:

```bash
export V2_PROJECT_ROOT=/path/to/EA_ATPO_V1
export V2_PYTHON=/path/to/EA_ATPO/.venv-real-pilot/bin/python
cd "$V2_PROJECT_ROOT"

NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
  "$V2_PYTHON" \
  experiments/vi_ea_atpo/phase2/manage_vi_services.py \
  --config experiments/ea_atpo_v2/outputs/phase1_gpu2_protocol_repair_20261119T173133Z/model_services_gpu2.yaml start

NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
  "$V2_PYTHON" \
  experiments/ea_atpo_v2/scripts/run_counterfactual_pilot.py \
  --execute --smoke-cases 3 --config experiments/ea_atpo_v2/configs/phase1_counterfactual.json \
  --output-dir experiments/ea_atpo_v2/outputs/phase1_gpu2_protocol_repair_20261119T173133Z/v2_smoke3

NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
  "$V2_PYTHON" \
  experiments/ea_atpo_v2/scripts/run_counterfactual_pilot.py \
  --execute --cases 10 --config experiments/ea_atpo_v2/configs/phase1_counterfactual.json \
  --output-dir experiments/ea_atpo_v2/outputs/phase1_gpu2_protocol_repair_20261119T173133Z/pilot10

"$V2_PYTHON" \
  experiments/ea_atpo_v2/scripts/build_q_dataset.py \
  --trajectories experiments/ea_atpo_v2/outputs/phase1_gpu2_protocol_repair_20261119T173133Z/pilot10/pilot_trajectories.jsonl \
  --config experiments/ea_atpo_v2/configs/phase1_counterfactual.json \
  --output experiments/ea_atpo_v2/outputs/phase1_gpu2_protocol_repair_20261119T173133Z/pilot10/q_dataset.jsonl

"$V2_PYTHON" \
  experiments/ea_atpo_v2/scripts/analyze_counterfactual_pilot.py \
  --trajectories experiments/ea_atpo_v2/outputs/phase1_gpu2_protocol_repair_20261119T173133Z/pilot10/pilot_trajectories.jsonl \
  --q-dataset experiments/ea_atpo_v2/outputs/phase1_gpu2_protocol_repair_20261119T173133Z/pilot10/q_dataset.jsonl \
  --output-dir experiments/ea_atpo_v2/outputs/phase1_gpu2_protocol_repair_20261119T173133Z/pilot10/analysis

NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
  "$V2_PYTHON" \
  experiments/vi_ea_atpo/phase2/manage_vi_services.py \
  --config experiments/ea_atpo_v2/outputs/phase1_gpu2_protocol_repair_20261119T173133Z/model_services_gpu2.yaml stop
```

The run manifest records the frozen selection, model revisions, data/config hashes, endpoint checks, and status. `analysis/action_summary.csv`, `analysis/paired_state_results.csv`, and `analysis/pilot_summary.json` are local, ignored outputs. Only code and this aggregate report are intended for GitHub; raw case-level outputs and model weights are not.

## Decision

The protocol and data path now work with real models, and the pilot provides 87 empirical action targets. ASK shows a small positive utility difference in this sample, but only 6/29 responses passed the non-UNKNOWN novelty proxy. DEFER did not improve accuracy over ANSWER in these 29 states and its registered cost made its return lower. These findings are exploratory, with only 10 independent cases and no repeated rollouts. Keep V2 in data-collection/quality-audit mode; do not infer a learned policy or start Q Regression from this pilot alone.
