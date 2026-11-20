# EA-ATPO V3 Oracle Expert Experiment Report

**Date:** 2026-11-21
**Branch:** `codex/oracle-expert-20261121`
**Final evaluation run:** `/pfs/pfs-spMCeu/image_team/workspace-copy/EA-ATPO_Oracle_20261121/run_08/`
**Evidence level:** CPU-only reanalysis of an already-used development set; not an independent confirmation.

## Executive summary

The planned Expert A/B SFT work was paused before formal training. The 291-row review queue is still entirely pending; its 169 disagreements with historical labels are not verified errors and were not used as supervision. The question of interest can be tested without training an Expert: does a frozen Router select Base errors that an ideal answer-key Oracle could repair?

The answer-key Oracle yields a clear *upper bound*: preserving a correct Base answer and replacing an incorrect one with the benchmark key gives 100% accuracy on the evaluated states. The frozen Learned Router with the actual Qwen3-32B-AWQ Expert has case-macro utility `0.491` at DEFER cost `0.05`, compared with `0.400` for Always ANSWER and `0.434` for Always ASK. Its paired case-cluster confidence intervals include zero against both fixed policies. At cost `0.60`, the learned policy's utility falls to `0.358`, below both fixed policies.

The optimistic Learned Router + Oracle safety-gate result is `0.576` utility at cost `0.05`, but the gate uses the answer key after the Router decision to know whether a proposed DEFER would spoil a correct Base answer. It is not deployable. Its paired intervals are positive against both fixed policies for costs `0.05–0.20`, and include zero at `0.60`.

**Conclusion:** the experiment confirms that *perfect selective correction has large theoretical value on this sample*. It does **not** show that the current learned Router has reliable practical decision value, because the learned Router + actual Expert comparisons are imprecise and this same Dev set was used for checkpoint selection. No independent-generalization claim is supported.

## 1. Why Expert SFT was paused

The V3 SFT queue has 291 state-level records, not 291 independent cases. All 291 are marked `PENDING` in the human-review workflow. The 169 disagreements with older labels are review-priority flags only. Neither disagreement direction nor a model-generated answer establishes which label is correct. Using them as SFT targets would risk teaching unverified labels. The Oracle comparison instead uses the original source benchmark's answer key strictly after frozen actions are produced.

Phase 0 found no formal Expert SFT/LoRA training process from this task to stop, so none was stopped. The prior 10-step AWQ LoRA compatibility smoke, code, configs, audit reports, and checkpoints remain intact. No Router training, Expert SFT, PPO/RL, model service request, or GPU computation was performed in this phase.

## 2. Gold / answer-key audit

### Provenance and trust boundary

The audited source is the restored ATPO `merged_train_dataset.jsonl`, 14,256 rows, SHA-256 `434a6734dce931262f2ed9e3201a3751e57db3609b1612ab9666835ac8832fdc`. Its row metadata labels 9,400 records `medqa` and 4,856 `mcqa`. The official ATPO README says its datasets were adapted from public benchmarks, but it does not provide a per-row licensing/provenance map. The upstream MedQA and MedMCQA repositories describe exam-style questions with answer keys and expose MIT license files; this supports the benchmark-answer-key provenance but does not by itself establish the licensing of every transformed ATPO row. We therefore retain source data and answer text only in controlled storage and do not upload them to public GitHub. [ATPO README](https://github.com/Quark-Medical/ATPO), [MedQA repository](https://github.com/jind11/MedQA), [MedMCQA repository](https://github.com/medmcqa/medmcqa)

The present Oracle source should be interpreted as **benchmark-provided answer keys**, not independently clinician-adjudicated truth. The MEDIQ paper describes deriving interactive cases from MedQA and CRAFT-MD, and its Hugging Face card labels that separate dataset CC-BY-4.0; neither source is the answer-key source used in this run. No claim of real-doctor review or clinical validity is made here. [MEDIQ paper](https://papers.nips.cc/paper_files/paper/2024/file/32b80425554e081204e5988ab1c97e9a-Paper-Conference.pdf), [MEDIQ dataset card](https://huggingface.co/datasets/stellalisy/mediQ)

### Reconciliation checks

The evaluation joined the source answer key only after frozen Router predictions were replayed. Across 43 Dev states and three actions:

- 129/129 source answer keys parsed;
- 129/129 initial public prompts exactly matched the corresponding source row;
- 129/129 recorded final-answer correctness labels reconciled to the source key;
- 129/129 model outputs were marked format-valid;
- 0 serialized action records contained `gold`, `ground_truth`, `atomic_facts`, or `answer_idx` fields;
- each case mapped to exactly one source row.

These are consistency checks against the source key, not an independent validation of the key's medical correctness. The 291 pending records were not used in the Oracle run, and no answer text or case-level Gold was added to this repository.

The four answer sources are kept distinct: the source dataset's `ground_truth.answer` is the benchmark key; historical simulated dialogue/patient replies are interaction records rather than Gold; Qwen Base and Expert final answers are predictions; and the 291 pending review-queue targets are unverified candidates. Only the first item was used as the offline Oracle key in this run.

## 3. Oracle definitions

Let `b(s) ∈ {0,1}` indicate whether the frozen Base answer matches the benchmark key for state `s`; let `g(s)` be the source answer key.

**Oracle Conditional** is the ideal rule:

```text
if b(s) = 1: ANSWER with the original Base answer; no Expert call
if b(s) = 0: DEFER and return g(s); count one DEFER call
```

This rule has 100% accuracy by construction and is a non-deployable upper bound because it requires knowing Base correctness, which is only available after joining the Gold key.

**Learned Router + Oracle safety gate** uses the frozen Router's proposed action. ANSWER and ASK use their recorded real branches. If it proposes DEFER, the offline Oracle preserves Base when Base is Gold-correct, otherwise returns the key. This safety gate also sees post-hoc correctness and is therefore another *optimistic, non-deployable upper bound*. It must not be described as a real Oracle model or a deployable policy.

**Always DEFER (direct answer-key Oracle)** returns the answer key for every state and pays one call per state. It is included to separate an always-call upper bound from selective Oracle Conditional.

The actual DEFER Expert is Qwen3-32B-AWQ; the actual Base is Qwen3-8B. The Oracle itself is a deterministic offline use of the answer key. No Expert weights were trained or invoked during this evaluation.

## 4. Split, leakage, and frozen model audit

The V2 counterfactual sample has 50 cases split by `case_id` into 35 Router Train cases and 15 Dev cases. All 50 are within the V3 source Train partition. The evaluated Dev subset consists of 43 public states and 129 paired ANSWER/ASK/DEFER outcomes from those 15 cases. The 15 Dev cases were previously used for early stopping/model selection, so this is a development reanalysis, not a test set. The V3 Internal Test and 400-case Frozen Holdout were not opened.

The Router feature boundary is the public state before action, represented by frozen Qwen3-8B embeddings. Gold, source answer keys, Expert outputs, and realized outcomes were not Router features. The three frozen Q-guided Distillation checkpoints (seeds 17, 42, 2026) were reloaded on CPU and replayed against saved predictions. Maximum score differences were `7.45e-8`, `5.96e-8`, and `7.45e-8`, respectively.

ASK branch outcomes were reused from the saved V2 counterfactual pilot under continuation policy `ASK_ONCE_THEN_FORCE_BASE_ANSWER_PATIENT_REVIEW_V2`; this phase did not re-query the Patient simulator. The historical manifest identifies the Patient role as the Qwen3-8B Base family but does not carry a separate pinned revision field, so this report does not claim an independently verified Patient revision.

| Artifact | SHA-256 |
|---|---|
| Seed 17 checkpoint | `9821e7ca701189d37df9567fbf8ffaff9eaddb537e97448d8019803e3b8e2d32` |
| Seed 42 checkpoint | `b461fab29c675070b3d1f02abb52b7ca84d8218c6a9c0dae4ac523c0bf0157b4` |
| Seed 2026 checkpoint | `3405648a8284a526db22b0e211e190302705443753d788e858b0d93b7ac83aeb` |
| Base / embedding model revision | `Qwen3-8B` / `b968826d9c46dd6066d109eabc6255188de91218` |
| Actual Expert revision | `Qwen3-32B-AWQ` / `0499c3ac83fdef8810b907a23894ba91e95eddd8` |

## 5. Metrics and cost accounting

The primary score is case-macro utility over the 15 case clusters. Within each case, all its states are averaged first; then the 15 case means are averaged. Raw state-weighted paired-cell denominators are also shown where relevant.

```text
Utility = terminal correctness − 0.01 × cumulative ASK count − c_DEFER × cumulative DEFER count
```

`c_DEFER ∈ {0.05, 0.10, 0.20, 0.60}`. ASK cost `0.01` is the frozen Router Q-target cost. The original rollout collector used ASK `0.05` and DEFER `0.60`; this reanalysis rebuilds utility from raw correctness and action counts, applies the stated costs once, and does not subtract costs from an already-net Q value. Token/latency cost is not included. Router actions remain frozen for every cost; there was no cost-specific threshold tuning.

All 95% intervals use 10,000 percentile bootstrap replicates resampling `case_id` clusters, keeping every state from a sampled case together. With only 15 clusters, intervals are exploratory and unstable.

## 6. Paired Base / actual Expert result

The raw state-count four-cell table is:

| Base | Qwen3-32B-AWQ correct | Qwen3-32B-AWQ wrong | Total |
|---|---:|---:|---:|
| Correct | 11 | 6 | 17 |
| Wrong | 9 | 17 | 26 |
| Total | 20 | 23 | 43 |

- Base accuracy: `17/43 = 39.5%` state-weighted; case-macro `0.400`.
- Actual Expert accuracy: `20/43 = 46.5%` state-weighted; case-macro `0.478`.
- Expert corrected `9/26 = 34.6%` of Base-error states; case-cluster 95% CI `[6.9%, 65.0%]`.
- Expert spoiled `6/17 = 35.3%` of Base-correct states; case-cluster 95% CI `[0.0%, 69.2%]`.
- Paired state-weighted Expert-minus-Base accuracy difference: `+7.0 pp`; case-cluster 95% CI `[-20.0, +34.9] pp`.

The point estimate is positive but the uncertainty is wide. The Expert is not perfectly reliable; its correction rate is not close to 100%, and a nontrivial fraction of correct Base answers are degraded.

## 7. Policy comparison

The following table reports case-macro estimates at `c_DEFER=0.05`. Rates average the three frozen Router seeds. Always ASK is included as an additional fixed-action baseline.

| Policy | Accuracy | Utility | ASK rate | DEFER proposals/calls | Mean total cost |
|---|---:|---:|---:|---:|---:|
| Always ANSWER | 0.400 | 0.400 | 0.000 | 0.000 | 0.000 |
| Always ASK (real continuation) | 0.444 | 0.434 | 1.000 | 0.000 | 0.010 |
| Always DEFER, actual Qwen Expert | 0.478 | 0.428 | 0.000 | 1.000 | 0.050 |
| Always DEFER, direct answer-key Oracle | 1.000 | 0.950 | 0.000 | 1.000 | 0.050 |
| Oracle Conditional (post-hoc Base correctness) | 1.000 | 0.970 | 0.000 | 0.600 | 0.030 |
| Frozen Learned Router + actual Expert | 0.507 | 0.491 | 0.474 | 0.241 | 0.017 |
| Frozen Learned Router + Oracle safety gate | 0.589 | 0.576 | 0.474 | 0.241 proposed; 0.167 called | 0.105 |

The frozen learned policy's proposed action frequencies are ANSWER `28.5%`, ASK `47.4%`, DEFER `24.1%`. Across seeds, DEFER proposals were 11/43, 9/43, and 10/43. With the post-hoc Oracle gate, effective calls were 9, 8, and 5 because proposed DEFERs on Base-correct states were preserved.

### Cost sensitivity

Accuracy and Router action rates stay fixed because no policy was retuned. Utility changes only through the predeclared DEFER cost.

| DEFER cost | Always ANSWER utility | Always ASK utility | Always DEFER actual Expert | Learned + actual Expert | Learned + Oracle gate | Oracle Conditional |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.400 | 0.434 | 0.428 | 0.491 | 0.576 | 0.970 |
| 0.10 | 0.400 | 0.434 | 0.378 | 0.479 | 0.568 | 0.940 |
| 0.20 | 0.400 | 0.434 | 0.278 | 0.455 | 0.551 | 0.880 |
| 0.60 | 0.400 | 0.434 | −0.122 | 0.358 | 0.484 | 0.640 |

At costs `0.05/0.10/0.20/0.60`, the Learned Router + actual Expert utility differences versus Always ANSWER are `+0.091/+0.079/+0.055/−0.042`; versus Always ASK they are `+0.056/+0.044/+0.020/−0.076`.

### Paired utility differences, clustered by case

| Cost | Learned + actual Expert vs Always ANSWER (95% CI) | Learned + actual Expert vs Always ASK (95% CI) | Learned + Oracle gate vs Always ANSWER (95% CI) | Learned + Oracle gate vs Always ASK (95% CI) |
|---:|---|---|---|---|
| 0.05 | +0.091 `[-0.034, 0.236]` | +0.056 `[-0.013, 0.157]` | +0.176 `[0.038, 0.323]` | +0.141 `[0.054, 0.247]` |
| 0.10 | +0.079 `[-0.043, 0.221]` | +0.044 `[-0.024, 0.143]` | +0.168 `[0.035, 0.309]` | +0.133 `[0.050, 0.235]` |
| 0.20 | +0.055 `[-0.063, 0.190]` | +0.020 `[-0.047, 0.115]` | +0.151 `[0.026, 0.283]` | +0.116 `[0.039, 0.212]` |
| 0.60 | −0.042 `[-0.150, 0.074]` | −0.076 `[-0.158, 0.016]` | +0.084 `[-0.011, 0.181]` | +0.050 `[-0.015, 0.129]` |

All Learned Router + actual Expert utility intervals include zero against both Always ANSWER and Always ASK. Although the Oracle-gated upper bound is above both fixed actions at costs `0.05–0.20` on these Dev cases, its gate is unavailable to a deployable Router and the Dev cases were reused for model selection.

Paired case-cluster accuracy difference versus Always ANSWER is `+0.107` for Learned Router + actual Expert (95% CI `[-0.015, +0.256]`) and `+0.189` for Learned Router + Oracle gate (95% CI `[+0.048, +0.341]`). The actual-Expert interval includes zero; the Oracle-gated interval reflects the non-deployable Gold-informed upper bound.

## 8. Router and DEFER diagnostics

The three frozen seeds have case-macro accuracy `0.544`, `0.511`, `0.467` (mean `0.507`, sample SD `0.039`) and utility at cost `0.05` `0.526`, `0.381`, `0.315` (mean `0.491`, sample SD `0.039`). Router behavior is variable even before generalization is tested.

Across pooled state/seed decisions, the actual Learned policy retains `92.2%` of Base-correct outcomes and reaches correctness on `23.1%` of Base-error states after its selected action. Among its DEFER calls, final Expert accuracy is `63.3%`, but useful corrections are only `40.0%` of calls; `45.5%` of deferred Base errors remain wrong. These pooled conditional metrics are descriptive and are not independent of the three seed policies.

For comparison, Always DEFER with the actual Expert retains only `64.7%` of Base-correct outputs; it corrects `34.6%` of Base errors overall. The Oracle-gated learned policy's `94.1%` Base-correct retention and 100% accuracy on its actual Oracle calls are consequences of the post-hoc Gold gate, not evidence of an operational safeguard.

## 9. What this experiment does and does not establish

1. **Can a perfect answer-key Expert create value?** Yes, as an ideal upper bound on this sample. Oracle Conditional has 100% accuracy and utility `0.64` even at cost `0.60`, because it pays only on Base errors. This assumes access to Base correctness and is not deployable.
2. **Does the current learned Router have demonstrated practical value?** No. Its actual-Expert point utility is above fixed baselines at costs `0.05–0.20`, but all paired 95% intervals include zero. At `0.60` its point utility is below Always ANSWER and Always ASK.
3. **Does Learned Router + Oracle show the Router can approach the upper bound?** It shows overlap between some frozen Router DEFER proposals and Base errors in this reused Dev set. Its outcome is an optimistic, Gold-informed gate and does not establish deployable selection.
4. **Is this a generalization result?** No. There are only 15 case clusters, and those Dev cases were used for model selection. The V3 Internal Test and 400-case Frozen Holdout remain unopened.
5. **Has the EA-ATPO core method been validated?** Only its answer-key Oracle potential has been demonstrated. A real Router's reliable selective-DEFER value has not been established. A new untouched case set and a frozen, Gold-blind routing rule are required.

## 10. Reproduction, integrity, and storage

Use the script and frozen config under `experiments/ea_atpo_v3/oracle_expert/`. Run the CPU-only unit tests and replay command in the adjacent `README.md`. The runner refuses to overwrite existing run directories and checks source/data/checkpoint hashes, split boundaries, prompt matches, action coverage, and label reconciliation.

The final run's `run_manifest.json` records:

- pre-run Git HEAD `fb4cff092bc165247ac6cd47d91e6a897aeede82` and a clean worktree;
- script SHA-256 `4d0078231c2784ec57949e553ec84f1845670369ae4370e6a77110a55f2ba618`;
- CPU-only execution; `model_generation_or_training=false`; `frozen_holdout_opened=false`;
- aggregate summary SHA-256 `f0879c2db11ef53287fb5c79a092d33a164580a7b4eb402e92adaf3fb2b19679`;
- private state result SHA-256 `d860a21b77f1d5a1bea49ae20a0fcc69aa5c62bf3d75358891b5e213de3edc18`.

Run 08 is stored with directory mode `0700` and files mode `0600` under `/pfs/pfs-spMCeu/image_team/workspace-copy/EA-ATPO_Oracle_20261121/run_08/`. It contains `aggregate_summary.json`, private state results, and the run manifest. Runs 01–07 are preserved. Public GitHub receives only code, configuration, this report, checksums, and a sanitized aggregate summary; no source case text, per-case correctness, Gold text, hidden facts, model weights, or credentials are included.

## 11. GitHub delivery

This report is part of branch `codex/oracle-expert-20261121`. The final commit and remote SHA verification are recorded in the task completion message and the branch history. No other worktree or branch is merged or modified by this experiment.
