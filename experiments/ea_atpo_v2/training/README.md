# EA-ATPO V2 training and evaluation framework

This directory adds the offline Q-regression, light policy-distillation, baseline, and evaluation layer. It does not change the V2 protocol, counterfactual collector, simulator, model-service launchers, or V1 code. The only action order is `ASK / ANSWER / DEFER`.

## Current data status

The framework has 19 unit tests using synthetic fixtures and adapters for Session A's real Q schema. Session A's latest committed expansion is `b6659b1` and the current read-only dataset is `../outputs/ask_unknown_repair_50case_20261120/merged/{q_dataset.jsonl,pilot_trajectories.jsonl}`. The merge contains 50 official train cases, 146 public states, and 438/438 valid action targets. The new merged-manifest audit is retained in `outputs/real_pilot50_merged_audit_20261120_rerun1/dataset_audit.json`.

Every state/action in this dataset has K=1 and `return_std=null`; each Q label is a single realized return, not a stable estimate of expected Q. The dataset uses `ASK_ONCE_THEN_FORCE_BASE_ANSWER_PATIENT_REVIEW_V2`. Its five source batches have matching config, Base/Expert revisions, continuation policy, and costs; the merged audit verifies their checksums and exact union. Do not mix it with the earlier V1 continuation policy. CPU smokes exercise the pipeline only and provide no generalization or medical-performance evidence. No formal Q or policy fit has been started.

The Session A report also flags three public states with semantic question/option anomalies (nine action trajectories). Keep those rows visible in audits and report a sensitivity result if the cases are later excluded; do not silently remove them.

The frozen official split has 70 train, 15 validation, and 15 internal-test cases, with 205 public train states. Formal fit CLIs require all 70 training case IDs, a passed audit for the exact Q file, one continuation policy ID, a versioned frozen-Base embedding cache, and at least K=3 valid rollouts per observed state/action. The current 50-case data therefore fails the formal-data gate by both case coverage and K. The minimum gate requires the remaining 20 train cases and 1,407 additional valid action rollouts if the remaining 59 train states are present: two more samples for each of the current 146×3 state/action cells (876), plus three for each of the remaining 59×3 cells (531). This yields 1,845 total outcomes across 615 state/action cells at K=3. It is an engineering threshold, not evidence that each target is a stable expected Q. The current run used temperature 0; repeated greedy calls may be identical. For stochastic expected-return estimates, lock a sampling configuration first; changing it requires one consistent cohort and cannot be silently mixed with these labels. Official validation/internal-test cases and the 400-case Frozen Holdout remain unopened.

## Data contract

`train_q.py` consumes the existing `build_q_dataset.py` JSONL schema:

```json
{
  "Q_ASK": 0.2,
  "Q_ANSWER": 0.5,
  "Q_DEFER": null,
  "valid_mask": {"ASK": true, "ANSWER": true, "DEFER": false},
  "return_std": {"Q_ASK": null, "Q_ANSWER": null, "Q_DEFER": null},
  "valid_sample_count": {"ASK": 1, "ANSWER": 1, "DEFER": 0},
  "attempted_sample_count": {"ASK": 1, "ANSWER": 1, "DEFER": 1},
  "metadata": {
    "case_id": "...",
    "state_id": "...",
    "public_state": [{"role": "user", "content": "..."}],
    "costs_included_in_targets": true
  }
}
```

The loader accepts V2 `Q_ASK`/`Q_ANSWER`/`Q_DEFER` fields or a `Q_target` mapping/list, supports the corresponding masks, valid/attempted sample counts, and `Q_*`-prefixed `return_std` keys. Missing targets stay `NaN` with a false mask. Rows with `costs_included_in_targets: true` are fit as supplied; the training path never deducts costs again. A single global `continuation_policy_id` is enforced so Q labels from different continuation policies cannot be silently combined.

`audit_q_dataset.py` checks action order, masks/counts, per-state/action K, sample standard deviation, Q means, unique rollout seeds, continuation-policy consistency, official split, and each raw rollout's `final_accuracy - ask_cost - defer_cost` return. With a single-run manifest it pins model/config/policy/cost provenance. With a merged manifest it verifies all source checksums, exact union of batch Q/trajectory rows, and identical model revisions, config, policy, and costs across batches. It records hashes without printing patient text. The current audit confirms all 438 targets are K=1, all `return_std` values are missing, and no action target is missing.

The existing Q/Actor CPU smoke used the audited V1 `PublicStateVectorizer`: lexical TF-IDF plus public-message length/structure statistics, fitted on gradient-training cases. These are sparse text/statistical features, not semantic embeddings. No validated frozen-Base state-embedding cache exists in the current artifacts; earlier generation-cache rows contain `embedding: null`.

`extract_state_embeddings.py` now provides the frozen-Base extractor/cache interface. It reads only `case_id`, `state_id`, and `public_state`, applies the pinned tokenizer chat template (`enable_thinking=false`, no generation prompt), mean-pools the final hidden state over unpadded tokens, and writes numeric vectors plus a SHA-256 manifest. Cache rows bind the vector to a canonical public-state hash, model revision, template/config hashes, pooling, truncation, dtype, batch size, device, and library versions. The trainer rejects missing manifests, stale state hashes, mixed configs, or a cache from a different embedding space.

The existing local Base snapshot is `/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/experiments/real_case_pilot_20261103/assets/models/Qwen3-8B`, pinned to revision `b968826d9c46dd6066d109eabc6255188de91218` (hidden size 4096). Tokenizer-only validation succeeded on CPU with the model-native chat template. No model weights were loaded and no embeddings were extracted in this task. If no validated cache is passed, the trainers use the text/statistical vectorizer only for smoke or explicit ablation; formal fitting requires the frozen-Base cache.

Example frozen embedding row (the extractor also writes a matching `.manifest.json` sidecar):

```json
{"case_id":"...","state_id":"...","feature_boundary":"public_state_before_action_only","state_hash":"<sha256>","rendered_prompt_hash":"<sha256>","embedding":[0.1,0.2],"model_id":"vi-qwen3-8b-base","model_revision":"<40-hex-commit>","pooling":"mean_last_hidden_state","max_length":2048,"prompt_template_hash":"<sha256>","embedding_config_hash":"<sha256>"}
```

The state reader rejects non-public roles and verifies that every state belongs to the requested official `train`, `validation`, or `internal_test` split. Validation/test feature extraction is allowed for inference only; those labels never enter fitting. The extractor starts no generation service. GPU extraction must use an explicitly allocated GPU; the default device is CPU.

The Q dataset builder admits official training cases only. Training then deterministically partitions those cases again by `case_id` into gradient-train and early-stop/calibration groups. All states for a case remain together. The official validation and internal-test sets are reserved for evaluation. Train/validation overlap is checked before optimization.

## Models and objectives

- `ConstantMeanQRegressor`: masked, sample-count-weighted per-action training means; no state-dependent parameters.
- `RidgeQRegressor`: one weighted ridge solve per action, with an unpenalized intercept and dual-space solve; default `alpha=1.0`. This is the default low-capacity Q model.
- `QRegressor`: two linear layers with GELU and hidden size 16 by default; masked Huber loss and case-level early stopping.
- Q loss weights each observed target by `valid_sample_count`; the target mean is not multiplied by cost again. `return_std` is preserved and checked against raw repeated returns.
- Action value calibration: action-wise affine calibration is fit on the case-disjoint early-stop/calibration partition. It calibrates gross expected correctness; registered costs are subtracted once when producing net values for a cost setting.
- `PolicyHead`: a two-layer logits head with hidden size 16 by default. Distillation uses `p_target = softmax(Q_target / temperature)` and cross-entropy against masked policy logits. Invalid actions are removed from both normalizations. States with fewer than two comparable actions are skipped; a validation split with no comparable states fails closed.
- Q and Actor splits are deterministic by `case_id`, using the same 20% internal validation fraction and seed `20261120`; no states from one case cross the split.
- Formal CLIs require the complete 70-case official train pool, a passed audit tied to the exact Q file and raw run manifest, a single continuation policy, a versioned Base embedding cache, and K≥3 for every observed state/action. `--smoke-only` is limited to ten cases and bypasses formal K/cohort gates.
- No large-model fine-tuning, PPO, GRPO, GAE, online Critic, or CONSULT path is implemented.

## Train commands

Run from the repository root. These commands are prepared for the next complete, single-policy training cohort. Use a new, empty `OUT` directory. Formal fitting remains gated until the dataset contains all 70 official training cases and K≥3 for every observed state/action. The current 50-case dataset has K=1 and must not be passed as a formal fit.

The current 50-case data audit can be reproduced with CPU only. The recorded report is at `outputs/real_pilot50_merged_audit_20261120_rerun1/dataset_audit.json`:

```bash
PY=/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/.venv-real-pilot/bin/python
PROJECT=/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO_V1
PILOT="$PROJECT/experiments/ea_atpo_v2/outputs/ask_unknown_repair_50case_20261120/merged"
SPLIT="$PROJECT/experiments/ea_atpo_value_router_20261105/artifacts/case_split.json"

CUDA_VISIBLE_DEVICES='' "$PY" -m experiments.ea_atpo_v2.training.audit_q_dataset \
  --q-dataset "$PILOT/q_dataset.jsonl" \
  --trajectories "$PILOT/pilot_trajectories.jsonl" \
  --run-manifest "$PILOT/merged_manifest.json" \
  --case-split "$SPLIT" \
  --report "$PROJECT/experiments/ea_atpo_v2/training/outputs/<new-audit>/dataset_audit.json" \
  --expected-cases 50 --expected-states 146 --expected-valid-labels 438
```

For the next expanded cohort, point the variables below at its merged output and use the exact `merged_manifest.json` that lists its source batches. The audit verifies source checksums, the exact union of batch Q/trajectory files, one config/model pair/continuation policy/cost schedule, official train-only cases, and cost accounting.

```bash
set -euo pipefail
PY=/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/.venv-real-pilot/bin/python
PROJECT=/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO_V1
RUN_DIR="$PROJECT/experiments/ea_atpo_v2/outputs/<expanded-70-train-case-run>/merged"
TRAJ="$RUN_DIR/pilot_trajectories.jsonl"
Q_DATA="$RUN_DIR/q_dataset.jsonl"
RUN_MANIFEST="$RUN_DIR/merged_manifest.json"
SPLIT="$PROJECT/experiments/ea_atpo_value_router_20261105/artifacts/case_split.json"
OUT="$PROJECT/experiments/ea_atpo_v2/training/outputs/<new-formal-prep-run>"  # choose a new path
mkdir -p "$OUT/audit" "$OUT/features"

"$PY" -m experiments.ea_atpo_v2.training.audit_q_dataset \
  --q-dataset "$Q_DATA" --trajectories "$TRAJ" \
  --run-manifest "$RUN_MANIFEST" --case-split "$SPLIT" \
  --report "$OUT/audit/dataset_audit.json"

# Run only after a scheduler/user has explicitly allocated this GPU to embedding extraction.
MODEL_PATH=/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/experiments/real_case_pilot_20261103/assets/models/Qwen3-8B
MODEL_ID=vi-qwen3-8b-base
MODEL_REVISION=b968826d9c46dd6066d109eabc6255188de91218
: "${ALLOCATED_GPU:?Set ALLOCATED_GPU only after an explicit GPU allocation}"
CUDA_VISIBLE_DEVICES="$ALLOCATED_GPU" "$PY" -m experiments.ea_atpo_v2.training.extract_state_embeddings \
  --states "$TRAJ" --split train --case-split "$SPLIT" \
  --model-path "$MODEL_PATH" --model-id "$MODEL_ID" --revision "$MODEL_REVISION" \
  --output "$OUT/features/train_public_embeddings.jsonl" \
  --device cuda:0 --dtype bfloat16 --max-length 2048 --batch-size 1 \
  --chat-template-kwargs-json '{"enable_thinking":false}'

# Compare low-capacity Q models on the same deterministic case_id split.
for MODEL in constant_mean ridge mlp; do
  "$PY" -m experiments.ea_atpo_v2.training.train_q \
    --q-dataset "$Q_DATA" --audit-report "$OUT/audit/dataset_audit.json" \
    --public-features "$OUT/features/train_public_embeddings.jsonl" \
    --real-data-confirmed --model "$MODEL" --minimum-k 3 \
    --validation-fraction 0.20 --seed 20261120 --ridge-alpha 1.0 --hidden-dim 16 \
    --output-dir "$OUT/q_$MODEL"
done

# Distill the same case-disjoint Q_target distribution into the lightweight policy head.
"$PY" -m experiments.ea_atpo_v2.training.train_policy \
  --q-dataset "$Q_DATA" --audit-report "$OUT/audit/dataset_audit.json" \
  --public-features "$OUT/features/train_public_embeddings.jsonl" \
  --real-data-confirmed --minimum-k 3 --temperature 1.0 \
  --validation-fraction 0.20 --seed 20261120 --hidden-dim 16 \
  --output-dir "$OUT/policy_distillation"
```

The embedding extractor writes a checksummed sidecar binding each vector to the public state, pinned Base revision, tokenizer chat template, pooling, truncation, and runtime config. It uses the frozen Qwen3-8B final hidden layer with mean pooling; no generation service is started. Train features include only official train public states. Validation and internal-test features must be extracted separately for inference. The current source snapshot is pinned to revision `b968826d9c46dd6066d109eabc6255188de91218`; no semantic vectors were extracted during this task.

The historical 10-case CPU Smoke remains retained under `outputs/real_pilot10_training_smoke_20261119T211801Z/` and its retry directory. Both confirm execution/checkpoint paths only; their small internal partitions are not generalization metrics.

## Prediction and evaluation

For each official evaluation split, extract features from that split's public states only, then predict those same states. Validation and internal-test cases are never included in Q fitting. The cache must use the identical Base revision and embedding config as the train cache. `predict.py` checks that config and reads only `public_state` from the paired trajectory file.

```bash
PY=/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/.venv-real-pilot/bin/python
PROJECT=/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO_V1
SPLIT="$PROJECT/experiments/ea_atpo_value_router_20261105/artifacts/case_split.json"
OUT="$PROJECT/experiments/ea_atpo_v2/training/outputs/<new-formal-prep-run>"
VAL_RUN="$PROJECT/experiments/ea_atpo_v2/outputs/<official-validation-run>"
VAL_TRAJ="$VAL_RUN/pilot_trajectories.jsonl"
VAL_MANIFEST="$VAL_RUN/run_manifest.json"
MODEL_PATH=/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/experiments/real_case_pilot_20261103/assets/models/Qwen3-8B
MODEL_ID=vi-qwen3-8b-base
MODEL_REVISION=b968826d9c46dd6066d109eabc6255188de91218
: "${ALLOCATED_GPU:?Set only after an explicit GPU allocation}"
mkdir -p "$OUT/validation"

CUDA_VISIBLE_DEVICES="$ALLOCATED_GPU" "$PY" -m experiments.ea_atpo_v2.training.extract_state_embeddings \
  --states "$VAL_TRAJ" --split validation --case-split "$SPLIT" \
  --model-path "$MODEL_PATH" --model-id "$MODEL_ID" --revision "$MODEL_REVISION" \
  --output "$OUT/validation/public_embeddings.jsonl" \
  --device cuda:0 --dtype bfloat16 --max-length 2048 --batch-size 1 \
  --chat-template-kwargs-json '{"enable_thinking":false}'

$PY -m experiments.ea_atpo_v2.training.predict \
  --checkpoint "$OUT/q_ridge/q_regressor.pt" --states "$VAL_TRAJ" \
  --public-features "$OUT/validation/public_embeddings.jsonl" \
  --output "$OUT/validation/q_predictions.jsonl"

$PY -m experiments.ea_atpo_v2.training.predict \
  --checkpoint "$OUT/policy_distillation/policy_head.pt" --states "$VAL_TRAJ" \
  --public-features "$OUT/validation/public_embeddings.jsonl" \
  --output "$OUT/validation/policy_predictions.jsonl"

$PY -m experiments.ea_atpo_v2.training.merge_predictions \
  --inputs "$OUT/validation/q_predictions.jsonl" \
           "$OUT/validation/policy_predictions.jsonl" \
           "$VAL_RUN/external_baseline_scores.jsonl" \
  --output "$OUT/validation/all_predictions.jsonl"

$PY -m experiments.ea_atpo_v2.training.evaluate_policies \
  --trajectories "$VAL_TRAJ" --predictions "$OUT/validation/all_predictions.jsonl" \
  --run-manifest "$VAL_MANIFEST" --case-split "$SPLIT" --split validation \
  --output-dir "$OUT/validation/evaluation"
```

The external baseline file must use the same state set and can provide a measured `uncertainty_score` and/or a separately evaluated `sft_action`. Missing inputs are reported as unavailable. No uncertainty values are synthesized. To select a threshold, first ensure the actual uncertainty scores are present, then rerun validation evaluation with `--select-uncertainty-threshold`; it writes a threshold lock. Never select a threshold on internal test.

```bash
$PY -m experiments.ea_atpo_v2.training.evaluate_policies \
  --trajectories "$VAL_TRAJ" --predictions "$OUT/validation/all_predictions.jsonl" \
  --run-manifest "$VAL_MANIFEST" --case-split "$SPLIT" --split validation \
  --select-uncertainty-threshold --output-dir "$OUT/validation/threshold_selection"
```

After locking the model, calibrator, and any selected threshold, evaluate once on the official internal-test split. Extract internal-test features separately using `--split internal_test`, generate Q and policy predictions, and merge them with that run's external baseline scores before evaluation. Use the same command form as validation with `--split internal_test`, the internal-test run's `--run-manifest`, and `--uncertainty-threshold <locked-validation-threshold>` only if a real uncertainty score and validation threshold exist. Do not use Frozen Holdout.

The evaluator uses the official `case_id` split and identical state/case coverage for every baseline. It includes Always ANSWER, Always DEFER, Static L2D (answer/defer only, Q advantage thresholded by `lambda_defer`), Uncertainty Threshold, SFT-only, and Q-guided Distillation when their required score inputs are present. Missing optional baseline inputs are reported as unavailable rather than fabricated.

The cost grid defaults to `lambda_defer = [0, 0.05, 0.10, 0.20, 0.40, 0.60]`; ASK cost remains 0.05. Utility is recomputed from raw `final_accuracy`, `ask_count`, `defer_count`, and cost components, not the already-costed `return`. Q and distilled-policy scores are shifted by the known cost delta once when evaluating another lambda. Raw correctness and per-action cost components are preserved in `normalized_evaluation.jsonl`.

Outputs include Accuracy and Utility with case-bootstrap 95% intervals, DEFER rate, ASK count, Expert Correction/Spoilage, Q MAE, advantage AUROC, action-wise calibration, Accuracy-Cost curve, and paired bootstrap comparisons against Always ANSWER. Bootstrap resampling uses cases, preserving within-case state pairing.

## Synthetic tests and GPU estimate

Run only the engineering tests with CPU visibility disabled:

```bash
CUDA_VISIBLE_DEVICES='' $PY -m unittest discover -s experiments/ea_atpo_v2/training/tests -v
```

The two-layer heads are small: with 3,009 TF-IDF features and 64 hidden units, each has about 193k parameters (under 1 MiB of FP32 weights). Q/Actor fitting on the current 205-state train pool needs only a few MiB of tensors and can run on CPU; a GPU head-only smoke would need under 1 GiB. Existing embeddings require no GPU. If embeddings must be extracted with Qwen3-8B, that is a separate inference job and prior service measurements were about 22 GiB; this framework does not launch it.

The current 10-case pipeline smoke disables CUDA and needs no GPU. The later formal workload is also a small frozen-feature head: the Q and policy heads are each under 1 MiB of FP32 weights at the documented dimensions; CPU is sufficient for fitting. GPU is only needed if a separate job extracts new Qwen3-8B embeddings. This task does not launch a model service or embedding job.
