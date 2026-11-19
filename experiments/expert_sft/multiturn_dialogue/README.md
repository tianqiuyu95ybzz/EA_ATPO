# Multi-turn Dialogue Expert SFT

This is a separate, reproducible track that trains only the final-answer behavior of the DEFER Expert from existing public ATPO dialogue states. It does not replace the earlier 1,000/300 answer-only experiment, change ATPO, train a Router, or run RL.

## Frozen inputs

- Training source: tracked `experiments/real_case_pilot_20261103/outputs/full/pilot_public_states.jsonl` and `pilot_trajectories.jsonl`.
- Case split: tracked `experiments/ea_atpo_value_router_20261105/artifacts/case_split.json`; 70 train, 15 validation, 15 internal-test cases. All splits are by `case_id`.
- The builder makes one sample per public state: 205 train, 42 validation, and 44 internal-test states from 100 cases. It reads the answer key only for the Assistant target and takes the current public dialogue prefix from the frozen state snapshot.
- The clean 400-case holdout remains unopened until a validation-only checkpoint lock exists. `build_clean_holdout_eval.py` enforces that order.

The generated JSONL contains third-party medical question text and is intentionally ignored by Git. Rebuild into a new directory without overwriting the frozen local copy:

```bash
python3 experiments/expert_sft/multiturn_dialogue/scripts/build_multiturn_sft_dataset.py \
  --output-dir experiments/expert_sft/multiturn_dialogue/data/rebuilt_YYYYMMDD
```

The tracked `data/dataset_manifest.json` and `data/recovery_manifest.json` include source/data hashes, split counts and reconstruction instructions, without copying patient-question text or Gold labels into the metadata bundle.

## Model order

1. First test the existing locked `Qwen/Qwen3-32B-AWQ` with GPTQModel's trainable `torch_awq` backend. vLLM serving compatibility is not evidence that LoRA gradients work. `train_multiturn_sft.py --mode awq --smoke-steps 10` checks load, adapter injection, finite forward loss, backward, nonzero gradients, a parameter update, adapter save, and exact adapter reload.
2. Only if that real AWQ test fails for a model/backend compatibility reason, use the local pinned BF16 `Qwen/Qwen3-32B` fallback. A resource-gate failure is not evidence that AWQ is unsupported.

The AWQ environment is separate from both the project `.venv` and any active serving environment. All environment files, package caches, temporary files, checkpoints and logs stay under this directory. Do not use GPU 1 or GPU 7. The runner requires at least 150,000 MiB free, utilization at most 20%, no other compute process on the selected card, and at least 32 GiB left after model load and gradient preflight.

```bash
bash experiments/expert_sft/multiturn_dialogue/scripts/setup_awq_env.sh
EA_ATPO_AWQ_GPU=3 bash experiments/expert_sft/multiturn_dialogue/scripts/run_awq_lora_smoke.sh
EA_ATPO_AWQ_GPU=3 bash experiments/expert_sft/multiturn_dialogue/scripts/run_awq_lora_formal.sh
```

The formal run saves each epoch and selects the lowest validation assistant-token cross-entropy. The independent holdout is not used for checkpoint selection. No SFT training has run yet; see `reports/MULTITURN_SFT_AND_DATA_RECOVERY_20261119.md`.

## Paired evaluation

Use the exact same visible prompt and greedy generation settings for Base8B, original AWQ, and the selected SFT Expert. The inference runner omits the assistant target from model input and reads Gold only after generation. `assemble_multiturn_comparison.py` checks state-level pairing, reports Helpful/Harmful/Oracle results, and bootstraps at the `case_id` cluster level so multiple states from one case are not treated as independent.

## Model data and redistribution

- `data/qwen3_32b_awq_source_lock.json` records the pinned original AWQ model revisions and hashes. The model weights remain in their existing PFS model directory and are not copied or committed.
- Do not commit generated dialogue JSONL/Parquet, model weights, adapters, optimizer states, runtime caches, credentials, or any holdout predictions. Commit scripts, configs, source/model hashes, safe reports and manifests so the data can be reconstructed from the tracked project inputs.

## Latest execution status (2026-11-19)

The audit of the restored official ATPO train source found 14,256 indexed single-turn cases; the public multi-turn data are the existing 100-case simulated Pilot snapshots, not human transcripts. The Pilot-derived SFT split has 205/42/44 state samples from 70/15/15 isolated cases. The source audit, current AWQ software/tokenizer preflight, and the reason model-level training did not run are recorded in [`outputs/data_audit_20261119Tverified/EXECUTION_STATUS.md`](outputs/data_audit_20261119Tverified/EXECUTION_STATUS.md). The independent AWQ environment is installed under this directory and locked in `environment.lock.txt`; the AWQ model has not been loaded into CUDA and no LoRA update or SFT has run because no eligible GPU passed the no-other-process and memory gate.
