# Reproduce Phase 3 smoke

From repository root, set `DATA=experiments/ea_atpo_v3/gold_dataset_admission_20261121/private/ddx_data` and `EXP=experiments/ea_atpo_v3/ddx_phase3_20261121`. Use the official source files matching the digests in `SPLIT_MANIFEST.json` and `OVERLAP_AUDIT.json`. Original datasets and per-case records are intentionally excluded from Git.

```sh
CUDA_VISIBLE_DEVICES='' python3 "$EXP/test_cpu.py" --zip "$DATA/release_validate_patients.zip" --evidences "$DATA/release_evidences.json" --conditions "$DATA/release_conditions.json"
python3 "$EXP/audit_overlap.py" --train "$DATA/release_train_patients.zip" --validate "$DATA/release_validate_patients.zip"
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8 /pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/.venv-real-pilot/bin/python "$EXP/run_cpu_base.py" --zip "$DATA/release_validate_patients.zip" --evidences "$DATA/release_evidences.json" --conditions "$DATA/release_conditions.json" --out "$EXP/private/new_base.jsonl" --limit 50
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8 /pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/.venv-real-pilot/bin/python "$EXP/run_cpu_ask.py" --zip "$DATA/release_validate_patients.zip" --evidences "$DATA/release_evidences.json" --conditions "$DATA/release_conditions.json" --base "$EXP/private/new_base.jsonl" --out "$EXP/private/new_ask.jsonl"
python3 "$EXP/aggregate.py" --base "$EXP/private/new_base.jsonl" --ask "$EXP/private/new_ask.jsonl"
```

Output files must not already exist. Base files require the original pinned Qwen3-8B local snapshot and compatible Python environment. Only the snapshot index hash is rechecked by the runners; independently verify all weight shards before formal use. Output is sensitive and remains local; no GPU or API is needed for this CPU smoke. This path is limited to 50 rows; it is not a full-validation runner, production simulator, or training configuration.
