# Medical evaluation readiness

Start with `MEDICAL_EVALUATION_READINESS.md`, then `PUBLIC_DATASET_COMPARISON.md` and `EA_ATPO_V3_NEXT_EXPERIMENT_PLAN.md`. `ROUTER_EXPERIMENT_CONFIG.yaml` is a disabled design artifact: it is not a trainer configuration authorized to run. No data, labels, rubric scores or patient dialogue are added to Git.

`verify_existing.py` reads the prior private review materials and prints aggregate integrity status only. An optional `--private-digests <new-private-file>` produces 40 individual output hashes under an ignored private path, never answer text; it refuses overwriting. Existing `private/per_output_hashes.jsonl` and `private/per_output_hashes_verified.jsonl` match. `EXISTING_INTEGRITY.json` contains only aggregate status and hash, not scores. The original 20 cases, 40 outputs and empty review forms remain unchanged.

The sources were checked on 2026-11-22. Study of public resource descriptions is not permission to use, transmit or import them. Source platform terms and data protection requirements may depend on date, jurisdiction and intended recipients; obtain a specific data-governance determination. No human evaluation, model generation, training, source-data download, or frozen-test access occurred in this task.
