# Reproduce the existing 50-case pilot audit

This is a CPU-only audit of the previous Qwen3-8B pilot. It does **not** evaluate the required SFT→ATPO checkpoint: that checkpoint and its provenance are unavailable. Raw case-level and invalid-output files are local under `private/` and excluded from Git. `classification_report.json` and `confusion_matrix.csv` contain aggregate counts over the public disease ontology only.

From the repository root, set `OLD=experiments/ea_atpo_v3/ddx_phase3_20261121/private`, `DATA=experiments/ea_atpo_v3/gold_dataset_admission_20261121/private/ddx_data`, and select a new, empty output directory with a `private/` child. Recreate the inputs from official sources and the existing private pilot records; outputs cannot be recreated from Git alone. Run:

```sh
python3 experiments/ea_atpo_v3/atpo_checkpoint_audit_20261121/audit_pilot.py --base "$OLD/base_cpu_smoke_retry.jsonl" --ask "$OLD/ask_cpu_smoke.jsonl" --archive "$DATA/release_validate_patients.zip" --evidences "$DATA/release_evidences.json" --conditions "$DATA/release_conditions.json" --out <new-output-directory>
```

`MODEL_SHA256SUMS` contains absolute paths and file hashes for the original pilot model and the separate, rejected ATPO smoke candidate; `sha256sum -c MODEL_SHA256SUMS` reads large local weight files and requires those paths. `SHA256SUMS` covers published artifacts, not raw data or weights. The raw invalid output and detailed case results are intentionally local only. No synthetic rows stand in for missing ATPO model inference.
