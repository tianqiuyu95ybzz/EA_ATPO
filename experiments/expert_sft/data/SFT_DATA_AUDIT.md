# SFT Data Audit

- Source: official ATPO train data at pinned commit `25af83688c06917f94245a4db06be7d0924c090c`.
- Frozen cases: 1,000 train and 300 validation. No historical holdout source rows or labels were read for dataset construction; only train/validation entries were consumed.
- Rebuilt Parquet SHA-256 matched the frozen source hash: `c586d8653460e8bef102f7c6231a1356dbfa88b5b35c3538219724efd4c1a5c5`.
- Case ID, public-state hash, and prompt hash matched the prior selection manifest for every selected row.
- Missing or malformed A-D labels: 0; the builder stops on any invalid label.
- Duplicate IDs/hashes, prompt duplicates, and train/validation overlap: 0.
- Hidden `atomic_facts`, answer explanations, and Gold reasoning are excluded from model input. Targets contain only the official answer letter in `Expert Final Answer: X` format.
- Gold reliability is limited to source/schema/integrity checks; labels were not re-adjudicated by clinicians.

Outputs: `train_sft.jsonl`, `validation_sft.jsonl`, and `sft_data_manifest.json`.
