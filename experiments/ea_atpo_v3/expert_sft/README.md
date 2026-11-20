# EA-ATPO V3 Expert SFT

This is an isolated, gated experiment branch. The source material is the existing simulated 100-case Pilot, not human patient transcripts. The frozen 400-case holdout is not opened by this workflow. See `reports/EXPERT_DATA_AUDIT.md` and `reports/V3_FINAL_DECISION.md` for actual execution status and limitations.

- `data/` contains privacy-safe audit, split, review and request manifests. `data/private/` is Git-ignored PFS storage for dialogue-bearing queues, Base requests and raw responses. Never commit that directory.
- `configs/` records separate A/B/C configurations and the Router admission placeholder. A, B and C must not share adapters, optimizers or outputs. Expert B has its own hash-verified PFS copy of the AWQ initialization under ignored `models/`.
- `scripts/` contains reproducible audits, the human-review gate, Base request preparation/generation, gated dataset construction, AWQ smoke, med_qwen metadata verification, paired metrics, Oracle extraction and Router admission checks.
- `outputs/` contains local logs, attempts and later checkpoints. It is Git-ignored. New attempts must use new directories; do not overwrite or clean historical artifacts.
- `reports/` distinguishes completed checks from blocked or unimplemented work. No V3 training/evaluation result may be inferred from the presence of a script or config.

The 291 Base outputs have been generated and verified, and the separate synthetic 10-step AWQ LoRA smoke passed with a loading warning recorded in its report. A/B training, prediction and Dev-calibration code is prepared but has not been exercised end to end on reviewed data or trained checkpoints. A separate synthetic GPU integration check of the A/B losses was interrupted and must be completed for the current code. Current execution also requires actual human Gold review before formal SFT. Only after Dev selection and freezing may Internal Test be evaluated once per Expert. Router training requires a separately satisfied admission decision; PPO/ATPO RL is outside this experiment.

Run the meaningful synthetic checks without loading a model or opening the holdout:

```bash
python3 -m unittest discover -s experiments/ea_atpo_v3/expert_sft/tests -v
```

GPU launch and failure-recovery commands are in `reports/BASE_PAIRING_STATUS.md` and `reports/AWQ_LORA_SMOKE_STATUS.md`. The resource snapshot is not a reservation; recheck the explicit allowlist, minimum free memory, utilization and other processes immediately before every model load. GPUs 1 and 7 are prohibited.
