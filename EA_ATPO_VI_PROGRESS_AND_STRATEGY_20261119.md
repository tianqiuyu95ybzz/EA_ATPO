# EA-ATPO / VI-EA-ATPO Progress Archive — 2026-11-19

This is the dated archive for the current Expert SFT implementation stage. It records the active research route, historical results, pinned models/data, actual environment and resource blockers, completed code/data work, and exact next commands.

For the maintained latest status and context-recovery prompt, see [EA_ATPO_VI_PROGRESS_AND_STRATEGY_LATEST.md](EA_ATPO_VI_PROGRESS_AND_STRATEGY_LATEST.md).

## State at archive time

- Historical AWQ pairing and Router evidence are retained as context only. The old 400-case holdout has already been used and is not a clean test.
- Official ATPO train data is restored and hash-checked. Answer-only SFT inputs for the frozen 1,000 Train and 300 Validation cases are built; no prior assistant answers or hidden-fact fields were found in the model messages.
- A new 400-case within-ATPO-train holdout was frozen before checkpoint selection, excluding the previous 1,700 cases and the Phase 2 cohort. Gold has not been materialized for it.
- Qwen3-32B BF16 at revision `9216db5781bf21249d130ec9da846c4624c16137` is downloading resumably into the project. It is not yet complete. The Qwen3-32B-AWQ checkpoint is not used for SFT.
- Current package versions are recorded in `experiments/expert_sft/reports/environment_audit_20261119.md`. BF16 execution and CUDA training have not been tested.
- No GPU passed the reserved free-memory, utilization, and no-compute-process gate. No existing process or model service was interrupted.
- SFT smoke/formal training, paired new-model evaluation, Router fitting, fresh-test evaluation, new-Expert Phase 2 pilot, VOI estimation/search, and RL are all Not Run.
- Code and configs for the gated steps are under `experiments/expert_sft/`; no original ATPO algorithm code was modified.

The current progress file is maintained incrementally. Historical result values, exact file hashes, model revisions, safety gates, service adaptation, and recovery commands are documented there.
