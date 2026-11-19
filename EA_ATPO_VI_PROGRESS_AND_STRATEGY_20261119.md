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

## 2026-11-19 addendum: multi-turn Expert SFT and data cleanup

The current SFT approach is revised: first use the 100-case existing ATPO public-dialogue Pilot to make case-isolated multi-turn Expert targets (70/15/15 cases, 205/42/44 state samples). The earlier 1,000/300 one-turn SFT artifacts remain unchanged and are not the primary data in this revised track. Gold is only an assistant target; current public patient replies are visible, future replies and hidden atomic facts are not.

The locked Qwen3-32B-AWQ becomes the first LoRA candidate using GPTQModel's `torch_awq` trainable backend. A separate Python environment and 10-step smoke script are prepared; the smoke must prove load, adapter injection, finite loss, backward, nonzero gradients, optimizer update, save and reload before formal training. If AWQ compatibility itself fails, use the pinned BF16 model fallback. No GPU/model smoke, SFT, paired model evaluation, Router, or RL has run. At the last check, all eligible GPUs were occupied; GPUs1 and7 remain forbidden.

CPU tokenization/chat-template validation passed for all 291 public-state examples (maximum sequence length 346 tokens; configured cap 4096). Four groups of byte-identical reconstruction Parquets and three identical metadata/metric files were deduplicated using relative links without breaking old paths. Unique checkpoints, historical reports and active service caches were preserved. The idempotent exact-SHA cleanup script has shared 54,332,055 bytes of duplicate payload through relative links and removed two stale bytecode files (104,507 bytes); a task-created 15 MB dependency/tokenizer cache was also removed. Generated medical JSONL/Parquet remains ignored by Git; tracked source Pilot states, case split, scripts, aggregate hashes and model SHA lock suffice to reconstruct it.

See `experiments/expert_sft/multiturn_dialogue/README.md` and `reports/MULTITURN_SFT_AND_DATA_RECOVERY_20261119.md` for commands and gates. GitHub handoff should include those scripts/configs/manifests/reports but omit model weights, medical case text, Gold-bearing generated JSONL, caches, and optimizer checkpoints.

Recovery bundle pushed to `ea-atpo-v1` at commit `a3f4c4f`.
