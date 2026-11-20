# Phase 0: project and resource audit

Audit time: 2026-11-20 approximately 19:07 Asia/Shanghai. This is a snapshot, not a GPU reservation.

- Project: `/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO_V1`.
- Baseline: branch `ea-atpo-v1`, `4c3a0516a16f326dd3ccdc386f611faa28e762e4` (`Record checkpoint pruning and data recovery`). Both `origin` and `ea-atpo` name the same GitHub repository. A live `git ls-remote origin refs/heads/ea-atpo-v1` returned the same commit. The V3 branch did not exist remotely at that check.
- The baseline worktree had an existing modification to `ea_atpo_stage/recipe/ea_atpo/collect_counterfactuals.py` and existing untracked experiment files. Those changes are not included in V3. Four other worktrees were present and were not modified or merged.
- New isolated worktree: `/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO_V1_expert_sft_20261120`, branch `codex/ea-atpo-v3-expert-sft-20261120`, created at the baseline above. `sources/` remains read-only per `AGENTS.md`.
- The live `nvidia-smi` snapshot showed eight NVIDIA B20Z GPUs, each with 183,359 MiB total. GPU 1 and 7 are explicitly prohibited, regardless of apparent availability. The V2 `resource_policy` also prohibits interrupting foreign processes. The previous AWQ-SFT runner requires at least 150,000 MiB free, GPU utilization at most 20%, **no other compute process**, and at least 32 GiB left after its model/gradient preflight. V3 must recheck before each launch.
- At the initial snapshot all GPUs had compute processes. A later snapshot showed GPU 5 temporarily at 182,625 MiB free with no listed compute PID, while a process had just occupied it in the earlier snapshot. Availability is volatile; this observation is not a reservation or permission to share the card. GPUs 0, 2, 3, 4, and 6 had existing compute processes; GPUs 1 and 7 remain prohibited.
- `CUDA_VISIBLE_DEVICES` was unset. No model, inference service, training job, or foreign-process management was started during this audit.
- Existing experiment trees and checkpoints remain in their historical PFS locations. The former multi-turn SFT reports and manifests are reference evidence; their outputs are not V3 results. The 400-case frozen holdout was not opened.

The current GPU gate must be checked immediately before any load, using the allowlist, memory/utilization limits and a fresh PID list. If the check fails, the smoke and training remain blocked; no BF16 or alternate-model fallback is authorized.
