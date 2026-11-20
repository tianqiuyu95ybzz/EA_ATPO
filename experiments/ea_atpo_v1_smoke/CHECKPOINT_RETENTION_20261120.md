# EA-ATPO local cleanup record (2026-11-20)

Removed only superseded intermediate checkpoint snapshots and rebuildable runtime caches. Each completed run retains its final checkpoint; historical reports, manifests, rollouts, recovery exports, and the V2 recovery bundle remain.

- Checkpoint snapshots removed: 12 directories (2.27 TB logical; 2.27 TB allocated).
- Rebuildable caches removed: 4 directories (0.56 GB logical).
- Total: 2.27 TB logical bytes removed.
- Retained: Attempt09 `global_step_10`; Attempt06 and Attempt07 `global_step_2`; Original ATPO Attempt20 `global_step_2`; Attempts02–05 each retain their sole `global_step_1`; Attempt09 HF Actor/Critic and Original ATPO inference export.
- Did not touch `.venv`, model caches/weights, source datasets, other experiments, any running process, or parallel-session files.
- The V2 50-case recovery bundle is committed separately and includes SHA-256 checksums; raw `outputs/` copies remain locally available.

## Removed checkpoint directories

- `experiments/ea_atpo_v1_smoke/ea_atpo_v1_attempt09_stability/checkpoints/global_step_1` (189.1 GB; 32 files)
- `experiments/ea_atpo_v1_smoke/ea_atpo_v1_attempt09_stability/checkpoints/global_step_2` (189.1 GB; 32 files)
- `experiments/ea_atpo_v1_smoke/ea_atpo_v1_attempt09_stability/checkpoints/global_step_3` (189.1 GB; 32 files)
- `experiments/ea_atpo_v1_smoke/ea_atpo_v1_attempt09_stability/checkpoints/global_step_4` (189.1 GB; 32 files)
- `experiments/ea_atpo_v1_smoke/ea_atpo_v1_attempt09_stability/checkpoints/global_step_5` (189.1 GB; 32 files)
- `experiments/ea_atpo_v1_smoke/ea_atpo_v1_attempt09_stability/checkpoints/global_step_6` (189.1 GB; 32 files)
- `experiments/ea_atpo_v1_smoke/ea_atpo_v1_attempt09_stability/checkpoints/global_step_7` (189.1 GB; 32 files)
- `experiments/ea_atpo_v1_smoke/ea_atpo_v1_attempt09_stability/checkpoints/global_step_8` (189.1 GB; 32 files)
- `experiments/ea_atpo_v1_smoke/ea_atpo_v1_attempt09_stability/checkpoints/global_step_9` (189.1 GB; 32 files)
- `experiments/ea_atpo_v1_smoke/ea_atpo_v1_attempt06/checkpoints/global_step_1` (189.1 GB; 32 files)
- `experiments/ea_atpo_v1_smoke/ea_atpo_v1_attempt07/checkpoints/global_step_1` (189.1 GB; 32 files)
- `experiments/ea_atpo_v1_smoke/original_atpo_attempt20/checkpoints/global_step_1` (189.1 GB; 32 files)

## Removed reproducible caches

- `experiments/ea_atpo_v1_smoke/qwen3_32b_awq_tp2_launch_20261114/cache` (233.7 MB)
- `experiments/ea_atpo_v2/outputs/phase1_gpu2_service_20261119T165122Z/runtime` (109.3 MB)
- `experiments/ea_atpo_v2/outputs/phase1_gpu2_protocol_repair_20261119T173133Z/runtime` (109.8 MB)
- `experiments/ea_atpo_v2/outputs/ask_unknown_repair_50case_20261120/runtime` (109.2 MB)

See `CLEANUP_MANIFEST_20261120.json` for per-path sizes and verification state.
