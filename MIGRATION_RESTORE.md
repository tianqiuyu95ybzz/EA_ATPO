# Restore EA-ATPO V3 on a new server

## GitHub checkout

After this branch is pushed, use:

```bash
git clone https://github.com/tianqiuyu95ybzz/EA_ATPO.git EA_ATPO
cd EA_ATPO
git fetch origin
git checkout --track origin/codex/ea-atpo-v3-migration-20261122
git rev-parse HEAD
sha256sum -c MIGRATION_SHA256SUMS.sha256
sha256sum -c MIGRATION_SHA256SUMS
```

The audit records the initial migration payload SHA and its remote verification.
For the current branch tip, compare `git rev-parse HEAD` with
`git ls-remote origin refs/heads/codex/ea-atpo-v3-migration-20261122` when you
restore. The manifest covers migration-added/modified public files and the V3
pilot/checkpoint summary reports named in the manifest; it excludes private
dialogues, per-case records, checkpoints, and model weights. Do not clone only
`main` if you need the V3 migration work.

## Restore boundary

GitHub restores code, reports, configurations, reviewed aggregates, and
reproduction commands. It intentionally does not restore private ReMeDi text,
the 20-case review packet, case mappings, case-level results, or model weights.
Before continuing, confirm PFS is mounted on the new server and locate the
restricted paths in `PFS_ARTIFACT_INDEX.md`. If those paths are unavailable,
stop and arrange an authorized private transfer; do not substitute public or
synthetic data.

## Model identity

The V3 pilot used original `Qwen/Qwen3-8B` revision
`b968826d9c46dd6066d109eabc6255188de91218`, with no adapter. The audited
Attempt20 Actor is ATPO-only from the raw base and is explicitly not an
SFT+ATPO model. A verified Qwen3-8B SFT+ATPO final checkpoint is currently
missing. Do not load either checkpoint as a silent replacement.

## Environment and safe smoke

The old pilot run records Python/PyTorch/Transformers only partially: the
runtime reported Transformers 5.20.2 and PyTorch 2.14.0+cu130, ran on CPU with
8 threads, and used `enable_thinking=false`, greedy decoding, 96 new tokens,
and a 3072-token context limit. The old host's system Python 3.12.3 has none of
the ML packages installed; this is not the pilot environment.

On the new host, first record `hostname`, GPU model/count/memory, driver, CUDA,
Python, PyTorch, Git, and PFS free space. Check `nvidia-smi` and existing
processes. Only run a single short synthetic-prompt load/generation check if a
GPU is clearly idle with adequate free memory. Do not run a 50-case evaluation,
training job, or Router fitting as part of restore verification.

The old V3 inference script is CPU-only and reads protected local files; do not
rerun it to reproduce a model-quality claim. The preserved original outputs
are the inputs to the pending human review.
