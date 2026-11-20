# Reproduce the Router audit

This is a CPU-only audit. It reads the existing result files from the source worktree and writes aggregate audit artifacts only. It does not train a model, call a GPU, open the Frozen Holdout, or load/modify checkpoints.

## Inputs

The read-only source worktree must contain the already existing ignored run artifacts under:

`experiments/ea_atpo_v2/router_training/outputs/real_q_router_20261120/`

It must also contain the original 50-case rollout recovery file under:

`experiments/ea_atpo_v2/recovery_bundle_20261120/pilot_trajectories.jsonl`

The audit validates input SHA256 values, case split and state keys before calculating metrics. It intentionally emits aggregate results without case identifiers or clinical text.

## Dependency

Python 3.10+ and NumPy are required. No PyTorch or Transformers dependency is used by the audit script. If NumPy is absent, install a pinned CPU-only NumPy in a workspace-local virtual environment, not a system location:

```bash
python3 -m venv .venv-router-audit
. .venv-router-audit/bin/activate
python -m pip install 'numpy==1.26.4'
```

## Run

From the repository root in the audit worktree:

```bash
python experiments/ea_atpo_v2/router_training/audits/20261120/audit_router_results.py \
  --source-root /pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO_V1_router_training \
  --output-dir experiments/ea_atpo_v2/router_training/audits/20261120/results
```

The command can be pointed at a new output directory for a non-overwriting rerun. It will refuse to use an output directory that overlaps the source worktree.

Expected summary: `status=passed`, `train_dev_overlap=0`, `train_states=103`, `dev_states=43`, and the corrected three-seed case-cluster bootstrap intervals recorded in `AUDIT_REPORT.md`.

## Verify the production bootstrap fix

The audit replay uses the same seed, replicate count and case-level result maps as the patched `cluster_bootstrap_delta` in `scripts/train_evaluate.py`. The direct validation performed for this audit extracted and ran only that function against the saved result maps; it did not execute the training script or import Torch. It reproduced the audit intervals exactly.

The script also compiles both Python files in memory for syntax validation. A full model training/evaluation run is deliberately outside this reproduction procedure.

## Verify delivered audit files

From this directory, validate the delivered file hashes with:

```bash
sha256sum -c MANIFEST.sha256
```

The manifest covers the report, reproduction instructions, audit script, aggregate result files, and the modified router evaluation script. It does not include itself, model weights, checkpoints, or case-level data.
