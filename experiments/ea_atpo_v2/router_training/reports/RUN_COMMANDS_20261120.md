# EA-ATPO V2 Router run commands

Run from the root of worktree `EA_ATPO_V1_router_training` on the PFS workspace. The output directory must be empty; scripts refuse to overwrite prior artifacts.

## 1. Recompute costed labels

    python3 -m experiments.ea_atpo_v2.router_training.scripts.prepare_costed_dataset \
      --q-source experiments/ea_atpo_v2/recovery_bundle_20261120/q_dataset.jsonl \
      --trajectories experiments/ea_atpo_v2/recovery_bundle_20261120/pilot_trajectories.jsonl \
      --official-case-split experiments/ea_atpo_value_router_20261105/artifacts/case_split.json \
      --output-dir experiments/ea_atpo_v2/router_training/outputs/real_q_router_20261120

## 2. Extract frozen Qwen3-8B public-state embeddings

    mkdir -p experiments/ea_atpo_v2/router_training/outputs/real_q_router_20261120/tmp \
      experiments/ea_atpo_v2/router_training/outputs/real_q_router_20261120/cache/hf
    CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 \
      HF_HOME="$PWD/experiments/ea_atpo_v2/router_training/outputs/real_q_router_20261120/cache/hf" \
      TRANSFORMERS_CACHE="$PWD/experiments/ea_atpo_v2/router_training/outputs/real_q_router_20261120/cache/hf/transformers" \
      TMPDIR="$PWD/experiments/ea_atpo_v2/router_training/outputs/real_q_router_20261120/tmp" \
      /pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO_V1/.venv/bin/python -u \
      -m experiments.ea_atpo_v2.training.extract_state_embeddings \
      --states experiments/ea_atpo_v2/router_training/outputs/real_q_router_20261120/public_states_only.jsonl \
      --split train \
      --case-split experiments/ea_atpo_value_router_20261105/artifacts/case_split.json \
      --model-path /pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/experiments/real_case_pilot_20261103/assets/models/Qwen3-8B \
      --model-id Qwen/Qwen3-8B \
      --revision b968826d9c46dd6066d109eabc6255188de91218 \
      --output experiments/ea_atpo_v2/router_training/outputs/real_q_router_20261120/qwen3_8b_frozen_public_embeddings.jsonl \
      --device cuda:0 --dtype bfloat16 --pooling mean_last_hidden_state \
      --max-length 2048 --batch-size 1 --truncation-side left --seed 20261120

## 3. Primary three-seed training

    CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
      /pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO_V1/.venv/bin/python -u \
      -m experiments.ea_atpo_v2.router_training.scripts.train_evaluate \
      --q-dataset experiments/ea_atpo_v2/router_training/outputs/real_q_router_20261120/q_dataset_cost_ask001_defer005.jsonl \
      --trajectories experiments/ea_atpo_v2/recovery_bundle_20261120/pilot_trajectories.jsonl \
      --split experiments/ea_atpo_v2/router_training/outputs/real_q_router_20261120/case_split_35_train_15_dev.json \
      --embedding-cache experiments/ea_atpo_v2/router_training/outputs/real_q_router_20261120/qwen3_8b_frozen_public_embeddings.jsonl \
      --output-dir experiments/ea_atpo_v2/router_training/outputs/real_q_router_20261120/training_final

## 4. Stronger-regularization sensitivity

    CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
      /pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO_V1/.venv/bin/python -u \
      -m experiments.ea_atpo_v2.router_training.scripts.train_evaluate \
      --q-dataset experiments/ea_atpo_v2/router_training/outputs/real_q_router_20261120/q_dataset_cost_ask001_defer005.jsonl \
      --trajectories experiments/ea_atpo_v2/recovery_bundle_20261120/pilot_trajectories.jsonl \
      --split experiments/ea_atpo_v2/router_training/outputs/real_q_router_20261120/case_split_35_train_15_dev.json \
      --embedding-cache experiments/ea_atpo_v2/router_training/outputs/real_q_router_20261120/qwen3_8b_frozen_public_embeddings.jsonl \
      --ridge-alpha 100 --hidden-dim 8 --weight-decay 0.1 \
      --output-dir experiments/ea_atpo_v2/router_training/outputs/real_q_router_20261120/regularization_sensitivity

## 5. Reload and compare checkpoints

    /pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO_V1/.venv/bin/python -u \
      -m experiments.ea_atpo_v2.router_training.scripts.verify_checkpoints \
      --q-dataset experiments/ea_atpo_v2/router_training/outputs/real_q_router_20261120/q_dataset_cost_ask001_defer005.jsonl \
      --split experiments/ea_atpo_v2/router_training/outputs/real_q_router_20261120/case_split_35_train_15_dev.json \
      --embedding-cache experiments/ea_atpo_v2/router_training/outputs/real_q_router_20261120/qwen3_8b_frozen_public_embeddings.jsonl \
      --training-dir experiments/ea_atpo_v2/router_training/outputs/real_q_router_20261120/training_final

## 6. Write hash-bound run manifest

Run after committing the code/config/report on this experiment branch:

    /pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO_V1/.venv/bin/python -u \
      -m experiments.ea_atpo_v2.router_training.scripts.write_run_manifest \
      --repo "$PWD" \
      --run-root experiments/ea_atpo_v2/router_training/outputs/real_q_router_20261120

The executed source bundle and processed-data checksums, embedding manifest, per-seed metrics, checkpoint paths, and reload audit are under the PFS outputs path above. The output tree is intentionally Git-ignored because it contains case-level labels and model artifacts.
