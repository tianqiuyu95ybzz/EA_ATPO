#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/setup_cuda_driver_link.sh"
ATPO_ROOT="$ROOT/ATPO-official"
CODE_ROOT="$ATPO_ROOT/codes"
DATA_ROOT="$ATPO_ROOT/dataset"
PYTHON="$ROOT/.venv/bin/python"
RUN_DIR="${EA_ATPO_RUN_DIR:-$ROOT/experiments/ea_atpo_v1_smoke/original_atpo}"
MODEL_PATH="${EA_ATPO_MODEL_PATH:-/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/experiments/real_case_pilot_20261103/assets/models/Qwen3-8B}"
GPU_INDEX="${EA_ATPO_GPU:-}"
MAX_GPU_UTIL_PCT="${EA_ATPO_MAX_GPU_UTIL_PCT:-15}"
MIN_GPU_FREE_MIB="${EA_ATPO_MIN_FREE_MIB:-155000}"

if [[ -z "$GPU_INDEX" ]]; then
  echo "Set EA_ATPO_GPU to a GPU index with sufficient free memory and low utilization." >&2
  exit 2
fi
if [[ "$GPU_INDEX" == "1" || "$GPU_INDEX" == "7" ]]; then
  echo "GPU $GPU_INDEX is excluded from this project run." >&2
  exit 2
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "Missing project environment: $PYTHON" >&2
  exit 2
fi
if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "Qwen3 model not found at EA_ATPO_MODEL_PATH=$MODEL_PATH" >&2
  exit 2
fi
if [[ ! -f "$CODE_ROOT/recipe/atpo/mt_main.py" ]]; then
  echo "Official ATPO checkout missing at $ATPO_ROOT (expected commit 25af83688c06917f94245a4db06be7d0924c090c)." >&2
  exit 2
fi
if [[ ! -f "$DATA_ROOT/merged_train_dataset.jsonl" || ! -f "$DATA_ROOT/medqa_test_dataset.jsonl" ]]; then
  echo "Official ATPO data files are missing under $DATA_ROOT." >&2
  exit 2
fi

GPU_STATUS="$(nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits)"
FREE_MIB="${GPU_STATUS%%,*}"
UTIL_PCT="${GPU_STATUS##*, }"
if (( FREE_MIB < MIN_GPU_FREE_MIB || UTIL_PCT > MAX_GPU_UTIL_PCT )); then
  echo "Refusing to start on GPU $GPU_INDEX (free=${FREE_MIB}MiB, utilization=${UTIL_PCT}%)." >&2
  echo "This smoke test requires ${MIN_GPU_FREE_MIB}MiB free and utilization <= ${MAX_GPU_UTIL_PCT}%; choose another GPU or wait." >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export HF_HOME="$ROOT/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export NCCL_DEBUG="${EA_ATPO_NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${EA_ATPO_NCCL_DEBUG_SUBSYS:-INIT,ENV,NET}"
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=0
export EA_ATPO_NATIVE_RMSNORM="${EA_ATPO_NATIVE_RMSNORM:-0}"
NATIVE_RMSNORM_DIR="$ROOT/experiments/ea_atpo_v1_smoke/native_rmsnorm_fallback"
export PYTHONPATH="$NATIVE_RMSNORM_DIR${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$RUN_DIR/data" "$RUN_DIR/logs" "$RUN_DIR/checkpoints" "$RUN_DIR/rollouts" "$RUN_DIR/validation"

TRAIN_PARQUET="$RUN_DIR/data/train.parquet"
TEST_PARQUET="$RUN_DIR/data/test.parquet"
if [[ ! -s "$TRAIN_PARQUET" ]]; then
  "$PYTHON" "$CODE_ROOT/recipe/atpo/data_utils/mix_data_jsonl.py" \
    train "$TRAIN_PARQUET" 0.00015 "$DATA_ROOT/merged_train_dataset"
fi
if [[ ! -s "$TEST_PARQUET" ]]; then
  "$PYTHON" "$CODE_ROOT/recipe/atpo/data_utils/mix_data_jsonl.py" \
    test "$TEST_PARQUET" 0.001 "$DATA_ROOT/medqa_test_dataset"
fi

cd "$CODE_ROOT"
LOG_FILE="$RUN_DIR/logs/original_atpo_smoke_$(date -u +%Y%m%dT%H%M%SZ).log"
set -o pipefail
# SGLang sizes its KV cache from the available-memory baseline measured before
# weight loading, not from physical VRAM. Earlier runs at 0.75 reached graph
# capture but lost the remaining headroom to shared-GPU allocations, so the
# retry uses 0.60 and caps concurrent requests at the two smoke trajectories.
# Triton avoids the failing FlashInfer JIT link; the smoke-only native RMSNorm
# fallback avoids the old sgl-kernel wheel's missing SM100 cubin.
"$PYTHON" -m recipe.atpo.mt_main \
  data.train_files="[$TRAIN_PARQUET]" \
  data.val_files="[$TEST_PARQUET]" \
  data.return_raw_chat=True \
  data.train_batch_size=1 \
  data.max_prompt_length=4096 \
  data.max_response_length=256 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  +data.seed=1 \
  algorithm.adv_estimator=gae \
  algorithm.use_kl_in_reward=False \
  algorithm.kl_ctrl.kl_coef=0.001 \
  algorithm.gamma=1.0 \
  algorithm.lam=1.0 \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model_type=qwen3 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=2 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4352 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.response_length=256 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.60 \
  actor_rollout_ref.rollout.max_model_len=4352 \
  actor_rollout_ref.rollout.max_num_batched_tokens=4352 \
  actor_rollout_ref.rollout.max_num_seqs=2 \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=triton \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=1 \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.top_p=0.8 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.calculate_log_probs=False \
  actor_rollout_ref.rollout.agent.agent_loop_config_path=recipe/atpo/agent.yaml \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4352 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  critic.model.path="$MODEL_PATH" \
  critic.model.use_remove_padding=True \
  critic.model.fsdp_config.param_offload=False \
  critic.model.fsdp_config.optimizer_offload=False \
  critic.optim.lr=1e-5 \
  critic.ppo_micro_batch_size_per_gpu=1 \
  critic.ppo_max_token_len_per_gpu=4352 \
  critic.ppo_epochs=1 \
  critic.loss_agg_mode=token-mean \
  tree_search.M_trajectories=2 \
  tree_search.N_candidates=2 \
  tree_search.variance_threshold=1.2 \
  tree_search.pruning_enabled=False \
  tree_search.only_use_critic_value=True \
  tree_search.call_critic_enabled=True \
  extra_params.use_critic_in_loop=True \
  reward_coefficients.correctness_score=3 \
  reward_model.reward_manager=mt_reward_manager \
  trainer.logger='[console]' \
  trainer.project_name=ea_atpo_v1_smoke \
  trainer.experiment_name=original_atpo_baseline \
  ray_kwargs.ray_init.num_cpus=4 \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.critic_warmup=0 \
  trainer.val_before_train=False \
  trainer.save_freq=1 \
  trainer.test_freq=-1 \
  trainer.default_local_dir="$RUN_DIR/checkpoints" \
  trainer.rollout_data_dir="$RUN_DIR/rollouts" \
  trainer.validation_data_dir="$RUN_DIR/validation" \
  trainer.total_epochs=1 \
  2>&1 | tee "$LOG_FILE"
