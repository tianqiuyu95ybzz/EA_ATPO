#!/usr/bin/env bash
set -euo pipefail

ROOT=/kwkj-k8s/llm_team/tqy/mywork/ATPO-official
cd "$ROOT/codes-repro"

export CUDA_VISIBLE_DEVICES=1,2,3
export RAY_ADDRESS=
export API_KEY=EMPTY
export ATPO_USER_API_URL=http://127.0.0.1:30000/v1/completions
export ATPO_USER_MODEL=Qwen3-8B
export PATH="$ROOT/.venv-atpo/bin:$PATH"
export PYTHONUNBUFFERED=1

curl --fail --silent --max-time 5 http://127.0.0.1:30000/health >/dev/null

exec python3 -m recipe.atpo.mt_main \
  actor_rollout_ref.model.path="$ROOT/checkpoints/sft-qwen3-4b-full-hf" \
  critic.model.path="$ROOT/checkpoints/sft-qwen3-4b-full-hf" \
  data.train_files="['./data/train.parquet']" \
  data.val_files="['./data/test.parquet']" \
  data.return_raw_chat=True \
  data.train_batch_size=3 \
  data.max_prompt_length=1024 \
  data.max_response_length=512 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  +data.seed=1 \
  actor_rollout_ref.model_type=qwen3 \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=3 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=3 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.top_p=0.8 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.calculate_log_probs=False \
  actor_rollout_ref.rollout.agent.agent_loop_config_path=recipe/atpo/agent.yaml \
  actor_rollout_ref.rollout.agent.num_workers=3 \
  critic.model.use_remove_padding=False \
  critic.model.fsdp_config.param_offload=True \
  critic.model.fsdp_config.optimizer_offload=True \
  critic.optim.lr=1e-5 \
  critic.ppo_mini_batch_size=3 \
  critic.ppo_micro_batch_size_per_gpu=1 \
  critic.ppo_epochs=1 \
  critic.loss_agg_mode=token-mean \
  tree_search.M_trajectories=2 \
  tree_search.N_candidates=2 \
  tree_search.variance_threshold=1.2 \
  tree_search.pruning_enabled=True \
  tree_search.only_use_critic_value=True \
  tree_search.call_critic_enabled=True \
  extra_params.use_critic_in_loop=True \
  custom_reward_function.path=recipe/atpo/mt_reward_fn.py \
  custom_reward_function.name=mt_reward_fn \
  reward_model.reward_manager=mt_reward_manager \
  reward_model.loop_enable=False \
  reward_coefficients.correctness_score=3 \
  algorithm.adv_estimator=gae \
  algorithm.use_kl_in_reward=False \
  algorithm.kl_ctrl.kl_coef=0.001 \
  algorithm.gamma=1.0 \
  algorithm.lam=1.0 \
  trainer.logger="['console']" \
  trainer.project_name=medical-atpo \
  trainer.experiment_name=official-smoke-4b \
  trainer.n_gpus_per_node=3 \
  trainer.nnodes=1 \
  trainer.critic_warmup=0 \
  trainer.val_before_train=False \
  trainer.log_val_generations=0 \
  trainer.save_freq=1 \
  trainer.test_freq=-1 \
  trainer.total_epochs=1 \
  trainer.total_training_steps=1 \
  trainer.default_local_dir="$ROOT/checkpoints/official-atpo-smoke" \
  trainer.rollout_data_dir="$ROOT/outputs/official-atpo-smoke/rollout" \
  ray_kwargs.ray_init.num_cpus=32 \
  +ray_kwargs.ray_init.include_dashboard=False \
  +ray_kwargs.ray_init._temp_dir=/tmp/ray-atpo-tqy-smoke2
