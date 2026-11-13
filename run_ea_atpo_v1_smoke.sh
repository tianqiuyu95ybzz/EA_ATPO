#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/setup_cuda_driver_link.sh"
cd "$ROOT"
ATPO_ROOT="$ROOT/ATPO-official"
CODE_ROOT="$ATPO_ROOT/codes"
DATA_ROOT="$ATPO_ROOT/dataset"
PYTHON="$ROOT/.venv/bin/python"
RUN_DIR="${EA_ATPO_RUN_DIR:-$ROOT/experiments/ea_atpo_v1_smoke/ea_atpo_v1_attempt01}"
RUN_DIR="$(realpath -m "$RUN_DIR")"
MODEL_PATH="${EA_ATPO_MODEL_PATH:-/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/experiments/real_case_pilot_20261103/assets/models/Qwen3-8B}"
GPU_LIST="${EA_ATPO_GPUS:-${EA_ATPO_GPU:-}}"
MAX_GPU_UTIL_PCT="${EA_ATPO_MAX_GPU_UTIL_PCT:-15}"
MIN_GPU_FREE_MIB="${EA_ATPO_MIN_FREE_MIB:-100000}"

if [[ -z "$GPU_LIST" ]]; then
  echo "Set EA_ATPO_GPU or EA_ATPO_GPUS to allowed GPU index/indices with sufficient free memory." >&2
  exit 2
fi
if [[ ! "$GPU_LIST" =~ ^[0-7](,[0-7])*$ ]]; then
  echo "Invalid GPU list '$GPU_LIST'; use comma-separated physical GPU indices, such as 4,6." >&2
  exit 2
fi
IFS=',' read -r -a GPU_IDS <<< "$GPU_LIST"
GPU_COUNT="${#GPU_IDS[@]}"
SEEN_GPU_IDS=()
for GPU_INDEX in "${GPU_IDS[@]}"; do
  if [[ "$GPU_INDEX" == "1" || "$GPU_INDEX" == "7" ]]; then
    echo "GPU $GPU_INDEX is excluded from this project run." >&2
    exit 2
  fi
  for SEEN_GPU in "${SEEN_GPU_IDS[@]}"; do
    if [[ "$GPU_INDEX" == "$SEEN_GPU" ]]; then
      echo "GPU $GPU_INDEX occurs more than once in EA_ATPO_GPUS." >&2
      exit 2
    fi
  done
  SEEN_GPU_IDS+=("$GPU_INDEX")
done
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

for GPU_INDEX in "${GPU_IDS[@]}"; do
  GPU_STATUS="$(nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits)"
  FREE_MIB="${GPU_STATUS%%,*}"
  UTIL_PCT="${GPU_STATUS##*, }"
  if (( FREE_MIB < MIN_GPU_FREE_MIB || UTIL_PCT > MAX_GPU_UTIL_PCT )); then
    echo "Refusing to start on GPU $GPU_INDEX (free=${FREE_MIB}MiB, utilization=${UTIL_PCT}%)." >&2
    echo "Each selected GPU requires ${MIN_GPU_FREE_MIB}MiB free and utilization <= ${MAX_GPU_UTIL_PCT}%; choose another GPU or wait." >&2
    exit 2
  fi
done

export CUDA_VISIBLE_DEVICES="$GPU_LIST"
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
export PYTHONPATH="$ROOT/ea_atpo_stage:$CODE_ROOT:$NATIVE_RMSNORM_DIR${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$RUN_DIR/data" "$RUN_DIR/logs" "$RUN_DIR/checkpoints" "$RUN_DIR/rollouts" "$RUN_DIR/validation"
LOG_FILE="$RUN_DIR/logs/ea_atpo_v1_smoke_$(date -u +%Y%m%dT%H%M%SZ).log"

TRAIN_PARQUET="$RUN_DIR/data/train.parquet"
TEST_PARQUET="$RUN_DIR/data/test.parquet"
has_api_key_column() {
  "$PYTHON" - "$1" <<'PY'
import sys
import pyarrow.parquet as pq

try:
    raise SystemExit(0 if "api_key" in pq.read_schema(sys.argv[1]).names else 1)
except Exception:
    raise SystemExit(1)
PY
}

# Preserve the official converted-data schema with a local non-secret sentinel.
# The EA Patient simulator runs through the same local SGLang server and does
# not use this field for authentication.
LOCAL_SMOKE_API_KEY="ea-atpo-local-smoke-placeholder"
if [[ ! -s "$TRAIN_PARQUET" ]] || ! has_api_key_column "$TRAIN_PARQUET"; then
  API_KEY="$LOCAL_SMOKE_API_KEY" "$PYTHON" "$CODE_ROOT/recipe/atpo/data_utils/mix_data_jsonl.py" \
    train "$TRAIN_PARQUET" 0.00015 "$DATA_ROOT/merged_train_dataset"
fi
if [[ ! -s "$TEST_PARQUET" ]] || ! has_api_key_column "$TEST_PARQUET"; then
  API_KEY="$LOCAL_SMOKE_API_KEY" "$PYTHON" "$CODE_ROOT/recipe/atpo/data_utils/mix_data_jsonl.py" \
    test "$TEST_PARQUET" 0.001 "$DATA_ROOT/medqa_test_dataset"
fi

"$PYTHON" - "$RUN_DIR/run_manifest.json" "$TRAIN_PARQUET" "$TEST_PARQUET" "$MODEL_PATH" "$GPU_LIST" "$LOG_FILE" <<'PY'
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

manifest_path, train_path, test_path, model_path, gpu_list, log_path = sys.argv[1:]

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def command(*args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"

packages = {}
for name in ("torch", "transformers", "ray", "sglang", "vllm", "datasets", "pyarrow"):
    try:
        packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        packages[name] = None

manifest = {
    "status": "launching",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "run_dir": str(Path(manifest_path).parent.resolve()),
    "git_commit": command("git", "-C", str(Path.cwd()), "rev-parse", "HEAD"),
    "official_atpo_commit": command("git", "-C", str(Path.cwd() / "ATPO-official"), "rev-parse", "HEAD"),
    "gpu_indices": [int(item) for item in gpu_list.split(",")],
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "gpu_snapshot": command("nvidia-smi", "--query-gpu=index,name,memory.free,memory.used", "--format=csv,noheader"),
    "python": sys.version,
    "packages": packages,
    "model": {
        "path": str(Path(model_path).resolve()),
        "config_sha256": sha256(str(Path(model_path) / "config.json")),
        "tokenizer_config_sha256": sha256(str(Path(model_path) / "tokenizer_config.json")),
    },
    "data": {
        "official_train_jsonl_sha256": sha256("ATPO-official/dataset/merged_train_dataset.jsonl"),
        "official_test_jsonl_sha256": sha256("ATPO-official/dataset/medqa_test_dataset.jsonl"),
        "train_parquet": {"path": train_path, "sha256": sha256(train_path)},
        "test_parquet": {"path": test_path, "sha256": sha256(test_path)},
    },
    "training_config": {
        "train_rows": 2,
        "validation_rows": 1,
        "seed": 1,
        "actor_critic_model": "Qwen3-8B",
        "max_actor_actions": 3,
        "max_asks": 1,
        "max_consults": 1,
        "M_trajectories": 2,
        "N_candidates": 2,
        "response_length": 512,
        "costs": {"ask": 0.05, "consult": 0.20, "defer": 0.60},
        "correctness_reward": 1.0,
        "costs_charged_in": "EA tree edge rewards only; episode scalar is reporting-only because advantages/returns are precomputed",
        "agent_loop": "recipe.ea_atpo.ea_agent_loop.EAATPOAgentLoop",
        "reward_manager": "recipe.ea_atpo.ea_reward_manager.EARewardManager",
        "reward_accounting": "copied from the EA tree trace once; precomputed Actor advantages and Critic returns are unchanged",
    },
    "log_path": log_path,
}
Path(manifest_path).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Wrote run manifest: {manifest_path}")
PY

cd "$CODE_ROOT"
set -o pipefail
# SGLang sizes its KV cache from the available-memory baseline measured before
# weight loading, not from physical VRAM. The two-GPU retry reached graph
# capture at 0.60 but the first FSDP-to-SGLang weight sync lacked 1.16 GiB.
# Use 0.35 to reserve more per-GPU headroom; this still yields KV capacity far
# above the two 4,352-token smoke requests. Multi-stage wake-up releases KV
# cache while FSDP materializes weights, then restores it for generation.
# Triton avoids the failing FlashInfer JIT link; the smoke-only native RMSNorm
# fallback avoids the old sgl-kernel wheel's missing SM100 cubin.
# Critic updates before Actor in the trainer. Its manual FSDP parameter and
# optimizer offload releases Critic state before Actor's first Adam allocation.
"$PYTHON" -m recipe.ea_atpo.mt_main_ea \
  data.train_files="[$TRAIN_PARQUET]" \
  data.val_files="[$TEST_PARQUET]" \
  data.return_raw_chat=True \
  data.train_batch_size=1 \
  data.max_prompt_length=4096 \
  data.max_response_length=512 \
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
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4608 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$GPU_COUNT" \
  actor_rollout_ref.rollout.response_length=512 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
  actor_rollout_ref.rollout.max_model_len=4608 \
  actor_rollout_ref.rollout.max_num_batched_tokens=4608 \
  actor_rollout_ref.rollout.max_num_seqs=2 \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=triton \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=3 \
  actor_rollout_ref.rollout.multi_stage_wake_up=True \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.top_p=0.8 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="$ROOT/ea_atpo_stage/recipe/ea_atpo/agent.yaml" \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4608 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  critic.model.path="$MODEL_PATH" \
  critic.model.use_remove_padding=True \
  critic.model.fsdp_config.param_offload=True \
  critic.model.fsdp_config.optimizer_offload=True \
  critic.optim.lr=1e-5 \
  critic.ppo_micro_batch_size_per_gpu=1 \
  critic.ppo_max_token_len_per_gpu=4608 \
  critic.ppo_epochs=1 \
  critic.loss_agg_mode=token-mean \
  tree_search.M_trajectories=2 \
  tree_search.N_candidates=2 \
  tree_search.variance_threshold=1.2 \
  tree_search.pruning_enabled=False \
  tree_search.only_use_critic_value=True \
  tree_search.call_critic_enabled=True \
  extra_params.use_critic_in_loop=True \
  reward_coefficients.correctness_score=1 \
  +ea_atpo.max_actor_actions=3 \
  +ea_atpo.max_asks=1 \
  +ea_atpo.max_consults=1 \
  +ea_atpo.ask_cost=0.05 \
  +ea_atpo.consult_cost=0.20 \
  +ea_atpo.defer_cost=0.60 \
  +ea_atpo.correctness_reward=1.0 \
  +ea_atpo.trace_path="$RUN_DIR/traces/train.jsonl" \
  reward_model.reward_manager=ea_atpo \
  trainer.logger='[console]' \
  trainer.project_name=ea_atpo_v1_smoke \
  trainer.experiment_name=ea_atpo_v1_protocol_smoke \
  ray_kwargs.ray_init.num_cpus=4 \
  trainer.n_gpus_per_node="$GPU_COUNT" \
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
