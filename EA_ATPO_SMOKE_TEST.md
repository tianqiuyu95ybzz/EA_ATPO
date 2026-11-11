# EA-ATPO V1 Smoke Test Status

**Status: pending; the original ATPO smoke has not reached its first PPO update.** The official `sgl-kernel==0.2.9+cu128` imports with PyTorch `2.7.1+cu128`; isolated SM100 RMSNorm and Qwen3-shaped rotary calls pass. Attempts 13 and 14 completed Critic/Actor and SGLang Qwen3-8B initialization, KV allocation, and CUDA graph capture. Both OOMed when the first rollout woke the FSDP model back onto the GPU after shared allocations changed. No PPO step or parameter update occurred. Attempt 14 passed the 170,000 MiB / three-sample stability gate, so that gate alone is insufficient under current shared-GPU contention. GPUs 1 and 7 remain excluded.

## Source and environment

- Branch: `ea-atpo-v1`; official ATPO base commit: `25af83688c06917f94245a4db06be7d0924c090c`.
- The original ATPO source tree under `codes/` has not been edited.
- Python 3.12.3; PyTorch `2.7.1+cu128`; CUDA runtime 12.8; Transformers 4.54.1; Ray 2.47.1; SGLang 0.4.10.post2.
- `sgl-kernel==0.2.9+cu128` is the official CUDA 12.8 wheel, with its full SHA256 recorded in `run_manifest.json`; isolated SM100 RMSNorm and rotary calls pass. The `0.3.8+cu128` candidate was rejected and rolled back because its extension had a Torch 2.7 ABI error.
- FlashAttention `2.8.3.post1` is from the official Dao-AILab release wheel for CPython 3.12 / Torch 2.7 / CUDA 12, with its SHA256 recorded in the run manifest.
- SGLang's optional `torch_memory_saver==0.0.8` is pinned, installed, and import/API checked. `uv pip check` passes for all 214 installed packages.
- `ninja==1.13.2` is installed in `.venv/bin`; the launcher now exposes that binary to child workers through `PATH`.
- GPU 1 and GPU 7 are excluded. The direct runner rejects those indices; the automatic runner checks only 0, 2, 3, 4, 5, and 6.
- The watcher checks only GPUs 0, 2, 3, 4, 5, and 6. Attempt 12's temporary 110,000 MiB gate was withdrawn after OOM. Attempts 13 and 14 showed that even a 170,000 MiB gate with three stable samples can race with later shared-GPU allocations; do not retry on the same memory plan without sufficient headroom.
- Model, dataset, Ray cache, checkpoints, logs, and outputs are on the project PFS workspace. The short `/tmp` Ray symlink points to the Ray cache on PFS to satisfy the Unix socket path limit.

## Data and model

- Official project data: `dataset/merged_train_dataset.jsonl` (14,256 rows) and `dataset/medqa_test_dataset.jsonl` (1,268 rows). Their SHA256 digests are in `experiments/ea_atpo_v1_smoke/run_manifest.json`.
- Smoke Parquet files contain two training examples and one validation example. Original JSONL files remain unchanged.
- Model: local Qwen3-8B snapshot under the prior EA_ATPO experiment workspace. Config, weights-index, tokenizer-config, and generation-config hashes are recorded in the manifest.

## Attempts and findings

1. Corrected an invalid Hydra override (`algorithm.use_kl_loss`) before worker startup; the official config places this setting under `actor_rollout_ref.actor.use_kl_loss`.
2. Routed Ray's Unix socket path through a short symlink to the PFS cache.
3. Installed the official FlashAttention wheel after the first worker reported the missing import.
4. On physical GPU 6, Actor FSDP initialization ran out of memory before rollout or optimizer step because a concurrent process occupied about 89.1 GiB.
5. On physical GPU 2, Actor/Critic initialized, but SGLang failed because the enabled optional `torch_memory_saver` dependency was absent. Pinned and installed version 0.0.8.
6. With the dependency present, SGLang loaded Qwen3-8B but the KV-cache profiler returned zero tokens at `gpu_memory_utilization=0.15` and `0.25`. Its code computes the cache budget from post-load available memory and a per-worker available-memory baseline captured before loading weights; it does not use total physical VRAM in that expression. The previous report's estimate based on total physical VRAM was incorrect and is withdrawn.
7. At `gpu_memory_utilization=0.75`, SGLang allocated 287,404 KV-cache tokens (K=19.73 GB, V=19.73 GB). Initialization then failed with `FlashAttention v3 Backend requires SM>=80 and SM<=90` on B200/SM100. FlashInfer is installed, and the retry explicitly sets `engine_kwargs.sglang.attention_backend=flashinfer`.
8. On physical GPU 2, FlashInfer initialized and allocated a 250,054-token KV cache (K=17.17 GB, V=17.17 GB). CUDA graph capture then failed with `FileNotFoundError: 'ninja'`: the binary existed at `.venv/bin/ninja`, but that directory was missing from `PATH` inherited by Ray workers. The launcher now prepends the project environment's `bin` directory.
9. The SGLang wrapper retained its default CUDA graph capture through batch size 512 despite `engine_kwargs.sglang.cuda_graph_max_bs=8`. Inspection of `codes/verl/workers/rollout/sglang_rollout/sglang_rollout.py` showed that this wrapper only consumes `attention_backend` from `engine_kwargs`; it passes `max_num_seqs` as SGLang's `max_running_requests`. The runner now sets `max_num_seqs=8` for the smoke.
10. After adding `.venv/bin` to `PATH`, FlashInfer's JIT compiled CUDA objects but the final host link failed with an undefined reference to `BatchDecodeWithPagedKVCacheKernel`. The next retry selects SGLang Triton attention to avoid the failing FlashInfer JIT path while preserving the model, rollout, reward, and PPO configuration.

All completed attempts stopped before the first training step. The original baseline must pass before any EA-ATPO training-path implementation begins.

## Artifacts

- Smoke data: `experiments/ea_atpo_v1_smoke/original_atpo/data/`
- Logs: `experiments/ea_atpo_v1_smoke/original_atpo/logs/`
- Run manifest: `experiments/ea_atpo_v1_smoke/run_manifest.json`
- Retry runner: `run_original_atpo_smoke_when_gpu_free.sh`

### Attempt 10: Triton attention reaches RMSNorm failure

10. On physical GPU 2 (NVIDIA B20Z, compute capability 10.0), SGLang loaded the model, allocated KV cache, and reduced CUDA graph capture sizes via `max_num_seqs=8`. Capture then failed in `sgl_kernel.rmsnorm` with `no kernel image is available for execution on the device`. Inspection of the installed `common_ops.abi3.so` confirmed its embedded cubins target SM75, SM80, SM89, SM90, and SM90a, not SM100. The run ended before PPO; no parameter updates occurred. The error, versions, GPU, and next action are recorded in the manifest.

### Attempt 11 preparation: native RMSNorm compatibility fallback

11. Confirmed GPU2 is NVIDIA B20Z / compute capability 10.0 and the installed PyTorch 2.7.1+cu128 reports `sm_100`. A small BF16 tensor passed through SGLang’s existing `RMSNorm.forward_native` and returned finite output. Added a Smoke-only `sitecustomize.py` dispatch shim, enabled by the Smoke launcher; no original ATPO source under `codes/` was changed. The retry runner found physical GPU 5 (166259 MiB free, 0% utilization) and started attempt 11 there. It checks only GPUs 0, 2, 3, 4, 5, and 6; GPUs 1 and 7 remain excluded. Attempt 11 reached SGLang model load and KV allocation but failed before PPO on the unsupported rotary kernel described below.

### Attempt 11: model loaded; SM100 rotary kernel missing

11. On allowed physical GPU 5, Actor and Critic weights loaded; SGLang loaded Qwen3-8B and allocated 61,969 KV tokens (K/V 4.26 GB each). CUDA graph capture failed in `BatchQKApplyRotaryPosIdsCosSinCache` with `no kernel image is available for execution on the device`. The native RMSNorm shim passed its isolated check and bypassed the earlier RMSNorm failure, but `sgl-kernel==0.2.8` also lacked the rotary kernel image. No PPO step or parameter update occurred.

### Kernel wheel compatibility check

The official `sgl-kernel 0.3.8+cu128` wheel passed its published SHA256 check and its `common_ops` fatbin included SM100 targets plus the missing rotary kernel. Importing it against the current `torch 2.7.1+cu128` failed with undefined symbol `_ZN3c104cuda9SetDeviceEab`. The package was rolled back from a PFS backup; the environment is back on `sgl-kernel==0.2.8`. No PPO step ran. The next candidate is an official CUDA 12.8 wheel from the Torch 2.7 compatibility window.

### Torch 2.7 compatible SM100 wheel and isolated op check

The official `sgl-kernel 0.2.9+cu128` wheel was downloaded with resumable byte ranges and verified against SHA256 `5f875f1d3b48be1f2d15b50ec2ddf4e8dfb9b839be50292717954057e80af1ec`. `cuobjdump` confirmed SM100/SM100a cubins in `common_ops.abi3.so`, and the binary contains the rotary kernel used by SGLang. Installation with `--no-deps` succeeded; importing it with the pinned Torch version succeeded. On allowed physical GPU 0 (SM100), BF16 RMSNorm (`[2,4096]`) and Qwen3-shaped rotary (`[3,4,128]`) both executed and returned finite outputs. Attempt 13 later validated full SGLang startup and CUDA graph capture with these kernels. The remaining blocker is shared-GPU memory contention before rollout.

### Attempt 12: insufficient shared-GPU headroom

Attempt 12 ran on allowed physical GPU 4 after a one-time 110,000 MiB free-memory override (preflight: 116,551 MiB free, 0% utilization). Data loading and ATPO config validation passed; the run began the two-step baseline setup. Critic checkpoint shards loaded, then FSDP parameter synchronization requested 384 MiB with 263 MiB free. The OOM report attributed 64.51, 69.26, and 18.90 GiB to unrelated GPU processes, which were left untouched. This run did not reach Actor or SGLang initialization and made no PPO or parameter update. The reduced threshold is withdrawn; future retries use the 150,000 MiB free / <=60% utilization gate and allowed-GPU list only.

### Attempt 13: SGLang startup succeeds; shared memory blocks rollout

The watcher started attempt 13 on allowed physical GPU 2 after observing 161,981 MiB free and 0% utilization. Critic and Actor initialized; SGLang loaded Qwen3-8B, allocated its KV cache, captured CUDA graphs for batch sizes 1, 2, 4, and 8, and entered the two-step training loop. On the first rollout wake-up, FSDP needed to reload weights to the GPU and requested 738 MiB with only 167 MiB free. The OOM report included a 50.25 GiB process and a 20.18 GiB process alongside this run's 107.63 GiB allocation. The job stopped before the first rollout and before any PPO or parameter update. No unrelated process was killed.

### Attempt 14: stable preflight still raced with shared allocations

Attempt 14 used the 170,000 MiB / 60% gate with three checks 15 seconds apart; the runner only considers GPUs 0, 2, 3, 4, 5, and 6. The selected physical index was not preserved in the per-run log. Critic/Actor initialization, Qwen3-8B SGLang loading, KV allocation, and CUDA graph capture succeeded. At the first rollout wake-up, FSDP requested 738 MiB with only 45.56 MiB free; this process used 97.54 GiB while four other processes used 17.38, 17.63, 22.51, and 23.11 GiB. No PPO update occurred. The 170,000 MiB gate did not provide enough protection from shared-GPU allocations that arrived after preflight.
