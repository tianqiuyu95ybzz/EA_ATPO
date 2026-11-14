# EA-ATPO V1 Smoke Test Status

**Status: the original ATPO baseline, deterministic four-action protocol smoke, and EA real-model engineering smoke have completed. EA Attempt 07 ran two PPO updates on GPUs 4 and 6, saved complete step-1 and step-2 Actor/Critic checkpoints, and exited successfully without OOM. The model generated ASK, CONSULT, and ANSWER; it did not sample DEFER. Two of four rollouts ended after exhausting the ASK budget. Attempt 06 is retained unchanged. These runs validate execution only, not model quality.** The original Qwen3-8B baseline completed two PPO updates across GPUs 4 and 6. The pinned ATPO source remains unchanged, and GPUs 1 and 7 remain excluded. See Attempt 20 for baseline logs and Attempt 07 for the latest EA reports and raw logs.

## Source and environment

- Branch: `ea-atpo-v1`; official ATPO base commit: `25af83688c06917f94245a4db06be7d0924c090c`.
- The read-only original ATPO checkout is `ATPO-official/` at commit `25af83688c06917f94245a4db06be7d0924c090c`; the smoke launcher uses `ATPO-official/codes/` and `ATPO-official/dataset/`. No original source files are edited.
- Python 3.12.3; PyTorch `2.7.1+cu128`; CUDA runtime 12.8; Transformers 4.54.1; Ray 2.47.1; SGLang 0.4.10.post2.
- `sgl-kernel==0.2.9+cu128` is the official CUDA 12.8 wheel, with its full SHA256 recorded in `run_manifest.json`; isolated SM100 RMSNorm and rotary calls pass. The `0.3.8+cu128` candidate was rejected and rolled back because its extension had a Torch 2.7 ABI error.
- FlashAttention `2.8.3.post1` is from the official Dao-AILab release wheel for CPython 3.12 / Torch 2.7 / CUDA 12, with its SHA256 recorded in the run manifest.
- SGLang's optional `torch_memory_saver==0.0.8` is pinned, installed, and import/API checked. `uv pip check` passes for all 214 installed packages.
- `ninja==1.13.2` is installed in `.venv/bin`; the launcher now exposes that binary to child workers through `PATH`.
- GPU 1 and GPU 7 are excluded. The direct runner rejects those indices; the automatic runner checks only 0, 2, 3, 4, 5, and 6.
- The watcher checks only GPUs 0, 2, 3, 4, 5, and 6. Attempt 12's temporary 110,000 MiB gate was withdrawn after OOM. Attempts 13 and 14 showed that even a 170,000 MiB gate with three stable samples can race with later shared-GPU allocations. The next retry lowers SGLang's GPU memory fraction from 0.75 to 0.60, limits concurrent requests to the two smoke trajectories, and requires at least 155,000 MiB free with utilization at most 15% across five 15-second checks. This is a resource adjustment only; Critic CPU parameter offload is not enabled because the official worker code explicitly disables it to avoid incorrect results with gradient accumulation.
- Model, dataset, Ray cache, checkpoints, logs, and outputs are on the project PFS workspace. The short `/tmp` Ray symlink points to the Ray cache on PFS to satisfy the Unix socket path limit.

## Data and model

- Official project data: `ATPO-official/dataset/merged_train_dataset.jsonl` (14,256 rows) and `ATPO-official/dataset/medqa_test_dataset.jsonl` (1,268 rows). Their SHA256 digests are in `experiments/ea_atpo_v1_smoke/run_manifest.json`.
- Smoke Parquet files contain two training examples and one validation example. Original JSONL files remain unchanged.
- Model: local Qwen3-8B snapshot under the prior EA_ATPO experiment workspace. Config, weights-index, tokenizer-config, and generation-config hashes are recorded in the manifest.

## Next retry plan

The runner now reads the pinned source and JSONL data from the ignored, read-only `ATPO-official/` checkout. A 900-second single-GPU watcher ended at 2026-11-11 10:36 UTC without an eligible GPU, so attempt 15 did not start; its output and end snapshot are in `experiments/ea_atpo_v1_smoke/gpu_wait_attempt15_20261111T1021Z.log`. The two-GPU layout uses Actor/Critic FSDP world size 2 and SGLang tensor parallel size 2. This mapping is supported by the official code: `mt_main.py` sizes the Ray resource pool from `trainer.n_gpus_per_node`; `sglang_rollout.py` forms a `(dp, tp, pp)` device mesh; and `async_sglang_server.py` maps worker ranks into the TP group. Each selected GPU must maintain at least 100,000 MiB free for five consecutive 15-second checks. Utilization is logged but is not a launch gate for the requested shared-GPU attempt. Attempt 17 loaded the model and captured graphs at SGLang `gpu_memory_utilization=0.60`, but FSDP needed 1.16 GiB to materialize the rollout `state_dict` while only 1.02 GiB was free. The next retry lowers the fraction to `0.35`, keeps `max_num_seqs=2`, and enables `multi_stage_wake_up=True` so KV cache is restored after FSDP materializes and syncs weights. GPU 1 and 7 are excluded. Critic CPU offload remains disabled because the official FSDP worker documents incorrect results with gradient accumulation when enabled. GPUs 4 and 6 briefly showed 0% utilization with about 112,000 MiB free, but their utilization fluctuates; the pair watcher logs memory and utilization and only launches after the per-card memory gate remains stable.

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

At this point in the chronological log, early EA attempts had not reached a training step. The later baseline and EA training outcomes are recorded in the sections below.

## Artifacts

- Smoke data: `experiments/ea_atpo_v1_smoke/original_atpo/data/`
- Logs: `experiments/ea_atpo_v1_smoke/original_atpo/logs/`
- Run manifest: `experiments/ea_atpo_v1_smoke/run_manifest.json`
- Retry runner: `run_original_atpo_smoke_when_gpu_free.sh`
- Two-GPU retry runner: `run_original_atpo_smoke_pair_when_gpu_free.sh`

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

### Attempt 15 resource gate: not launched

The revised watcher checked only GPUs 0, 2, 3, 4, 5, and 6 for 900 seconds, requiring at least 155,000 MiB free and at most 15% utilization across five consecutive samples. No GPU met the gate. The watcher exited with code 3 before starting the smoke; no training or GPU allocation occurred. The end snapshot showed the best allowed card at 96,595 MiB free, below the gate. See `experiments/ea_atpo_v1_smoke/gpu_wait_attempt15_20261111T1021Z.log`.

### Attempt 16 strict pair wait: stopped before launch

The first GPU 4+6 watcher also required utilization at most 15%. It repeatedly lost that gate while memory stayed near 112,000 MiB per GPU, so it was stopped before launching. Its output is in `experiments/ea_atpo_v1_smoke/gpu_pair_wait_20261113T170714Z.log`. The next attempt records utilization but gates on per-card free memory only.

### Attempt 17: two-GPU initialization succeeds; first weight sync OOMs

The run used physical GPUs 4 and 6 with Ray/FSDP world size 2, SGLang TP size 2, `gpu_memory_utilization=0.60`, and `max_num_seqs=2`. Both training models loaded. SGLang loaded Qwen3-8B across TP0/TP1, allocated 683,410 KV tokens, and captured CUDA graphs for batch sizes 1 and 2, leaving 36.4 GB reported available before training. On the first rollout `wake_up`, FSDP tried to materialize the sharded Actor state dict for SGLang synchronization and requested 1.16 GiB with 1.02 GiB free. The failing device reported 68.86 GiB held by another process, 51.23 GiB by this training worker, and 57.15 GiB by the SGLang process. The run stopped before generating a rollout and made zero PPO updates. Full log: `experiments/ea_atpo_v1_smoke/original_atpo/logs/original_atpo_smoke_20261113T171537Z.log`.

### Attempt 18: dual-GPU weight sync passes; missing dataset field stops AgentLoop

Attempt 18 used physical GPUs 4 and 6, Ray/FSDP world size 2, SGLang TP size 2, `gpu_memory_utilization=0.35`, and `multi_stage_wake_up=True`. Critic and Actor loaded, SGLang initialized, and the first rollout reached `UserAssistantAgentLoop.run` without the prior weight-sync OOM. It then failed before model generation because the cached smoke Parquet had been created without `api_key`; upstream `mix_data_jsonl.py` only adds that field when `API_KEY` is set. No rollout or PPO update occurred. The smoke runner now checks this schema field and regenerates only the derived Parquet with a non-secret local placeholder. This baseline sets `max_assistant_turns=1`, so it terminates after the Actor's first answer and does not call ATPO's external Patient API. The official README says that API endpoint was anonymized and must be deployed separately; actual multi-turn Patient-service testing remains a separate step. Full log: `experiments/ea_atpo_v1_smoke/original_atpo/logs/original_atpo_smoke_20261113T172831Z.log`; pair wait log: `experiments/ea_atpo_v1_smoke/gpu_pair_wait_20261113T172729Z.log`.

### Attempt 19: rollout succeeds; Actor Adam state hits shared-GPU OOM

The schema fix worked: train/test Parquet included `api_key` with a local non-secret placeholder. On physical GPUs 4 and 6, FSDP world size 2 and SGLang TP size 2 initialized; the Actor generated two candidates and tree search returned two final trajectories. ATPO's trainer then completed the Critic update and entered the Actor update. During Adam's first `exp_avg_sq` allocation, it requested 370 MiB while the failing rank reported only 272 MiB free. An unrelated process occupied about 69.5 GiB and the SGLang process 3.34 GiB. The Actor optimizer step failed, so the global PPO step did not complete and no checkpoint was saved; the in-memory Critic update was not retained. No unrelated process was stopped. Full run log: `experiments/ea_atpo_v1_smoke/original_atpo_attempt19/logs/original_atpo_smoke_20261113T174217Z.log`; wait log: `experiments/ea_atpo_v1_smoke/gpu_pair_wait_20261113T174112Z.log`.

### Attempt 20 configuration: offload Critic state before Actor update

The run kept the two-GPU model split and all algorithm/reward settings unchanged. It set `critic.model.fsdp_config.param_offload=True` and `critic.model.fsdp_config.optimizer_offload=True`; the official FSDP worker manually moves the Critic model and optimizer to CPU after its update. This targets the previous peak because the trainer calls `update_critic` before `update_actor`. The Actor already uses its existing manual parameter and optimizer offload path. The run used the same 100,000 MiB per-GPU preflight gate on GPUs 4 and 6, utilization recorded but ungated, and GPUs 1 and 7 excluded.

### Attempt 20: two-GPU ATPO training smoke passes

On physical GPUs 4 and 6, the runner used Ray/FSDP world size 2 and SGLang TP size 2. At preflight, the cards had 111,715 MiB and 111,387 MiB free. The model and KV cache initialized; SGLang allocated 421,305 KV tokens (14.46 GB each for K and V per TP log). With Critic parameter and optimizer CPU offload enabled, both Critic updates completed before their corresponding Actor updates. Tree search produced two candidates for each of two training samples, and the job exited 0 at `Training Progress 2/2`.

Both steps recorded nonzero gradients and both Actor and Critic model files changed between `global_step_1` and `global_step_2` on both FSDP ranks. Step 1 reported Actor/Critic grad norms 30.88/1,409.06; step 2 reported 37.39/1,868.29. Both checkpoints include Actor and Critic model and optimizer states. Checkpoint root: `experiments/ea_atpo_v1_smoke/original_atpo_attempt20/checkpoints/` (353 GB on PFS); latest checkpoint index is 2. The raw model checkpoints remain in the workspace because of their size; the run log and manifest are published.

This is an engineering smoke only. All four rollouts hit the 256-token response limit and had `custom/valid_rate=0` and `custom/num_correct=0`; the observed losses and gradients do not support an accuracy claim. The run used `max_assistant_turns=1`, so it did not call the separately deployed Patient API. Four-actions, ASK/CONSULT/DEFER transitions, and EA rewards have not yet been exercised. Full run log: `experiments/ea_atpo_v1_smoke/original_atpo_attempt20/logs/original_atpo_smoke_20261113T175436Z.log`; wait log: `experiments/ea_atpo_v1_smoke/gpu_pair_wait_20261113T175330Z.log`.

## GitHub publication

The smoke setup, run manifest, and attempt logs were published to `https://github.com/tianqiuyu95ybzz/EA_ATPO/tree/ea-atpo-v1`. The branch is based on the target repository's `main` commit `ad9d0e31660b973a3e5ddfcadce6701f767e0f62`; published commit `ed4883b736b9f82dff9d5357adb6e347599983e1` was confirmed through both `git ls-remote` and the GitHub Branch API. The target repository's `main` branch and the original ATPO `codes/` source were not modified.

## EA-ATPO V1 protocol smoke

The deterministic loop smoke ran through the EA tree expansion, reward assignment, and ATPO output converter using the local Qwen3 tokenizer plus a stand-in generation client. It exercised ANSWER, ASK, CONSULT, and DEFER; ASK and CONSULT returned to the same Actor; DEFER ended with an Expert final answer; and role prompts excluded hidden facts and gold. It also checked that every Actor-generated response token (including action tags) has response mask 1, Patient/Expert observation tokens have mask 0, and action costs are charged once. The unit tests pass, but this is protocol evidence only, not model-performance evidence.

The real training launcher is `run_ea_atpo_v1_smoke.sh`. It uses the official two-row smoke subset derived from the unchanged ATPO train JSONL, Qwen3-8B Actor/Critic, and physical GPUs 4 and 6. Its run manifest hashes the source data, derived Parquet, and model configuration. The run will be reported separately after it exits; only then will model-generated action usage and Actor/Critic parameter updates be assessed.

## Real EA training attempt 1

Attempt 1 initialized the Qwen3-8B Actor/Critic and the SGLang rollout service across physical GPUs 4 and 6, then stopped before the first generated token. The EA loop had omitted `correctness_reward` from the search-runner configuration, causing a `KeyError`; there was no OOM, rollout, optimizer update, or model-quality result. The failure, GPU snapshot, hashes, and raw log are preserved in `experiments/ea_atpo_v1_smoke/ea_atpo_v1_attempt01/`. The missing field is now passed explicitly and all 21 EA protocol/data-collection tests pass. A clean retry uses a separate attempt directory so the failed run remains reproducible.

## Real EA training attempt 2

The two-GPU run initialized Qwen3-8B Actor/Critic and the SGLang TP=2 service without OOM, generated two real model rollouts, and completed one PPO update. Actor and Critic model/optimizer files were saved at `global_step_1`. Both episodes answered after ASK and CONSULT; 0/2 were correct, which is only an engineering smoke observation. The process then stopped during post-update metrics because the EA reward adapter had not populated legacy `is_valid` and related keys required by ATPO's unchanged metric function. The adapter now derives answer validity/correctness from completed EA episode traces and supplies the inapplicable legacy interaction counters as zero. The manifest, trace, rollout, raw log, and partial checkpoint remain under `experiments/ea_atpo_v1_smoke/ea_atpo_v1_attempt02/`.


## Real EA training attempt 3

Attempt 3 reproduced the metric failure after one PPO update. The second adapter attempt had added fields inside the reward manager, but ATPO removes per-sample `statistics` before that manager is called. The run again initialized the two-GPU models, generated two episodes, and saved Actor/Critic checkpoint state without OOM. The fix now attaches the post-rollout metric fields at the EA-only `_postprocess` boundary before ATPO discards statistics. Attempt 3 artifacts remain under `experiments/ea_atpo_v1_smoke/ea_atpo_v1_attempt03/`.


## Real EA training attempt 4

The third metric-bridge attempt was installed only in the training driver, while rollout uses a separate Ray AgentLoopWorker process, so the same post-update `is_valid` error remained. The run again loaded the two-GPU models, generated two episodes, and wrote Actor/Critic model and optimizer state after one update without OOM. The hook now installs when the EA loop module is imported inside the worker. Attempt 4 is preserved under `experiments/ea_atpo_v1_smoke/ea_atpo_v1_attempt04/`.


## Real EA training attempt 5

Attempt 5 reproduced the missing `is_valid` field after one PPO update. The bridge was still not on the actual metric path, so it is removed. Attempt 6 instead computes answer validity/correctness in the EA Reward Manager from the completed response and `reward_model.ground_truth`, then returns the exact fields required by the legacy logger. This is post-rollout evaluation metadata and does not change EA reward, Critic returns, Actor advantages, or model-visible context. Attempt 5 artifacts remain under `experiments/ea_atpo_v1_smoke/ea_atpo_v1_attempt05/`.

## Real EA training attempt 6

Attempt 6 completed two PPO updates with Qwen3-8B Actor/Critic FSDP across physical GPUs 4 and 6 and SGLang tensor parallel size 2. At launch, GPU 4 had 111,699 MiB free and GPU 6 had 111,311 MiB free (about 109 GiB per GPU). Peak trainer-reported allocated/reserved memory was 76.30/88.20 GiB, and no OOM occurred. Actor and Critic gradient norms were nonzero on both updates. Complete step-1 and step-2 checkpoints, including model and optimizer shards, were saved locally on PFS.

Across four generated rollouts (two training rows, two trajectories each), the model emitted ASK 6 times, CONSULT 2 times, and ANSWER 2 times; it did not emit DEFER. Two trajectories reached an answer and both were incorrect; two terminated with `ask_budget_exhausted` and no final answer. Step metrics were 2/2 valid and 0/2 correct on step 1, then 0/2 valid and 0/2 correct on step 2. These four rollouts are far too few for any accuracy or learning claim. The deterministic protocol smoke still provides the only DEFER execution coverage.

The launcher exited with code 0 and reported 2/2 training steps. Ray printed a worker-shutdown warning after the progress bar reached 100%; both checkpoint directories and all step metrics were already written. Full evidence and limitations are in `experiments/ea_atpo_v1_smoke/ea_atpo_v1_attempt06/RESULTS.md`; the 353 GiB checkpoint files remain on PFS and are not committed to GitHub.

## Validated EA smoke attempt 7

Attempt 07 first passed three deterministic tests, including four independent forced-action states. Those fixtures were excluded from PPO. It then passed a five-sample memory gate of 105,000 MiB per GPU and completed two real PPO updates on GPUs 4 and 6. Actor/Critic gradient norms were nonzero on both steps and complete checkpoints were saved. Normal policy rollouts again sampled ASK, CONSULT, and ANSWER but not DEFER; step 2 had zero valid episodes after repeated ASK exhausted its budget. The original ATPO train/test Parquet hashes match exactly. EA keeps Attempt 06's 512-token response cap; the original baseline cap was 256, while observed EA response lengths were 212.5 and 117 tokens with no clipping, so no matched-cap accuracy comparison is claimed.

Reports: `EA_ATPO_FOUR_ACTION_PROTOCOL_REPORT.md`, `EA_ATPO_REWARD_RETURN_ADVANTAGE_AUDIT.md`, `EA_ATPO_PPO_GRADIENT_COVERAGE.md`, and `EA_ATPO_V1_JOINT_TRAINING_SMOKE.md`. Configuration, CSV coverage, tests, GPU gate, traces, rollout dumps, logs, manifest, and local-only checkpoints are under `experiments/ea_atpo_v1_smoke/ea_atpo_v1_attempt07/`.
