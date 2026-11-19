# EA-ATPO V2 Real Service Smoke and Pilot Gate

**Date:** 2026-11-20 Asia/Shanghai  
**Scope:** GPU permission/resource recheck, real Base/Expert service startup, and the required V2 role smoke. No cache rows were reused. No model was trained.

## GPU permission and deployment choice

The V2 config still contains `resource_policy.gpu_1_and_7_prohibited: true` in `experiments/ea_atpo_v2/configs/phase1_counterfactual.json`. This is a project-level rule; the current shell has no `CUDA_VISIBLE_DEVICES`, `NVIDIA_VISIBLE_DEVICES`, or SLURM/PBS allocation variables, no `squeue`/`scontrol` scheduler client is installed, and `/dev/nvidia7` is accessible at the device-node level. These observations do not grant permission to override the project rule. GPU7 was not used.

The live snapshot showed GPU6 at 145,799 MiB free and 100% utilization, with another workspace's image-generation process using about 36.8 GiB. GPU7 had 145,337 MiB free and 100% utilization, with another workspace's process using about 37.3 GiB; it is also explicitly prohibited. Neither was used. GPU2 was the only eligible card: 182,625 MiB free, 0% utilization, and no compute PID. Its approved Phase 2 service config already assigns both Base and Expert to GPU2, requires at least 100,000 MiB free, at most 20% utilization, and no foreign compute process. The manager's read-only preflight passed with 182,621 MiB free and zero compute PIDs.

Other cards were not eligible substitutes at that snapshot: GPU0/1 had other users' processes, GPU3/4/5 were under active workloads, and the project service config selects GPU2. The paired service budget was 0.12 + 0.25 of one 183,359 MiB card (about 67.8 GiB reserved by the configured fractions), with an additional 24 GiB reserve. No other user's process was stopped or modified.

The two model directories, pinned revisions, tokenizers, and vLLM runtime were verified before startup. vLLM 0.33.0 and PyTorch 2.14.0 matched the service config. Base revision was `b968826d9c46dd6066d109eabc6255188de91218`; Expert revision was `0499c3ac83fdef8810b907a23894ba91e95eddd8`. Asset verification found five Base shards and four Expert shards, all present and size-consistent with their pinned indexes/locks.

## Service startup

Services were started on the permitted GPU2 using a run-specific copy of the existing service config. The original config and previous service logs were not overwritten. The run copy, service manifests, logs, runtime caches, and all experiment outputs are under:

`experiments/ea_atpo_v2/outputs/phase1_gpu2_service_20261119T165122Z/`

The established command path was `experiments/vi_ea_atpo/phase2/manage_vi_services.py start --config <run-dir>/model_services_gpu2.yaml`, invoked with the configured `.venv-real-pilot` Python and `NO_PROXY/no_proxy=127.0.0.1,localhost`. Runtime cache and temp targets were placed under the run directory. A short transient symlink in `/dev/shm` addressed the Unix-socket path length while pointing to the PFS temp directory; it was removed after shutdown. No model cache or log was written to `/tmp` or the system disk.

Both aliases became ready at the configured endpoints:

| Role | Alias | Endpoint | Result |
|---|---|---|---|
| Base | `vi-qwen3-8b-base` | `127.0.0.1:30481` | Ready; pinned revision matched |
| Expert | `vi-qwen3-32b-awq-expert` | `127.0.0.1:30482` | Ready; pinned revision matched |

The manager's generic API health check passed for Base, Patient-role, and Expert. These are service checks, separate from the V2 case smoke. It reported Base 5 completion tokens / 0.600 s, Patient 11 tokens / 0.075 s, and Expert 6 tokens / 0.097 s. GPU2 after service health checks showed 66,534 MiB used, 116,091 MiB free, and 0% utilization. The model loader reported 15.27 GiB for Base and 18.24 GiB for Expert; the aggregate GPU snapshot is the relevant shared-card allocation.

The Expert log contains non-fatal AWQ loader dtype conversion warnings (BF16 checkpoint tensors assigned to FP16 parameters); loading completed and the service health check passed. These warnings are retained in the raw log for follow-up.

## V2 real role smoke

The V2 runner preflight matched both configured model aliases and revisions, then exercised one frozen training state without scoring Gold. Five actual completion requests were sent: direct Base ANSWER; Base ASK; Patient reply; terminal Base answer after ASK; and Expert DEFER.

| Smoke path | Result | Tokens (prompt + completion) | Latency | Detail |
|---|---|---:|---:|---|
| Direct ANSWER | Invalid | 176 + 24 | 0.143 s | Output began with `Final Answer: None...`; `finish_reason=length`, so no A–D answer parsed |
| ASK question | Valid | 179 + 15 | 0.104 s | One specific question parsed |
| Patient reply | Non-empty, but unknown | 121 + 8 | 0.074 s | Simulator said it could not answer; no useful new fact was obtained |
| Base after ASK | Valid | 211 + 5 | 0.065 s | The ASK continuation completed and parsed as option D |
| DEFER | Invalid | 180 + 29 | 0.309 s | Expert returned a “none of the options” response without an A–D choice; `finish_reason=stop` |

The ASK path updated the internal public dialogue with the question and Patient reply before the terminal Base call. The Patient response was textually different from the prior state but explicitly unknown, so textual novelty is not counted as information gain. The V2 prompt-boundary checks kept Gold out of Base/Expert inputs and passed the hidden-fact checks; only the Patient role received its private fact prompt. No correctness reward was computed. The smoke manifest records `gold_read=false` for scoring; source answer metadata is only inspected by the existing static boundary audit, never sent to a model.

Overall smoke totals were 867 prompt tokens, 81 completion tokens, and 0.696 s summed model latency. All five API requests succeeded at the transport layer; two of the three terminal action answers failed the required parser. The smoke gate therefore **failed**. The Base failure is consistent with the 24-token terminal limit and non-compliance; the Expert failure is non-compliance even though generation stopped normally. No prompt, parser, or decoding changes were made after seeing these outputs.

## Pilot gate and cleanup

The 10-case / 29-state Pilot was **not started** because the required real-service smoke failed. This preserves the requested smoke-before-pilot gate. No pilot trajectory, Gold correctness label, action return, cost curve, or empirical Q target was created. Coverage remains **0/87** potential state-action Q labels. The failed smoke files are retained under `outputs/phase1_gpu2_service_20261119T165122Z/v2_role_smoke/`.

Only the two vLLM process groups recorded as owned by this run were stopped. Ports 30481/30482 are now closed, GPU2 returned to 0 MiB used by this run, and the transient `/dev/shm` symlink was removed. GPU6/7 processes remain untouched. No Q Regression, Actor distillation, PPO, or Expert SFT was started.

## Next gate

The GPU permission issue is resolved for this attempt by using the already approved GPU2 configuration; GPU7 remains prohibited and was not bypassed. The remaining gate is answer-format behavior on the V2 smoke. The current evidence does not justify starting the 10-case Pilot. Any change to output contract, parser, or generation limits requires a separately recorded protocol decision and a fresh smoke before the paired rollout.
