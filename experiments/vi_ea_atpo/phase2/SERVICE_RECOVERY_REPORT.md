# Phase 2 service recovery report

**Date:** 2026-11-18  
**Status:** Base and Expert services recovered and verified; Phase 2 formal Pilot remains blocked by a Patient simulator response-quality failure.  
**Git:** branch `ea-atpo-v1`, HEAD `2aebfa8ce33db872b9b4c204377a4a11cfb09c97`. Existing uncommitted and untracked work was preserved.

## Models and deployment

| Role | Verified checkpoint | Endpoint | Status |
|---|---|---|---|
| Base | Qwen3-8B, revision `b968826d9c46dd6066d109eabc6255188de91218`, 5 shards / 15.26 GiB | `127.0.0.1:30481`, alias `vi-qwen3-8b-base` | Running, PID 1121380 |
| Patient | ATPO simulator using Base model with a separate role prompt and `atomic_facts` request | Shared Base endpoint | Running as a role; no third weight set or process |
| Expert | Qwen3-32B-AWQ, revision `0499c3ac83fdef8810b907a23894ba91e95eddd8`, 4 shards / 18.00 GiB, AWQ 4-bit / group 128 / GEMM / zero-point | `127.0.0.1:30482`, alias `vi-qwen3-32b-awq-expert` | Running, PID 1137181 |

Both services run under the verified `.venv-real-pilot` environment on physical GPU 2 (NVIDIA B20Z, 183,359 MiB), with `CUDA_VISIBLE_DEVICES=2`, TP=1, vLLM 0.33.0, and PyTorch 2.14.0. Actual engine allocations were 21,590 MiB (Base) and 45,152 MiB (Expert). Both `/v1/models` aliases, checkpoint paths, revisions, runtime versions, and manifests matched. Minimal inference succeeded for Base, Patient, and Expert; output format parsing returned valid A–D values for Base and Expert.

At the final resource snapshot GPU 2 showed 67,854 MiB used, 114,771 MiB free, 1% utilization. This includes an unrelated approximately 1.1 GiB process that appeared after service startup; it was not interrupted. GPU 3 currently has 145,423 MiB free but 100% utilization. GPU 7 has 16,383 MiB free and 100% utilization. The prior recorded GPU 3 attempt had 56,783 MiB free and 100% utilization. These GPUs were not used for this deployment.

## Recovery sequence and fixes

1. Verified local model files, pinned revisions, shard indices and sizes, Base revision-tree evidence, AWQ source-lock and all-shards-verified evidence, tokenizer templates, Python environment, vLLM binary, CUDA mapping, and GPU 2 pre-start gate. No BF16 32B checkpoint was used.
2. The first Base vLLM launch exited before loading weights because its IPC socket path exceeded Linux's 107-character ZMQ limit. The manager now assigns a short role-specific `TMPDIR` under `/tmp/vi-ea-atpo-<uid>/`; the second launch loaded the checkpoint successfully.
3. An initial health-check implementation created `aiohttp.ClientSession` outside a running event loop. Client construction was moved into the async health-check coroutine.
4. The next health check exposed a missing success-path return from that coroutine. The manager now returns the complete role record after closing sessions; the follow-up health check passed.
5. The original GPU2 service pair remained under this deployment's PID/process-group records throughout recovery. No other user's process was stopped.
6. Lifecycle review found that a second `start` invocation could mark a matching service as reused but drop this manager's ownership flag. The manager now retains ownership only when the local service manifest and live PID identity prove that this deployment created the service; a foreign matching service remains unowned. A repeat `start` was not run against live services to avoid unnecessary inference.

The Expert log contains expected, non-fatal BF16-to-FP16 load conversion warnings from the AWQ path. Both model servers reached ready state and served valid requests. The Qwen model generation-config warning was also present in the historical successful Attempt04 service logs; the Phase 2 requests explicitly set temperature 0 and top-p 1, matching the recorded generation setup.

## Commands actually run

```bash
/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/.venv-real-pilot/bin/python experiments/vi_ea_atpo/phase2/manage_vi_services.py preflight --config experiments/vi_ea_atpo/configs/model_services.yaml
bash experiments/vi_ea_atpo/phase2/start_vi_services.sh
# Ran once during recovery (first exposed the async return bug), then repeated after the fix.
bash experiments/vi_ea_atpo/phase2/check_vi_services.sh
/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/.venv-real-pilot/bin/python experiments/vi_ea_atpo/phase2/run_information_pilot.py
/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/.venv-real-pilot/bin/python experiments/vi_ea_atpo/phase2/run_information_pilot.py --smoke-cases 2
/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/.venv-real-pilot/bin/python experiments/vi_ea_atpo/phase2/run_information_pilot.py --smoke-cases 2 --output-dir experiments/vi_ea_atpo/phase2/smoke_recovery_v2
/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/.venv-real-pilot/bin/python experiments/vi_ea_atpo/phase2/test_information_pilot.py
/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/.venv-real-pilot/bin/python -m py_compile experiments/vi_ea_atpo/phase2/manage_vi_services.py experiments/vi_ea_atpo/phase2/run_information_pilot.py ea_atpo_stage/recipe/ea_atpo/collect_counterfactuals.py
```

The first smoke run and the post-prompt-change recheck used the same two frozen case IDs (one train and one validation), not four distinct cases. The first rollout remains in `phase2/intervention_rollouts.jsonl`; the second remains in `phase2/smoke_recovery_v2/intervention_rollouts.jsonl`. Each run wrote 12 branch rows, with 21 successful API completion events reconstructed from the unique trace events. No test split was used.

## Artifacts and current action

- Deployment config: `experiments/vi_ea_atpo/configs/model_services.yaml`
- Lifecycle scripts: `phase2/start_vi_services.sh`, `phase2/check_vi_services.sh`, `phase2/stop_vi_services.sh`
- Service identity and process state: `phase2/services/{base,expert}/service_manifest.json`, `phase2/services/state.json`
- Health requests: `phase2/services/health_check.json`, with separate Base, Patient, and Expert log records
- First startup and recovery logs: `phase2/services/logs/base_vllm.log`, `expert_vllm.log`, `patient_role.log`
- Smoke branch results and decision: `phase2/SMOKE_TEST_REPORT.md`
- Patient factual-response acceptance failure records: `phase2/smoke_quality_failure.log`

**Services are healthy and left running.** The 150-case Pilot, Phase 3/4, and all RL training remain unrun. Pilot readiness is currently **No** until the Patient simulator can return supported facts reliably under the frozen role policy and that behavior is revalidated without expanding the cohort.
