# VI-EA-ATPO Phase 2 model deployment plan

**Updated:** 2026-11-18  
**Deployment state:** Base and AWQ Expert are running on physical GPU 2.  
**Pilot state:** Infrastructure is available; formal Pilot is not ready because the Patient simulator missed a fact present in the frozen validation case.

## Pinned assets and runtime

| Role | Local model | Pinned revision | Verified files | Runtime settings |
|---|---|---|---|---|
| Base | `/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/experiments/real_case_pilot_20261103/assets/models/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` | 5 indexed shards, 16,381,516,776 bytes (15.26 GiB), all sizes match the cached Hugging Face revision tree | vLLM 0.33.0, BF16, TP=1, max length 4096, max sequences 1, memory fraction 0.12, `TRITON_ATTN` |
| Expert | `/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/experiments/real_case_pilot_20261103/assets/models/Qwen3-32B-AWQ` | `0499c3ac83fdef8810b907a23894ba91e95eddd8` | 4 indexed shards, 19,325,481,744 bytes (18.00 GiB); source lock and completed-shard record match all indexed files | vLLM 0.33.0, model-native `auto` dtype, AWQ, TP=1, max length 4096, max sequences 1, memory fraction 0.25, `TRITON_ATTN` |

The Expert `config.json` records AWQ, 4 bits, group size 128, GEMM version, and zero point enabled. These values are read from the checkpoint; the deployment does not invent quantization parameters. The 32B BF16 candidate is not used.

The verified runtime is `/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/.venv-real-pilot`: Python 3.12.3, vLLM 0.33.0, PyTorch 2.14.0, Transformers 5.20.2, aiohttp 3.14.5, and PyYAML 6.0.3.

## Roles, endpoints, and GPU mapping

| Role | Endpoint / alias | Process | Physical GPU |
|---|---|---:|---:|
| Base | `http://127.0.0.1:30481/v1` / `vi-qwen3-8b-base` | PID 1121380 | 2 |
| Patient | Shared Base endpoint / `vi-qwen3-8b-base` | Shared service; separate Patient prompt and request | 2 |
| Expert | `http://127.0.0.1:30482/v1` / `vi-qwen3-32b-awq-expert` | PID 1137181 | 2 |

Both vLLM processes use `CUDA_VISIBLE_DEVICES=2`; for each process CUDA logical device 0 maps to physical GPU 2 (NVIDIA B20Z, compute capability 10.0). TP=1 was sufficient on this 183,359 MiB device. The service logs report 21,590 MiB for the Base engine and 45,152 MiB for the AWQ Expert engine after load. After smoke inference, the GPU snapshot showed 67,854 MiB used and 114,771 MiB free; it also included an unrelated approximately 1.1 GiB process. No other user's process was signaled.

Patient is the original ATPO simulated role, not a third model: `ea_atpo_stage/recipe/ea_atpo/collect_counterfactuals.py::_patient_messages` sends only the selected source row's `atomic_facts` and the doctor's question to the shared Base service. Role prompts, request sessions, state, and logs are kept separate. The current prompt revision is recorded in the Phase 2 execution manifest.

The historical Attempt04 manifests confirm the same model revisions, vLLM version, BF16/AWQ choice, `max_model_len=4096`, `max_num_seqs=1`, memory fractions, attention backend, `temperature=0`, `top_p=1`, and `enable_thinking=false`. Historical startup logs also show the same vLLM warning that model `generation_config.json` supplies default `temperature=0.6`, `top_k=20`, and `top_p=0.95`; requests in both runs explicitly provide temperature 0 and top-p 1. The warning is retained as a service-level observation, not treated as a Phase 2-specific mismatch.

## GPU selection and legacy launcher findings

- GPU 2 passed the new pre-start gate with 182,621 MiB free, 0% utilization, and no foreign compute process. The two models were loaded sequentially there.
- The recorded earlier GPU 3 Expert attempt had 56,783 MiB free and 100% utilization, below its 81,920 MiB / 25% historical gate. In the current snapshot GPU 3 had 145,423 MiB free but remained at 100% utilization, so it still fails the utilization gate.
- In the current snapshot GPU 7 had 16,383 MiB free and 100% utilization; it fails both the 70,000 MiB AWQ Expert memory gate and the 30% utilization gate in `start_qwen32_awq_complementarity_service.sh`.
- `audit/start_audit_model_services.sh` is an obsolete BF16 launcher: its default Expert device is GPU 6, but its generated manifest hard-codes the Expert physical index as GPU 2. Do not use this launcher or its BF16 model for this experiment.
- GPU 3 and GPU 7 were not used. No foreign process was stopped to free them.

## Reproduction commands

Run from the `EA_ATPO_V1` repository root, using the pinned environment:

```bash
/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/.venv-real-pilot/bin/python experiments/vi_ea_atpo/phase2/manage_vi_services.py preflight --config experiments/vi_ea_atpo/configs/model_services.yaml
bash experiments/vi_ea_atpo/phase2/start_vi_services.sh
bash experiments/vi_ea_atpo/phase2/check_vi_services.sh
/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/.venv-real-pilot/bin/python experiments/vi_ea_atpo/phase2/run_information_pilot.py
```

The start script refuses to load new services unless the configured GPU identity, memory, utilization, process occupancy, model files, environment, and ports pass. It reuses a matching service only when both the served alias and its local revision manifest match. It writes separate Base/Expert logs and Patient role-call logs, verifies `/v1/models`, then makes a minimal inference request for all three roles. `stop_vi_services.sh` only signals process groups recorded as owned by this deployment and rechecks their process identity before signaling.

`model_services.yaml`, the three lifecycle wrappers, service manifests, and `services/state.json` are the reproducible deployment record. Service logs and manifests are under `experiments/vi_ea_atpo/phase2/services/`.

## Pilot gate

The deployment itself is healthy and remains available. The two-case execution exercised all required branches and cost accounting, but the Patient simulator returned UNKNOWN for a validation question whose source facts included a relevant trauma history. A prompt revision did not correct that case. The service is therefore **not yet approved for the 150-case Pilot**; see `SMOKE_TEST_REPORT.md`. No additional cases, full Pilot, PPO, or ATPO RL were run.
