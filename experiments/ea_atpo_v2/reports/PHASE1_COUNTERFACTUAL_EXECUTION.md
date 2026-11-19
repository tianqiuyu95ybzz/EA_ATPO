# EA-ATPO V2 Phase 1: Data Audit, Protocol, and Counterfactual Pilot Status

**Execution date:** 2026-11-19
**Repository starting state:** branch `ea-atpo-v1`, commit `62be94d`
**Final status:** protocol and Q-data interface implemented; real model rollout blocked at service preflight.
**Scope:** no SFT, Router, Q network, Critic, PPO, GRPO, GAE, online value update, tree search run, or Holdout use.

## Summary

The V2 implementation is isolated under `experiments/ea_atpo_v2/`. It reuses the frozen 70/15/15 case split and existing public-state snapshots, and defines only ASK / ANSWER / DEFER. The protocol forces one deterministic continuation after ASK: the Patient simulator answers from hidden atomic facts, then the same Qwen3-8B Base service gives a terminal answer. ANSWER and DEFER terminate immediately. DEFER calls the Qwen3-32B-AWQ Expert only with the visible state.

Ten frozen training cases contain **29 public states** in the deterministic Phase 1 sample. The data and role-boundary preflight passed, including **116 static Base/Expert/Patient prompt-boundary checks**. Initial endpoint probes showed HTTP 503, but follow-up identified those responses as coming from the local HTTP proxy, not the model services. Direct loopback checks show the configured services are not listening. The `--execute` gate stopped before any completion request; **zero new model rollouts were produced**. No model process was started, and no existing process was stopped or modified.

The existing 3,492-rollout V1 file is not treated as V2 supervision: it contains the old four-action protocol, including CONSULT, and its ASK continuation semantics do not match this frozen V2 policy. Those results remain intact and are used only as historical context.

## Data and split audit

| Data | Audited count | Use in this phase |
|---|---:|---|
| Official ATPO `merged_train_dataset.jsonl` | 14,256 rows | Indexed source prompts, Patient atomic facts, and post-rollout answer key. This is single-turn source data. |
| Frozen public states from the simulated Pilot | 291 states / 100 cases | Starting states only; 205 states from 70 train cases are eligible. |
| SFT samples already present | 291 state samples: 205 train / 42 validation / 44 internal-test | Audited as existing state-level SFT material, not used as V2 rollout labels. The samples are not 291 independent cases. |
| Frozen case split | 70 train / 15 validation / 15 internal-test cases | Case-level split; Phase 1 selects 10 cases from train only. Validation and internal test are not used. |
| V1 action trajectories | 3,492 rollouts, historically 3,465 valid | Read only for protocol provenance; excluded from V2 Q targets. |
| Frozen 400-case Holdout | 400 cases | Not opened, materialized, or used. |

For the selected 10 cases, the script verified that each public-state `case_id` maps to the indexed raw source row and that each frozen initial prompt exactly matches that source prompt. The selected rows cover 29 states from 10 distinct train cases. Each selected case has two or three states. The preflight checked that case splits do not overlap and that no validation or internal-test case enters the selected set.

The raw answer key is read for correctness only after all three branch calls for a state finish. It is not passed to Base, Patient, Expert, or serialized into the V2 trajectory output. In the selected source rows, the Gold answer field has the expected single-letter A–D format; this is a format audit, not model evaluation.

## Reused and newly implemented modules

| Area | Status |
|---|---|
| Original ATPO `TreeSearchManager` and `TreeSearchRunner` | Audited and preserved. The manager holds active/completed trajectories and advances them breadth-first; the runner expands Actor candidates, obtains environment responses, evaluates nodes, prunes, backs up values, and computes returns/advantages. V2 Phase 1 does not invoke this training/search path. |
| V1 EA `PublicState`, role prompts, hidden-fact checks, response parsing | Audited and reused as design reference only. V1 remains four-action; its runner was not imported as the V2 policy engine. |
| Existing Base/Patient/Expert API convention | Reused through the pinned Phase 2 service configuration: Base and simulated Patient share the Qwen3-8B service under different role prompts; Expert uses Qwen3-32B-AWQ. |
| New V2 protocol | Implemented in `src/v2_protocol.py`, with only ASK / ANSWER / DEFER, max-three ASK legality, strict action parsing, public-state Expert input, and the frozen post-ASK terminal answer rule. |
| New paired rollout runner | Implemented in `scripts/run_counterfactual_pilot.py`; independent branches start from the same saved public state. It does not call the Expert except for DEFER. It writes raw run output only under ignored project-local `outputs/`. |
| Q Regression data API | Implemented in `src/q_dataset.py` and `scripts/build_q_dataset.py`. Targets are means of valid paired net returns; masks remain false and values remain null where data are absent. Costs are already included in targets. |
| V1 collector edit | Preserved untouched. No V1 or original ATPO file was modified for V2. |

## Protocol and value definition

The protocol is frozen in `configs/phase1_counterfactual.json`:

| Action | Calls | Terminal rule | Registered cost |
|---|---|---|---:|
| ANSWER | Base final answer | Terminates | 0.00 |
| ASK | Base generates one question → Patient replies → Base final answer | Terminates after one fixed continuation; no repeated ASK or DEFER in this branch | 0.05 |
| DEFER | Expert final answer | Terminates | 0.60 |

Base prompts replace the legacy V1 four-action system instruction with the V2 action contract while preserving the original public case prompt and dialogue. Net return is `final_accuracy - 0.05 * ask_count - 0.60 * defer_count`. Accuracy, each cost component, token counts, latency, and role call counts are logged separately. The selected costs carry forward the prior EA utility units; they were frozen before any V2 outcome and were not tuned on this sample. Tokens and latency are descriptive resource measures, not a second hidden cost deduction.

For a single repetition, the action record uses `sample_count=1` and `return_std=null`. An invalid format, timeout, or role failure yields `valid=false`, a failure reason, and a null return. No expected return is imputed. ASK checks record whether the Patient reply is non-empty, whether its exact normalized text is novel relative to the public state, and whether it is an explicit UNKNOWN response. Those operational checks do not claim semantic information gain.

## Leakage and action tests

The preflight created the Base ANSWER, Base ASK, Patient, and Expert DEFER prompt views for every selected state without requesting model completions. All **116** structural checks passed:

- Base and Expert views were built only from the frozen public state; privileged field names and exact unrevealed atomic-fact values were rejected.
- Patient views were built separately from atomic facts and a question. Gold was not included.
- The answer key and hidden facts are never arguments to Base/Expert prompt builders.
- ASK after the maximum count is masked before a model request.
- No CONSULT action exists in the V2 enum or configuration.
- The Expert route is only entered by the DEFER branch.

Eleven CPU-only engineering tests passed, covering action legality, terminal behavior, strict parsing, legacy action-prompt removal, role separation, one-time cost application, Q-target masking, case split enforcement, mixed-policy rejection, and loopback proxy bypass. These are protocol tests; they are not model outputs or evidence of clinical performance.

## Actual service and resource checks

Read-only endpoint checks were made against the configured services:

| Role | Configured alias | Direct result |
|---|---|---|
| Base | `vi-qwen3-8b-base` at port 30481 | Connection refused; no listener |
| Patient | shared Base endpoint | Connection refused; no listener |
| Expert | `vi-qwen3-32b-awq-expert` at port 30482 | Connection refused; no listener |

### 503 source and parallel-service startup failure

The shell has `HTTP_PROXY`, `HTTPS_PROXY`, and `ALL_PROXY` pointed at `127.0.0.1:18118`, without localhost excluded. Requests through that proxy returned an empty-body HTTP 503 for ports 30481/30482. With proxying explicitly bypassed, both ports return `ConnectionRefusedError [Errno 111]`; `ss` shows no service listener. The V2 runner used Python's environment-aware `urllib` before this diagnosis, so it could mistake the proxy's 503 for an origin-service response. It now uses an explicit empty `ProxyHandler` for loopback model requests and records direct connection errors. The new regression test covers this behavior.

I also checked the parallel task without modifying its code, logs, or output. Its attempted services used ports 32106 and 32107 with aliases `ea-atpo-v2-base8b-gpu6` and `ea-atpo-v2-expert32b-gpu7`; neither port was listening. Its Base and Expert startup logs both report `zmq.error.ZMQError: ipc path is longer than 107 characters (sizeof(sockaddr_un.sun_path))`. The VLLM service processes exited before serving requests. This is an IPC socket-path failure, not a malformed OpenAI request or confirmed model-weight load error. No healthy Base/Expert endpoint was available to reuse.

The pinned model directories and runtime executable exist. The established launch command is `bash experiments/vi_ea_atpo/phase2/start_vi_services.sh`; it uses `manage_vi_services.py start` with `experiments/vi_ea_atpo/configs/model_services.yaml`, serving Qwen3-8B on 30481 and Qwen3-32B-AWQ on 30482. The config maps both services to physical GPU2, allocates 0.12 and 0.25 GPU memory fractions, and requires at least 100,000 MiB free, at most 20% utilization before startup, and no foreign compute processes. GPU2 currently has about 127,675 MiB free but 100% utilization and another compute process, so this launch command would refuse. GPU6 has about 168,173 MiB free and 0% utilization, but a foreign process still uses about 14 GiB; it fails the same no-sharing gate. GPU7 is excluded and has a foreign process. No safe card is available, so neither existing launcher nor a new V2 service was started. No process was interrupted. All V2 artifacts remain under project PFS; no system-disk cache or output was written.

The final proxy-bypassing static preflight is saved in the ignored local manifest at `outputs/phase1_20261119T140623460757Z/run_manifest.json`; the execution-gate attempt is recorded at `outputs/phase1_20261119T140623978525Z/run_manifest.json`. These local artifacts include selected case identifiers and must not be committed. The folder is covered by the existing `**/outputs/` ignore rule.

## Q Regression readiness and scale estimate

The data interface is ready; **the training data are not** because no V2 action return was collected.

- The blocked pilot is preselected at 10 train cases / 29 states. It would yield up to 87 state-action records at one sample each and at most 145 model calls (ANSWER 1, ASK 3, DEFER 1 per state), if endpoints become available.
- The complete currently available train pool is 70 cases / 205 states. One valid sample per action would require up to 615 action rollouts; three repetitions would require up to 1,845 rollouts. The matching 42 validation states and 44 internal-test states remain reserved for later validation/diagnostics.
- The 10-case pilot is an engineering feasibility set, too small to fit or assess a meaningful Q regressor. Q training should wait for successful paired labels across the larger train pool and fixed continuation policy. Reusing V1 Q labels would violate policy consistency.
- Previously recorded service memory was approximately 21.6 GiB for Qwen3-8B and 45.2 GiB for Qwen3-32B-AWQ, plus runtime headroom. A dedicated idle 183 GiB B20Z card with at least 100,000 MiB free is the configured deployment gate for the two services; current cards with foreign processes are not eligible under that gate.

## Reproduction

From the repository root, run the static and service preflight:

```bash
python3 -m unittest discover -s experiments/ea_atpo_v2/tests -v
python3 experiments/ea_atpo_v2/scripts/run_counterfactual_pilot.py --preflight --cases 10
```

After the configured pinned Base and Expert endpoints are available and match `/v1/models`, execute:

```bash
python3 experiments/ea_atpo_v2/scripts/run_counterfactual_pilot.py --execute --cases 10
```

The runner refuses to start if either model alias is unavailable. It does not launch services or touch GPUs. Once real trajectories exist, build masked Q-regression rows with:

```bash
python3 experiments/ea_atpo_v2/scripts/build_q_dataset.py \
  --trajectories experiments/ea_atpo_v2/outputs/<run>/pilot_trajectories.jsonl \
  --output experiments/ea_atpo_v2/outputs/<run>/q_regression_dataset.jsonl
```

## Conclusion for this phase

- **Real model execution:** none in V2 Phase 1; service gate blocked it before generation.
- **Static data audit:** passed on the restored source and frozen training-state subset.
- **Protocol and boundary tests:** 11/11 passed; engineering-only.
- **ASK produces a clinically useful new fact:** not evaluated; no Patient calls ran.
- **Three action values:** not available; no V2 returns exist.
- **Q Regression:** interface exists, but fitting is not justified yet.
- **Tree search and policy distillation:** not run; they remain later phases.
- **Frozen Holdout:** unopened.

The next executable step is to restore the two pinned services on a safely available dedicated card, rerun `--preflight`, then run the frozen 10-case pilot. Do not advance to Q Regression training until the paired rollouts are valid and continuation-policy-consistent.

## Live service and resource recheck (2026-11-19 14:20 UTC)

The service state was rechecked after the earlier endpoint diagnosis. The V2 configuration still pins Base to `vi-qwen3-8b-base` at `127.0.0.1:30481` (revision `b968826d9c46dd6066d109eabc6255188de91218`) and Expert to `vi-qwen3-32b-awq-expert` at `127.0.0.1:30482` (revision `0499c3ac83fdef8810b907a23894ba91e95eddd8`); Patient shares Base. Both local model directories are present with their configs, tokenizer files, indexes, and weight shards (5 Base shards, 4 Expert shards).

At this recheck, `HTTP_PROXY`/`ALL_PROXY` still point to `127.0.0.1:18118`, where the proxy listener is active. A proxied GET to Base returned HTTP 503 from `127.0.0.1:18118`; the same GET with proxy bypass returned connection refused to port 30481. Direct checks found no listeners on 30481, 30482, 32106, or 32107. No Base/Expert generation request was sent. This confirms the observed 503 is generated by the local proxy while the configured model endpoints are absent; it does not indicate a malformed inference payload. The origin request format and model-load behavior therefore remain untested. The previous parallel-service attempt's logs recorded a separate startup failure, `zmq.error.ZMQError: ipc path is longer than 107 characters (sizeof(sockaddr_un.sun_path))`; those service processes are no longer listening. Its code and output were not changed.

GPU6 had 166,879 MiB free and GPU7 had 179,357 MiB free, both at 0% utilization, but neither meets the no-foreign-process condition: another user's `rtdm_facescore_continue100` process held about 15,726 MiB on GPU6 and processes held about 3,238 MiB on GPU7. GPU7 also remains excluded by the prior project constraint. The other GPUs were not safe substitutes under the existing deployment gate: GPU2 had 127,675 MiB free but 90% utilization and foreign compute processes; GPU3 had 106,885 MiB free with a foreign process; GPU4 had only 51,119 MiB free; GPU0/1/5 were heavily utilized. No GPU process was stopped, and no inference request was sent to any foreign service. The only other listening VLLM endpoint found was port 18951 serving `Qwen/Qwen3-VL-8B-Instruct`, which is not the pinned Base model and was not reused.

The established launcher remains `bash experiments/vi_ea_atpo/phase2/start_vi_services.sh`; its config targets GPU2 and ports 30481/30482, so its utilization/foreign-process preflight would fail on the current snapshot. That config also sets `runtime.tmp_root` to `/tmp/vi-ea-atpo`; it must be redirected to a project-local path before any future launch, consistent with the storage constraint. No service was started.

A fresh read-only V2 preflight at `outputs/phase1_20261119T142044510359Z/run_manifest.json` selected the frozen 10 training cases / 29 public states and ended as `blocked_endpoint_preflight`; `model_execution_started` is false. The unchanged CPU regression suite passed 11/11 tests and Python compilation passed. The smoke path is implemented but has not run against a model. Consequently there are still zero V2 real trajectories, zero action returns, zero empirical Q labels out of the potential 87 state-action labels, no measured token/latency costs, and no observed ASK response. No Q Regression, distillation, PPO, or SFT work was started.
