# V2 Cache Reuse Audit and GPU Gate

**Date:** 2026-11-19  
**V2 commit at audit:** `9d73052`  
**Scope:** Read-only audit of the completed three-case cache and current service/GPU state. No model was started and no prediction cache was modified.

## Cache integrity and provenance

The cache at `experiments/ea_atpo_v2/parallel_cache/pilot3_20261119T134631Z/` passes all 17 entries in `checksums.sha256`. Its manifest pins the same public-state JSONL and case-split files currently used by V2; both file hashes match. The cache contains three train cases and nine states. All nine `(case_id, state_id)` pairs exist in the frozen public-state source, and none belongs to validation, internal-test, or frozen holdout data.

The cache validation manifest reports all checks passed: Base and Expert predictions align for all nine states, prompt/state hashes validate against the cache's frozen-state builder, there are no request errors, and all 9/9 outputs per model parse. The manifest records `gold_source_opened=false`, `hidden_atomic_fact_source_opened=false`, and `holdout_opened=false`. The prediction JSONL files contain no Gold or hidden-fact fields. Checksums and manifests are retained as supplied; this audit did not rewrite them.

Cached model IDs and revisions match the V2 pins:

| Role | Model | Revision | Cached rows | Parse-valid |
|---|---|---|---:|---:|
| Base | Qwen3-8B | `b968826d9c46dd6066d109eabc6255188de91218` | 9 | 9 |
| Expert | Qwen3-32B-AWQ | `0499c3ac83fdef8810b907a23894ba91e95eddd8` | 9 | 9 |

The cache model fingerprint records the locked weight and tokenizer hashes for both models. No cached row was used as an accuracy label; the cache did not open Gold, so its outputs have no evaluated accuracy, correction, or degradation rate.

## Compatibility with the frozen V2 pilot

The V2 run manifest selects 10 train cases / 29 states. The cache selects 3 cases / 9 states. The case intersection is **0**, and the state-ID intersection is **0**. Thus no cache row refers to one of the 29 target states. Although both artifacts come from the same frozen public-state source, there is no paired target state on which to compare state hashes; the cache's own state/prompt hash validation does not make a different selected state interchangeable.

There are additional protocol mismatches:

- The cache uses one shared terminal-answer prompt and `Final Answer: <A-D>` format for Base and Expert. V2 uses a forced-ANSWER Base prompt and a separate DEFER Expert prompt with `Expert Final Answer: <A-D>`.
- Cache decoding is temperature 0, top-p 1, `max_tokens=64`, with logprobs and top-20 logprobs. V2 uses separate limits (`answer=24`, `ASK=160`, `patient=256`, `DEFER=32`) and does not request logprobs.
- The cache has no ASK question, patient response, or post-ASK Base continuation.

**Reusable predictions for this frozen V2 pilot: 0/18.** The cached 18 direct answers remain valid as a separate engineering cache run, but must not be spliced into the V2 counterfactual trajectories or Q labels.

Cache-only service telemetry, not V2 pilot metrics: Base and Expert each returned 45 completion tokens over nine requests; mean request latency was 0.099 s for Base and 0.131 s for Expert. These figures use the cache's different prompts/decoding and do not represent ASK, paired returns, or full V2 costs.

## Service and GPU gate

The cache manifest records the earlier services as started on GPU6/GPU7 and then cleanly stopped. At this audit, ports 32106/32107 and the V2-configured 30481/30482 have no listeners. The cache's startup scripts resolved their output directories from their own script path; the recorded model and runtime paths are the existing sibling `EA_ATPO` assets, and the cache manifest confirms both models loaded. No stale working-directory failure was found in that completed run. The earlier long Unix-socket-path startup failure was subsequently worked around in the cache run using short TMP symlinks; this audit did not edit those scripts or their output directory.

Current GPU snapshot:

| GPU | Free memory | Utilization | Other process | V2 status |
|---:|---:|---:|---|---|
| 6 | 166,879 MiB | 0% | PID 234877, another user's `rtdm_facescore_continue100` training process, about 15,726 MiB | No V2 Base started; foreign workload is left untouched |
| 7 | 179,357 MiB | 0% | PIDs 234877 and 301818, another user's process(es), about 3,238 MiB combined | **Prohibited** by V2 `resource_policy.gpu_1_and_7_prohibited=true` and the prior explicit GPU7 restriction |

The user instruction explicitly says to report this GPU7 permission conflict and not bypass it. Therefore the Expert service was not restarted. No Base-only inference was started because the requested paired smoke/pilot cannot complete without the Expert and the cache has no compatible target rows. No other process was terminated, signaled, or sent inference traffic.

The established V1/VI service launcher targets GPU2 and ports 30481/30482; its current GPU2 utilization exceeds its configured startup threshold. It also sets `runtime.tmp_root=/tmp/vi-ea-atpo`, so it is not suitable to launch unchanged under the project-local storage requirement. The old cache run's services are already closed; there is no healthy endpoint to reuse now.

## Experimental outcome

- Base ANSWER / Expert DEFER / ASK Patient / post-ASK Base smoke: **not run**.
- Frozen V2 pilot: **not run**; 0 of 29 states have three V2 actions.
- ASK new-information rate, action returns, accuracy/correction/degradation, costs, and latency: **not estimable** for V2.
- Empirical Q labels: **0/87** potential state-action labels.
- Q Regression, Actor distillation, PPO, Expert SFT: **not started**.

Next execution requires the GPU7 prohibition to be explicitly lifted by the project owner, a safe GPU6/7 resource window that does not disturb the other user's work, and project-local runtime/cache paths. After that, run the V2 role smoke first; only if it passes should the 10-case / 29-state rollout start. The cache must remain excluded from that pilot unless a future target state, prompt, model revision, and decoding configuration all match exactly.
