# Base answer pairing status

The frozen Base is Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`. The local tokenizer and tokenizer configuration match the historical pinned fingerprints. V2 batch manifests record this revision, no thinking, temperature 0, top-p 1, maximum 24 answer tokens, and the same public-state, case-split and source-file SHA-256 values. The existing 146 V2 ANSWER records all map exactly to the corresponding Train cases/states; each contains one valid, non-empty raw answer. Their full raw outputs and per-output SHA-256 hashes have been copied to a **private candidate-only** JSONL under this V3 directory. This adds no new SFT samples.

Those 146 responses are **not approved for V3 reuse**: the old per-request trace does not save the exact request messages or a prompt hash, and the recorded Git commit does not lock the runtime's uncommitted code. Thus the actual prompt sent cannot be independently established byte for byte. The remaining 145 snapshots also lack this V2 output. The safe resolution is to generate all 291 Base answers under a single frozen V3 prompt, tokenizer, decoding setup and runtime, recording each raw response, source state and request hash. No historical output is silently substituted.

**Base generation completed** on an allowed GPU 5 after both no-sharing pre-load checks passed. All 291 requests received one format-valid raw output: 205 Train, 42 Dev and 44 Internal Test. An independent check verified all 291 raw SHA-256 values and exact sample/request-hash correspondence with the frozen plan. The model and tokenizer fingerprints matched the pinned files. The V2 runtime used vLLM 0.33.0, PyTorch 2.14.0+cu130, TRITON_ATTN, temperature 0, top-p 1, 24-token maximum, constrained answer format and thinking disabled. The maximum observed device use was 21,540 MiB and the whole attempt, including fingerprinting/startup, took 298.879 seconds. The raw outputs remain private; the safe fingerprint and file manifest are in `data/base_generation_manifest.json`.

The review checker requires a separate complete Base manifest with the pinned revision, raw-answer file checksum and passing model/tokenizer/prompt/decoding/data/case/completeness checks. Candidate records carry `UNVERIFIED_PROMPT_AT_RUNTIME` and are rejected by that gate. The eventual manifest must be generated from evidence, not populated by hand merely to pass the checker.

A private plan freezes 291 public-only V2 ANSWER prompts and a SHA-256 for each request; the plan excludes Gold and review metadata. `data/base_request_manifest.json` records its checksum and decoding settings. The generator checked all Base shards and tokenizer hashes, checked the GPU gate twice, monitored GPU process ownership and free memory during generation, and produced a `VERIFIED` manifest. A deliberate earlier check against occupied GPU 0 stopped at the gate before loading weights.

Earlier attempts are retained: attempt 01 failed before inference because its Unix socket path exceeded 107 bytes; attempt 02 selected FlashInfer instead of the intended V2 backend and was stopped by signaling only this run's own processes; attempt 03 was stopped after an unsupported environment variable was detected, before generation. During attempt 02 FlashInfer attempted a JIT build under `/pfs/pfs-spMCeu/image_team/persistent-cache-agnes-intern-image/cache/flashinfer/0.7.2.post1/100a/cached_ops/sampling`. This shared cache was not cleaned or otherwise altered after stopping the attempt; its pre-existing state is unknown. Attempt 04 explicitly sets the V2 backend and keeps runtime caches under its own PFS directory. Short IPC aliases point to V3 private runtime directories and remain in place. No Base answers from attempts 01–03 were used.

After an allowed GPU is actually free and the project gate passes, use a fresh output directory and the existing isolated V2 vLLM runtime (replace `GPU` with one of 0, 2, 3, 4, 5, 6):

```bash
CUDA_VISIBLE_DEVICES=GPU /pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/.venv-real-pilot/bin/python \
  experiments/ea_atpo_v3/expert_sft/scripts/generate_base.py --gpu GPU \
  --model-dir /pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/experiments/real_case_pilot_20261103/assets/models/Qwen3-8B \
  --plan experiments/ea_atpo_v3/expert_sft/data/private/base_request_plan.jsonl \
  --plan-manifest experiments/ea_atpo_v3/expert_sft/data/base_request_manifest.json \
  --output-dir experiments/ea_atpo_v3/expert_sft/data/private/base_generation_NEW
```

The successful run has been copied into a new still-`PENDING` review queue. No reviewed Gold or paired training dataset exists yet; no Base accuracy is claimed from the historical, unreviewed answer labels.
