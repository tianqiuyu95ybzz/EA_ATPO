# Base answer pairing status

The frozen Base is Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`. The local tokenizer and tokenizer configuration match the historical pinned fingerprints. V2 batch manifests record this revision, no thinking, temperature 0, top-p 1, maximum 24 answer tokens, and the same public-state, case-split and source-file SHA-256 values. The existing 146 V2 ANSWER records all map exactly to the corresponding Train cases/states; each contains one valid, non-empty raw answer. Their full raw outputs and per-output SHA-256 hashes have been copied to a **private candidate-only** JSONL under this V3 directory. This adds no new SFT samples.

Those 146 responses are **not approved for V3 reuse**: the old per-request trace does not save the exact request messages or a prompt hash, and the recorded Git commit does not lock the runtime's uncommitted code. Thus the actual prompt sent cannot be independently established byte for byte. The remaining 145 snapshots also lack this V2 output. The safe resolution is to generate all 291 Base answers under a single frozen V3 prompt, tokenizer, decoding setup and runtime, recording each raw response, source state and request hash. No historical output is silently substituted.

No eligible GPU passed the project gate at the latest check. Generation has not started, the review queue's Base fields remain empty and no paired dataset exists. The earlier 0%, 100% or other accuracy figures from different historical experiments are not V3 results.

The review checker requires a separate complete Base manifest with the pinned revision, raw-answer file checksum and passing model/tokenizer/prompt/decoding/data/case/completeness checks. Candidate records carry `UNVERIFIED_PROMPT_AT_RUNTIME` and are rejected by that gate. The eventual manifest must be generated from evidence, not populated by hand merely to pass the checker.

A private plan now freezes 291 public-only V2 ANSWER prompts and a SHA-256 for each request; the plan excludes Gold and review metadata. `data/base_request_manifest.json` records its checksum and decoding settings. The generator checks all Base shards and tokenizer hashes against the recorded fingerprint, checks the GPU gate twice, records raw responses/hashes and emits a `VERIFIED` manifest only on complete success. Its real GPU/model execution has **not** been validated here. A deliberate check against occupied GPU 0 stopped at the safety gate before loading weights; its preflight evidence remains in the private V3 directory.

After an allowed GPU is actually free and the project gate passes, use a fresh output directory and the existing isolated V2 vLLM runtime (replace `GPU` with one of 0, 2, 3, 4, 5, 6):

```bash
CUDA_VISIBLE_DEVICES=GPU /pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/.venv-real-pilot/bin/python \
  experiments/ea_atpo_v3/expert_sft/scripts/generate_base.py --gpu GPU \
  --model-dir /pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/experiments/real_case_pilot_20261103/assets/models/Qwen3-8B \
  --plan experiments/ea_atpo_v3/expert_sft/data/private/base_request_plan.jsonl \
  --plan-manifest experiments/ea_atpo_v3/expert_sft/data/base_request_manifest.json \
  --output-dir experiments/ea_atpo_v3/expert_sft/data/private/base_generation_NEW
```

Successful completion is a prerequisite to copying verified raw Base answers into a new version of the human review queue. Model loading, model output format, GPU peak and actual duration remain unmeasured until the run is performed.
