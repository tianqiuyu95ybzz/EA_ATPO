# Gold review status

The private review queue was created for all 291 source snapshots (205 Train, 42 Dev, 44 Internal Test). All 291 are `PENDING`; zero have a verified Base answer, reviewed Gold, reviewed Base correctness, reviewed action, evidence, reviewer or timestamp. Risk remains `UNKNOWN`. No status or correctness was inferred from an old answer key or an old metric. The queue also includes each original public conversation and historical answer and is therefore excluded from Git.

The queue was created with mode `0600` in a mode `0700` PFS directory. `data/gold_review_queue_manifest.json` records the initial queue checksum and aggregate status without publishing the queue contents. The three previously noted V2 source anomalies require particular attention, as do all potential corrections and all KEEP cases. Because Base outputs and human risk judgments are not yet available, no candidate-specific risk or action counts can be honestly reported.

The review checker was executed against the initial queue: it correctly blocked formal use with 291 pending rows and 291 missing verified Base answers. It must be run again on a new frozen human-reviewed copy together with a verified Base-output artifact. It cannot replace the separate model/tokenizer/prompt/decoding provenance gate or establish that a reviewer is a clinician.

Private queue:

`experiments/ea_atpo_v3/expert_sft/data/private/gold_review_queue.csv`

See `GOLD_REVIEW_PROTOCOL.md` for the required fields, review rules, versioning and privacy constraints. No actual human review has been claimed, and no formal SFT or Internal Test is authorized by the current state.
