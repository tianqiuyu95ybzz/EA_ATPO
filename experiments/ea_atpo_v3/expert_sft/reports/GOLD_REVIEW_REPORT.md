# Gold review status

The private review queue was created for all 291 source snapshots (205 Train, 42 Dev, 44 Internal Test). A new queue version now contains the 291 individually verified raw Base answers. All 291 are still `PENDING`; none has reviewed Gold, reviewed Base correctness, reviewed action, evidence, reviewer or timestamp. Risk remains `UNKNOWN`. No status or correctness was inferred from an old answer key or an old metric. The queue also includes each original public conversation and historical answer and is therefore excluded from Git.

The queue was created with mode `0600` in a mode `0700` PFS directory. `data/gold_review_queue_manifest.json` records the original blank-Base queue, and `data/gold_review_priority_manifest.json` records the later queue checksum. The priority list marks 169 states where Base's answer differs from the **unreviewed historical** label and 122 where it agrees. These are review-priority counts, not correctness labels or Base accuracy. All 291 still require human review, including possible KEEP cases; the three previously noted V2 source anomalies and high-risk cases require particular attention. Risk and action labels remain undecided.

The review checker was run on the updated queue with the verified Base artifact: it correctly blocked formal use solely because 291 rows are pending. It must be run again on a new frozen human-reviewed copy. It cannot establish that a reviewer is a clinician.

Private queue:

`experiments/ea_atpo_v3/expert_sft/data/private/gold_review_queue_with_base.csv`

The original blank-Base queue remains preserved. `data/private/gold_review_priority.csv` orders historical disagreements first; it does not assign a target action or correctness label.

See `GOLD_REVIEW_PROTOCOL.md` for the required fields, review rules, versioning and privacy constraints. No actual human review has been claimed, and no formal SFT or Internal Test is authorized by the current state.
