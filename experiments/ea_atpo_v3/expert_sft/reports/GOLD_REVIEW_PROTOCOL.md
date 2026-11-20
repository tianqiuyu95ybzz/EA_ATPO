# Gold review protocol

The queue contains simulated medical case text and historical answer labels. It lives only in the private PFS directory, must not be committed, and initially has all 291 rows `PENDING`. It is not a clinical Gold standard and no clinician has been represented as having reviewed it.

1. Populate `base_answer` only from a separately verified, complete raw Base-answer artifact. Preserve the exact raw bytes/text and its per-row SHA-256. Do not infer correctness from the historical V2 metric.
2. A named human reviewer examines the full visible case/dialogue, available answer choices, original Gold and Base answer for **every** row, including every possible KEEP, REVISE, high-risk or ambiguous case. Record a specific evidence basis, risk level, reviewer, timezone-aware timestamp and relevant note. Clinical credentials must be stated truthfully outside the data; do not call non-clinician review physician-certified.
3. Use `APPROVED` only when an answer and its correctness are supported: `base_correct=true` means `KEEP`, and `base_correct=false` means `REVISE`. Preserve the original Base response verbatim for KEEP at runtime. A corrected answer is not invented if the evidence is inadequate.
4. Use `AMBIGUOUS` plus `ABSTAIN` with no `reviewed_gold` for unresolved ambiguity; alternatively use `REJECTED` and leave both action and reviewed Gold blank to exclude the row. `PENDING` is not a training label. No automatic review or risk classification is substituted for a human decision.
5. A reviewed queue is private and may be edited only by its reviewer. Do not overwrite the initial queue or prior reviewed versions: copy it to a new private version before review. Freeze and hash the final version. The checker rejects missing rows, changed source text, missing Base provenance records, inconsistent actions, unreviewed risk/evidence and any PENDING row. It is a structural gate, not proof of clinical quality.

The separate pairing gate must verify Base model revision, tokenizer, exact prompt, sampling settings, data version, case/state identity and output completeness. The review checker by itself cannot establish those provenance claims.

After Base generation succeeds, `scripts/enrich_gold_review_queue.py` can create a **new** private queue version from the original queue and the verified Base output/manifest. It checks coverage, case and split identities, raw-answer hashes and review status; it does not approve any row. The original queue is preserved.

At inference, conversation and Base answer are the only task inputs to Expert A. `original_gold`, `reviewed_gold`, `base_correct`, action labels, evidence, risk, reviewer and timestamp are labels or audit metadata and are never model inputs. Expert B/C see only the public conversation.
