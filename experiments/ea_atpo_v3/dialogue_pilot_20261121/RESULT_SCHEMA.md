# V3 dialogue pilot data and evaluation schema

## Private sample JSONL

`private/pilot_samples.jsonl` contains one complete source dialogue per row:

- `sample_id`: stable local identifier derived from the ReMeDi source ID.
- `source_dialogue_id`: original ReMeDi `dialogue` ID.
- `source_order_index`: original array position, retained for provenance.
- `initial_patient_question`: first patient turn.
- `messages`: all valid patient/doctor turns in source order with zero-based `turn_index`.
- `historical_clinician_response_references`: each original doctor turn with its turn index; none is certified Gold.
- `ask_replay_pairs`: only an immediately adjacent, question-like doctor turn followed by a non-empty patient turn; both original turn indices and texts are retained.

The ASK question proxy is the exact historical clinician question in the pair. If an evaluated Base generates a different question, do not attach this patient reply unless a predeclared exact/validated match rule succeeds. Otherwise record a replay-mismatch failure and provide no patient response. This prevents using an unrelated later patient turn.

## Prediction JSONL from C

One row per `sample_id` × `policy` × replicate:

```json
{
  "sample_id": "remedi_<source-id>",
  "policy": "ANSWER|ASK|DEFER|EA-ATPO-simple-rule",
  "replicate_id": "0",
  "blind_id": "random opaque ID",
  "action_trace": ["ASK", "ANSWER"],
  "ask_count": 1,
  "defer_count": 0,
  "final_output": "private model output",
  "failure_reason": null,
  "ask_replay_pair_index": 0,
  "replay_match": true
}
```

`ANSWER` uses Base direct answer. `ASK` replays only the linked historical pair and then lets Base continue. `DEFER` uses the live Expert model; the historical clinician response is a reference proxy only, never a replacement for the generated answer or verified Gold. The same fixed rating prompt must be used for all anonymized outputs.

## Blind rating JSONL

Policy identity must not appear in the blind ratings file:

```json
{"blind_id":"random opaque ID","answer_quality_1_to_5":4,"ask_information_useful":true}
```

`evaluate_pilot.py` joins by `blind_id` and reports aggregate quality, ASK information usefulness, action counts, failures, and `Utility = (quality−1)/4 − 0.01×ASK_count − 0.05×DEFER_count`. This utility is a normalized **blind quality proxy**, not accuracy. The script rejects `gold`, `ground_truth`, `correctness`, `final_correct`, and `accuracy` fields.
