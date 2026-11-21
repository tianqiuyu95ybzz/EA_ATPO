# A → C handoff: ReMeDi V3 pilot

The private, deterministic 100-dialogue pilot is ready now. Do not wait for the remaining report or GitHub push to begin service setup.

Absolute sample path:

`/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO_V1_v3_pilot_prep_20261121/experiments/ea_atpo_v3/dialogue_pilot_20261121/private/pilot_samples.jsonl`

Private manifest:

`/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO_V1_v3_pilot_prep_20261121/experiments/ea_atpo_v3/dialogue_pilot_20261121/private/pilot_manifest.json`

- 100 unique ReMeDi source dialogues; original `source_dialogue_id` preserved.
- 2,299 original turns, in source order; each starts with an initial patient turn and contains later doctor replies.
- 281 linked ASK replay pairs. Each is exactly an adjacent, question-like historical doctor turn followed by the next non-empty patient turn. The pair is a replay proxy, not a generated-question match.
- Doctor turns are `historical_clinician_response_references`, not verified Gold.
- Raw dialogue text and manifest are private, mode 0600/0700; do not upload to public GitHub.
- Source SHA-256 `b27b6b1222255cc87cbc6cc0fb6ce7de98e0b6341f5ea61892bb92f247383458`; sample SHA-256 `84085116662104c50525558d1b17ff54f0f4d6f4720bec1ae2a8e7f1c8c7b494`.

Format and scoring instructions are in `RESULT_SCHEMA.md`; the fixed blind quality prompt is `blind_evaluation_prompt.md`. The evaluation aggregator is `scripts/evaluate_pilot.py`. It computes a quality-based utility proxy and will not treat historical doctor text as truth.
