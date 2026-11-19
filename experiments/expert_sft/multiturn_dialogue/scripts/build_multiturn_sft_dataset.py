#!/usr/bin/env python3
"""Build answer-only SFT examples from frozen, public ATPO dialogue states.

Only public state snapshots are used as model inputs. Gold is read from the
corresponding rollout record and appears only as the assistant target.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
EXP = ROOT / "experiments/expert_sft"
MULTI = EXP / "multiturn_dialogue"
DEFAULT_STATES = ROOT / "experiments/real_case_pilot_20261103/outputs/full/pilot_public_states.jsonl"
DEFAULT_TRAJECTORIES = ROOT / "experiments/real_case_pilot_20261103/outputs/full/pilot_trajectories.jsonl"
DEFAULT_SPLIT = ROOT / "experiments/ea_atpo_value_router_20261105/artifacts/case_split.json"
DEFAULT_HOLDOUT = EXP / "data/fresh_test_selection.json"

SYSTEM = (
    "You are a medical multiple-choice diagnostic expert. Use only the visible case, "
    "answer options, and public conversation below. Do not assume hidden patient facts "
    "or use an answer key. Return exactly one line in this format: "
    "Expert Final Answer: <one uppercase option letter A-D>. Do not include reasoning."
)


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: malformed JSONL") from exc


def public_case_text(state: dict[str, Any]) -> tuple[str, int]:
    initial = state.get("initial_messages")
    dialogue = state.get("dialogue")
    trace = state.get("state_construction_trace")
    if not isinstance(initial, list) or not isinstance(dialogue, list) or not isinstance(trace, list):
        raise ValueError(f"State {state.get('state_id')} misses public message fields")
    user_messages = [m.get("content") for m in initial if isinstance(m, dict) and m.get("role") == "user"]
    if len(user_messages) != 1 or not isinstance(user_messages[0], str) or not user_messages[0].strip():
        raise ValueError(f"State {state.get('state_id')} must have exactly one public user case message")
    # The original Base system message is deliberately omitted. It contains agent
    # role instructions, whereas the Expert gets a new, fixed system instruction.
    lines = ["Visible case and answer options:\n" + user_messages[0].strip()]
    if len(dialogue) != len(trace) or len(dialogue) % 2:
        raise ValueError(f"State {state.get('state_id')} has an incomplete dialogue/trace pair")
    expected_roles = ("assistant", "user")
    expected_actors = ("base", "patient")
    for index, (message, event) in enumerate(zip(dialogue, trace)):
        role = message.get("role") if isinstance(message, dict) else None
        actor = event.get("actor") if isinstance(event, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        response = event.get("response") if isinstance(event, dict) else None
        slot = index % 2
        if role != expected_roles[slot] or actor != expected_actors[slot]:
            raise ValueError(f"State {state.get('state_id')} has unexpected public role/actor at message {index}")
        if not isinstance(content, str) or not content.strip() or content != response:
            raise ValueError(f"State {state.get('state_id')} public text does not match its construction event")
        label = "Clinician question" if actor == "base" else "Patient reply"
        lines.append(f"{label}: {content.strip()}")
    return "\n\n".join(lines), len(dialogue) // 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=Path, default=DEFAULT_STATES)
    parser.add_argument("--trajectories", type=Path, default=DEFAULT_TRAJECTORIES)
    parser.add_argument("--case-split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--fresh-holdout-selection", type=Path, default=DEFAULT_HOLDOUT)
    parser.add_argument("--output-dir", type=Path, default=MULTI / "data")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if MULTI.resolve() not in output.parents:
        raise ValueError(f"All outputs must remain under {MULTI}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing data directory: {output}")

    split_doc = json.loads(args.case_split.read_text(encoding="utf-8"))
    expected_source_hash = split_doc.get("pilot_trajectory_sha256")
    trajectory_hash = sha_file(args.trajectories)
    if expected_source_hash != trajectory_hash:
        raise ValueError("Pilot trajectory hash does not match the frozen case-split artifact")
    split_ids = {name: set(map(str, split_doc[name])) for name in ("train", "validation", "internal_test")}
    if any(split_ids[a] & split_ids[b] for i, a in enumerate(split_ids) for b in list(split_ids)[i + 1 :]):
        raise ValueError("Frozen case split contains case_id overlap")
    if [len(split_ids[k]) for k in ("train", "validation", "internal_test")] != [70, 15, 15]:
        raise ValueError("Expected the existing 70/15/15 case_id split")

    gold_by_case: dict[str, str] = {}
    trajectory_rows = Counter()
    trajectory_action_repeats: dict[tuple[str, str], Counter[str]] = {}
    for row in jsonl(args.trajectories):
        if row.get("record_type") != "trajectory":
            continue
        case_id = str(row.get("case_id", ""))
        gold = str(row.get("gold", "")).strip().upper()
        if not case_id or gold not in {"A", "B", "C", "D"}:
            raise ValueError("Trajectory row has a missing case_id or non A-D Gold target")
        previous = gold_by_case.setdefault(case_id, gold)
        if previous != gold:
            raise ValueError(f"Gold label changes within case_id {case_id}")
        trajectory_rows[case_id] += 1
        state_id = str(row.get("state_id", ""))
        action = str(row.get("action", ""))
        if not state_id or action not in {"ANSWER", "ASK", "CONSULT", "DEFER"}:
            raise ValueError("Trajectory row has an invalid state_id or action")
        trajectory_action_repeats.setdefault((case_id, state_id), Counter())[action] += 1
    if set(gold_by_case) != set().union(*split_ids.values()):
        raise ValueError("Frozen split case ids do not exactly match the trajectory cases")
    if any(set(counts) != {"ANSWER", "ASK", "CONSULT", "DEFER"} or any(counts[action] != 3 for action in ("ANSWER", "ASK", "CONSULT", "DEFER")) for counts in trajectory_action_repeats.values()):
        raise ValueError("Expected three repetitions of each of the four actions per recorded public state")

    holdout = json.loads(args.fresh_holdout_selection.read_text(encoding="utf-8"))
    if holdout.get("status") != "frozen_before_sft_checkpoint_selection":
        raise ValueError("The fresh 400-case selection is not marked as pre-SFT frozen")
    holdout_ids = {str(r["case_id"]) for r in holdout.get("rows", [])}
    holdout_source_rows = {int(r["source_row_index"]) for r in holdout.get("rows", [])}
    if len(holdout_ids) != 400 or len(holdout_source_rows) != 400:
        raise ValueError("Expected 400 distinct cases in the frozen fresh holdout")

    state_rows: dict[str, dict[str, Any]] = {}
    state_source_rows: set[int] = set()
    for state in jsonl(args.states):
        case_id = str(state.get("case_id", ""))
        state_id = str(state.get("state_id", ""))
        if not case_id or not state_id or case_id not in gold_by_case:
            raise ValueError("Public state is not linked to a known training trajectory case")
        if state_id in state_rows:
            raise ValueError(f"Duplicate public state_id {state_id}")
        if int(state.get("source_row_index", -1)) in holdout_source_rows or case_id in holdout_ids:
            raise ValueError("Training dialogue overlaps the pre-frozen fresh holdout")
        text, dialogue_turns = public_case_text(state)
        state_rows[state_id] = {
            "case_id": case_id,
            "source_row_index": int(state["source_row_index"]),
            "state_id": state_id,
            "state_index": int(state["state_index"]),
            "split": next(name for name, ids in split_ids.items() if case_id in ids),
            "dialogue_turns": dialogue_turns,
            "public_text": text,
            "prompt_hash": sha_text(text),
        }
        state_source_rows.add(int(state["source_row_index"]))
    if len(state_rows) != 291 or {r["case_id"] for r in state_rows.values()} != set(gold_by_case):
        raise ValueError("Expected 291 frozen public states from all 100 training cases")
    if set(trajectory_action_repeats) != {(r["case_id"], r["state_id"]) for r in state_rows.values()}:
        raise ValueError("The trajectory action grid does not cover exactly the archived public states")

    by_split: dict[str, list[dict[str, Any]]] = {k: [] for k in split_ids}
    for row in sorted(state_rows.values(), key=lambda r: (r["case_id"], r["state_index"], r["state_id"])):
        prompt = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": row["public_text"]},
        ]
        messages = prompt + [{"role": "assistant", "content": f"Expert Final Answer: {gold_by_case[row['case_id']]}"}]
        by_split[row["split"]].append({
            "case_id": row["case_id"],
            "source_row_index": row["source_row_index"],
            "state_id": row["state_id"],
            "state_index": row["state_index"],
            "split": row["split"],
            "dialogue_turns": row["dialogue_turns"],
            "prompt_sha256": row["prompt_hash"],
            "messages": messages,
        })

    hashes = {name: {row["prompt_sha256"] for row in rows} for name, rows in by_split.items()}
    if any(hashes[a] & hashes[b] for i, a in enumerate(hashes) for b in list(hashes)[i + 1 :]):
        raise ValueError("Exact public prompt is repeated across case_id splits")
    output.mkdir(parents=True, exist_ok=True)
    filenames = {"train": "train_sft.jsonl", "validation": "validation_sft.jsonl", "internal_test": "internal_test_sft.jsonl"}
    for split, rows in by_split.items():
        path = output / filenames[split]
        with path.open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "status": "built_from_existing_public_multiturn_pilot_states",
        "source": {
            "pilot_public_states": portable_path(args.states),
            "pilot_public_states_sha256": sha_file(args.states),
            "pilot_trajectories": portable_path(args.trajectories),
            "pilot_trajectories_sha256": trajectory_hash,
            "frozen_case_split": portable_path(args.case_split),
            "frozen_case_split_sha256": sha_file(args.case_split),
            "fresh_holdout_selection_sha256": sha_file(args.fresh_holdout_selection),
        },
        "case_split": {name: {"cases": len(split_ids[name]), "states": len(by_split[name]), "cases_sha256": sha_text("\n".join(sorted(split_ids[name])))} for name in split_ids},
        "total": {"cases": len(gold_by_case), "states": len(state_rows), "trajectory_rollouts": sum(trajectory_rows.values()), "holdout_cases": len(holdout_ids)},
        "dialogue_turn_counts": dict(Counter(row["dialogue_turns"] for row in state_rows.values())),
        "files": {name: {"path": filenames[name], "rows": len(rows), "sha256": sha_file(output / filenames[name])} for name, rows in by_split.items()},
        "input_boundary": {
            "state_source": "pilot_public_states.initial_messages[user] + public dialogue messages only",
            "public_dialogue_actor_mapping_verified": "assistant/base -> Clinician question; user/patient -> Patient reply",
            "hidden_atomic_facts_read": False,
            "gold_read_for_input": False,
            "gold_used_only_as_assistant_target": True,
            "future_patient_replies_or_action_traces_in_input": False,
            "expert_advice_or_base_final_answers_in_input": False,
        },
        "holdout": {"selection_status": holdout["status"], "fresh_holdout_cases": 400, "source_row_overlap_with_dialogue_cases": 0, "gold_materialized": False},
        "target_protocol": "Expert Final Answer: A-D; assistant-only causal cross-entropy",
        "no_synthetic_or_rephrased_dialogue": True,
    }
    (output / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "case_split": manifest["case_split"], "total": manifest["total"], "dialogue_turn_counts": manifest["dialogue_turn_counts"], "holdout": manifest["holdout"], "manifest": str(output / "dataset_manifest.json")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
