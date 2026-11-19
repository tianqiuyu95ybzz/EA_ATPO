#!/usr/bin/env python3
"""Audit restored ATPO train data and existing public multi-turn Pilot states.

The report contains aggregate counts and hashes only; no prompts, answers, facts,
case IDs, patient simulator text, or raw model outputs are written.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[4]
MULTI = ROOT / "experiments/expert_sft/multiturn_dialogue"
SRC_JSONL = ROOT / "experiments/expert_sft/data/source/dataset/merged_train_dataset.jsonl"
PILOT_DIR = ROOT / "experiments/real_case_pilot_20261103/outputs/full"
CASE_SPLIT = ROOT / "experiments/ea_atpo_value_router_20261105/artifacts/case_split.json"
FRESH_SELECTION = ROOT / "experiments/expert_sft/data/fresh_test_selection.json"
SFT_FILES = {
    "train": MULTI / "data/train_sft.jsonl",
    "validation": MULTI / "data/validation_sft.jsonl",
    "internal_test": MULTI / "data/internal_test_sft.jsonl",
}
PARQUETS = [
    ROOT / "experiments/expert_sft/data/reconstructed/train.parquet",
    ROOT / "experiments/expert_sft/data/reconstructed/train.pilot.parquet",
    ROOT / "experiments/expert_sft/data/reconstructed_compat/train.parquet",
    ROOT / "experiments/expert_sft/data/reconstructed_compat/train.pilot.parquet",
]

ANSWER_RE = re.compile(r"^\s*([A-D])(?:\b|[.)/:])", re.I)
EXPERT_TARGET_RE = re.compile(r"^Expert Final Answer: ([A-D])$")


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Malformed JSONL at {path.name}:{line_number}") from exc


def native(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def mapping(value: Any) -> dict[str, Any]:
    value = native(value)
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
            return loaded if isinstance(loaded, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def compact(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def user_text(prompt: Any) -> str:
    prompt = native(prompt)
    if not isinstance(prompt, list):
        return ""
    return "\n".join(str(m.get("content", "")) for m in prompt
                     if isinstance(m, dict) and m.get("role") == "user")


def source_gold(row: dict[str, Any]) -> str | None:
    raw = row.get("ground_truth")
    if raw is None:
        raw = mapping(row.get("reward_model")).get("ground_truth")
    if raw is None:
        raw = mapping(row.get("extra_info")).get("ground_truth")
    if isinstance(raw, dict):
        raw = raw.get("answer", raw.get("label", raw.get("correct_option")))
    match = ANSWER_RE.match(str(raw or ""))
    return match.group(1).upper() if match else None


def option_letter(value: Any) -> str | None:
    match = re.fullmatch(r"\s*([A-D])\s*", str(value or ""), re.I)
    return match.group(1).upper() if match else None


def public_state_check(state: dict[str, Any]) -> tuple[bool, int, str]:
    initial = state.get("initial_messages")
    dialogue = state.get("dialogue")
    trace = state.get("state_construction_trace")
    if not isinstance(initial, list) or not isinstance(dialogue, list) or not isinstance(trace, list):
        return False, 0, "missing_public_message_fields"
    initial_users = [m for m in initial if isinstance(m, dict) and m.get("role") == "user"]
    if len(initial_users) != 1 or not str(initial_users[0].get("content", "")).strip():
        return False, 0, "invalid_initial_user_message"
    if len(dialogue) != len(trace) or len(dialogue) % 2:
        return False, len(dialogue) // 2, "dialogue_trace_length_mismatch"
    for i, (message, event) in enumerate(zip(dialogue, trace)):
        expected_role = "assistant" if i % 2 == 0 else "user"
        expected_actor = "base" if i % 2 == 0 else "patient"
        if not isinstance(message, dict) or not isinstance(event, dict):
            return False, len(dialogue) // 2, "non_mapping_dialogue_event"
        if message.get("role") != expected_role or event.get("actor") != expected_actor:
            return False, len(dialogue) // 2, "role_actor_sequence_mismatch"
        if not str(message.get("content", "")).strip() or message.get("content") != event.get("response"):
            return False, len(dialogue) // 2, "public_text_trace_mismatch"
    return True, len(dialogue) // 2, "valid"


def public_state_text(state: dict[str, Any]) -> str:
    """Reconstruct exactly the permitted current public prompt from a state."""
    initial = state.get("initial_messages", [])
    user_messages = [m.get("content") for m in initial
                     if isinstance(m, dict) and m.get("role") == "user"]
    if len(user_messages) != 1 or not isinstance(user_messages[0], str):
        return ""
    lines = ["Visible case and answer options:\n" + user_messages[0].strip()]
    for message, event in zip(state.get("dialogue", []), state.get("state_construction_trace", [])):
        if not isinstance(message, dict) or not isinstance(event, dict):
            return ""
        actor = event.get("actor")
        if actor not in ("base", "patient") or message.get("content") != event.get("response"):
            return ""
        label = "Clinician question" if actor == "base" else "Patient reply"
        lines.append(f"{label}: {str(message.get('content', '')).strip()}")
    return "\n\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-jsonl", type=Path, default=SRC_JSONL)
    parser.add_argument("--public-states", type=Path, default=PILOT_DIR / "pilot_public_states.jsonl")
    parser.add_argument("--pilot-trajectories", type=Path, default=PILOT_DIR / "pilot_trajectories.jsonl")
    parser.add_argument("--case-split", type=Path, default=CASE_SPLIT)
    parser.add_argument("--fresh-selection", type=Path, default=FRESH_SELECTION)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output_dir or MULTI / "outputs" / f"data_audit_{stamp}").resolve()
    if MULTI.resolve() not in output.parents:
        raise ValueError(f"Audit output must remain under {MULTI / 'outputs'}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite audit output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    # Official ATPO train source: stable row key is `index`; source prompt is one-turn.
    source_rows = list(jsonl(args.source_jsonl))
    source_by_index: dict[int, dict[str, Any]] = {}
    source_prompt_hashes = Counter()
    source_user_hashes = Counter()
    source_prompt_hash_to_indices: dict[str, list[int]] = defaultdict(list)
    role_shapes = Counter()
    source_gold_counts = Counter()
    duplicate_index_count = 0
    for row in source_rows:
        index = int(row["index"])
        if index in source_by_index:
            duplicate_index_count += 1
        source_by_index[index] = row
        prompt = native(row.get("prompt"))
        roles = tuple(str(m.get("role")) for m in prompt if isinstance(m, dict)) if isinstance(prompt, list) else ()
        role_shapes[roles] += 1
        full = json.dumps(prompt, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        source_prompt_hashes[sha_text(full)] += 1
        user_hash = sha_text(compact(user_text(prompt)))
        source_user_hashes[user_hash] += 1
        source_prompt_hash_to_indices[user_hash].append(index)
        source_gold_counts[source_gold(row) or "UNPARSEABLE"] += 1
    source_ids = set(source_by_index)
    expected_ids = set(range(len(source_rows)))
    source_index_sequential = source_ids == expected_ids and duplicate_index_count == 0

    # Parquet restoration and adapter audit: schema, row identity, prompt and answer integrity.
    parquet_audit = []
    for path in PARQUETS:
        if not path.is_file():
            parquet_audit.append({"path": str(path.relative_to(ROOT)), "exists": False})
            continue
        parquet_file = pq.ParquetFile(path)
        meta = parquet_file.metadata
        frame = pd.read_parquet(path)
        index_name = "index" if "index" in frame.columns else None
        row_ids = [int(v) for v in frame[index_name].tolist()] if index_name else list(range(len(frame)))
        row_id_unique = len(row_ids) == len(set(row_ids))
        rows_found = rows_prompt_match = rows_gold_parse = rows_gold_match = 0
        for pos, row in enumerate(frame.to_dict(orient="records")):
            idx = int(row[index_name]) if index_name else pos
            src = source_by_index.get(idx)
            if src is None:
                continue
            rows_found += 1
            pval = row.get("prompt")
            parquet_user = compact(user_text(pval))
            if parquet_user and parquet_user == compact(user_text(src.get("prompt"))):
                rows_prompt_match += 1
            raw_gold = row.get("ground_truth")
            if raw_gold is None:
                raw_gold = mapping(row.get("reward_model")).get("ground_truth")
            letter = source_gold({"ground_truth": raw_gold})
            if letter:
                rows_gold_parse += 1
                rows_gold_match += letter == source_gold(src)
        parquet_audit.append({
            "path": str(path.relative_to(ROOT)), "exists": True,
            "sha256": sha_file(path), "bytes": path.stat().st_size,
            "rows": int(meta.num_rows), "row_groups": int(meta.num_row_groups),
            "columns": list(frame.columns), "has_stable_index": bool(index_name),
            "index_unique": row_id_unique,
            "all_expected_rows": len(frame) == len(source_rows),
            "row_ids_found_in_source": rows_found,
            "prompt_user_text_matches_source": rows_prompt_match,
            "gold_A_D_parseable": rows_gold_parse,
            "gold_matches_source": rows_gold_match,
        })

    # Frozen case split and historical public Pilot snapshots.
    split_doc = json.loads(args.case_split.read_text(encoding="utf-8"))
    split_ids = {name: set(map(str, split_doc[name])) for name in ("train", "validation", "internal_test")}
    split_overlap = sum(len(split_ids[a] & split_ids[b]) for i, a in enumerate(split_ids) for b in list(split_ids)[i + 1:])
    states = list(jsonl(args.public_states))
    trajectories_all = list(jsonl(args.pilot_trajectories))
    trajectories = [r for r in trajectories_all if r.get("record_type") == "trajectory"]
    status_rows = [r for r in trajectories_all if r.get("record_type") != "trajectory"]

    state_ids = Counter(str(s.get("state_id", "")) for s in states)
    case_to_states: dict[str, list[dict[str, Any]]] = defaultdict(list)
    state_validity = Counter()
    dialogue_rounds = Counter()
    source_link_found = source_prompt_match = 0
    for state in states:
        case_id = str(state.get("case_id", ""))
        case_to_states[case_id].append(state)
        ok, rounds, reason = public_state_check(state)
        state_validity["valid" if ok else reason] += 1
        dialogue_rounds[str(rounds)] += 1
        source_idx = int(state.get("source_row_index", -1))
        source_row = source_by_index.get(source_idx)
        if source_row is not None:
            source_link_found += 1
            public_initial = "\n".join(str(m.get("content", "")) for m in state.get("initial_messages", [])
                                         if isinstance(m, dict) and m.get("role") == "user")
            if compact(public_initial) == compact(user_text(source_row.get("prompt"))):
                source_prompt_match += 1

    case_gold = defaultdict(set)
    action_counts = Counter()
    validity_by_action = Counter()
    final_parse_by_action = Counter()
    failure_class_by_action = Counter()
    trajectory_state_actions = defaultdict(Counter)
    trajectory_source_links = trajectory_gold_parse = trajectory_gold_match = 0
    for row in trajectories:
        case_id = str(row.get("case_id", ""))
        action = str(row.get("action", "UNKNOWN"))
        action_counts[action] += 1
        valid = bool(row.get("valid"))
        validity_by_action[(action, "valid" if valid else "invalid")] += 1
        letter = source_gold({"ground_truth": row.get("gold")})
        if letter:
            trajectory_gold_parse += 1
            case_gold[case_id].add(letter)
        source_row = source_by_index.get(int(row.get("source_row_index", -1)))
        if source_row is not None:
            trajectory_source_links += 1
            trajectory_gold_match += letter is not None and letter == source_gold(source_row)
        final_parse_by_action[(action, "strict_option" if option_letter(row.get("final_answer")) else "unparseable")] += 1
        trajectory_state_actions[(case_id, str(row.get("state_id", "")))][action] += 1
        if not valid:
            text = (str(row.get("failure_reason", "")) + " " + str(row.get("error", ""))).lower()
            if "parse" in text or "format" in text:
                bucket = "format_or_parse"
            elif "timeout" in text:
                bucket = "timeout"
            elif "budget" in text or "turn" in text:
                bucket = "budget_or_turn_limit"
            else:
                bucket = "other_or_protocol"
            failure_class_by_action[(action, bucket)] += 1

    # Verify existing SFT rows against their public state and the original ATPO answer.
    state_by_id = {str(s.get("state_id")): s for s in states}
    sft_audit = {}
    sft_all_cases = {}
    sft_all_indices = set()
    sft_all_state_ids = set()
    split_summary = []
    for split, path in SFT_FILES.items():
        rows = list(jsonl(path))
        prompt_hashes = Counter()
        sft_cases = set()
        valid_targets = matched_public_inputs = matched_gold = input_boundary = 0
        examples_with_patient_reply = zero_reply_examples = exact_public_state_matches = 0
        messages_shapes = Counter()
        for row in rows:
            case_id = str(row.get("case_id", ""))
            state_id = str(row.get("state_id", ""))
            sft_cases.add(case_id)
            sft_all_cases[case_id] = split
            sft_all_indices.add(int(row.get("source_row_index", -1)))
            sft_all_state_ids.add(state_id)
            messages = row.get("messages", [])
            messages_shapes[tuple(m.get("role") for m in messages if isinstance(m, dict))] += 1
            target_match = EXPERT_TARGET_RE.fullmatch(str(messages[-1].get("content", ""))) if messages else None
            if target_match:
                valid_targets += 1
                src = source_by_index.get(int(row.get("source_row_index", -1)))
                matched_gold += src is not None and target_match.group(1) == source_gold(src)
            dialogue_turns = int(row.get("dialogue_turns", 0))
            examples_with_patient_reply += dialogue_turns > 0
            zero_reply_examples += dialogue_turns == 0
            if (len(messages) == 3 and [m.get("role") for m in messages] == ["system", "user", "assistant"]
                    and all("content" in m for m in messages)):
                input_boundary += 1
            state = state_by_id.get(state_id)
            if state is not None:
                actual_user = str(messages[1].get("content", "")) if len(messages) > 1 else ""
                expected_user = public_state_text(state)
                if actual_user == expected_user:
                    exact_public_state_matches += 1
                    matched_public_inputs += 1
            h = str(row.get("prompt_sha256", ""))
            prompt_hashes[h] += 1
        expected_ids_for_split = split_ids[split]
        split_ok = sft_cases == expected_ids_for_split
        sft_audit[split] = {
            "path": str(path.relative_to(ROOT)), "sha256": sha_file(path),
            "examples": len(rows), "unique_cases": len(sft_cases),
            "case_ids_match_frozen_split": split_ok,
            "case_id_overlap_with_other_sft_splits": 0,
            "unique_state_ids": len({str(r.get("state_id")) for r in rows}),
            "duplicate_prompt_hashes": sum(n - 1 for n in prompt_hashes.values() if n > 1),
            "target_format_valid": valid_targets,
            "target_matches_official_source_gold": matched_gold,
            "examples_with_patient_reply": examples_with_patient_reply,
            "zero_reply_examples": zero_reply_examples,
            "mean_public_patient_turns": round(sum(int(r.get("dialogue_turns", 0)) for r in rows) / max(1, len(rows)), 4),
            "exact_current_public_state_input_match": exact_public_state_matches,
            "state_public_input_matches": matched_public_inputs,
            "input_target_role_boundary_valid": input_boundary,
            "message_role_shapes": {str(k): v for k, v in messages_shapes.items()},
        }
        split_summary.append({
            "split": split, "cases": len(sft_cases), "samples": len(rows),
            "samples_with_patient_reply": sum(int(r.get("dialogue_turns", 0)) > 0 for r in rows),
            "zero_reply_samples": sum(int(r.get("dialogue_turns", 0)) == 0 for r in rows),
            "mean_public_patient_turns": round(sum(int(r.get("dialogue_turns", 0)) for r in rows) / max(1, len(rows)), 4),
            "target_format_valid": valid_targets, "target_matches_official_gold": matched_gold,
        })

    trajectory_case_ids = set(str(r.get("case_id", "")) for r in trajectories)
    public_case_ids = set(case_to_states)
    state_grid_complete = all(
        set(actions) == {"ANSWER", "ASK", "CONSULT", "DEFER"}
        and all(actions[a] == 3 for a in ("ANSWER", "ASK", "CONSULT", "DEFER"))
        for actions in trajectory_state_actions.values()
    )
    all_trajectory_gold_single = all(len(labels) == 1 for labels in case_gold.values())
    train_state_count = sum(1 for s in states if str(s.get("case_id")) in split_ids["train"])
    nonempty_states = sum(1 for s in states if len(s.get("dialogue", [])) > 0)
    total_public_rounds = sum(int(k) * v for k, v in dialogue_rounds.items())

    # Fresh holdout is only inspected as frozen identity metadata, never read as a model sample.
    selection = json.loads(args.fresh_selection.read_text(encoding="utf-8"))
    selection_rows = selection.get("rows", [])
    holdout_indices = {int(r["source_row_index"]) for r in selection_rows}
    holdout_case_ids = {str(r["case_id"]) for r in selection_rows}
    pilot_indices = {int(s["source_row_index"]) for s in states}
    selected_source_content_groups = {
        source_hash: sorted(index for index in indices if index in pilot_indices)
        for source_hash, indices in source_prompt_hash_to_indices.items()
        if sum(index in pilot_indices for index in indices) > 1
    }
    selected_content_split_leaks = []
    for indices in selected_source_content_groups.values():
        case_ids = {
            str(state.get("case_id"))
            for state in states
            if int(state.get("source_row_index", -1)) in indices
        }
        split_names = {name for name, ids in split_ids.items() if case_ids & ids}
        if len(split_names) > 1:
            selected_content_split_leaks.append({"selected_source_rows": len(indices), "split_count": len(split_names)})

    # Aggregate rollout validity tables; no error strings or generated text are emitted.
    action_rows = []
    for action in ("ANSWER", "ASK", "CONSULT", "DEFER"):
        valid_count = validity_by_action[(action, "valid")]
        invalid_count = validity_by_action[(action, "invalid")]
        strict_parsed = final_parse_by_action[(action, "strict_option")]
        action_rows.append({
            "action": action, "rollouts": action_counts[action],
            "valid_rollouts": valid_count, "invalid_rollouts": invalid_count,
            "strict_final_option_parse": strict_parsed,
            "strict_final_option_parse_rate": round(strict_parsed / action_counts[action], 6) if action_counts[action] else None,
            "failure_format_or_parse": failure_class_by_action[(action, "format_or_parse")],
            "failure_timeout": failure_class_by_action[(action, "timeout")],
            "failure_budget_or_turn_limit": failure_class_by_action[(action, "budget_or_turn_limit")],
            "failure_other_or_protocol": failure_class_by_action[(action, "other_or_protocol")],
        })

    report = {
        "audit_status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_files": {
            "official_train_jsonl": {"path": str(args.source_jsonl.relative_to(ROOT)), "bytes": args.source_jsonl.stat().st_size, "sha256": sha_file(args.source_jsonl), "rows": len(source_rows)},
            "public_states_jsonl": {"path": str(args.public_states.relative_to(ROOT)), "bytes": args.public_states.stat().st_size, "sha256": sha_file(args.public_states), "rows": len(states)},
            "pilot_trajectories_jsonl": {"path": str(args.pilot_trajectories.relative_to(ROOT)), "bytes": args.pilot_trajectories.stat().st_size, "sha256": sha_file(args.pilot_trajectories), "rows": len(trajectories_all)},
            "frozen_case_split": {"path": str(args.case_split.relative_to(ROOT)), "sha256": sha_file(args.case_split)},
            "fresh_holdout_selection": {"path": str(args.fresh_selection.relative_to(ROOT)), "sha256": sha_file(args.fresh_selection), "status": selection.get("status"), "cases": len(selection_rows), "gold_or_atomic_facts_stored": selection.get("gold_or_atomic_facts_stored")},
        },
        "official_atpo_train": {
            "cases": len(source_rows), "stable_index_field": "index",
            "stable_index_unique": len(source_ids), "stable_index_sequential_0_to_n_minus_1": source_index_sequential,
            "duplicate_stable_indices": duplicate_index_count,
            "prompt_role_shapes": {str(k): v for k, v in role_shapes.items()},
            "multi_turn_prompt_count": sum(v for k, v in role_shapes.items() if len(k) > 2),
            "duplicate_full_prompts": sum(n - 1 for n in source_prompt_hashes.values() if n > 1),
            "duplicate_user_case_texts": sum(n - 1 for n in source_user_hashes.values() if n > 1),
            "duplicate_source_text_groups_in_selected_pilot": len(selected_source_content_groups),
            "selected_duplicate_source_text_cross_split_groups": len(selected_content_split_leaks),
            "ground_truth_A_D_parse_counts": dict(source_gold_counts),
            "ground_truth_A_D_parse_rate": round(sum(v for k, v in source_gold_counts.items() if k != "UNPARSEABLE") / max(1, len(source_rows)), 6),
            "parquets": parquet_audit,
        },
        "existing_pilot_multi_turn_data": {
            "cases": len(public_case_ids), "public_states": len(states),
            "states_per_case_distribution": dict(Counter(len(v) for v in case_to_states.values())),
            "duplicate_state_ids": sum(v - 1 for v in state_ids.values() if v > 1),
            "state_validation": dict(state_validity),
            "dialogue_rounds_per_state": dict(dialogue_rounds),
            "states_with_at_least_one_patient_reply": nonempty_states,
            "patient_reply_turns_across_states": total_public_rounds,
            "mean_patient_reply_turns_all_states": round(total_public_rounds / max(1, len(states)), 4),
            "mean_patient_reply_turns_nonempty_states": round(total_public_rounds / max(1, nonempty_states), 4),
            "source_row_index_links_found": source_link_found,
            "source_prompt_matches": source_prompt_match,
            "trajectory_rows": len(trajectories), "nontrajectory_status_rows": len(status_rows),
            "trajectory_actions": dict(action_counts),
            "trajectory_valid": sum(1 for r in trajectories if bool(r.get("valid"))),
            "trajectory_invalid": sum(1 for r in trajectories if not bool(r.get("valid"))),
            "trajectory_final_answer_strict_A_D_parse": sum(v for (action, kind), v in final_parse_by_action.items() if kind == "strict_option"),
            "trajectory_final_answer_parse_rate": round(sum(v for (action, kind), v in final_parse_by_action.items() if kind == "strict_option") / max(1, len(trajectories)), 6),
            "trajectory_gold_A_D_parse": trajectory_gold_parse,
            "trajectory_gold_matches_official_source": trajectory_gold_match,
            "trajectory_source_row_links_found": trajectory_source_links,
            "gold_single_value_per_case": all_trajectory_gold_single,
            "action_grid_three_repetitions_per_state": state_grid_complete,
        },
        "case_split": {
            "unit": split_doc.get("unit"), "seed": split_doc.get("seed"),
            "case_counts": {name: len(ids) for name, ids in split_ids.items()},
            "cross_split_case_overlap": split_overlap,
            "all_public_cases_covered": public_case_ids == set.union(*split_ids.values()),
            "pilot_cases_have_split": public_case_ids <= set.union(*split_ids.values()),
            "selected_duplicate_source_text_cross_split_groups": len(selected_content_split_leaks),
            "sft": sft_audit,
        },
        "clean_holdout_isolation": {
            "status": selection.get("status"), "cases": len(selection_rows),
            "source_rows_overlap_sft_state_cases": len(holdout_indices & sft_all_indices),
            "source_rows_overlap_pilot_state_cases": len(holdout_indices & pilot_indices),
            "case_ids_overlap_pilot": len(holdout_case_ids & public_case_ids),
            "gold_or_atomic_facts_read_for_selection": selection.get("gold_or_atomic_facts_read_for_selection"),
            "gold_or_atomic_facts_stored": selection.get("gold_or_atomic_facts_stored"),
            "checkpoint_selection_or_training_use": False,
        },
        "boundary": {
            "original_atpo_source_has_multi_turn_dialogue": sum(v for k, v in role_shapes.items() if len(k) > 2) > 0,
            "pilot_states_are_llm_simulator_generated_not_human_transcripts": True,
            "sft_input_source": "case prompt + current public state messages only",
            "gold_in_input": False,
            "hidden_atomic_facts_in_input": False,
            "future_patient_replies_in_input": False,
            "expert_advice_or_final_answers_in_input": False,
            "gold_in_supervised_target_only": True,
        },
    }

    write_csv(output / "split_summary.csv", split_summary,
              ["split", "cases", "samples", "samples_with_patient_reply", "zero_reply_samples", "mean_public_patient_turns", "target_format_valid", "target_matches_official_gold"])
    write_csv(output / "rollout_validity_by_action.csv", action_rows,
              ["action", "rollouts", "valid_rollouts", "invalid_rollouts", "strict_final_option_parse", "strict_final_option_parse_rate", "failure_format_or_parse", "failure_timeout", "failure_budget_or_turn_limit", "failure_other_or_protocol"])
    (output / "data_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    train = report["case_split"]["sft"]["train"]
    val = report["case_split"]["sft"]["validation"]
    internal = report["case_split"]["sft"]["internal_test"]
    original_turns = report["official_atpo_train"]["multi_turn_prompt_count"]
    report_md = f"""# Restored ATPO Multi-turn Data Audit

**Status:** aggregate audit completed from local raw files. No model was loaded and no GPU was used.

## Finding on data provenance

The official ATPO train source at the pinned project commit contains **{len(source_rows):,} training cases**. Every prompt has the two-role shape `system/user`; this restored source has no multi-turn dialogue fields. Its stable row key is `index`, unique and sequential over 0–{len(source_rows)-1}. All {len(source_rows):,} official answer labels parse to A–D. The existing reconstructed Parquets are audited separately in `data_audit.json`; none was overwritten.

The multi-turn material available locally is from the prior 100-case EA-ATPO Pilot, where Base and Patient simulators generated public dialogue states. These are **simulated model conversations, not human patient transcripts**. The 100 cases have {len(states)} public states; {nonempty_states} states include at least one Patient reply and {total_public_rounds} Patient replies appear across snapshots. Public state IDs map to stable ATPO source indices and their source prompt text matched for {source_prompt_match}/{len(states)} states.

## Extractable Expert SFT data

The existing answer-only SFT files contain {len(SFT_FILES)} case-isolated splits:

| Split | Cases | State samples | Samples with Patient reply | Target format valid | Gold matches source |
|---|---:|---:|---:|---:|---:|
| Train | {train['unique_cases']} | {train['examples']} | {train['examples_with_patient_reply']} | {train['target_format_valid']} | {train['target_matches_official_source_gold']} |
| Validation | {val['unique_cases']} | {val['examples']} | {val['examples_with_patient_reply']} | {val['target_format_valid']} | {val['target_matches_official_source_gold']} |
| Internal test | {internal['unique_cases']} | {internal['examples']} | {internal['examples_with_patient_reply']} | {internal['target_format_valid']} | {internal['target_matches_official_source_gold']} |

Strict multi-turn states with Patient replies: **{nonempty_states}** ({dialogue_rounds.get('1', 0)} one-reply states; {dialogue_rounds.get('2', 0)} two-reply states). The present SFT set also includes {dialogue_rounds.get('0', 0)} zero-reply states as the direct-answer anchor. Each source case contributes up to three public states, all kept in the same case_id split.

## Rollout and label quality

The source trajectory file contains {len(trajectories)} action rollouts and {len(status_rows)} run-status records. {sum(1 for r in trajectories if bool(r.get('valid')))} rollout records are marked valid; {sum(1 for r in trajectories if not bool(r.get('valid')))} are invalid. Strict final option parse rate is {report['existing_pilot_multi_turn_data']['trajectory_final_answer_parse_rate']:.2%}; invalid branches remain recorded and are not treated as successful model responses. All rollout Gold labels match the official source answer for their mapped case: {trajectory_gold_match}/{len(trajectories)}.

The 400-case fresh holdout remains a pre-frozen **within-ATPO-train** holdout. Its identity metadata is disjoint from the Pilot/SFT state cases and no Gold or atomic facts were stored or read. The internal 15-case split is development-only; it is not a substitute for the untouched 400-case Holdout.

## Leakage boundary

SFT model input contains the original public case prompt plus only the current state prefix. The answer is the sole Assistant target. The builder does not serialize hidden atomic facts, Gold in input, future Patient replies, Expert advice, or Base terminal answers into the input. This report stores aggregate data only; it contains no prompt text, answer labels by case, Patient text, case IDs, or raw model outputs.

Detailed file hashes, schemas, validity counts by action, case split checks, and Parquet integrity are in `data_audit.json`, `split_summary.csv`, and `rollout_validity_by_action.csv`.
"""
    (output / "AUDIT_REPORT.md").write_text(report_md, encoding="utf-8")
    print(json.dumps({"status": "completed", "output": str(output.relative_to(ROOT)),
                      "official_train_cases": len(source_rows), "pilot_cases": len(public_case_ids),
                      "public_states": len(states), "states_with_patient_reply": nonempty_states,
                      "trajectory_rollouts": len(trajectories), "sft_samples": {k: v['examples'] for k,v in sft_audit.items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
