#!/usr/bin/env python3
"""Select a private, replayable ReMeDi pilot from an already-downloaded file."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

QUESTION_RE = re.compile(
    r"[?？]|吗\s*[，。！？!? ]*$|(?:有没有|是否|什么时候|何时|多久|多少|哪里|哪儿|什么|怎么|如何|为什么|为何|请问|几天|几次|多大|多长)"
)


def normalize(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).lower()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_source(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("expected ReMeDi JSON array")
    return data


def parse_turns(doc: dict[str, Any]) -> tuple[list[dict[str, Any]] | None, str | None]:
    info = doc.get("information")
    if not isinstance(info, list) or not info:
        return None, "missing_or_empty_information"
    turns = []
    for i, item in enumerate(info):
        if not isinstance(item, dict):
            return None, "non_object_turn"
        role, text = item.get("role"), item.get("sentence")
        if role not in ("patient", "doctor"):
            return None, "unknown_role"
        if not isinstance(text, str) or not text.strip():
            return None, "empty_turn_text"
        turns.append({"turn_index": i, "role": role, "text": text})
    if turns[0]["role"] != "patient":
        return None, "first_turn_not_patient"
    if not any(t["role"] == "doctor" for t in turns[1:]):
        return None, "no_later_doctor_reply"
    return turns, None


def dialogue_fingerprint(turns: list[dict[str, Any]]) -> str:
    stable = [(t["role"], normalize(t["text"])) for t in turns]
    return hashlib.sha256(json.dumps(stable, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def replay_pairs(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs = []
    for i in range(len(turns) - 1):
        doctor, patient = turns[i], turns[i + 1]
        if doctor["role"] != "doctor" or patient["role"] != "patient":
            continue
        if not QUESTION_RE.search(doctor["text"]):
            continue
        pairs.append({
            "doctor_turn_index": doctor["turn_index"],
            "patient_turn_index": patient["turn_index"],
            "historical_clinician_question": doctor["text"],
            "linked_historical_patient_reply": patient["text"],
            "link_basis": "immediately_adjacent_doctor_question_then_patient_turn",
            "question_detection": "punctuation_or_chinese_interrogative_heuristic",
        })
    return pairs


def make_record(doc: dict[str, Any], source_order: int, turns: list[dict[str, Any]], pairs: list[dict[str, Any]], fingerprint: str) -> dict[str, Any]:
    doctors = [t for t in turns if t["role"] == "doctor"]
    return {
        "schema_version": 1,
        "sample_id": f"remedi_{doc['dialogue']}",
        "dataset": "ReMeDi",
        "source_dialogue_id": doc["dialogue"],
        "source_order_index": source_order,
        "dialogue_fingerprint_sha256": fingerprint,
        "initial_patient_question": turns[0]["text"],
        "messages": turns,
        "historical_clinician_response_references": [
            {
                "turn_index": t["turn_index"],
                "text": t["text"],
                "scope": "historical clinician message in the same source dialogue",
                "verified_expert_gold": False,
            }
            for t in doctors
        ],
        "ask_replay_pairs": pairs,
        "ask_replay_limit_reached": False,
        "data_semantics": {
            "historical_clinician_responses_are_verified_gold": False,
            "patient_reply_pairing_is_turn_adjacency_based": True,
            "ask_question_is_historical_clinician_question_proxy": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="must be a private directory")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--seed", type=str, default="20261121")
    args = parser.parse_args()
    if not 1 <= args.max_samples <= 100:
        raise ValueError("max-samples must be between 1 and 100")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite a non-empty output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(args.output_dir, 0o700)

    source = read_source(args.input)
    seen_fingerprints: set[str] = set()
    reject_counts: Counter[str] = Counter()
    duplicate_count = 0
    valid_base = 0
    ask_eligible: list[tuple[str, dict[str, Any], int, list[dict[str, Any]], list[dict[str, Any]]]] = []
    doctor_question_pair_count = 0
    for source_order, doc in enumerate(source):
        if not isinstance(doc, dict) or not isinstance(doc.get("dialogue"), (str, int)):
            reject_counts["missing_original_dialogue_id"] += 1
            continue
        turns, reason = parse_turns(doc)
        if reason:
            reject_counts[reason] += 1
            continue
        assert turns is not None
        fingerprint = dialogue_fingerprint(turns)
        if fingerprint in seen_fingerprints:
            duplicate_count += 1
            reject_counts["exact_duplicate_dialogue"] += 1
            continue
        seen_fingerprints.add(fingerprint)
        valid_base += 1
        pairs = replay_pairs(turns)
        if not pairs:
            reject_counts["no_adjacent_doctor_question_patient_reply_pair"] += 1
            continue
        doctor_question_pair_count += len(pairs)
        tie = hashlib.sha256(f"EA_ATPO-ReMeDi-pilot:{args.seed}:{doc['dialogue']}".encode()).hexdigest()
        ask_eligible.append((tie, doc, source_order, turns, pairs))

    ask_eligible.sort(key=lambda x: (x[0], str(x[1]["dialogue"])))
    selected = ask_eligible[: args.max_samples]
    records = [make_record(doc, order, turns, pairs, dialogue_fingerprint(turns)) for _, doc, order, turns, pairs in selected]

    samples_path = args.output_dir / "pilot_samples.jsonl"
    manifest_path = args.output_dir / "pilot_manifest.json"
    with samples_path.open("x", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.chmod(samples_path, stat.S_IRUSR | stat.S_IWUSR)
    manifest = {
        "schema_version": 1,
        "dataset": "ReMeDi",
        "source_file_name": args.input.name,
        "source_sha256": sha256(args.input),
        "selection": {
            "max_samples": args.max_samples,
            "selected_samples": len(records),
            "seed": args.seed,
            "sort_key": "sha256('EA_ATPO-ReMeDi-pilot:{seed}:{original_dialogue_id}') ascending",
            "requires_initial_patient_turn": True,
            "requires_later_doctor_reply": True,
            "requires_adjacent_heuristic_doctor_question_patient_reply": True,
            "deduplication": "NFKC + whitespace removal + lowercase, preserving role sequence",
        },
        "audit_counts": {
            "raw_dialogues": len(source),
            "structurally_valid_dialogues_before_exact_dedup": valid_base + duplicate_count,
            "unique_structurally_valid_dialogues": valid_base,
            "exact_duplicate_dialogues_excluded": duplicate_count,
            "unique_dialogues_with_eligible_ask_replay": len(ask_eligible),
            "eligible_adjacent_question_reply_pairs_before_selection": doctor_question_pair_count,
            "selected_ask_replay_pairs": sum(len(r["ask_replay_pairs"]) for r in records),
            "excluded_reasons": dict(sorted(reject_counts.items())),
        },
        "selected_source_ids": [r["source_dialogue_id"] for r in records],
        "sample_file": {
            "path": str(samples_path.resolve()),
            "sha256": sha256(samples_path),
            "bytes": samples_path.stat().st_size,
            "rows": len(records),
            "contains_private_dialogue_text": True,
        },
        "semantics": {
            "historical_doctor_text_is_reference_only_not_verified_expert_gold": True,
            "ask_replay_uses_only_an_adjacent_historical_doctor_question_and_patient_reply": True,
            "no_gold_or_dialogue_synthesis_performed": True,
        },
        "privacy": {
            "storage_class": "private local data; do not upload to public GitHub",
            "mode": "0600",
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.chmod(manifest_path, stat.S_IRUSR | stat.S_IWUSR)
    print(json.dumps({
        "status": "created",
        "samples": len(records),
        "ask_replay_pairs": sum(len(r["ask_replay_pairs"]) for r in records),
        "unique_eligible_dialogues": len(ask_eligible),
        "private_output_dir": str(args.output_dir.resolve()),
        "samples_sha256": sha256(samples_path),
        "manifest_sha256": sha256(manifest_path),
    }, indent=2))


if __name__ == "__main__":
    main()
