"""Restore ReMeDi's published supervision into a private local data bundle."""

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import hashlib
from itertools import groupby
import json
from pathlib import Path
import re

SPLITS = ("train", "dev", "test")
ROW = re.compile(
    r"<\|currentuser\|>\s*(.*?)\s*<\|endofcurrentuser\|>"
    r".*?<\|response\|>\s*(.*?)\s*<\|endofresponse\|>"
)


def normalize(text):
    return re.sub(r"\s+", "", text)


def groups(record):
    return [
        {"role": role, "text": "".join(turn["sentence"] for turn in turns),
         "turns": [{"source_turn": turn["turn"], "source_text": turn["sentence"],
                    "official_labels": turn["actions"]}
                   for turn in turns]}
        for role, run in groupby(record["information"], lambda turn: turn["role"])
        if (turns := list(run))
    ]


def raw_signature(dialogue):
    return tuple(
        (normalize(dialogue[i - 1]["text"]) if i and dialogue[i - 1]["role"] == "patient" else "",
         normalize(turn["text"]))
        for i, turn in enumerate(dialogue) if turn["role"] == "doctor"
    )


def split_signatures(path):
    for block in re.split(r"\n\s*\n", path.read_text(encoding="utf-8").strip()):
        rows = [ROW.search(line) for line in block.splitlines()]
        if not rows or any(row is None for row in rows):
            raise ValueError(f"invalid processed row in {path.name}")
        yield tuple((normalize(row[1]), normalize(row[2])) for row in rows)


def has_label(group, name):
    return any(label.get("intent") == name for turn in group["turns"] for label in turn["official_labels"])


def write_row(handle, row):
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare(source, previous, output):
    raw = json.loads((source / "ReMeDi-base.json").read_text(encoding="utf-8"))
    previous_ids = {
        json.loads(line)["case_id"].removeprefix("remedi:")
        for line in previous.read_text(encoding="utf-8").splitlines()
    }
    if len(previous_ids) != 50:
        raise ValueError("expected the 50 hash-verified prior pilot cases")

    raw_groups = [groups(row) for row in raw]
    source_matches = defaultdict(list)
    for index, dialogue in enumerate(raw_groups):
        source_matches[raw_signature(dialogue)].append(index)
    split_blocks = {s: list(split_signatures(source / f"{s}_human_annotation.txt")) for s in SPLITS}
    all_blocks = Counter(block for blocks in split_blocks.values() for block in blocks)
    history_ids_unmapped = set(previous_ids)
    stats = {}
    output.mkdir(mode=0o700, parents=False, exist_ok=False)

    with ExitStack() as files:
        for split, blocks in split_blocks.items():
            rows = files.enter_context((output / f"{split}-semantic.jsonl").open("x", encoding="utf-8"))
            replay = files.enter_context((output / f"{split}-replay-index.jsonl").open("x", encoding="utf-8"))
            counts = Counter(official_dialogues=len(blocks), official_doctor_rows=sum(map(len, blocks)))
            for block in blocks:
                matches = source_matches.get(block, [])
                if len(matches) != 1 or all_blocks[block] != 1:
                    counts["excluded_ambiguous_or_unmatched"] += 1
                    continue
                index = matches[0]
                dialogue = raw_groups[index]
                identifier = str(raw[index]["dialogue"])
                if any(turn["role"] not in ("doctor", "patient") or not turn["text"].strip()
                       for turn in dialogue):
                    counts["excluded_invalid"] += 1
                    continue
                historical = identifier in previous_ids
                history_ids_unmapped.discard(identifier)
                counts["prepared_dialogues"] += 1
                counts["prior_pilot_dialogues"] += historical
                counts["source_utterances"] += sum(len(group["turns"]) for group in dialogue)
                counts["doctor_groups_with_diagnosis"] += sum(
                    g["role"] == "doctor" and has_label(g, "Diagnosis") for g in dialogue
                )
                write_row(rows, {
                    "source_dialogue": identifier, "split": split, "groups": dialogue,
                    "seen_in_historical_pilot": historical,
                })
                for i in range(1, len(dialogue) - 2):
                    window = dialogue[i - 1:i + 3]
                    if [g["role"] for g in window] != ["patient", "doctor", "patient", "doctor"]:
                        continue
                    if not has_label(dialogue[i], "Inquire"):
                        continue
                    counts["labeled_replay_windows"] += 1
                    if historical:
                        continue
                    counts["new_replay_windows"] += 1
                    write_row(replay, {
                        "source_dialogue": identifier, "split": split, "question_group": i,
                        "historical_feedback_group": i + 1, "observed_next_doctor_group": i + 2,
                        "defer_reference": None, "generated_answer_quality_score": None,
                    })
            stats[split] = dict(counts)

    aggregate = {
        "raw_dialogues": len(raw),
        "raw_utterances": sum(len(d["information"]) for d in raw),
        "raw_utterances_with_labels": sum(bool(t["actions"]) for d in raw for t in d["information"]),
        "official_label_counts": dict(Counter(
            a.get("intent") or "(missing)" for d in raw for t in d["information"] for a in t["actions"]
        )),
        "splits": stats, "prior_pilot_excluded_from_replay": len(previous_ids),
        "prior_pilot_not_in_prepared_unique_mapping": len(history_ids_unmapped),
    }
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    with (output / "SHA256SUMS").open("x", encoding="utf-8") as manifest:
        for path in sorted(output.glob("*.jsonl")):
            manifest.write(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--previous-pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.previous_pilot, args.output)
