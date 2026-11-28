"""Build private ReMeDi state/action outcomes and public aggregate reports."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
from itertools import chain
import json
import math
from pathlib import Path
import re
import unicodedata

SPLITS = ("train", "dev", "test")
DIRECT = {"Diagnosis", "QuestionAnswering"}
ENTITY_SLOTS = {"disease", "disease_history", "symptom", "check_item", "medicine",
                "medicine_category", "range_body", "pathogeny", "side_effect", "treatment"}
UNSPECIFIED = {"", "none", "null", "nan", "unknown", "ns", "未知", "不详", "不确定", "?"}
THRESHOLD = 0.8


def read_lines(path):
    with path.open(encoding="utf-8") as handle:
        yield from map(json.loads, handle)


def normalize(value):
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value)).casefold()


def label_slot(label):
    slot = normalize(label.get("slot"))
    return slot if slot not in UNSPECIFIED | {"other"} else ""


def value_pair(label):
    return tuple(value if value not in UNSPECIFIED else ""
                 for value in (normalize(label.get("value1")), normalize(label.get("value2"))))


def annotations(group):
    return [label for turn in group["turns"] for label in turn["official_labels"]]


def facts(labels):
    return {(label_slot(a), *value_pair(a)) for a in labels
            if a.get("intent") == "Inform" and label_slot(a) and value_pair(a)[0]}


def visible(groups):
    return [{"role": "user" if g["role"] == "patient" else "assistant", "content": g["text"]}
            for g in groups]


def write_json(handle, row):
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def shingles(text):
    text = "".join(c for c in normalize(text) if c.isalnum())
    return {text[i:i + 5] for i in range(len(text) - 4)} if len(text) >= 5 else {text}


def audit_near(data, raw, pilot_ids):
    # Global frequency order and a Jaccard prefix filter, followed by exact scoring.
    entries = [(str(row["source_dialogue"]), split, shingles("".join(
        group["text"] for group in row["groups"]))) for split, rows in data.items() for row in rows]
    entries += [(str(row["dialogue"]), "pilot", shingles("".join(
        turn["sentence"] for turn in row["information"]))) for row in raw
        if str(row["dialogue"]) in pilot_ids]
    frequencies = Counter(chain.from_iterable(tokens for _, _, tokens in entries))
    index = defaultdict(list)
    pairs = Counter()
    flagged = set()
    for i, (cid, split, tokens) in enumerate(entries):
        prefix_length = len(tokens) - math.ceil(THRESHOLD * len(tokens)) + 1
        prefix = sorted(tokens, key=lambda t: (frequencies[t], t))[:prefix_length]
        for j in set(chain.from_iterable(index[t] for t in prefix)):
            other, other_split, other_tokens = entries[j]
            if cid == other or split == other_split:
                continue
            if min(len(tokens), len(other_tokens)) < THRESHOLD * max(len(tokens), len(other_tokens)):
                continue
            score = len(tokens & other_tokens) / len(tokens | other_tokens)
            if score >= THRESHOLD:
                kind = "+".join(sorted((split, other_split)))
                pairs[kind] += 1
                flagged.update((cid, other))
        for token in prefix:
            index[token].append(i)
    return flagged, dict(sorted(pairs.items()))


def response_reference(group):
    actions = set()
    slots = set()
    entities = set()
    matched = set()
    span_total = 0
    span_matched = 0
    for turn in group["turns"]:
        source = turn["source_text"]
        for label in turn["official_labels"]:
            if label.get("intent") not in DIRECT:
                continue
            actions.add(label["intent"])
            slot = label_slot(label)
            if slot:
                slots.add(slot)
            value = value_pair(label)[0]
            if value and slot in ENTITY_SLOTS:
                entities.add((slot, value))
                if value in normalize(source):
                    matched.add((slot, value))
            span = label.get("range")
            reference = normalize(label.get("text"))
            if (isinstance(span, list) and len(span) == 2 and all(type(x) is int for x in span)
                    and 0 <= span[0] <= span[1] <= len(source) and reference):
                span_total += 1
                start, end = span
                if reference in (normalize(source[start:end]), normalize(source[start:end + 1])):
                    span_matched += 1
    return actions, slots, entities, matched, span_total, span_matched


def ask_statistics(bucket, history, labels, feedback, later, prior_questions):
    targets = {(label_slot(a), *value_pair(a)) for a in labels
               if a.get("intent") == "Inquire" and label_slot(a)}
    slots = {target[0] for target in targets}
    known_slots = {fact[0] for fact in history}
    observed = facts(annotations(feedback)) if feedback else set()
    new = observed - history
    new_slots = {fact[0] for fact in new}
    matching = {fact[0] for fact in observed} & slots
    # Same-slot outcomes are measured explicitly. Generic slots do not imply a subject match.
    bucket["inquire_groups"] += 1
    bucket["feedback_present"] += feedback is not None
    bucket["feedback_absent"] += feedback is None
    bucket["target_evaluable"] += bool(slots)
    bucket["target_evaluable_with_feedback"] += bool(slots) and feedback is not None
    bucket["target_slot_obtained"] += bool(matching)
    bucket["target_slots_asked"] += len(slots)
    bucket["target_slots_obtained"] += len(matching)
    bucket["feedback_no_new_fact"] += feedback is not None and not new
    bucket["feedback_new_fact"] += bool(new)
    bucket["feedback_target_mismatch"] += feedback is not None and bool(slots) and not matching
    bucket["new_slot_value_tuples"] += len(new)
    bucket["new_information_slots_per_window_sum"] += len(new_slots)
    bucket["distinct_known_slots_before_sum"] += len(known_slots)
    bucket["distinct_known_slots_after_sum"] += len(known_slots | new_slots)
    bucket["previous_query_signature_repeated"] += bool(targets & prior_questions)
    bucket["asked_slot_already_known"] += bool(slots & known_slots)
    bucket["no_subsequent_doctor_group"] += later is None
    if later:
        bucket.update("subsequent_" + category for category in sorted({
            a.get("intent") or "missing" for a in annotations(later)
        } or {"unlabeled"}))
    return targets, observed, new, matching


def write_csv(path, rows):
    columns = ["split", "cohort"] + sorted(set(chain.from_iterable(rows)) - {"split", "cohort"})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row.get(key, 0) for key in columns} for row in rows)


def rates(bucket, definitions):
    for name, numerator, denominator in definitions:
        bucket[name] = round(bucket[numerator] / bucket[denominator], 6) if bucket[denominator] else ""


def build(source, destination):
    prepared = source / "prepared-with-spans"
    for line in (prepared / "SHA256SUMS").read_text().splitlines():
        expected, filename = line.split(maxsplit=1)
        if hashlib.sha256((prepared / filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f"checksum mismatch: {filename}")
    data = {split: list(read_lines(prepared / f"{split}-semantic.jsonl")) for split in SPLITS}
    raw = json.loads((source / "ReMeDi-base.json").read_text(encoding="utf-8"))
    pilot = {row["case_id"].removeprefix("remedi:") for row in read_lines(source / "historical-pilot-50.jsonl")}
    assert len(pilot) == 50
    assert all("patient_id" not in row for row in raw)
    flagged, duplicates = audit_near(data, raw, pilot)
    public = destination / "aggregate"
    private = destination / "private"
    public.mkdir(parents=True, exist_ok=False)
    private.mkdir(mode=0o700)
    outputs = {"dataset": [], "ask": [], "answer": []}
    seen = set()
    measured_replays = 0

    for split, dialogue_rows in data.items():
        cohorts = {name: {table: Counter() for table in outputs} for name in ("all_prepared", "eligible")}
        replay_index = set()
        for row in read_lines(prepared / f"{split}-replay-index.jsonl"):
            assert row["split"] == split and row["source_dialogue"] not in pilot
            replay_index.add((row["source_dialogue"], row["question_group"]))
        actual_replay = set()
        with ((private / f"{split}-behavior.jsonl").open("x", encoding="utf-8") as behavior,
              (private / f"{split}-observed-outcome.jsonl").open("x", encoding="utf-8") as outcome,
              (private / f"{split}-feature-input.jsonl").open("x", encoding="utf-8") as features,
              (private / f"{split}-counterfactual.jsonl").open("x", encoding="utf-8") as counterfactual):
            for row in dialogue_rows:
                cid = str(row["source_dialogue"])
                assert row["split"] == split and cid not in seen
                seen.add(cid)
                historical = cid in pilot
                assert historical == row["seen_in_historical_pilot"]
                eligible = not historical and cid not in flagged
                live = ["all_prepared", "eligible"] if eligible else ["all_prepared"]
                groups = row["groups"]
                history, prior_questions = set(), set()
                prior_diagnosis = False
                for name in live:
                    bucket = cohorts[name]["dataset"]
                    bucket["dialogues"] += 1
                    bucket["historical_pilot_dialogues"] += historical
                    bucket["near_duplicate_flagged_dialogues"] += cid in flagged
                for i, group in enumerate(groups):
                    labels = annotations(group)
                    if group["role"] == "patient":
                        history.update(facts(labels))
                        continue
                    if not i or groups[i - 1]["role"] != "patient":
                        continue
                    original = sorted({a.get("intent") or "missing" for a in labels} or {"unlabeled"})
                    inquire = "Inquire" in original
                    direct = bool(DIRECT.intersection(original))
                    normalized = "MIXED" if inquire and direct else "ASK" if inquire else "ANSWER" if direct else "OTHER"
                    children = groups[i + 1:i + 3]
                    feedback = children[0] if children and children[0]["role"] == "patient" else None
                    later = children[1] if feedback and len(children) == 2 and children[1]["role"] == "doctor" else None
                    context = visible(groups[:i])
                    key = {"dialogue_id": cid, "turn_id": i, "split": split}
                    state = {
                        **key, "context_before_action": context,
                        "current_slots": sorted({fact[0] for fact in history}),
                        "current_slot_values": sorted(history),
                        "original_doctor_action": original, "normalized_action": normalized,
                        "doctor_utterance": group["text"], "action_slot_values": labels,
                        "source": {"file": "ReMeDi-base.json", "processed_file": f"{split}_human_annotation.txt",
                                   "source_dialogue_id": cid, "doctor_source_turns": group["turns"],
                                   "official_group_index": i},
                        "eligible": eligible,
                    }
                    write_json(behavior, state)
                    feature_eligible = eligible and not prior_diagnosis
                    if feature_eligible:
                        write_json(features, {**key, "public_state": context})
                    prior_diagnosis |= "Diagnosis" in original
                    write_json(counterfactual, {**key, "ANSWER": None, "ASK": None, "DEFER": None,
                                                "q_value": None, "trainable": False})
                    for name in live:
                        bucket = cohorts[name]["dataset"]
                        bucket["doctor_states"] += 1
                        bucket["feature_inputs_exported"] += feature_eligible
                        bucket[normalized.lower()] += 1
                        bucket["direct_answer_reference"] += normalized == "ANSWER"
                        bucket["single_action_behavior_examples"] += normalized in {"ASK", "ANSWER"}
                    if inquire:
                        for name in live:
                            targets, observed, new, matching = ask_statistics(
                                cohorts[name]["ask"], history, labels, feedback, later, prior_questions
                            )
                            bucket = cohorts[name]["dataset"]
                            bucket["ask_feedback_available"] += feedback is not None
                            bucket["ask_new_info_available"] += bool(new)
                            bucket["ask_target_and_new_info"] += any(fact[0] in matching for fact in new)
                        prior_questions.update(targets)
                        if later and not historical:
                            actual_replay.add((cid, i))
                            for name in live:
                                cohorts[name]["dataset"]["indexed_replay_windows"] += 1
                        write_json(outcome, {
                            **key, "ask_target_slots": sorted({t[0] for t in targets}),
                            "patient_feedback": feedback,
                            "feedback_slot_values": sorted(observed),
                            "feedback_new_slot_values": sorted(new),
                            "target_slots_obtained": sorted(matching),
                            "context_after_feedback": visible(groups[:i + 2]) if feedback else None,
                            "subsequent_doctor_action": annotations(later) if later else None,
                            "subsequent_doctor_response": later["text"] if later else None,
                            "eligible": eligible,
                        })
                    if normalized == "ANSWER":
                        actions, slots, entities, matched, nspan, okspan = response_reference(group)
                        for name in live:
                            bucket = cohorts[name]["answer"]
                            bucket["answer_reference_groups"] += 1
                            bucket["independent_predictions_evaluated"] += 0
                            bucket["medical_correctness_evaluated"] += 0
                            bucket["action_reference_available"] += bool(actions)
                            bucket["slot_reference_available"] += bool(slots)
                            bucket["entity_reference_available"] += bool(entities)
                            bucket["annotated_medical_slot_value_pairs"] += len(entities)
                            bucket["literal_source_entity_matches"] += len(matched)
                            bucket["valid_reference_spans"] += nspan
                            bucket["source_span_matches"] += okspan
                            bucket.update("reference_" + action for action in sorted(actions))
                        write_json(outcome, {**key, "answer_reference_entities": sorted(entities),
                                            "answer_entities_matched_in_source": sorted(matched),
                                            "independent_prediction_metrics": None, "eligible": eligible})
        assert actual_replay == replay_index
        measured_replays += len(actual_replay)
        for name, tables in cohorts.items():
            tables["dataset"].update(defer_available=0, comparable_action_outcomes=0, q_guided_examples=0)
            for table, counts in tables.items():
                outputs[table].append({"split": split, "cohort": name, **counts})

    assert len(seen) == 1517 and measured_replays == 3506
    summaries = defaultdict(list)
    for table, rows in outputs.items():
        for name in ("all_prepared", "eligible"):
            total = Counter()
            for row in rows:
                if row["cohort"] == name:
                    total.update({key: value for key, value in row.items() if isinstance(value, int)})
            summaries[table].extend([{"split": "total", "cohort": name, **total}])
        rows.extend(summaries[table])

    specs = {
        "dataset": ("remedi_state_action_dataset_statistics.csv", [
            ("ask_feedback_coverage_over_all_states", "ask_feedback_available", "doctor_states"),
            ("answer_reference_coverage_over_all_states", "direct_answer_reference", "doctor_states"),
            ("defer_coverage_over_all_states", "defer_available", "doctor_states"),
        ]),
        "ask": ("remedi_ask_replay_analysis.csv", [
            ("feedback_coverage", "feedback_present", "inquire_groups"),
            ("target_slot_acquisition_rate_given_labeled_target_and_feedback", "target_slot_obtained", "target_evaluable_with_feedback"),
            ("target_slot_acquisition_rate_over_asked_slots", "target_slots_obtained", "target_slots_asked"),
            ("query_signature_repeat_rate_among_labeled_targets", "previous_query_signature_repeated", "target_evaluable"),
            ("known_slot_inquiry_rate_among_labeled_targets", "asked_slot_already_known", "target_evaluable"),
            ("mean_distinct_known_slots_before", "distinct_known_slots_before_sum", "inquire_groups"),
            ("mean_distinct_known_slots_after", "distinct_known_slots_after_sum", "inquire_groups"),
        ]),
        "answer": ("remedi_answer_structure_analysis.csv", [
            ("action_reference_coverage", "action_reference_available", "answer_reference_groups"),
            ("slot_reference_coverage", "slot_reference_available", "answer_reference_groups"),
            ("entity_reference_coverage", "entity_reference_available", "answer_reference_groups"),
            ("literal_entity_alignment_rate", "literal_source_entity_matches", "annotated_medical_slot_value_pairs"),
            ("source_span_alignment_rate", "source_span_matches", "valid_reference_spans"),
        ]),
    }
    for table, (filename, rules) in specs.items():
        for row in outputs[table]:
            rates(row, rules)
            if table == "answer":
                for metric in ("action_consistency", "slot_consistency",
                               "medical_entity_coverage", "medical_correctness"):
                    row[metric] = ""
        write_csv(public / filename, outputs[table])
    metadata = {"near_threshold": THRESHOLD, "near_pairs": duplicates,
                "near_flagged_prepared_ids": len(flagged & seen),
                "dialogue_ids_checked": len(seen), "prior_pilot_cases": len(pilot),
                "original_replay_windows_verified": measured_replays}
    (public / "audit_summary.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    with (private / "SHA256SUMS").open("x", encoding="utf-8") as manifest:
        for path in sorted(private.glob("*.jsonl")):
            manifest.write(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n")
    print(json.dumps(metadata))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    build(args.source, args.output)
