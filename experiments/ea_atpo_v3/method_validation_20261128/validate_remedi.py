"""Run bounded ReMeDi Train/Dev ASK and ANSWER checks. Test is never opened."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import random
from time import perf_counter

from build_remedi_state_action import (
    label_slot, normalize, read_lines, response_reference, value_pair,
)

SPLITS = ("train", "dev")
SEED = 20261128
SLOT_NAMES = (
    "symptom", "disease", "disease_history", "medicine", "medicine_category", "check_item",
    "treatment", "range_body", "pathogeny", "side_effect", "precaution", "time", "frequency",
    "degree", "effect", "dose", "temperature", "medical_place", "department",
)


def key(row):
    return row["dialogue_id"], row["turn_id"]


def rank(row):
    return hashlib.sha256(f"{SEED}:{row['dialogue_id']}:{row['turn_id']}".encode()).digest()


def dump_csv(path, rows):
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def rate(n, d):
    return round(n / d, 6) if d else ""


def ask_report(behavior, outcomes):
    by_slot = defaultdict(Counter)
    by_result = defaultdict(Counter)
    prior = defaultdict(set)
    for row in behavior:
        if not row["eligible"]:
            continue
        normalized = row["normalized_action"]
        if normalized not in ("ASK", "MIXED"):
            continue
        cohort = (row["split"], row["dialogue_id"])
        outcome = outcomes[key(row)]
        new = outcome["feedback_new_slot_values"]
        observed = outcome["feedback_slot_values"]
        targets = defaultdict(set)
        for label in row["action_slot_values"]:
            if label.get("intent") == "Inquire" and (slot := label_slot(label)):
                targets[slot].add((slot, *value_pair(label)))
        prior_questions = prior[cohort]
        for slot, signatures in targets.items():
            if normalized == "ASK":
                bucket = by_slot[row["split"], slot]
                bucket["asked"] += 1
                bucket["feedback"] += outcome["patient_feedback"] is not None
                bucket["target_observed"] += any(fact[0] == slot for fact in observed)
                bucket["target_new"] += any(fact[0] == slot for fact in new)
                bucket["repeat_signature"] += bool(signatures & prior_questions)
                bucket["previously_known_slot"] += slot in row["current_slots"]
            prior_questions.update(signatures)
        if normalized != "ASK":
            continue
        if outcome["patient_feedback"] is None:
            result = "no_feedback"
        elif any(fact[0] in targets for fact in new):
            result = "target_new"
        elif new:
            result = "non_target_new"
        else:
            result = "no_new_fact"
        bucket = by_result[row["split"], result]
        bucket["windows"] += 1
        bucket["known_slots_before"] += len(row["current_slots"])
        bucket["known_slots_after"] += len(set(row["current_slots"]) | {fact[0] for fact in new})
        following = outcome["subsequent_doctor_action"]
        categories = ({"absent"} if following is None else
                      {label.get("intent") or "missing" for label in following} or {"unlabeled"})
        bucket.update("next_" + label for label in sorted(categories))
    slot_rows = [
        {"split": s, "target_slot": slot, **c,
         "target_observed_given_feedback": rate(c["target_observed"], c["feedback"]),
         "target_new_given_feedback": rate(c["target_new"], c["feedback"])}
        for (s, slot), c in sorted(by_slot.items())
    ]
    result_rows = [
        {"split": s, "feedback_outcome": result, **c,
         "mean_slots_before": rate(c["known_slots_before"], c["windows"]),
         "mean_slots_after": rate(c["known_slots_after"], c["windows"])}
        for (s, result), c in sorted(by_result.items())
    ]
    return slot_rows, result_rows


def reference(row):
    source_turns = row["source"]["doctor_source_turns"]
    _, slots, entities, _, _, _ = response_reference({"turns": source_turns})
    return sorted(slots), sorted(entities)


def verify_inputs(source):
    expected = {}
    for line in (source / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        if name.startswith(("train-", "dev-")):
            expected[name] = digest
    for split in SPLITS:
        for kind in ("behavior", "observed-outcome", "feature-input"):
            name = f"{split}-{kind}.jsonl"
            if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected[name]:
                raise ValueError(f"checksum mismatch: {name}")


def prepare(source, run):
    verify_inputs(source)
    run.mkdir(mode=0o700, parents=False)
    slot_report, outcome_report, selection = [], [], []
    for split in SPLITS:
        behavior = list(read_lines(source / f"{split}-behavior.jsonl"))
        by_key = {key(row): row for row in behavior}
        outcomes = {key(row): row for row in read_lines(source / f"{split}-observed-outcome.jsonl")}
        features = {key(row): row["public_state"] for row in read_lines(source / f"{split}-feature-input.jsonl")}
        slot_rows, outcome_rows = ask_report(behavior, outcomes)
        slot_report.extend(slot_rows)
        outcome_report.extend(outcome_rows)
        answers = []
        pairs = []
        for row in behavior:
            loc = key(row)
            if loc not in features:
                continue
            if row["normalized_action"] == "ANSWER":
                slots, entities = reference(row)
                if slots or entities:
                    answers.append({"kind": "answer", "split": split, "dialogue_id": loc[0],
                                    "turn_id": loc[1], "slots": slots, "entities": entities,
                                    "before": features[loc]})
            if split != "dev" or row["normalized_action"] != "ASK":
                continue
            observed = outcomes[loc]
            later = by_key.get((loc[0], loc[1] + 2))
            if not later or later["normalized_action"] != "ANSWER" or observed["context_after_feedback"] is None:
                continue
            slots, entities = reference(later)
            if slots or entities:
                pairs.append({"kind": "pair", "split": split, "dialogue_id": loc[0],
                              "turn_id": loc[1], "slots": slots, "entities": entities,
                              "before": features[loc], "after": observed["context_after_feedback"]})
        selection.extend(sorted(answers, key=rank)[:12] if split == "train" else sorted(answers, key=rank))
        pair_dialogues = list(dict((row["dialogue_id"], row) for row in reversed(sorted(pairs, key=rank))).values())
        selection.extend(sorted(pair_dialogues, key=rank)[:8])
    dump_csv(run / "ask_by_target_slot.csv", slot_report)
    dump_csv(run / "ask_outcomes_and_next_actions.csv", outcome_report)
    (run / "selection-private.json").write_text(json.dumps(selection, ensure_ascii=False), encoding="utf-8")
    counts = Counter(f'{x["split"]}_{x["kind"]}' for x in selection)
    print(json.dumps(counts), flush=True)


def parse(text):
    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict) or not isinstance(value.get("answer"), str):
        return None
    slots = value.get("slots")
    if not isinstance(slots, list) or any(not isinstance(slot, str) for slot in slots):
        return None
    return {normalize(slot) for slot in slots}, normalize(value["answer"])


def score(row, parsed):
    predicted, answer = parsed if parsed is not None else (set(), "")
    reference_slots = set(row["slots"])
    reference_entities = {tuple(x) for x in row["entities"]}
    return {
        "slot_hits": len(reference_slots & predicted), "slot_reference": len(reference_slots),
        "slot_predicted": len(predicted),
        "entity_hits": sum(value in answer for _, value in reference_entities),
        "entity_reference": len(reference_entities),
        "parsed": parsed is not None,
    }


def prompt_for(history):
    writer = "\n".join(("患者：" if row["role"] == "user" else "医生：") + row["content"] for row in history)
    return (
        "根据以下既有中文咨询，给出当前时点直接回复。限一句话，不编造患者信息。"
        '仅输出JSON，例如 {"slots":["symptom"],"answer":"一句回复"}。'
        "slots 仅列出你的回答中涉及的这些槽名：" + ", ".join(SLOT_NAMES) + "。\n" + writer
    )


def summarize(run, results, peak):
    rows = []
    for split, kind, state in (("train", "answer", "before"), ("dev", "answer", "before"),
                               ("dev", "pair", "before"), ("dev", "pair", "after")):
        bucket = [r for r in results if (r["split"], r["kind"], r["state"]) == (split, kind, state)]
        if not bucket:
            continue
        sums = {name: sum(r[name] for r in bucket) for name in (
            "slot_hits", "slot_reference", "slot_predicted", "entity_hits", "entity_reference", "parsed")}
        rows.append({"split": split, "kind": kind, "state": state, "rows": len(bucket), **sums,
                     "slot_precision": rate(sums["slot_hits"], sums["slot_predicted"]),
                     "slot_recall": rate(sums["slot_hits"], sums["slot_reference"]),
                     "entity_coverage": rate(sums["entity_hits"], sums["entity_reference"])})
    dump_csv(run / "answer_structure_results.csv", rows)
    pairs = defaultdict(dict)
    for row in results:
        if row["kind"] == "pair":
            pairs[row["dialogue_id"]][row["state"]] = row
    scenarios = {}
    for metric in ("slot", "entity"):
        diffs = []
        for pair in pairs.values():
            before, after = pair["before"], pair["after"]
            if before[f"{metric}_reference"]:
                diffs.append((after[f"{metric}_hits"] - before[f"{metric}_hits"]) / before[f"{metric}_reference"])
        if diffs:
            rng = random.Random(SEED)
            boot = sorted(sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(10000))
            scenarios[metric] = {"pairs": len(diffs), "improved": sum(d > 0 for d in diffs),
                                 "worsened": sum(d < 0 for d in diffs), "tied": sum(d == 0 for d in diffs),
                                 "mean_delta": sum(diffs) / len(diffs), "ci95": [boot[250], boot[9749]]}
    combined = {"generations": len(results), "peak_allocated_mib": peak, "pairs": scenarios}
    (run / "paired_aggregate.json").write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(combined), flush=True)


def infer(run, weights):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(SEED)
    assert torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported()
    free, total = torch.cuda.mem_get_info()
    assert free >= 28 * 2**30
    torch.cuda.set_per_process_memory_fraction(24 * 2**30 / total)
    tokenizer = AutoTokenizer.from_pretrained(weights, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        weights, local_files_only=True, use_safetensors=True, dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation="eager",
    ).eval()
    results = []
    start = perf_counter()
    with (run / "predictions-private.jsonl").open("x", encoding="utf-8") as private:
        for row in json.loads((run / "selection-private.json").read_text(encoding="utf-8")):
            for state in ("before", "after") if row["kind"] == "pair" else ("before",):
                inputs = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt_for(row[state])}],
                    tokenize=True, add_generation_prompt=True, enable_thinking=False,
                    return_dict=True, return_tensors="pt",
                ).to("cuda")
                if inputs.input_ids.shape[1] > 3072:
                    raise ValueError("context longer than fixed evaluation budget")
                with torch.inference_mode():
                    output = model.generate(**inputs, max_new_tokens=96, do_sample=False)
                new = output[0, inputs.input_ids.shape[1]:]
                text = tokenizer.decode(new, skip_special_tokens=True)
                result = {name: row[name] for name in ("dialogue_id", "turn_id", "kind", "split")}
                result.update(state=state, **score(row, parse(text)))
                results.append(result)
                private.write(json.dumps({**result, "text": text}, ensure_ascii=False) + "\n")
                private.flush()
    summarize(run, results, torch.cuda.max_memory_allocated() // 2**20)
    print(f"inference_seconds={perf_counter() - start:.2f}", flush=True)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("mode", choices=("prepare", "infer"))
    args.add_argument("--run", type=Path, required=True)
    args.add_argument("--source", type=Path)
    args.add_argument("--weights", type=Path)
    opts = args.parse_args()
    os.umask(0o077)
    if opts.mode == "prepare":
        prepare(opts.source, opts.run)
    else:
        infer(opts.run, opts.weights)
