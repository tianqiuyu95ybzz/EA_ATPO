#!/usr/bin/env python3
"""Emit the manual, leakage-safe audit of the 23 Unknown ASK replies in pilot10.

Annotations were made by comparing each public question/state to the Patient-visible
source facts. This file stores no source facts, answer key, or correctness labels.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ANNOTATIONS = {
    "row-10617:item-10617:a5e89df75a70ee2c": ("fact_present_under_matched", "Question targets a documented cardiac-history item; simulator returned Unknown."),
    "row-10617:item-10617:85cd1e1bdbdf978d": ("already_public_or_partial", "Public state already includes cough and hoarseness; infection-specific detail remains unstated."),
    "row-10617:item-10617:3ed24673e1583ee4": ("relevant_fact_absent", "No leg or ankle swelling information is stated in the available patient record."),
    "row-10708:item-10708:061efd11dc9f1854": ("unrelated_or_invalid", "GI-history question does not meaningfully target the visible injury-location task."),
    "row-11537:item-11537:8798c6d2e6cb8b87": ("vague_or_multicomponent", "Asks the patient to interpret another person's motives and combines several judgments."),
    "row-1465:item-1465:b87a337da1e79154": ("fact_present_under_matched", "Question targets a documented cough symptom; simulator returned Unknown."),
    "row-1465:item-1465:4b7b9e9613fc4394": ("relevant_fact_absent", "No prior COPD or interstitial-lung-disease diagnosis is stated."),
    "row-3359:item-3359:e36146ed72ce04d7": ("already_public_or_partial", "Public state already contains hypertension and neurological-event information."),
    "row-3359:item-3359:77298eae04435288": ("already_public_or_partial", "Hypertension is already stated in the public case; the broader risk-factor query is partly redundant."),
    "row-3359:item-3359:b9fbdcd7588275f0": ("unrelated_or_invalid", "Meta-question critiques the benchmark options rather than asking about patient history."),
    "row-7908:item-7908:a9fd348104e5e70d": ("relevant_fact_absent", "Renal impairment and renin-related comorbidity are not stated."),
    "row-7908:item-7908:092b29ea8ca5de29": ("relevant_fact_absent", "No kidney-disease history is available in the patient facts."),
    "row-8099:item-8099:fd78143527b434a2": ("relevant_fact_absent", "Requested hypoglycemia and GI history are not stated."),
    "row-8099:item-8099:473c2a095435a98c": ("relevant_fact_absent", "No family history of similar symptoms or a genetic condition is stated."),
    "row-8099:item-8099:f9b885e81e883b75": ("relevant_fact_absent", "Abdominal pain and diarrhea history are not stated."),
    "row-8565:item-8565:2e78c3ca451081ec": ("relevant_fact_absent", "Prior atopic or allergic history is not stated."),
    "row-8565:item-8565:dc94faeaf94f186b": ("fact_present_under_matched", "Question partly overlaps documented itch-related night waking; simulator returned Unknown."),
    "row-895:item-895:bab5207b2fe64947": ("fact_present_under_matched", "Known meal and a symptom cluster among companions are relevant partial facts; contamination itself is unverified."),
    "row-895:item-895:b224c095c6f3ffe6": ("relevant_fact_absent", "No recent travel or outbreak exposure is stated."),
    "row-895:item-895:19570f608fa576d3": ("fact_present_under_matched", "Known shared meal and companion symptoms support a partial common-source response."),
    "row-9706:item-9706:e87c500143b371a7": ("relevant_fact_absent", "Requested persistent depressive symptoms are not directly stated."),
    "row-9706:item-9706:61ca07e72edcc9db": ("relevant_fact_absent", "Worthlessness, guilt, and anhedonia are not directly stated."),
    "row-9706:item-9706:a9a90855bf3d73c9": ("vague_or_multicomponent", "Combines multiple personality judgments and asks for interpretation not established by the record."),
}

CATEGORY_ORDER = (
    "fact_present_under_matched", "already_public_or_partial", "unrelated_or_invalid",
    "relevant_fact_absent", "vague_or_multicomponent",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    unknowns = [row for row in rows if row.get("action") == "ASK" and row.get("ask_response_unknown") is True]
    by_id = {str(row["state_id"]): row for row in unknowns}
    if len(unknowns) != 23 or set(by_id) != set(ANNOTATIONS):
        raise SystemExit(f"Expected the frozen 23 Unknown states, found {len(unknowns)}; annotation id mismatch")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    category_counts = Counter()
    per_case: dict[str, Counter] = defaultdict(Counter)
    with (args.output_dir / "ask_unknown_classifications.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case_id", "state_id", "state_index", "question", "category", "rationale"))
        writer.writeheader()
        for state_id, row in sorted(by_id.items(), key=lambda pair: (str(pair[1]["case_id"]), int(pair[1]["state_index"]))):
            category, rationale = ANNOTATIONS[state_id]
            category_counts[category] += 1
            per_case[str(row["case_id"])][category] += 1
            writer.writerow({"case_id": row["case_id"], "state_id": state_id,
                             "state_index": row["state_index"], "question": row.get("ask_question", ""),
                             "category": category, "rationale": rationale})
    case_counts = Counter(str(row["case_id"]) for row in rows if row.get("action") == "ASK")
    with (args.output_dir / "ask_unknown_by_case.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ("case_id", "ask_states", "unknown_count", "unknown_rate", *CATEGORY_ORDER)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case_id in sorted(case_counts):
            counts = per_case[case_id]
            unknown_count = sum(counts.values())
            writer.writerow({"case_id": case_id, "ask_states": case_counts[case_id],
                             "unknown_count": unknown_count,
                             "unknown_rate": unknown_count / case_counts[case_id],
                             **{category: counts[category] for category in CATEGORY_ORDER}})
    summary = {
        "annotation_type": "manual protocol/fact-availability audit; not clinical adjudication",
        "source_trajectory_file": str(args.input.resolve()),
        "unknown_count": len(unknowns), "unique_case_count": len(per_case),
        "categories": {category: {"count": category_counts[category], "rate": category_counts[category] / len(unknowns)}
                       for category in CATEGORY_ORDER},
        "gold_used_for_annotation": False,
        "gold_serialized": False,
        "hidden_fact_text_serialized": False,
        "category_total_check": sum(category_counts.values()) == len(unknowns),
    }
    (args.output_dir / "ask_unknown_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
