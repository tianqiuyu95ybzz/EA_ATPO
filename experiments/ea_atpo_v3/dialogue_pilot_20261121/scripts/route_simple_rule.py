#!/usr/bin/env python3
"""Generate a deterministic, pre-outcome heuristic action for each private sample."""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

RED_FLAGS = re.compile(r"胸痛|呼吸困难|意识不清|大出血|抽搐|突发偏瘫")
AGE = re.compile(r"(?:\d{1,3}\s*岁|年龄|多大)")
DURATION = re.compile(r"(?:\d+\s*(?:天|周|月|年)|多久|几天|何时|什么时候|从.{0,8}(?:开始|出现))")
MEDS = re.compile(r"(?:药物|用药|服用|药|过敏)")


def route(row: dict, confidence: float | None, *, expert_available: bool) -> tuple[str, str]:
    text = row["initial_patient_question"]
    if RED_FLAGS.search(text) and expert_available:
        return "DEFER", "listed_red_flag_keyword"
    if confidence is not None and confidence < 0.35 and expert_available:
        return "DEFER", "base_confidence_below_0_35"
    if confidence is not None and confidence < 0.65 and row.get("ask_replay_pairs"):
        return "ASK", "base_confidence_below_0_65_with_replay"
    if any(not rx.search(text) for rx in (AGE, DURATION, MEDS)) and row.get("ask_replay_pairs"):
        return "ASK", "missing_one_or_more_basic_history_fields_and_replay_exists"
    return "ANSWER", "no_rule_triggered"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--base-predictions", type=Path, help="optional JSONL with sample_id and predecision confidence in [0,1]")
    p.add_argument("--expert-available", action="store_true")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    confidence_by_id = {}
    if args.base_predictions:
        for line in args.base_predictions.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            prediction = json.loads(line)
            sid, confidence = prediction.get("sample_id"), prediction.get("confidence")
            if not sid or sid in confidence_by_id:
                raise ValueError("Base predictions must have unique sample_id values")
            if not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0:
                raise ValueError("confidence must be a number in [0,1]")
            if {"gold", "ground_truth", "correctness", "final_correct", "accuracy"} & set(prediction):
                raise ValueError("Base prediction contains a forbidden outcome label")
            confidence_by_id[sid] = float(confidence)
    out = []
    for line in args.samples.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        confidence = confidence_by_id.get(row["sample_id"])
        action, reason = route(row, confidence, expert_available=args.expert_available)
        pair_index = 0 if action == "ASK" and row.get("ask_replay_pairs") else None
        clinician_index = row["historical_clinician_response_references"][0]["turn_index"] if action == "DEFER" and row.get("historical_clinician_response_references") else None
        out.append({
            "sample_id": row["sample_id"],
            "policy": "EA-ATPO-simple-rule",
            "chosen_action": action,
            "reason_code": reason,
            "base_confidence": confidence,
            "confidence_thresholds": {"defer_below": 0.35, "ask_below": 0.65},
            "ask_replay_pair_index": pair_index,
            "historical_clinician_reference_turn_index_for_comparison_only": clinician_index,
            "router_features": {"initial_patient_question_only": True},
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.chmod(args.output, 0o600)
    print(json.dumps({"samples": len(out), "actions": {a: sum(r["chosen_action"] == a for r in out) for a in ("ANSWER", "ASK", "DEFER")}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
