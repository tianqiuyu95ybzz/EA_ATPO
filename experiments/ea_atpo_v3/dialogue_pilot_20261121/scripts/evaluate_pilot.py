#!/usr/bin/env python3
"""Aggregate blind-rated pilot output without treating clinician text as Gold."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

FORBIDDEN_LABEL_FIELDS = {"gold", "ground_truth", "correctness", "final_correct", "accuracy", "answer_idx"}
ALLOWED_ACTIONS = {"ANSWER", "ASK", "DEFER"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"line {i} must be an object")
        forbidden = FORBIDDEN_LABEL_FIELDS & set(row)
        if forbidden:
            raise ValueError(f"line {i} contains unsupported Gold/correctness fields: {sorted(forbidden)}")
        rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", type=Path, required=True, help="private predictions JSONL; see DATA_AUDIT.md schema")
    p.add_argument("--ratings", type=Path, required=True, help="blind ratings JSONL keyed by blind_id")
    p.add_argument("--output-dir", type=Path, required=True, help="private output directory")
    p.add_argument("--ask-cost", type=float, default=0.01)
    p.add_argument("--defer-cost", type=float, default=0.05)
    args = p.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite a non-empty private output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(args.output_dir, 0o700)

    predictions, ratings = read_jsonl(args.predictions), read_jsonl(args.ratings)
    by_blind_id = {}
    for row in ratings:
        bid = row.get("blind_id")
        if not bid or bid in by_blind_id:
            raise ValueError("ratings require unique non-empty blind_id values")
        by_blind_id[bid] = row
    per_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    failures = []
    seen_keys = set()
    seen_blind_ids = set()
    for row in predictions:
        sample_id, policy = row.get("sample_id"), row.get("policy")
        if not sample_id or not policy:
            raise ValueError("prediction rows require sample_id and policy")
        key = (sample_id, policy, row.get("replicate_id", "0"))
        if key in seen_keys:
            raise ValueError("duplicate sample/policy/replicate prediction")
        seen_keys.add(key)
        blind_id = row.get("blind_id")
        if blind_id is not None:
            if not isinstance(blind_id, str) or not blind_id or blind_id in seen_blind_ids:
                raise ValueError("each rated prediction must have a unique non-empty blind_id")
            seen_blind_ids.add(blind_id)
        trace = row.get("action_trace") or ([row["chosen_action"]] if row.get("chosen_action") else [])
        if not isinstance(trace, list) or any(a not in ALLOWED_ACTIONS for a in trace):
            raise ValueError("action_trace must contain only ANSWER/ASK/DEFER")
        if any(a in {"ANSWER", "DEFER"} for a in trace[:-1]):
            raise ValueError("ANSWER and DEFER are terminal and must be the last action")
        ask_count = sum(a == "ASK" for a in trace)
        defer_count = sum(a == "DEFER" for a in trace)
        if row.get("ask_count") is not None and int(row["ask_count"]) != ask_count:
            raise ValueError("ask_count does not match action_trace")
        if row.get("defer_count") is not None and int(row["defer_count"]) != defer_count:
            raise ValueError("defer_count does not match action_trace")
        rating = by_blind_id.get(blind_id) if blind_id else None
        quality = rating.get("answer_quality_1_to_5") if rating else None
        if quality is not None and (not isinstance(quality, (int, float)) or not 1 <= float(quality) <= 5):
            raise ValueError("answer_quality_1_to_5 must be in [1,5]")
        useful = rating.get("ask_information_useful") if rating else None
        if useful not in (None, True, False):
            raise ValueError("ask_information_useful must be boolean or null")
        failed = bool(row.get("failure_reason")) or bool(row.get("failed", False))
        record = {
            "sample_id": sample_id,
            "policy": policy,
            "action_trace": trace,
            "ask_count": ask_count,
            "defer_count": defer_count,
            "failed": failed,
            "failure_reason": row.get("failure_reason"),
            "quality_score": float(quality) if quality is not None else None,
            "ask_information_useful": useful,
        }
        per_policy[policy].append(record)
        if failed:
            failures.append({"sample_id": sample_id, "policy": policy, "failure_reason": row.get("failure_reason") or "unspecified_failure"})

    summary = {
        "schema_version": 1,
        "quality_metric": "blind evaluator score 1-5; normalized to [0,1] as (score-1)/4 for Utility",
        "utility": "normalized blind answer-quality score - ask_cost * ASK_count - defer_cost * DEFER_count",
        "costs": {"ASK": args.ask_cost, "DEFER": args.defer_cost, "ANSWER": 0.0},
        "medical_correctness_claimed": False,
        "historical_clinician_text_is_verified_gold": False,
        "policies": {},
        "total_prediction_rows": len(predictions),
        "total_rating_rows": len(ratings),
        "ratings_without_prediction_match": sorted(set(by_blind_id) - {r.get("blind_id") for r in predictions if r.get("blind_id")}),
        "failures": {"count": len(failures), "file": "failures_private.jsonl"},
    }
    for policy, rows in sorted(per_policy.items()):
        action_counts = Counter(a for r in rows for a in r["action_trace"])
        ask_rows = [r for r in rows if r["ask_count"] > 0]
        scored = [r for r in rows if r["quality_score"] is not None]
        useful_rated = [r for r in ask_rows if r["ask_information_useful"] is not None]
        normalized = [(r["quality_score"] - 1.0) / 4.0 for r in scored]
        utilities = [
            (r["quality_score"] - 1.0) / 4.0 - args.ask_cost * r["ask_count"] - args.defer_cost * r["defer_count"]
            for r in scored
        ]
        summary["policies"][policy] = {
            "sample_rows": len(rows),
            "unique_samples": len({r["sample_id"] for r in rows}),
            "failed_rows": sum(r["failed"] for r in rows),
            "action_counts": {a: action_counts[a] for a in ("ANSWER", "ASK", "DEFER")},
            "action_episode_rates": {a: sum(a in r["action_trace"] for r in rows) / len(rows) if rows else None for a in ("ANSWER", "ASK", "DEFER")},
            "mean_ask_count": sum(r["ask_count"] for r in rows) / len(rows) if rows else None,
            "answer_quality": {"n_rated": len(scored), "mean_1_to_5": sum(r["quality_score"] for r in scored) / len(scored) if scored else None,
                               "mean_normalized_0_to_1": sum(normalized) / len(normalized) if normalized else None},
            "ask_information": {"ask_rows": len(ask_rows), "n_blind_rated": len(useful_rated),
                                "useful_rate": sum(r["ask_information_useful"] is True for r in useful_rated) / len(useful_rated) if useful_rated else None},
            "mean_utility_on_scored_rows": sum(utilities) / len(utilities) if utilities else None,
        }
    failures_path = args.output_dir / "failures_private.jsonl"
    with failures_path.open("x", encoding="utf-8") as f:
        for row in failures:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.chmod(failures_path, 0o600)
    summary_path = args.output_dir / "summary_private.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.chmod(summary_path, 0o600)
    print(json.dumps({"status": "aggregated", "prediction_rows": len(predictions), "policies": list(summary["policies"]), "failure_rows": len(failures), "private_output": str(args.output_dir.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
