#!/usr/bin/env python3
"""Recompute complete CONSULT-chain costs and paired CIs from frozen outputs."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


AUDIT_DIR = Path(__file__).resolve().parent
PILOT_DIR = AUDIT_DIR.parent
OUT_DIR = AUDIT_DIR / "outputs/advice_utilization"
INPUTS = OUT_DIR / "frozen_consult_inputs.jsonl"
TRAJECTORIES = OUT_DIR / "advice_utilization_trajectories.jsonl"
PILOT_TRAJECTORIES = PILOT_DIR / "outputs/full/pilot_trajectories.jsonl"
MANIFEST = OUT_DIR / "run_manifest.json"
METHODS = ("PILOT_ANSWER", "CONSULT_NAIVE", "CONSULT_RETAIN", "CONSULT_CRITICAL")
LIVE_METHODS = METHODS[1:]
REPEATS = 10_000


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cluster_bootstrap(
    rows: list[dict[str, Any]], stat: Callable[[list[dict[str, Any]]], float], seed: int
) -> tuple[float, float]:
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case[str(row["case_id"])].append(row)
    case_ids = sorted(by_case)
    rng = random.Random(seed)
    values = []
    for _ in range(REPEATS):
        sampled = rng.choices(case_ids, k=len(case_ids))
        replicate = [row for case_id in sampled for row in by_case[case_id]]
        value = stat(replicate)
        if value is not None:
            values.append(float(value))
    values.sort()
    return values[int(0.025 * len(values))], values[min(len(values) - 1, int(0.975 * len(values)))]


def ratio(rows: list[dict[str, Any]], numerator: Callable, denominator: Callable) -> float:
    eligible = [row for row in rows if denominator(row)]
    return sum(bool(numerator(row)) for row in eligible) / len(eligible) if eligible else 0.0


def correct(row: dict[str, Any], method: str) -> bool:
    return bool(row["pilot_answer_correct"]) if method == "PILOT_ANSWER" else bool(row["methods"][method]["correct"])


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    pilot_rows = read_jsonl(PILOT_TRAJECTORIES)
    pilot_digest = sha256(PILOT_TRAJECTORIES)
    if pilot_digest != manifest["pilot_trajectory_sha256"]:
        raise RuntimeError("Pilot source hash differs from the frozen experiment manifest")

    pilot_consults = {
        (str(row["case_id"]), str(row["state_id"]), int(row["rollout_index"])): row
        for row in pilot_rows
        if row.get("record_type") == "trajectory" and row.get("action") == "CONSULT"
    }
    inputs = read_jsonl(INPUTS)
    trajectories = read_jsonl(TRAJECTORIES)
    output_by_key = {(row["sample_id"], row["method"]): row for row in trajectories}
    if len(inputs) != 200 or len(output_by_key) != 600:
        raise RuntimeError(f"expected 200 frozen states and 600 unique calls; found {len(inputs)}, {len(output_by_key)}")

    wide: list[dict[str, Any]] = []
    by_state_cost: list[dict[str, Any]] = []
    for item in inputs:
        source_key = (item["case_id"], item["state_id"], int(item["rollout_index"]))
        source = pilot_consults.get(source_key)
        if source is None:
            raise RuntimeError(f"missing paired source CONSULT trace: {source_key}")
        setup_events = [
            event for event in source.get("trace", [])
            if event.get("action") == "CONSULT" and event.get("actor") in ("base", "expert")
        ]
        if not setup_events:
            raise RuntimeError(f"no recorded CONSULT setup calls: {source_key}")
        setup_prompt = sum(int(event.get("prompt_tokens", 0)) for event in setup_events)
        setup_completion = sum(int(event.get("completion_tokens", 0)) for event in setup_events)
        setup_latency = sum(float(event.get("latency_s", 0.0)) for event in setup_events)
        method_rows = {method: output_by_key[(item["sample_id"], method)] for method in LIVE_METHODS}
        cost_item: dict[str, Any] = {**item, "methods": method_rows, "costs": {}}
        direct_prompt = int(item["pilot_answer_prompt_tokens"])
        direct_completion = int(item["pilot_answer_completion_tokens"])
        direct_latency = float(item["pilot_answer_latency_s"])
        cost_item["costs"]["PILOT_ANSWER"] = {
            "setup_prompt_tokens": 0, "setup_completion_tokens": 0, "setup_total_tokens": 0,
            "setup_latency_s": 0.0,
            "continuation_prompt_tokens": direct_prompt,
            "continuation_completion_tokens": direct_completion,
            "continuation_total_tokens": direct_prompt + direct_completion,
            "continuation_latency_s": direct_latency,
            "full_chain_total_tokens": direct_prompt + direct_completion,
            "full_chain_latency_s": direct_latency,
        }
        by_state_cost.append({
            "sample_id": item["sample_id"], "case_id": item["case_id"], "method": "PILOT_ANSWER",
            "valid": bool(item["pilot_answer_valid"]), "correct": bool(item["pilot_answer_correct"]),
            "setup_prompt_tokens": 0, "setup_completion_tokens": 0, "setup_total_tokens": 0,
            "setup_latency_s": 0.0, "continuation_prompt_tokens": direct_prompt,
            "continuation_completion_tokens": direct_completion, "continuation_total_tokens": direct_prompt + direct_completion,
            "continuation_latency_s": direct_latency, "full_chain_total_tokens": direct_prompt + direct_completion,
            "full_chain_latency_s": direct_latency,
        })
        for method, output in method_rows.items():
            continuation_prompt = int(output.get("prompt_tokens", 0))
            continuation_completion = int(output.get("completion_tokens", 0))
            continuation_latency = float(output.get("latency_s", 0.0))
            cost_item["costs"][method] = {
                "setup_prompt_tokens": setup_prompt,
                "setup_completion_tokens": setup_completion,
                "setup_total_tokens": setup_prompt + setup_completion,
                "setup_latency_s": setup_latency,
                "continuation_prompt_tokens": continuation_prompt,
                "continuation_completion_tokens": continuation_completion,
                "continuation_total_tokens": continuation_prompt + continuation_completion,
                "continuation_latency_s": continuation_latency,
                "full_chain_total_tokens": setup_prompt + setup_completion + continuation_prompt + continuation_completion,
                "full_chain_latency_s": setup_latency + continuation_latency,
            }
            by_state_cost.append({
                "sample_id": item["sample_id"], "case_id": item["case_id"], "method": method,
                "valid": bool(output.get("valid")), "correct": bool(output.get("correct")),
                **cost_item["costs"][method],
            })
        wide.append(cost_item)

    cost_path = OUT_DIR / "advice_utilization_cost_by_state.csv"
    with cost_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(by_state_cost[0]))
        writer.writeheader()
        writer.writerows(by_state_cost)

    summary_path = OUT_DIR / "advice_utilization_cost_summary.csv"
    summary_rows = []
    cost_fields = (
        "setup_prompt_tokens", "setup_completion_tokens", "setup_total_tokens", "setup_latency_s",
        "continuation_prompt_tokens", "continuation_completion_tokens", "continuation_total_tokens",
        "continuation_latency_s", "full_chain_total_tokens", "full_chain_latency_s",
    )
    for method in METHODS:
        method_costs = [row["costs"][method] for row in wide]
        for metric in cost_fields:
            values = [float(row[metric]) for row in method_costs]
            low, high = cluster_bootstrap(
                wide, lambda rows, m=method, f=metric: statistics.mean(float(r["costs"][m][f]) for r in rows),
                20261105 + METHODS.index(method) * 100 + cost_fields.index(metric),
            )
            summary_rows.append({"method": method, "metric": metric, "mean": statistics.mean(values), "ci95_low": low, "ci95_high": high, "n_states": len(values)})
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    pair_rows = []
    pairs = [(METHODS[i], METHODS[j]) for i in range(1, len(METHODS)) for j in range(i)]
    for pair_index, (method, reference) in enumerate(pairs):
        metric_fns: dict[str, Callable[[list[dict[str, Any]]], float]] = {
            "accuracy_delta": lambda rows, m=method, r=reference: statistics.mean(float(correct(x, m)) - float(correct(x, r)) for x in rows),
            "full_chain_total_tokens_delta": lambda rows, m=method, r=reference: statistics.mean(float(x["costs"][m]["full_chain_total_tokens"]) - float(x["costs"][r]["full_chain_total_tokens"]) for x in rows),
            "full_chain_latency_delta_s": lambda rows, m=method, r=reference: statistics.mean(float(x["costs"][m]["full_chain_latency_s"]) - float(x["costs"][r]["full_chain_latency_s"]) for x in rows),
            "correction_rate_delta_given_pilot_answer_wrong": lambda rows, m=method, r=reference: ratio(rows, lambda x: correct(x, m), lambda x: not bool(x["pilot_answer_correct"])) - ratio(rows, lambda x: correct(x, r), lambda x: not bool(x["pilot_answer_correct"])),
            "degradation_rate_delta_given_pilot_answer_correct": lambda rows, m=method, r=reference: ratio(rows, lambda x: bool(x["pilot_answer_correct"]) and not correct(x, m), lambda x: bool(x["pilot_answer_correct"])) - ratio(rows, lambda x: bool(x["pilot_answer_correct"]) and not correct(x, r), lambda x: bool(x["pilot_answer_correct"])),
            "answer_change_fraction_of_all_states_delta": lambda rows, m=method, r=reference: statistics.mean((float(m != "PILOT_ANSWER" and x["methods"][m]["valid"] and x["methods"][m]["predicted"] != x["pilot_answer"])) - (float(r != "PILOT_ANSWER" and x["methods"][r]["valid"] and x["methods"][r]["predicted"] != x["pilot_answer"])) for x in rows),
            "valid_rate_delta": lambda rows, m=method, r=reference: statistics.mean(float(bool(x["pilot_answer_valid"]) if m == "PILOT_ANSWER" else bool(x["methods"][m]["valid"])) - float(bool(x["pilot_answer_valid"]) if r == "PILOT_ANSWER" else bool(x["methods"][r]["valid"])) for x in rows),
        }
        for metric_index, (metric, stat) in enumerate(metric_fns.items()):
            point = stat(wide)
            low, high = cluster_bootstrap(wide, stat, 20261105 + 1000 + pair_index * 20 + metric_index)
            pair_rows.append({"method": method, "reference": reference, "metric": metric, "difference": point, "ci95_low": low, "ci95_high": high, "n_states": len(wide), "n_cases": len({row['case_id'] for row in wide})})
    pair_path = OUT_DIR / "advice_utilization_paired_differences.csv"
    with pair_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(pair_rows[0]))
        writer.writeheader()
        writer.writerows(pair_rows)

    manifest.update({
        "cost_accounting": "For CONSULT variants, add recorded Base consultation-question and Expert-response calls from the matching Pilot CONSULT trace to the fresh continuation request. Prior state-acquisition costs are shared/sunk and excluded. Direct ANSWER uses the paired Pilot answer request.",
        "latency_accounting": "Sum of the recorded sequential request latencies; direct ANSWER is historical batched-service timing, while consultation setup is from historical Pilot and continuation is from the new sequential service. Treat as observed, not controlled throughput.",
        "cost_by_state_path": str(cost_path.resolve()),
        "cost_by_state_sha256": sha256(cost_path),
        "cost_summary_path": str(summary_path.resolve()),
        "cost_summary_sha256": sha256(summary_path),
        "paired_differences_path": str(pair_path.resolve()),
        "paired_differences_sha256": sha256(pair_path),
        "bootstrap": {"unit": "case cluster", "repeats": REPEATS, "sampled_cases": len({row['case_id'] for row in wide}), "seed_base": 20261105},
    })
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cost_summary": str(summary_path), "paired_differences": str(pair_path), "manifest_updated": str(MANIFEST)}, indent=2))


if __name__ == "__main__":
    main()
