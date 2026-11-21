#!/usr/bin/env python3
"""CPU-only, aggregate-only recomputation of historical V2 fixed-policy metrics.

Reads private counterfactual trajectories in place and writes only aggregate rows.
It does not load model checkpoints, fit models, call services, or print case IDs/text.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ACTION_ORDER = ("ASK", "ANSWER", "DEFER")
ASK_COST_GRID = (0.0, 0.01, 0.03, 0.05)
DEFER_COST_GRID = (0.0, 0.05, 0.10, 0.20, 0.40, 0.60)
BOOTSTRAP_REPLICATES = 10_000


def case_means(values: np.ndarray, case_ids: list[str]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for case_id, value in zip(case_ids, values):
        grouped[case_id].append(float(value))
    return {case_id: float(np.mean(grouped[case_id])) for case_id in sorted(grouped)}


def bootstrap_ci(values: dict[str, float], seed: int) -> tuple[float, float]:
    ordered = np.asarray([values[k] for k in sorted(values)], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(ordered), size=(BOOTSTRAP_REPLICATES, len(ordered)))
    samples = ordered[draws].mean(axis=1)
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return float(lo), float(hi)


def read_trajectories(path: Path):
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    total = invalid = return_mismatches = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        total += 1
        if row.get("valid") is not True or row.get("final_accuracy") not in (0, 1, 0.0, 1.0):
            invalid += 1
        case_id, state_id, action = str(row["case_id"]), str(row["state_id"]), str(row["action"])
        key = (case_id, state_id)
        if action in grouped[key]:
            raise ValueError("duplicate state/action row detected")
        costs = row.get("cost_components") or {}
        ask_cost = float(costs.get("ask_cost", 0.0))
        defer_cost = float(costs.get("defer_cost", 0.0))
        raw_return = float(row["final_accuracy"]) - ask_cost - defer_cost
        if not np.isclose(raw_return, float(row["return"]), atol=1e-12, rtol=0.0):
            return_mismatches += 1
        grouped[key][action] = row
    return grouped, total, invalid, return_mismatches


def summarize(grouped, ask_cost: float, defer_cost: float, seed_base: int, scenario_index: int):
    keys = sorted(grouped)
    case_ids = [key[0] for key in keys]
    accuracy = np.asarray(
        [[float(grouped[key][action]["final_accuracy"]) for action in ACTION_ORDER] for key in keys],
        dtype=np.float64,
    )
    utilities = accuracy - np.asarray([ask_cost, 0.0, defer_cost])[None, :]
    best = np.isclose(utilities, utilities.max(axis=1, keepdims=True), atol=1e-12, rtol=0.0)
    oracle_prob = best / best.sum(axis=1, keepdims=True)
    policies = {
        "Always ANSWER": np.tile(np.asarray([0.0, 1.0, 0.0]), (len(keys), 1)),
        "Always ASK": np.tile(np.asarray([1.0, 0.0, 0.0]), (len(keys), 1)),
        "Always DEFER": np.tile(np.asarray([0.0, 0.0, 1.0]), (len(keys), 1)),
        "Retrospective Oracle": oracle_prob,
    }
    rows = []
    metric_order = ("accuracy", "utility", "ask_rate", "defer_rate")
    for policy_name, probs in policies.items():
        vals_by_metric = {
            "accuracy": np.sum(probs * accuracy, axis=1),
            "utility": np.sum(probs * utilities, axis=1),
            "ask_rate": probs[:, 0],
            "defer_rate": probs[:, 2],
        }
        row: dict[str, Any] = {
            "ask_cost": ask_cost,
            "defer_cost": defer_cost,
            "policy": policy_name,
            "n_cases": len(set(case_ids)),
            "n_states": len(keys),
        }
        for metric_index, metric in enumerate(metric_order):
            per_case = case_means(vals_by_metric[metric], case_ids)
            row[metric] = float(np.mean(list(per_case.values())))
            lo, hi = bootstrap_ci(per_case, seed_base + scenario_index * 7 + metric_index)
            row[f"{metric}_ci95_low"] = lo
            row[f"{metric}_ci95_high"] = hi
            row[f"{metric}_state_pooled"] = float(np.mean(vals_by_metric[metric]))
        if policy_name == "Retrospective Oracle":
            for metric, vals in (
                ("oracle_accuracy_gain_vs_answer", vals_by_metric["accuracy"] - accuracy[:, 1]),
                ("oracle_utility_gain_vs_answer", vals_by_metric["utility"] - utilities[:, 1]),
            ):
                per_case = case_means(vals, case_ids)
                row[metric] = float(np.mean(list(per_case.values())))
                lo, hi = bootstrap_ci(per_case, seed_base + scenario_index * 11 + 91 + int(metric.endswith("utility_gain_vs_answer")))
                row[f"{metric}_ci95_low"] = lo
                row[f"{metric}_ci95_high"] = hi
            row["oracle_unique_best_states"] = int((best.sum(axis=1) == 1).sum())
            row["oracle_tied_best_states"] = int((best.sum(axis=1) > 1).sum())
        else:
            row.update({
                "oracle_accuracy_gain_vs_answer": 0.0,
                "oracle_utility_gain_vs_answer": 0.0,
                "oracle_unique_best_states": "",
                "oracle_tied_best_states": "",
            })
        rows.append(row)
    return rows


def read_reference(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return {(float(r["ask_cost"]), float(r["defer_cost"]), r["policy"]): r for r in csv.DictReader(handle)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectories", type=Path, required=True)
    ap.add_argument("--reference-csv", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--bootstrap-seed-base", type=int, default=1300)
    args = ap.parse_args()
    grouped, total, invalid, return_mismatches = read_trajectories(args.trajectories)
    if not grouped:
        raise ValueError("no trajectory rows found")
    case_set = {case for case, _ in grouped}
    action_coverage = defaultdict(int)
    for key, actions in grouped.items():
        if set(actions) != set(ACTION_ORDER):
            raise ValueError("state does not contain exactly the three actions")
        for action in actions:
            action_coverage[action] += 1
    if invalid or return_mismatches:
        raise ValueError("invalid rows or raw-return mismatches were found")

    result_rows = []
    scenario_index = 0
    for ask_cost in ASK_COST_GRID:
        for defer_cost in DEFER_COST_GRID:
            result_rows.extend(summarize(grouped, ask_cost, defer_cost, args.bootstrap_seed_base, scenario_index))
            scenario_index += 1

    reference = read_reference(args.reference_csv)
    fields = list(dict.fromkeys(key for row in result_rows for key in row))
    deltas = []
    for row in result_rows:
        ref = reference[(float(row["ask_cost"]), float(row["defer_cost"]), row["policy"])]
        for field in fields:
            if field in {"policy"}:
                continue
            row_value = row.get(field, "")
            ref_value = ref.get(field, "")
            if ref_value == "" and row_value == "":
                continue
            if row_value == "" or ref_value == "":
                deltas.append(float("inf"))
                continue
            delta = abs(float(row_value) - float(ref_value))
            deltas.append(delta)
    max_abs_difference = max(deltas, default=0.0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "windowb_cost_recomputed.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(result_rows)
    summary = {
        "status": "passed" if max_abs_difference <= 1e-12 else "mismatch",
        "rows_read": total,
        "valid_rows": total - invalid,
        "case_count": len(case_set),
        "state_count": len(grouped),
        "action_coverage": dict(action_coverage),
        "invalid_rows": invalid,
        "raw_return_mismatches": return_mismatches,
        "reference_rows": len(reference),
        "recomputed_rows": len(result_rows),
        "max_abs_difference_vs_saved_aggregate_csv": max_abs_difference,
        "bootstrap_unit": "case_id, all states kept within case",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed_base": args.bootstrap_seed_base,
        "case_ids_or_case_text_written": False,
        "model_or_gpu_used": False,
    }
    (args.output_dir / "windowb_metric_recalculation.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if summary["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
