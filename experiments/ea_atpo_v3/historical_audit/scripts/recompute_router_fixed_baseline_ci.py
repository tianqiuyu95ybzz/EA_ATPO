#!/usr/bin/env python3
"""Recompute paired case-cluster intervals for saved Router Dev predictions.

No model is loaded or trained. Output contains policy-level aggregates only.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

REPLICATES = 5_000
SEED = 20261120
METRICS = ("accuracy", "utility")
POLICIES = ("always_answer", "always_ask", "always_defer")
LEARNED = ("q_guided_distillation", "mlp_relative")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-results", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    data = json.loads(args.training_results.read_text(encoding="utf-8"))
    results = [r for r in data["results"] if r.get("feature") == "embedding"]
    by_model: dict[str, list[dict]] = {}
    for model in (*POLICIES, *LEARNED):
        by_model[model] = [r for r in results if r.get("model") == model]
    for model in LEARNED:
        if len(by_model[model]) != 3:
            raise ValueError("expected three fixed-seed learned policy outputs")
    for model in POLICIES:
        if len(by_model[model]) != 1:
            raise ValueError("expected one shared fixed-policy output")

    rows = []
    for learned in LEARNED:
        seed_rows = sorted(by_model[learned], key=lambda r: int(r["seed"]))
        shared_cases = set(seed_rows[0]["dev_case_metrics"])
        if any(set(r["dev_case_metrics"]) != shared_cases for r in seed_rows[1:]):
            raise ValueError("learned seeds do not cover the same Dev cases")
        for baseline in POLICIES:
            base = by_model[baseline][0]
            if set(base["dev_case_metrics"]) != shared_cases:
                raise ValueError("fixed baseline and learned predictions do not use identical Dev cases")
            for metric in METRICS:
                delta_by_case = []
                for case in sorted(shared_cases):
                    learned_case = np.mean([r["dev_case_metrics"][case][metric] for r in seed_rows])
                    baseline_case = base["dev_case_metrics"][case][metric]
                    delta_by_case.append(float(learned_case - baseline_case))
                values = np.asarray(delta_by_case, dtype=np.float64)
                rng = np.random.default_rng(SEED)
                idx = rng.integers(0, len(values), size=(REPLICATES, len(values)))
                samples = values[idx].mean(axis=1)
                lo, hi = np.quantile(samples, [0.025, 0.975])
                rows.append({
                    "feature": "embedding",
                    "learned_policy": learned,
                    "baseline": baseline,
                    "metric": metric,
                    "n_cases": len(values),
                    "point_delta": float(values.mean()),
                    "ci95_low": float(lo),
                    "ci95_high": float(hi),
                    "bootstrap_unit": "case_id; three fixed seed policies averaged within case",
                    "bootstrap_replicates": REPLICATES,
                    "bootstrap_seed": SEED,
                })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({
        "status": "passed",
        "policies": list(LEARNED),
        "fixed_baselines": list(POLICIES),
        "case_count": len(by_model["always_answer"][0]["dev_case_metrics"]),
        "paired_comparisons": len(rows),
        "rows_written": len(rows),
        "case_ids_or_case_text_written": False,
        "model_or_gpu_used": False,
    }, indent=2))


if __name__ == "__main__":
    main()
