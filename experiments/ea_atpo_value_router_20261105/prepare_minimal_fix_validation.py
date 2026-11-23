#!/usr/bin/env python3
"""Freeze a small policy candidate set and disjoint validation rows.

Candidate choice is based on the case-disjoint Pilot validation split. A second
50-case online validation run selects among the frozen profiles; the existing
200-case stepwise set remains reserved for final evaluation.
"""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from diagnose_router_failure import model_diagnostics
from router_core import ACTIONS
from train_router import COST_COMPONENTS


HERE = Path(__file__).resolve().parent
PILOT_DIR = HERE.parent / "real_case_pilot_20261103"
OUT = HERE / "outputs/failure_analysis"
STEPWISE_MANIFEST = HERE / "outputs/stepwise_test_v2/run_manifest.json"
DIRECT_MANIFEST = PILOT_DIR / "audit/outputs/direct_eval/expert_reliability_manifest.json"
TEST_DATA = PILOT_DIR / "assets/test.pilot.parquet"
REFERENCE_COSTS = {"ANSWER": 0.0, "ASK": 0.01, "CONSULT": 0.02, "DEFER": 0.02}


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    _q_rows, diag = model_diagnostics()
    ids = diag["ids_by_split"]["validation"]
    states, targets, model, predictions = diag["states"], diag["targets"], diag["model"], diag["predictions"]
    train_ids = diag["ids_by_split"]["train"]
    priors = {a: float(np.mean([targets[sid][a]["q"] for sid in train_ids])) for a in ACTIONS}
    fallback_cost = model["cost_means"]

    grid = []
    for gamma in (0.5, 0.75, 1.0):
        for ask_cost in (0.01, 0.03, 0.05):
            for expert_cost in (0.02, 0.05, 0.10):
                for margin in (0.0, 0.03, 0.06, 0.10):
                    per_case = defaultdict(list)
                    action_counts = {a: 0 for a in ACTIONS}
                    for sid in ids:
                        q = {}
                        pred_u = {}
                        actual_u = {}
                        for a in ACTIONS:
                            p = predictions[sid][a]
                            q[a] = min(1.0, max(0.0, priors[a] + gamma * (p["q"] - priors[a])))
                            c = p["costs"]
                            pred_u[a] = q[a] - ask_cost * c["patient_turns"] - expert_cost * (c["consult_tokens"] + c["defer_tokens"])
                            target = targets[sid][a]
                            actual_costs = {}
                            for comp in COST_COMPONENTS:
                                value = float(target["costs"][comp])
                                actual_costs[comp] = value if np.isfinite(value) else float(fallback_cost[a][comp])
                            actual_u[a] = (float(target["q"]) - ask_cost * actual_costs["patient_turns"]
                                           - expert_cost * (actual_costs["consult_tokens"] + actual_costs["defer_tokens"]))
                        best_alt = max((a for a in ACTIONS if a != "ANSWER"), key=lambda a: (pred_u[a], -ACTIONS.index(a)))
                        chosen = best_alt if pred_u[best_alt] - pred_u["ANSWER"] >= margin else "ANSWER"
                        action_counts[chosen] += 1
                        per_case[str(states[sid]["case_id"])].append(actual_u[chosen])
                    case_means = [float(np.mean(v)) for v in per_case.values()]
                    grid.append({
                        "q_shrinkage": gamma, "lambda_ask": ask_cost, "lambda_expert_shared": expert_cost,
                        "min_intervention_advantage": margin,
                        "validation_cases": len(per_case), "validation_states": len(ids),
                        "validation_case_macro_utility": float(np.mean(case_means)),
                        "validation_case_sd_utility": float(np.std(case_means, ddof=1)) if len(case_means) > 1 else 0.0,
                        "action_counts": json.dumps(action_counts, sort_keys=True),
                    })
    grid.sort(key=lambda r: (-r["validation_case_macro_utility"], r["q_shrinkage"], r["lambda_ask"], r["lambda_expert_shared"], r["min_intervention_advantage"]))
    write_csv(OUT / "offline_validation_policy_grid.csv", grid)
    best = grid[0]
    costs = {"ANSWER": 0.0, "ASK": best["lambda_ask"], "CONSULT": best["lambda_expert_shared"], "DEFER": best["lambda_expert_shared"]}
    configs = {
        "candidate_q_shrink_050": {"q_shrinkage": 0.5, "q_priors": priors, "costs": REFERENCE_COSTS, "max_consults": 1},
        "candidate_higher_cost": {"q_shrinkage": 1.0, "q_priors": priors, "costs": {"ANSWER": 0.0, "ASK": 0.03, "CONSULT": 0.10, "DEFER": 0.10}, "max_consults": 1},
        "candidate_gap_guard": {"q_shrinkage": 1.0, "q_priors": priors, "costs": REFERENCE_COSTS, "min_intervention_advantage": 0.06, "max_consults": 1},
        "candidate_one_ask": {"q_shrinkage": 1.0, "q_priors": priors, "costs": REFERENCE_COSTS, "max_asks": 1, "max_consults": 1},
        "candidate_combined": {"q_shrinkage": best["q_shrinkage"], "q_priors": priors, "costs": costs,
                               "min_intervention_advantage": best["min_intervention_advantage"],
                               "max_asks": 1, "max_consults": 1},
    }
    (OUT / "policy_candidate_configs.json").write_text(json.dumps(configs, indent=2) + "\n", encoding="utf-8")
    (OUT / "offline_selected_candidate.json").write_text(json.dumps({"selected_by": "case_macro_validation_reference_utility",
        "validation_split": "artifacts/case_split.json:validation", "candidate": best, "q_priors_from": "training cases only",
        "policy_config": configs["candidate_combined"]}, indent=2) + "\n", encoding="utf-8")

    data = pd.read_parquet(TEST_DATA)
    direct = json.loads(DIRECT_MANIFEST.read_text(encoding="utf-8"))
    old_test = json.loads(STEPWISE_MANIFEST.read_text(encoding="utf-8"))
    direct_rows = {int(x) for x in direct["sampled_source_rows"]}
    final_test_rows = {int(x) for x in old_test["selected_source_rows"]}
    eligible = set(range(len(data))) - direct_rows - final_test_rows
    if direct_rows & final_test_rows:
        raise RuntimeError("existing strong-expert subset overlaps the final Router test set")
    validation_rows = sorted(random.Random(20261107).sample(sorted(eligible), 50))
    if set(validation_rows) & final_test_rows or set(validation_rows) & direct_rows:
        raise RuntimeError("new validation selection overlaps an existing held-out set")
    (OUT / "validation_source_rows.json").write_text(json.dumps(validation_rows, indent=2) + "\n", encoding="utf-8")
    (OUT / "final_test_source_rows.json").write_text(json.dumps(sorted(final_test_rows), indent=2) + "\n", encoding="utf-8")
    print("offline validation grid best:", json.dumps(best, sort_keys=True))
    print("frozen candidate policies:", ", ".join(configs))
    print("online validation rows:", len(validation_rows), "final test rows:", len(final_test_rows),
          "direct-expert rows:", len(direct_rows), "all disjoint")


if __name__ == "__main__":
    main()
