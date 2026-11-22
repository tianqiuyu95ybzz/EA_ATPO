#!/usr/bin/env python3
"""Summarize the frozen minimal-fix policy on the reserved 200-case test set."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from router_core import load_jsonl


HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs/failure_analysis"
RUN = OUT / "final_test_one_ask"
BOOTSTRAPS = 10000


def bootstrap_ci(values, seed):
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(BOOTSTRAPS)
    for lo in range(0, BOOTSTRAPS, 250):
        n = min(250, BOOTSTRAPS - lo)
        means[lo:lo+n] = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def usage(row):
    result = {f"{actor}_{part}": 0.0 for actor in ("base", "patient", "expert") for part in ("prompt_tokens", "completion_tokens", "latency_s")}
    for event in row.get("trace", []):
        actor = event.get("actor")
        if actor in ("base", "patient", "expert"):
            for part in ("prompt_tokens", "completion_tokens", "latency_s"):
                result[f"{actor}_{part}"] += float(event.get(part, 0) or 0)
    result["total_tokens"] = sum(result[f"{a}_{p}_tokens"] for a in ("base", "patient", "expert") for p in ("prompt", "completion"))
    return result


def main():
    episodes = load_jsonl(RUN / "policy_episodes.jsonl")
    by_case = {}
    for e in episodes:
        by_case.setdefault(str(e["case_id"]), {})[str(e["policy"])] = e
    cases = sorted(by_case)
    policies = sorted({p for rows in by_case.values() for p in rows})
    if len(cases) != 200 or any(len(by_case[c]) != len(policies) for c in cases):
        raise RuntimeError("reserved test set is incomplete or policy rows are not paired")
    baseline = "always_answer"
    summaries = []
    paired = []
    vectors = {}
    for policy_index, policy in enumerate(policies):
        selected = [by_case[c][policy] for c in cases]
        acc = np.asarray([float(bool(e.get("correctness"))) for e in selected])
        utility = np.asarray([
            float(bool(e.get("correctness"))) - 0.01 * float(e.get("patient_turns", 0))
            - 0.02 * float(e.get("consult_expert_tokens", 0)) / 1000
            - 0.02 * float(e.get("defer_expert_tokens", 0)) / 1000
            for e in selected
        ])
        use = [usage(e) for e in selected]
        token_count = np.asarray([u["total_tokens"] for u in use])
        actions = Counter(a for e in selected for a in (e.get("decision_actions") or []))
        vectors[policy] = {"accuracy": acc, "utility": utility, "tokens": token_count}
        action_episodes = {a: sum(a in (e.get("decision_actions") or []) for e in selected) for a in ("ASK", "CONSULT", "DEFER")}
        summary = {
            "policy": policy, "n_cases": len(cases), "accuracy": float(acc.mean()),
            "accuracy_ci95_low": bootstrap_ci(acc, 20261130 + policy_index)[0],
            "accuracy_ci95_high": bootstrap_ci(acc, 20261130 + policy_index)[1],
            "reference_utility": float(utility.mean()),
            "mean_total_tokens": float(token_count.mean()),
            "mean_base_tokens": float(np.mean([u["base_prompt_tokens"] + u["base_completion_tokens"] for u in use])),
            "mean_patient_tokens": float(np.mean([u["patient_prompt_tokens"] + u["patient_completion_tokens"] for u in use])),
            "mean_expert_tokens": float(np.mean([u["expert_prompt_tokens"] + u["expert_completion_tokens"] for u in use])),
            "patient_turns_per_case": float(np.mean([e.get("patient_turns", 0) for e in selected])),
            "expert_calls_per_case": float(np.mean([e.get("expert_calls", 0) for e in selected])),
            "ASK_episode_rate": action_episodes["ASK"] / len(cases),
            "CONSULT_episode_rate": action_episodes["CONSULT"] / len(cases),
            "DEFER_episode_rate": action_episodes["DEFER"] / len(cases),
            "ASK_decisions": int(actions["ASK"]), "CONSULT_decisions": int(actions["CONSULT"]),
            "DEFER_decisions": int(actions["DEFER"]), "ANSWER_decisions": int(actions["ANSWER"]),
            "ASK_ge2_episodes": sum(int(e.get("ASK_count", 0) >= 2) for e in selected),
            "CONSULT_ge2_episodes": sum(int(e.get("CONSULT_count", 0) >= 2) for e in selected),
            "invalid_episodes": sum(not bool(e.get("valid")) for e in selected),
        }
        if policy != baseline:
            delta_acc = acc - vectors[baseline]["accuracy"]
            delta_util = utility - vectors[baseline]["utility"]
            summary.update({
                "accuracy_delta_vs_answer": float(delta_acc.mean()),
                "accuracy_delta_ci95_low": bootstrap_ci(delta_acc, 20261230 + policy_index)[0],
                "accuracy_delta_ci95_high": bootstrap_ci(delta_acc, 20261230 + policy_index)[1],
                "utility_delta_vs_answer": float(delta_util.mean()),
                "utility_delta_ci95_low": bootstrap_ci(delta_util, 20261330 + policy_index)[0],
                "utility_delta_ci95_high": bootstrap_ci(delta_util, 20261330 + policy_index)[1],
                "corrected_answer_errors": int(((vectors[baseline]["accuracy"] == 0) & (acc == 1)).sum()),
                "answer_errors": int((vectors[baseline]["accuracy"] == 0).sum()),
                "degraded_answer_correct": int(((vectors[baseline]["accuracy"] == 1) & (acc == 0)).sum()),
                "answer_correct": int((vectors[baseline]["accuracy"] == 1).sum()),
            })
        summaries.append(summary)
    fields = list(dict.fromkeys(k for r in summaries for k in r))
    with (OUT / "minimal_fix_final_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(summaries)

    baseline_rows = [by_case[c][baseline] for c in cases]
    candidate_rows = [by_case[c]["candidate_one_ask"] for c in cases]
    b = np.asarray([float(bool(e.get("correctness"))) for e in baseline_rows])
    x = np.asarray([float(bool(e.get("correctness"))) for e in candidate_rows])
    cells = {
        "both_correct": int(((b == 1) & (x == 1)).sum()),
        "baseline_correct_router_wrong": int(((b == 1) & (x == 0)).sum()),
        "baseline_wrong_router_correct": int(((b == 0) & (x == 1)).sum()),
        "both_wrong": int(((b == 0) & (x == 0)).sum()),
    }
    print("test cases", len(cases), "paired cells", json.dumps(cells, sort_keys=True))
    for r in summaries:
        print(r["policy"], "accuracy", r["accuracy"], "CI", r["accuracy_ci95_low"], r["accuracy_ci95_high"],
              "utility", r["reference_utility"], "tokens", r["mean_total_tokens"],
              "delta", r.get("accuracy_delta_vs_answer"), r.get("accuracy_delta_ci95_low"), r.get("accuracy_delta_ci95_high"),
              "utility_delta", r.get("utility_delta_vs_answer"), r.get("utility_delta_ci95_low"), r.get("utility_delta_ci95_high"))


if __name__ == "__main__":
    main()
