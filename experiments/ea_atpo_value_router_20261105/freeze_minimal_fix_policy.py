#!/usr/bin/env python3
"""Select and freeze one minimal-fix profile using only online validation data."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from router_core import file_sha256, load_jsonl


HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs/failure_analysis"
RUN = OUT / "online_validation"
POLICY_CONFIGS = OUT / "policy_candidate_configs.json"
BOOTSTRAPS = 10000


def ci(values, seed):
    a = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.asarray([a[rng.integers(0, len(a), size=len(a))].mean() for _ in range(BOOTSTRAPS)])
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def usage(row):
    result = {f"{a}_{k}": 0.0 for a in ("base", "patient", "expert") for k in ("prompt_tokens", "completion_tokens", "latency_s")}
    for e in row.get("trace", []):
        actor = e.get("actor")
        if actor in ("base", "patient", "expert"):
            for k in ("prompt_tokens", "completion_tokens", "latency_s"):
                result[f"{actor}_{k}"] += float(e.get(k, 0) or 0)
    result["total_tokens"] = sum(result[f"{a}_{k}"] for a in ("base", "patient", "expert") for k in ("prompt_tokens", "completion_tokens"))
    return result


def main():
    rows = load_jsonl(RUN / "policy_episodes.jsonl")
    cfgs = json.loads(POLICY_CONFIGS.read_text(encoding="utf-8"))
    by_case = {}
    for r in rows:
        by_case.setdefault(str(r["case_id"]), {})[str(r["policy"])] = r
    policies = sorted({str(r["policy"]) for r in rows})
    if len(by_case) != 50 or any(len(v) != len(policies) for v in by_case.values()):
        raise RuntimeError("online validation cases or policy rows are incomplete")
    if "always_answer" not in policies:
        raise RuntimeError("validation run lacks Always Answer reference")
    summaries = []
    values_by_policy = {}
    correct_by_policy = {}
    for policy in policies:
        selected = {case: by_case[case][policy] for case in by_case}
        correct = np.asarray([float(bool(selected[c].get("correctness"))) for c in sorted(selected)])
        utilities, tokens, patient_turns, consults, defers = [], [], [], [], []
        action_counts = Counter()
        for case in sorted(selected):
            row = selected[case]
            u = usage(row)
            util = (float(bool(row.get("correctness"))) - 0.01 * float(row.get("patient_turns", 0))
                    - 0.02 * float(row.get("consult_expert_tokens", 0)) / 1000
                    - 0.02 * float(row.get("defer_expert_tokens", 0)) / 1000)
            utilities.append(util)
            tokens.append(u["total_tokens"])
            patient_turns.append(float(row.get("patient_turns", 0)))
            consults.append(int(row.get("CONSULT_count", 0)))
            defers.append(int(row.get("DEFER_count", 0)))
            action_counts.update(row.get("decision_actions") or [])
        values_by_policy[policy] = np.asarray(utilities)
        correct_by_policy[policy] = correct
        summary = {
            "policy": policy, "n_cases": len(selected), "accuracy": float(correct.mean()),
            "accuracy_ci95_low": ci(correct.tolist(), 20261108 + policies.index(policy))[0],
            "accuracy_ci95_high": ci(correct.tolist(), 20261108 + policies.index(policy))[1],
            "reference_utility": float(np.mean(utilities)), "mean_total_tokens": float(np.mean(tokens)),
            "patient_turns_per_case": float(np.mean(patient_turns)),
            "consult_calls_per_case": float(np.mean(consults)), "defer_calls_per_case": float(np.mean(defers)),
            "action_decisions": json.dumps({a: int(action_counts[a]) for a in ("ANSWER", "ASK", "CONSULT", "DEFER")}, sort_keys=True),
        }
        if policy != "always_answer":
            delta = values_by_policy[policy] - values_by_policy["always_answer"]
            acc_delta = correct_by_policy[policy] - correct_by_policy["always_answer"]
            summary.update({
                "paired_utility_delta_vs_answer": float(delta.mean()),
                "paired_utility_delta_ci95_low": ci(delta.tolist(), 20262108 + policies.index(policy))[0],
                "paired_utility_delta_ci95_high": ci(delta.tolist(), 20262108 + policies.index(policy))[1],
                "paired_accuracy_delta_vs_answer": float(acc_delta.mean()),
                "paired_accuracy_delta_ci95_low": ci(acc_delta.tolist(), 20263108 + policies.index(policy))[0],
                "paired_accuracy_delta_ci95_high": ci(acc_delta.tolist(), 20263108 + policies.index(policy))[1],
            })
        summaries.append(summary)
    write_path = OUT / "online_validation_selection.csv"
    with write_path.open("w", newline="", encoding="utf-8") as f:
        fields = list(dict.fromkeys(key for row in summaries for key in row))
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(summaries)
    # Selection objective is fixed in advance: highest case-mean reference utility;
    # ties prefer higher accuracy, then lower tokens, then stable policy name.
    selected = max(summaries, key=lambda r: (r["reference_utility"], r["accuracy"], -r["mean_total_tokens"], r["policy"]))
    policy = selected["policy"]
    payload = {
        "selection_status": "frozen_after_online_validation",
        "selected_policy": policy,
        "selection_objective": "maximize mean(correctness - 0.01*patient_turns - 0.02*consult_expert_tokens/1000 - 0.02*defer_expert_tokens/1000) on the 50-case online validation set; ties prefer accuracy then fewer tokens",
        "validation_cases": len(by_case), "validation_episode_sha256": file_sha256(RUN / "policy_episodes.jsonl"),
        "validation_summary_sha256": file_sha256(write_path), "selected_validation_metrics": selected,
        "selected_policy_config": cfgs.get(policy),
        "candidate_source": "pre-frozen profile grid; only online validation outcomes used for final profile selection",
    }
    (OUT / "frozen_final_policy.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("frozen policy:", policy)
    print("validation accuracy:", selected["accuracy"], "reference utility:", selected["reference_utility"],
          "tokens/case:", selected["mean_total_tokens"])
    print("summary:", write_path)


if __name__ == "__main__":
    main()
