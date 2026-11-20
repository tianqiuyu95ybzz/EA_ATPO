#!/usr/bin/env python3
"""Recompute previously reported Dev metrics from saved labels and outcomes.

This is an audit of the already-used 15-case Dev set, not a new evaluation.
The script extracts only identifiers, action correctness, cumulative costs,
costed-Q values, and saved predictions from prior rollout records. It never
sends clinical text to a model or emits case-level content. Frozen Holdout is
not read.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


ACTION_ORDER = ("ASK", "ANSWER", "DEFER")
SEEDS = (17, 42, 2026)


def auc(labels, scores):
    y = [bool(x) for x in labels]
    s = [float(x) for x in scores]
    pos, neg = sum(y), len(y) - sum(y)
    if not pos or not neg:
        return None
    wins = 0.0
    for i, label in enumerate(y):
        if label:
            wins += sum(1.0 if s[i] > s[j] else 0.5 if s[i] == s[j] else 0.0
                        for j, other in enumerate(y) if not other)
    return wins / (pos * neg)


def mean(values):
    return sum(values) / len(values) if values else 0.0


def case_metrics(actions, rows, q_by_key, correctness):
    grouped = defaultdict(lambda: defaultdict(list))
    for row, action in zip(rows, actions):
        key = (row["case_id"], row["state_id"])
        grouped[row["case_id"]]["accuracy"].append(correctness[key][action])
        grouped[row["case_id"]]["utility"].append(q_by_key[key][action])
        for candidate in ACTION_ORDER:
            grouped[row["case_id"]][f"{candidate.lower()}_rate"].append(float(action == candidate))
    return {cid: {metric: mean(values) for metric, values in fields.items()}
            for cid, fields in grouped.items()}


def summarize(case_map):
    keys = sorted(next(iter(case_map.values())))
    return {k: mean([case_map[c][k] for c in case_map]) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path.cwd())
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    root = args.root.resolve()
    base = root / "experiments/ea_atpo_v2/router_training/outputs/real_q_router_20261120"
    split = json.loads((base / "case_split_35_train_15_dev.json").read_text())
    dev_ids = set(map(str, split["dev"]))
    q_by_key, correct_by_key, rows = {}, {}, []
    with (base / "q_dataset_cost_ask001_defer005.jsonl").open(encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            md = item["metadata"]
            cid, sid = str(md["case_id"]), str(md["state_id"])
            if cid not in dev_ids:
                continue
            q = {a: float(item[f"Q_{a}"]) for a in ACTION_ORDER}
            q_by_key[(cid, sid)] = q
            rows.append({"case_id": cid, "state_id": sid})
    if len(rows) != 43 or len({r["case_id"] for r in rows}) != 15:
        raise ValueError("the expected 43-state/15-case historical Dev set was not found")

    # Recover correctness from the already observed rollout outcomes, not from
    # the per-action unit cost: Q includes cumulative ASK/DEFER counts.
    trajectory_path = root / "experiments/ea_atpo_v2/recovery_bundle_20261120/pilot_trajectories.jsonl"
    with trajectory_path.open(encoding="utf-8") as f:
        for line in f:
            tr = json.loads(line)
            cid, sid, action = str(tr["case_id"]), str(tr["state_id"]), str(tr["action"])
            if cid not in dev_ids or action not in ACTION_ORDER or not tr.get("valid", False):
                continue
            key = (cid, sid)
            correct_by_key.setdefault(key, {})[action] = int(round(float(tr["final_accuracy"])))
            expected_q = (float(tr["final_accuracy"])
                          - .01 * int(tr.get("ask_count", 0))
                          - .05 * int(tr.get("defer_count", 0)))
            if abs(expected_q - q_by_key[key][action]) > 1e-8:
                raise ValueError(f"historical costed Q does not reconcile for action {action}")
    if any(set(correct_by_key.get((r["case_id"], r["state_id"]), {})) != set(ACTION_ORDER) for r in rows):
        raise ValueError("historical rollout correctness is incomplete for a Dev state")

    pred_path = base / "training_final/dev_predictions.jsonl"
    training_results = json.loads((base / "training_final/training_results.json").read_text())
    saved = defaultdict(dict)
    with pred_path.open(encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            if p.get("feature") != "embedding" or p.get("model") not in ("mlp_relative", "q_guided_distillation"):
                continue
            if p.get("case_id") not in dev_ids:
                continue
            saved[(p["model"], int(p["seed"]))][(p["case_id"], p["state_id"])] = p

    # Keep state order deterministic so every policy is compared on identical rows.
    rows.sort(key=lambda r: (r["case_id"], r["state_id"]))
    baseline_case_maps = {}
    baseline_results = {}
    for action in ACTION_ORDER:
        action_vec = [action] * len(rows)
        cm = case_metrics(action_vec, rows, q_by_key, correct_by_key)
        baseline_case_maps[action] = cm
        baseline_results[action] = summarize(cm)

    out = {"status": "recomputed_prior_dev_only", "cases": 15, "states": 43,
           "source": "Q targets already used for training/model selection; no new cases",
           "accuracy_reconstruction": "correctness comes from terminal rollout outcome; Q is reconciled to correctness - 0.01*cumulative_ASK - 0.05*cumulative_DEFER",
           "base_expert_four_cells_state_count": {"base_correct_expert_correct": 0,
             "base_correct_expert_wrong": 0, "base_wrong_expert_correct": 0, "base_wrong_expert_wrong": 0},
           "baselines_case_macro": baseline_results, "policies": {},
           "bootstrap": {"unit": "case_id", "replicates": 5000, "seed": 20261120,
                         "ci_source": "saved original training_results.json; point and case metrics are reconciled here"}}
    cell_names = {(1,1): "base_correct_expert_correct", (1,0): "base_correct_expert_wrong",
                  (0,1): "base_wrong_expert_correct", (0,0): "base_wrong_expert_wrong"}
    for key, name in cell_names.items():
        out["base_expert_four_cells_state_count"][name] = sum(
            (correct_by_key[(r["case_id"], r["state_id"])]["ANSWER"],
             correct_by_key[(r["case_id"], r["state_id"])]["DEFER"]) == key for r in rows)
    out["base_expert_four_cells_case_weighted"] = {
        name: mean([
            mean([int((correct_by_key[(r["case_id"], r["state_id"])]["ANSWER"],
                            correct_by_key[(r["case_id"], r["state_id"])]["DEFER"]) == key)
                              for r in rows if r["case_id"] == cid])
            for cid in sorted(dev_ids)]) for key, name in cell_names.items()}

    for family in ("mlp_relative", "q_guided_distillation"):
        seed_maps = []
        seed_rates_state_weighted = []
        for seed in SEEDS:
            map_rows = saved[(family, seed)]
            if set(map_rows) != {(r["case_id"], r["state_id"]) for r in rows}:
                raise ValueError(f"saved predictions incomplete for {family} seed {seed}")
            if family == "q_guided_distillation":
                actions = [map_rows[(r["case_id"], r["state_id"])]["predicted_action"] for r in rows]
            else:
                actions = [map_rows[(r["case_id"], r["state_id"])]["predicted_action"] for r in rows]
            cm = case_metrics(actions, rows, q_by_key, correct_by_key)
            seed_maps.append(cm)
            seed_rates_state_weighted.append({f"{a.lower()}_rate": mean([float(x == a) for x in actions]) for a in ACTION_ORDER})
        metric_means = {metric: mean([summarize(x)[metric] for x in seed_maps])
                        for metric in ("accuracy", "utility", "ask_rate", "answer_rate", "defer_rate")}
        # State-weighted action frequencies are also reported to expose denominator choices.
        state_rates = {k: mean([row[k] for row in seed_rates_state_weighted])
                       for k in seed_rates_state_weighted[0]}
        per_seed = {str(seed): summarize(cm) for seed, cm in zip(SEEDS, seed_maps)}
        source_runs = [x for x in training_results["results"]
                       if x.get("feature") == "embedding" and x.get("model") == family]
        source_summary = next(x for x in training_results["policy_summary_mean_sd_and_case_clustered_ci"]
                              if x.get("feature") == "embedding" and x.get("model") == family)
        max_case_metric_diff = 0.0
        for source in source_runs:
            seed_map = seed_maps[SEEDS.index(int(source["seed"]))]
            for cid, metrics in seed_map.items():
                recorded = source["dev_case_metrics"][cid]
                for metric, recorded_metric in (("accuracy", "accuracy"), ("utility", "utility"),
                                                ("ask_rate", "ask"), ("answer_rate", "answer"),
                                                ("defer_rate", "defer")):
                    max_case_metric_diff = max(max_case_metric_diff,
                                               abs(metrics[metric] - float(recorded[recorded_metric])))
        result = {"case_macro_mean_over_three_seeds": metric_means,
                  "state_weighted_action_rates_mean_over_three_seeds": state_rates,
                  "per_seed_case_macro": per_seed,
                  "saved_case_metrics_max_abs_difference": max_case_metric_diff,
                  "saved_case_cluster_bootstrap_ci_vs_always_answer": {
                      "accuracy": source_summary["paired_vs_always_answer_delta_accuracy_ci95"],
                      "utility": source_summary["paired_vs_always_answer_delta_utility_ci95"]},
                  "bootstrap_ci_recomputed_here": False}
        if family == "mlp_relative":
            target_ask = [q_by_key[(r["case_id"], r["state_id"])]["ASK"]-
                          q_by_key[(r["case_id"], r["state_id"])]["ANSWER"] for r in rows]
            target_defer = [q_by_key[(r["case_id"], r["state_id"])]["DEFER"]-
                            q_by_key[(r["case_id"], r["state_id"])]["ANSWER"] for r in rows]
            mae_by_seed, auc_ask_by_seed, auc_defer_by_seed = [], [], []
            for seed in SEEDS:
                p_ask, p_defer = [], []
                for r in rows:
                    k=(r["case_id"],r["state_id"])
                    scores=saved[(family,seed)][k]["predicted_scores"]
                    p_ask.append(float(scores[0])); p_defer.append(float(scores[2]))
                mae_by_seed.append(mean([abs(p-y) for p,y in zip(p_ask,target_ask)] +
                                        [abs(p-y) for p,y in zip(p_defer,target_defer)]))
                auc_ask_by_seed.append(auc([y > 0 for y in target_ask],p_ask))
                auc_defer_by_seed.append(auc([y > 0 for y in target_defer],p_defer))
            result["relative_advantage_mae_state_weighted_seed_mean"] = mean(mae_by_seed)
            result["relative_advantage_positive_auroc_state_level"] = {
                "ASK": mean(auc_ask_by_seed), "DEFER": mean(auc_defer_by_seed)}
        else:
            result["q_prediction_mae"] = None
            result["q_prediction_mae_note"] = "Distillation head outputs policy logits, not Q values; Q MAE/AUROC are undefined."
        out["policies"][family] = result

    target = args.output if args.output.is_absolute() else root / args.output
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
