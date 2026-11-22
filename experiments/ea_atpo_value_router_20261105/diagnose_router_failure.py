#!/usr/bin/env python3
"""Audit the fitted EA-ATPO Router and its existing stepwise trajectories.

This script is read-only with respect to the registered checkpoint and runs.
It writes fresh diagnostics under outputs/failure_analysis/.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from router_core import ACTIONS, fit_reliability_oof, load_jsonl, predict_kernel_ridge
from train_router import (
    COST_COMPONENTS,
    _cost_observations,
    _fit_model,
    aggregate_records,
    predict,
)


HERE = Path(__file__).resolve().parent
PILOT = HERE.parent / "real_case_pilot_20261103/outputs/full/pilot_trajectories.jsonl"
SPLIT = HERE / "artifacts/case_split.json"
STEPWISE = HERE / "outputs/stepwise_test_v2/policy_episodes.jsonl"
FINAL_STEPWISE = HERE / "outputs/failure_analysis/final_test_one_ask/policy_episodes.jsonl"
OUT = HERE / "outputs/failure_analysis"
REFERENCE_COST = {"ASK": 0.01, "CONSULT": 0.02, "DEFER": 0.02}
POLICY_ORDER = (
    "always_answer", "uncertainty_threshold", "fixed_cost_router",
    "learned_value_no_cost", "learned_value_router", "learned_value_higher_cost",
    "ablation_no_expert_reliability", "ablation_no_consult", "ablation_no_defer",
    "candidate_one_ask",
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _corr(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return math.nan
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


def _rank(values: list[float]) -> list[float]:
    a = np.asarray(values, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    i = 0
    while i < len(a):
        j = i + 1
        while j < len(a) and a[order[j]] == a[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2 + 1
        i = j
    return ranks.tolist()


def _best(values: dict[str, float]) -> str:
    return max(ACTIONS, key=lambda a: (values[a], -ACTIONS.index(a)))


def _branch_utility(q: float, costs: dict[str, float]) -> float:
    return (q - REFERENCE_COST["ASK"] * costs["patient_turns"]
            - REFERENCE_COST["CONSULT"] * costs["consult_tokens"]
            - REFERENCE_COST["DEFER"] * costs["defer_tokens"])


def model_diagnostics() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = load_jsonl(PILOT)
    states, targets = aggregate_records(records)
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    split_cases = {k: set(map(str, split[k])) for k in ("train", "validation", "internal_test")}
    sets = list(split_cases.values())
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    all_cases = {str(s["case_id"]) for s in states.values()}
    assert set.union(*sets) == all_cases, "case split does not cover exactly the pilot cases"
    ids_by_split = {
        name: sorted(sid for sid, state in states.items() if str(state["case_id"]) in cases)
        for name, cases in split_cases.items()
    }
    model, fit_summary = _fit_model(ids_by_split["train"], states, targets, use_reliability=True)
    predictions: dict[str, dict[str, dict[str, Any]]] = {}
    train_ids = ids_by_split["train"]
    train_states = [states[sid] for sid in train_ids]
    base_x = model["vectorizer"].transform(train_states)
    case_ids = [str(states[sid]["case_id"]) for sid in train_ids]
    defer_q = np.asarray([float(targets[sid]["DEFER"]["q"]) for sid in train_ids])
    rel_oof, _, _ = fit_reliability_oof(base_x, defer_q, case_ids, 1.0, folds=5, seed=20261105)
    train_x = np.column_stack([base_x, rel_oof])
    for i, sid in enumerate(train_ids):
        predictions[sid] = {}
        for action in ACTIONS:
            raw = train_x[i:i+1] @ model["action_models"][action]["x_train"].T @ model["action_models"][action]["dual"]
            predictions[sid][action] = {
                "q": float(np.clip(raw[0, 0], 0.0, 1.0)),
                "costs": {c: float(max(0.0, raw[0, j+1])) for j, c in enumerate(COST_COMPONENTS)},
            }
    for name in ("validation", "internal_test"):
        predictions.update(predict(model, [states[sid] for sid in ids_by_split[name]]))

    raw_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        if row.get("record_type") == "trajectory":
            raw_groups[(str(row["state_id"]), str(row["action"]))].append(row)

    rows: list[dict[str, Any]] = []
    selected_by_split: dict[str, dict[str, int]] = {}
    gap_by_split: dict[str, list[float]] = {}
    for name, state_ids in ids_by_split.items():
        selected_by_split[name] = Counter()
        gaps = []
        chosen_matches = []
        for sid in state_ids:
            target_action_utils = {}
            pred_action_utils = {}
            for action in ACTIONS:
                item = targets[sid][action]
                target_cost = {}
                for component in COST_COMPONENTS:
                    value = float(item["costs"][component])
                    if not math.isfinite(value):
                        value = float(model["cost_means"][action][component])
                    target_cost[component] = value
                target_action_utils[action] = _branch_utility(float(item["q"]), target_cost)
                pred = predictions[sid][action]
                pred_action_utils[action] = _branch_utility(float(pred["q"]), pred["costs"])
            pbest, ebest = _best(pred_action_utils), _best(target_action_utils)
            selected_by_split[name][pbest] += 1
            chosen_matches.append(float(pbest == ebest))
            sorted_p = sorted(pred_action_utils.values(), reverse=True)
            gaps.append(float(sorted_p[0] - sorted_p[1]))
        gap_by_split[name] = gaps
        for action in ACTIONS:
            group = [sid for sid in state_ids if action in targets[sid]]
            valid_n = sum(bool(r.get("valid")) for sid in group for r in raw_groups[(sid, action)])
            rollout_n = sum(len(raw_groups[(sid, action)]) for sid in group)
            actual_q = [float(targets[sid][action]["q"]) for sid in group]
            predicted_q = [float(predictions[sid][action]["q"]) for sid in group]
            err = np.asarray(predicted_q) - np.asarray(actual_q)
            q_quantiles = np.quantile(actual_q, [0.0, 0.25, 0.5, 0.75, 1.0]) if actual_q else [math.nan] * 5
            pred_quantiles = np.quantile(predicted_q, [0.0, 0.25, 0.5, 0.75, 1.0]) if predicted_q else [math.nan] * 5
            rows.append({
                "split": name, "metric_type": "Q", "action": action, "component": "correctness",
                "case_count": len(split_cases[name]), "state_action_count": len(group),
                "raw_rollouts": rollout_n, "valid_rollouts": valid_n,
                "valid_rate": valid_n / rollout_n if rollout_n else math.nan,
                "empirical_mean": float(np.mean(actual_q)) if actual_q else math.nan,
                "empirical_sd": float(np.std(actual_q, ddof=1)) if len(actual_q) > 1 else math.nan,
                "empirical_q_min": float(q_quantiles[0]), "empirical_q_p25": float(q_quantiles[1]),
                "empirical_q_median": float(q_quantiles[2]), "empirical_q_p75": float(q_quantiles[3]),
                "empirical_q_max": float(q_quantiles[4]),
                "predicted_mean": float(np.mean(predicted_q)) if predicted_q else math.nan,
                "predicted_q_p25": float(pred_quantiles[1]), "predicted_q_median": float(pred_quantiles[2]),
                "predicted_q_p75": float(pred_quantiles[3]),
                "prediction_bias": float(np.mean(err)) if len(err) else math.nan,
                "mae": float(np.mean(np.abs(err))) if len(err) else math.nan,
                "mse": float(np.mean(err ** 2)) if len(err) else math.nan,
                "pearson_r": _corr(predicted_q, actual_q),
                "spearman_r": _corr(_rank(predicted_q), _rank(actual_q)),
                "predicted_action_rate": selected_by_split[name][action] / len(state_ids) if state_ids else math.nan,
                "empirical_best_action_rate": sum(_best({a: _branch_utility(
                    float(targets[sid][a]["q"]),
                    {c: float(targets[sid][a]["costs"][c]) if math.isfinite(float(targets[sid][a]["costs"][c])) else float(model["cost_means"][a][c]) for c in COST_COMPONENTS}
                ) for a in ACTIONS}) == action for sid in state_ids) / len(state_ids) if state_ids else math.nan,
                "predicted_empirical_best_agreement": float(np.mean(chosen_matches)) if chosen_matches else math.nan,
                "mean_predicted_best_second_gap": float(np.mean(gaps)) if gaps else math.nan,
            })
            for component in COST_COMPONENTS:
                observed, predicted_cost = [], []
                for sid in group:
                    raw_cost = float(targets[sid][action]["costs"][component])
                    observed.append(raw_cost if math.isfinite(raw_cost) else float(model["cost_means"][action][component]))
                    predicted_cost.append(float(predictions[sid][action]["costs"][component]))
                ce = np.asarray(predicted_cost) - np.asarray(observed)
                rows.append({
                    "split": name, "metric_type": "COST", "action": action, "component": component,
                    "case_count": len(split_cases[name]), "state_action_count": len(group),
                    "raw_rollouts": rollout_n, "valid_rollouts": valid_n,
                    "valid_rate": valid_n / rollout_n if rollout_n else math.nan,
                    "empirical_mean": float(np.mean(observed)) if observed else math.nan,
                    "empirical_sd": float(np.std(observed, ddof=1)) if len(observed) > 1 else math.nan,
                    "empirical_q_min": math.nan, "empirical_q_p25": math.nan,
                    "empirical_q_median": math.nan, "empirical_q_p75": math.nan, "empirical_q_max": math.nan,
                    "predicted_mean": float(np.mean(predicted_cost)) if predicted_cost else math.nan,
                    "predicted_q_p25": math.nan, "predicted_q_median": math.nan, "predicted_q_p75": math.nan,
                    "prediction_bias": float(np.mean(ce)) if len(ce) else math.nan,
                    "mae": float(np.mean(np.abs(ce))) if len(ce) else math.nan,
                    "mse": float(np.mean(ce ** 2)) if len(ce) else math.nan,
                    "pearson_r": _corr(predicted_cost, observed),
                    "spearman_r": _corr(_rank(predicted_cost), _rank(observed)),
                    "predicted_action_rate": selected_by_split[name][action] / len(state_ids) if state_ids else math.nan,
                    "empirical_best_action_rate": math.nan,
                    "predicted_empirical_best_agreement": float(np.mean(chosen_matches)) if chosen_matches else math.nan,
                    "mean_predicted_best_second_gap": float(np.mean(gaps)) if gaps else math.nan,
                })
        # A separate action-choice summary row makes it clear that this is a
        # state-level decision metric rather than an action-target regression.
        rows.append({
            "split": name, "metric_type": "ACTION_SELECTION", "action": "ALL", "component": "reference_utility",
            "case_count": len(split_cases[name]), "state_action_count": len(state_ids),
            "raw_rollouts": sum(len(raw_groups[(sid, a)]) for sid in state_ids for a in ACTIONS),
            "valid_rollouts": sum(bool(r.get("valid")) for sid in state_ids for a in ACTIONS for r in raw_groups[(sid, a)]),
            "valid_rate": math.nan, "empirical_mean": math.nan, "empirical_sd": math.nan,
            "empirical_q_min": math.nan, "empirical_q_p25": math.nan, "empirical_q_median": math.nan,
            "empirical_q_p75": math.nan, "empirical_q_max": math.nan,
            "predicted_mean": math.nan, "predicted_q_p25": math.nan, "predicted_q_median": math.nan,
            "predicted_q_p75": math.nan, "prediction_bias": math.nan, "mae": math.nan, "mse": math.nan,
            "pearson_r": math.nan, "spearman_r": math.nan,
            "predicted_action_rate": json.dumps({a: selected_by_split[name][a] / max(1, len(state_ids)) for a in ACTIONS}, sort_keys=True),
            "empirical_best_action_rate": json.dumps({a: sum(_best({b: _branch_utility(
                float(targets[sid][b]["q"]),
                {c: float(targets[sid][b]["costs"][c]) if math.isfinite(float(targets[sid][b]["costs"][c])) else float(model["cost_means"][b][c]) for c in COST_COMPONENTS}
            ) for b in ACTIONS}) == a for sid in state_ids) / max(1, len(state_ids)) for a in ACTIONS}, sort_keys=True),
            "predicted_empirical_best_agreement": float(np.mean(chosen_matches)) if chosen_matches else math.nan,
            "mean_predicted_best_second_gap": float(np.mean(gaps)) if gaps else math.nan,
        })
    return rows, {"fit_summary": fit_summary, "state_count": len(states), "case_count": len(all_cases),
                  "ids_by_split": ids_by_split, "targets": targets, "states": states, "model": model,
                  "predictions": predictions, "raw_groups": raw_groups, "split_cases": split_cases}


def _trace_usage(episode: dict[str, Any]) -> dict[str, float]:
    usage = {f"{actor}_{part}": 0.0 for actor in ("base", "patient", "expert") for part in ("prompt_tokens", "completion_tokens", "latency_s")}
    for event in episode.get("trace", []):
        actor = event.get("actor")
        if actor in ("base", "patient", "expert"):
            for part in ("prompt_tokens", "completion_tokens", "latency_s"):
                usage[f"{actor}_{part}"] += float(event.get(part, 0) or 0)
    usage["total_tokens"] = sum(usage[f"{a}_{p}_tokens"] for a in ("base", "patient", "expert") for p in ("prompt", "completion"))
    return usage


def stepwise_diagnostics() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    episodes = load_jsonl(STEPWISE)
    if FINAL_STEPWISE.exists():
        # The final rerun uses the exact frozen 200 source rows. Its Always
        # Answer and unmodified Router episodes supersede the earlier copies
        # only for the same case IDs; original files remain untouched.
        episodes.extend(load_jsonl(FINAL_STEPWISE))
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for e in episodes:
        grouped[str(e["case_id"])][str(e["policy"])] = e
    policies = [p for p in POLICY_ORDER if any(p in x for x in grouped.values())]
    cost_rows: list[dict[str, Any]] = []
    seq_rows: list[dict[str, Any]] = []
    case_index: dict[str, dict[str, dict[str, Any]]] = {}
    for policy in policies:
        these = [g[policy] for g in grouped.values() if policy in g]
        case_index[policy] = {str(e["case_id"]): e for e in these}

        def add_cost_row(scope: str, action: str, subset: list[dict[str, Any]], occurrences: int = 0) -> None:
            if not subset:
                return
            us = [_trace_usage(e) for e in subset]
            vals = {
                "policy": policy, "scope": scope, "action": action,
                "episodes": len(subset), "action_occurrences": occurrences,
                "accuracy": float(np.mean([float(bool(e.get("correctness"))) for e in subset])),
                "valid_rate": float(np.mean([float(bool(e.get("valid"))) for e in subset])),
                "patient_turns_per_episode": float(np.mean([float(e.get("patient_turns", 0)) for e in subset])),
                "expert_calls_per_episode": float(np.mean([float(e.get("expert_calls", 0)) for e in subset])),
                "base_prompt_tokens_per_episode": float(np.mean([u["base_prompt_tokens"] for u in us])),
                "base_completion_tokens_per_episode": float(np.mean([u["base_completion_tokens"] for u in us])),
                "patient_prompt_tokens_per_episode": float(np.mean([u["patient_prompt_tokens"] for u in us])),
                "patient_completion_tokens_per_episode": float(np.mean([u["patient_completion_tokens"] for u in us])),
                "expert_prompt_tokens_per_episode": float(np.mean([u["expert_prompt_tokens"] for u in us])),
                "expert_completion_tokens_per_episode": float(np.mean([u["expert_completion_tokens"] for u in us])),
                "total_tokens_per_episode": float(np.mean([u["total_tokens"] for u in us])),
                "inference_latency_s_per_episode": float(np.mean([u["base_latency_s"] + u["patient_latency_s"] + u["expert_latency_s"] for u in us])),
                "ask_ge2_episodes": sum(int(e.get("ASK_count", 0) >= 2) for e in subset),
                "consult_ge2_episodes": sum(int(e.get("CONSULT_count", 0) >= 2) for e in subset),
            }
            cost_rows.append(vals)

        add_cost_row("policy_total", "ALL", these)
        first_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        contains_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        sequences: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
        action_occurrences = Counter()
        for e in these:
            first_groups[str(e.get("first_action") or "NONE")].append(e)
            actions = list(e.get("decision_actions") or [])
            sequences[tuple(actions)].append(e)
            for a in actions:
                action_occurrences[a] += 1
            for a in set(actions):
                contains_groups[a].append(e)
        for action, subset in sorted(first_groups.items()):
            add_cost_row("first_action", action, subset)
        for action, subset in sorted(contains_groups.items()):
            add_cost_row("episode_contains_action", action, subset, action_occurrences[action])
        for seq, subset in sorted(sequences.items(), key=lambda x: (-len(x[1]), x[0])):
            us = [_trace_usage(e) for e in subset]
            seq_rows.append({
                "policy": policy, "action_sequence": " -> ".join(seq), "episodes": len(subset),
                "correct": sum(bool(e.get("correctness")) for e in subset),
                "accuracy": float(np.mean([float(bool(e.get("correctness"))) for e in subset])),
                "mean_total_tokens": float(np.mean([u["total_tokens"] for u in us])),
                "mean_patient_turns": float(np.mean([float(e.get("patient_turns", 0)) for e in subset])),
                "mean_expert_calls": float(np.mean([float(e.get("expert_calls", 0)) for e in subset])),
                "invalid_episodes": sum(not bool(e.get("valid")) for e in subset),
            })

    split_doc = json.loads(SPLIT.read_text(encoding="utf-8"))
    pilot_records = [r for r in load_jsonl(PILOT) if r.get("record_type") == "trajectory"]
    for split_name in ("train", "validation", "internal_test"):
        case_set = set(map(str, split_doc[split_name]))
        for action in ACTIONS:
            subset = [r for r in pilot_records if str(r["case_id"]) in case_set and r["action"] == action]
            if not subset:
                continue
            us = [_trace_usage(r) for r in subset]
            ask_counts = [sum(e.get("actor") == "base" and e.get("action") == "ASK" for e in r.get("trace", [])) for r in subset]
            consult_counts = [sum(e.get("actor") == "base" and e.get("action") == "CONSULT" for e in r.get("trace", [])) for r in subset]
            cost_rows.append({
                "policy": f"pilot_{split_name}", "scope": "pilot_counterfactual", "action": action,
                "episodes": len(subset), "action_occurrences": len(subset),
                "accuracy": float(np.mean([float(bool(r.get("correctness", r.get("outcome", 0)))) for r in subset])),
                "valid_rate": float(np.mean([float(bool(r.get("valid"))) for r in subset])),
                "patient_turns_per_episode": float(np.mean([float(r.get("patient_turns", 0)) for r in subset])),
                "expert_calls_per_episode": float(np.mean([sum(e.get("actor") == "expert" for e in r.get("trace", [])) for r in subset])),
                "base_prompt_tokens_per_episode": float(np.mean([u["base_prompt_tokens"] for u in us])),
                "base_completion_tokens_per_episode": float(np.mean([u["base_completion_tokens"] for u in us])),
                "patient_prompt_tokens_per_episode": float(np.mean([u["patient_prompt_tokens"] for u in us])),
                "patient_completion_tokens_per_episode": float(np.mean([u["patient_completion_tokens"] for u in us])),
                "expert_prompt_tokens_per_episode": float(np.mean([u["expert_prompt_tokens"] for u in us])),
                "expert_completion_tokens_per_episode": float(np.mean([u["expert_completion_tokens"] for u in us])),
                "total_tokens_per_episode": float(np.mean([u["total_tokens"] for u in us])),
                "inference_latency_s_per_episode": float(np.mean([u["base_latency_s"] + u["patient_latency_s"] + u["expert_latency_s"] for u in us])),
                "ask_ge2_episodes": sum(x >= 2 for x in ask_counts),
                "consult_ge2_episodes": sum(x >= 2 for x in consult_counts),
            })

    paired_rows: list[dict[str, Any]] = []
    baseline = case_index["always_answer"]
    for policy in policies:
        if policy == "always_answer":
            continue
        current = case_index[policy]
        common = sorted(set(baseline) & set(current))
        cells: dict[str, list[str]] = defaultdict(list)
        for case in common:
            b = bool(baseline[case].get("correctness"))
            r = bool(current[case].get("correctness"))
            key = ("both_correct" if b and r else "baseline_correct_router_wrong" if b else
                   "baseline_wrong_router_correct" if r else "both_wrong")
            cells[key].append(case)
        for cell in ("both_correct", "baseline_correct_router_wrong", "baseline_wrong_router_correct", "both_wrong"):
            paired_rows.append({"record_type": "paired_cell", "baseline_policy": "always_answer",
                                "router_policy": policy, "quadrant": cell, "case_id": "",
                                "baseline_correct": "", "router_correct": "", "baseline_first_action": "",
                                "router_first_action": "", "router_sequence": "", "router_total_tokens": "",
                                "n_cases": len(cells[cell]), "case_ids": json.dumps(cells[cell])})
        for case in common:
            b, r = baseline[case], current[case]
            paired_rows.append({
                "record_type": "case", "baseline_policy": "always_answer", "router_policy": policy,
                "quadrant": ("both_correct" if b["correctness"] and r["correctness"] else
                             "baseline_correct_router_wrong" if b["correctness"] else
                             "baseline_wrong_router_correct" if r["correctness"] else "both_wrong"),
                "case_id": case, "baseline_correct": int(bool(b.get("correctness"))),
                "router_correct": int(bool(r.get("correctness"))),
                "baseline_first_action": b.get("first_action"), "router_first_action": r.get("first_action"),
                "router_sequence": " -> ".join(r.get("decision_actions") or []),
                "router_total_tokens": _trace_usage(r)["total_tokens"], "n_cases": 1, "case_ids": "",
            })
    return cost_rows, seq_rows, paired_rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    q_rows, _ = model_diagnostics()
    cost_rows, seq_rows, paired_rows = stepwise_diagnostics()
    write_csv(OUT / "Q_VALUE_DIAGNOSTICS.csv", q_rows)
    write_csv(OUT / "ACTION_COST_BREAKDOWN.csv", cost_rows)
    write_csv(OUT / "PAIRED_ERROR_ANALYSIS.csv", paired_rows)
    write_csv(OUT / "ACTION_SEQUENCE_BREAKDOWN.csv", seq_rows)
    print(f"wrote Q diagnostics: {len(q_rows)} rows")
    print(f"wrote action cost breakdown: {len(cost_rows)} rows")
    print(f"wrote paired case analysis: {len(paired_rows)} rows")
    print(f"wrote action sequence analysis: {len(seq_rows)} rows")


if __name__ == "__main__":
    main()
