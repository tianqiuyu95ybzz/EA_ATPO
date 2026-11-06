#!/usr/bin/env python3
"""Fit the four-action outcome-value model with case-disjoint splits."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from router_core import (
    ACTIONS,
    PublicStateVectorizer,
    file_sha256,
    fit_kernel_ridge,
    fit_reliability_oof,
    load_jsonl,
    predict_kernel_ridge,
    stable_split,
    write_json,
)


HERE = Path(__file__).resolve().parent
DEFAULT_PILOT = HERE.parent / "real_case_pilot_20261103/outputs/full/pilot_trajectories.jsonl"
DEFAULT_OUT = HERE / "artifacts"
ALPHA = 1.0
MAX_FEATURES = 3000
REFERENCE_COST = {"ASK": 0.01, "CONSULT": 0.02, "DEFER": 0.02}
COST_COMPONENTS = ("patient_turns", "consult_tokens", "defer_tokens")


def _cost_observations(row: dict[str, Any]) -> dict[str, tuple[float | None, bool]]:
    action = str(row["action"])
    valid = bool(row.get("valid"))
    trace = row.get("trace") or []
    patient_turns = float(sum(x.get("actor") == "patient" for x in trace))
    consult_tokens = float(sum(
        int(x.get("prompt_tokens", 0) or 0) + int(x.get("completion_tokens", 0) or 0)
        for x in trace if x.get("actor") == "expert" and x.get("action") == "CONSULT"
    )) / 1000.0
    defer_tokens = float(sum(
        int(x.get("prompt_tokens", 0) or 0) + int(x.get("completion_tokens", 0) or 0)
        for x in trace if x.get("actor") == "expert" and x.get("action") == "DEFER"
    )) / 1000.0
    values = {"patient_turns": patient_turns, "consult_tokens": consult_tokens, "defer_tokens": defer_tokens}
    reachable = {
        "ANSWER": set(),
        "ASK": set(COST_COMPONENTS),
        "CONSULT": set(COST_COMPONENTS),
        "DEFER": {"defer_tokens"},
    }[action]
    observed_event = {
        "patient_turns": patient_turns > 0,
        "consult_tokens": consult_tokens > 0,
        "defer_tokens": defer_tokens > 0,
    }
    result = {}
    for component, value in values.items():
        missing = bool(not valid and component in reachable and not observed_event[component])
        result[component] = (None if missing else value, missing)
    return result


def aggregate_records(records: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    states: dict[str, dict[str, Any]] = {}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        if row.get("record_type") != "trajectory":
            continue
        sid = str(row["state_id"])
        public_state = row.get("public_state")
        if not isinstance(public_state, list):
            raise ValueError(f"state {sid} has no public_state message list")
        prior = states.setdefault(sid, {
            "state_id": sid,
            "case_id": str(row["case_id"]),
            "public_state": public_state,
        })
        if prior["case_id"] != str(row["case_id"]) or prior["public_state"] != public_state:
            raise ValueError(f"state_id collision or nonidentical public state: {sid}")
        grouped[(sid, str(row["action"]))].append(row)

    targets: dict[str, dict[str, Any]] = defaultdict(dict)
    for (sid, action), group in grouped.items():
        q = [float(row.get("correctness", row.get("outcome", 0.0)) or 0.0) for row in group]
        q_mean = float(np.mean(q))
        observed_costs: dict[str, list[float]] = {key: [] for key in COST_COMPONENTS}
        missing_counts: dict[str, int] = {key: 0 for key in COST_COMPONENTS}
        for row in group:
            observations = _cost_observations(row)
            for component in COST_COMPONENTS:
                value, missing = observations[component]
                if missing:
                    missing_counts[component] += 1
                elif value is not None:
                    observed_costs[component].append(float(value))
        cost_means = {key: float(np.mean(v)) if v else math.nan for key, v in observed_costs.items()}
        targets[sid][action] = {"q": q_mean, "costs": cost_means, "cost_missing_n": missing_counts, "n": float(len(group))}
    return states, targets


def _fill_cost_targets(
    state_ids: list[str], targets: dict[str, dict[str, Any]], actions: tuple[str, ...],
    fallback: dict[str, dict[str, float]] | None = None,
) -> tuple[dict[str, dict[str, dict[str, float]]], dict[str, dict[str, float]]]:
    observed: dict[str, dict[str, list[float]]] = {a: {c: [] for c in COST_COMPONENTS} for a in actions}
    for sid in state_ids:
        for action in actions:
            item = targets.get(sid, {}).get(action)
            if item:
                for component in COST_COMPONENTS:
                    value = float(item["costs"][component])
                    if math.isfinite(value):
                        observed[action][component].append(value)
    means = {a: dict((fallback or {}).get(a, {})) for a in actions}
    for action in actions:
        for component in COST_COMPONENTS:
            if observed[action][component]:
                means[action][component] = float(np.mean(observed[action][component]))
            means[action].setdefault(component, 0.0)
    filled: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for sid in state_ids:
        for action in actions:
            item = targets.get(sid, {}).get(action)
            if item is None:
                continue
            filled[sid][action] = {}
            for component in COST_COMPONENTS:
                cost = float(item["costs"][component])
                filled[sid][action][component] = cost if math.isfinite(cost) else means[action][component]
    return filled, means


def _fit_model(
    state_ids: list[str], states: dict[str, dict[str, Any]], targets: dict[str, dict[str, Any]],
    *, use_reliability: bool, alpha: float = ALPHA, max_features: int = MAX_FEATURES,
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected_states = [states[sid] for sid in state_ids]
    vectorizer = PublicStateVectorizer(max_features).fit(selected_states)
    base_x = vectorizer.transform(selected_states)
    case_ids = [str(s["case_id"]) for s in selected_states]
    costs, cost_means = _fill_cost_targets(state_ids, targets, ACTIONS)
    defer_q = np.asarray([float(targets[sid]["DEFER"]["q"]) for sid in state_ids])
    if use_reliability:
        reliability_oof, rel_x, rel_dual = fit_reliability_oof(
            base_x, defer_q, case_ids, alpha, folds=5, seed=20261105
        )
        model_x = np.column_stack([base_x, reliability_oof])
    else:
        reliability_oof = None
        rel_x = rel_dual = None
        model_x = base_x

    action_models: dict[str, dict[str, np.ndarray]] = {}
    for action in ACTIONS:
        q = np.asarray([float(targets[sid][action]["q"]) for sid in state_ids])
        c = np.asarray([[float(costs[sid][action][component]) for component in COST_COMPONENTS] for sid in state_ids])
        dual = fit_kernel_ridge(model_x, np.column_stack([q, c]), alpha)
        action_models[action] = {"x_train": model_x.copy(), "dual": dual}

    model = {
        "vectorizer": vectorizer,
        "use_reliability": use_reliability,
        "rel_x": rel_x,
        "rel_dual": rel_dual,
        "action_models": action_models,
        "cost_means": cost_means,
        "train_state_ids": list(state_ids),
    }
    metadata = {"cost_means": cost_means, "training_states": len(state_ids), "training_cases": len(set(case_ids))}
    if reliability_oof is not None:
        metadata["reliability_oof_mean"] = float(np.mean(reliability_oof))
        metadata["reliability_oof_sd"] = float(np.std(reliability_oof))
    return model, metadata


def predict(model: dict[str, Any], states: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    if not states:
        return {}
    base_x = model["vectorizer"].transform(states)
    if model["use_reliability"]:
        reliability = np.clip(
            predict_kernel_ridge(base_x, model["rel_x"], model["rel_dual"]), 0.0, 1.0
        )
        features = np.column_stack([base_x, reliability])
    else:
        reliability = np.full(len(states), np.nan)
        features = base_x
    result: dict[str, dict[str, dict[str, float]]] = {}
    for i, state in enumerate(states):
        sid = str(state["state_id"])
        result[sid] = {}
        for action in ACTIONS:
            raw = predict_kernel_ridge(
                features[i : i + 1], model["action_models"][action]["x_train"], model["action_models"][action]["dual"]
            )
            result[sid][action] = {
                "q": float(np.clip(raw[0, 0], 0.0, 1.0)),
                "costs": {component: float(max(0.0, raw[0, index + 1])) for index, component in enumerate(COST_COMPONENTS)},
                "expert_reliability": float(reliability[i]),
            }
    return result


def _save_model(model: dict[str, Any], out_dir: Path, metadata: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    if model["use_reliability"]:
        arrays["reliability_x"] = model["rel_x"]
        arrays["reliability_dual"] = model["rel_dual"]
    for action, item in model["action_models"].items():
        arrays[f"{action}_x_train"] = item["x_train"]
        arrays[f"{action}_dual"] = item["dual"]
    np.savez_compressed(out_dir / "router_checkpoint.npz", **arrays)
    payload = {
        "vectorizer": model["vectorizer"].to_json(),
        "use_reliability": model["use_reliability"],
        "actions": list(ACTIONS),
        "alpha": ALPHA,
        "cost_means": model["cost_means"],
        "train_state_ids": model["train_state_ids"],
        "training": metadata,
    }
    write_json(out_dir / "router_checkpoint.json", payload)


def _load_model(out_dir: Path) -> dict[str, Any]:
    from router_core import PublicStateVectorizer
    meta = json.loads((out_dir / "router_checkpoint.json").read_text(encoding="utf-8"))
    arrays = np.load(out_dir / "router_checkpoint.npz")
    action_models = {
        action: {"x_train": arrays[f"{action}_x_train"], "dual": arrays[f"{action}_dual"]}
        for action in ACTIONS
    }
    return {
        "vectorizer": PublicStateVectorizer.from_json(meta["vectorizer"]),
        "use_reliability": bool(meta["use_reliability"]),
        "rel_x": arrays["reliability_x"] if "reliability_x" in arrays else None,
        "rel_dual": arrays["reliability_dual"] if "reliability_dual" in arrays else None,
        "action_models": action_models,
        "cost_means": meta["cost_means"],
        "train_state_ids": meta["train_state_ids"],
    }


def _mean_utility(targets, state_ids, action, costs):
    q, component_values = [], {c: [] for c in COST_COMPONENTS}
    filled, _means = _fill_cost_targets(state_ids, targets, ACTIONS)
    for sid in state_ids:
        q.append(float(targets[sid][action]["q"]))
        for component in COST_COMPONENTS:
            component_values[component].append(float(filled[sid][action][component]))
    utility = np.asarray(q)
    utility -= float(costs.get("ASK", 0.0)) * np.asarray(component_values["patient_turns"])
    utility -= float(costs.get("CONSULT", 0.0)) * np.asarray(component_values["consult_tokens"])
    utility -= float(costs.get("DEFER", 0.0)) * np.asarray(component_values["defer_tokens"])
    return float(np.mean(utility))


def _choose(values: dict[str, dict[str, float]], costs: dict[str, float], allowed=ACTIONS) -> str:
    rank = {a: i for i, a in enumerate(ACTIONS)}
    def utility(action: str) -> float:
        item = values[action]
        components = item["costs"]
        return (item["q"]
                - costs.get("ASK", 0.0) * components["patient_turns"]
                - costs.get("CONSULT", 0.0) * components["consult_tokens"]
                - costs.get("DEFER", 0.0) * components["defer_tokens"])
    return max(allowed, key=lambda a: (utility(a), -rank[a]))


def _offline_metrics(case_states, target_map, prediction_map, costs, cost_fallback, actions=ACTIONS):
    result = {"n_states": len(case_states), "policy_accuracy": 0.0, "mean_predicted_utility": 0.0, "mean_observed_utility": 0.0, "action_rates": {a: 0.0 for a in ACTIONS}}
    actual_utils, pred_utils, chosen_q = [], [], []
    for state in case_states:
        sid = str(state["state_id"])
        action = _choose(prediction_map[sid], costs, actions)
        actual = target_map[sid][action]
        actual_costs = {}
        for component in COST_COMPONENTS:
            value = float(actual["costs"][component])
            if not math.isfinite(value):
                value = float(cost_fallback.get(action, {}).get(component, 0.0))
            actual_costs[component] = value
        predicted_costs = prediction_map[sid][action]["costs"]
        actual_utils.append(float(actual["q"])
                            - costs.get("ASK", 0.0) * actual_costs["patient_turns"]
                            - costs.get("CONSULT", 0.0) * actual_costs["consult_tokens"]
                            - costs.get("DEFER", 0.0) * actual_costs["defer_tokens"])
        pred_utils.append(prediction_map[sid][action]["q"]
                          - costs.get("ASK", 0.0) * predicted_costs["patient_turns"]
                          - costs.get("CONSULT", 0.0) * predicted_costs["consult_tokens"]
                          - costs.get("DEFER", 0.0) * predicted_costs["defer_tokens"])
        chosen_q.append(float(actual["q"]))
        result["action_rates"][action] += 1
    n = max(1, len(case_states))
    result["policy_accuracy"] = float(np.mean(chosen_q)) if chosen_q else math.nan
    result["mean_predicted_utility"] = float(np.mean(pred_utils)) if pred_utils else math.nan
    result["mean_observed_utility"] = float(np.mean(actual_utils)) if actual_utils else math.nan
    result["action_rates"] = {k: v / n for k, v in result["action_rates"].items()}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=20261105)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_jsonl(args.pilot)
    states, targets = aggregate_records(records)
    case_ids = sorted({str(s["case_id"]) for s in states.values()})
    split = stable_split(case_ids, args.seed)
    write_json(args.output_dir / "case_split.json", {
        "seed": args.seed,
        "unit": "case_id",
        "train": split["train"], "validation": split["validation"], "internal_test": split["internal_test"],
        "counts": {k: len(v) for k, v in split.items()},
        "pilot_trajectory_sha256": file_sha256(args.pilot),
    })

    states_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for state in states.values():
        states_by_case[state["case_id"]].append(state)
    state_ids_by_split = {
        name: sorted(sid for sid, state in states.items() if state["case_id"] in cases)
        for name, cases in split.items()
    }
    train_ids = state_ids_by_split["train"]
    train_model, train_fit_summary = _fit_model(train_ids, states, targets, use_reliability=True)
    val_states = [states[sid] for sid in state_ids_by_split["validation"]]
    int_states = [states[sid] for sid in state_ids_by_split["internal_test"]]
    val_pred = predict(train_model, val_states)
    int_pred = predict(train_model, int_states)

    # Freeze the simple uncertainty-threshold baseline using only validation cases.
    train_means = {a: _mean_utility(targets, train_ids, a, REFERENCE_COST) for a in ACTIONS}
    threshold_alt = max((a for a in ACTIONS if a != "ANSWER"), key=lambda a: (train_means[a], -ACTIONS.index(a)))
    candidate_thresholds = sorted({0.0, 1.0, *[float(val_pred[s["state_id"]]["ANSWER"]["q"]) for s in val_states]})
    val_costs, _ = _fill_cost_targets(state_ids_by_split["validation"], targets, ACTIONS, train_model["cost_means"])
    best_threshold, best_utility = 1.0, -float("inf")
    for threshold in candidate_thresholds:
        vals = []
        for state in val_states:
            sid = str(state["state_id"])
            chosen = "ANSWER" if val_pred[sid]["ANSWER"]["q"] >= threshold else threshold_alt
            chosen_cost = val_costs[sid][chosen]
            vals.append(float(targets[sid][chosen]["q"])
                        - REFERENCE_COST["ASK"] * chosen_cost["patient_turns"]
                        - REFERENCE_COST["CONSULT"] * chosen_cost["consult_tokens"]
                        - REFERENCE_COST["DEFER"] * chosen_cost["defer_tokens"])
        score = float(np.mean(vals)) if vals else -float("inf")
        # Prefer the less-interventionist/higher threshold when tied.
        if score > best_utility + 1e-12 or (abs(score - best_utility) <= 1e-12 and threshold > best_threshold):
            best_threshold, best_utility = threshold, score
    threshold_config = {
        "uncertainty_threshold": 1.0 - best_threshold,
        "equivalent_answer_confidence_threshold": best_threshold,
        "alternative_action": threshold_alt,
        "selection_metric": "validation mean utility under reference cost",
        "validation_cases": split["validation"],
        "validation_mean_utility": best_utility,
        "cost": REFERENCE_COST,
    }
    write_json(args.output_dir / "baseline_policy_config.json", threshold_config)

    offline_rows = []
    for split_name, prediction, current_states in (
        ("validation", val_pred, val_states), ("internal_test", int_pred, int_states)
    ):
        # No-cost and reference-cost model-selected action values are evaluated
        # against the held-out counterfactual labels only for diagnostics.
        for cost_name, cost_set in (("no_cost", {a: 0.0 for a in ACTIONS}), ("reference", REFERENCE_COST)):
            metric = _offline_metrics(current_states, targets, prediction, cost_set, train_model["cost_means"])
            offline_rows.append({"split": split_name, "cost_set": cost_name, **{k: json.dumps(v, sort_keys=True) if isinstance(v, dict) else v for k, v in metric.items()}})
    with (args.output_dir / "offline_split_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(offline_rows[0]))
        writer.writeheader(); writer.writerows(offline_rows)

    # Freeze a final model on all Pilot cases after using the internal validation
    # splits solely for tuning the threshold baseline.
    all_ids = sorted(states)
    final_model, final_fit_summary = _fit_model(all_ids, states, targets, use_reliability=True)
    _save_model(final_model, args.output_dir / "checkpoint", {**final_fit_summary, "method": "case-cross-fitted advice reliability + action-specific kernel ridge for Q and three cost components", "pilot_cases": len(case_ids)})

    # Save an independently fit no-reliability model for the pre-registered feature ablation.
    no_rel_model, no_rel_summary = _fit_model(all_ids, states, targets, use_reliability=False)
    _save_model(no_rel_model, args.output_dir / "checkpoint_no_reliability", {**no_rel_summary, "method": "action-specific kernel ridge for Q/cost components without advice reliability feature", "pilot_cases": len(case_ids)})

    # One audit table records every state/action target and the exact action costs.
    final_costs, imputed_means = _fill_cost_targets(all_ids, targets, ACTIONS)
    output_rows = []
    missing_cost_component_observations = {component: 0 for component in COST_COMPONENTS}
    n_invalid = 0
    grouped_raw = defaultdict(list)
    for row in records:
        if row.get("record_type") == "trajectory":
            grouped_raw[(str(row["state_id"]), str(row["action"]))].append(row)
            n_invalid += int(not bool(row.get("valid")))
    for sid in all_ids:
        for action in ACTIONS:
            item = targets[sid][action]
            output_rows.append({
                "case_id": states[sid]["case_id"], "state_id": sid, "action": action,
                "q_mean": item["q"],
                "expected_patient_turns": final_costs[sid][action]["patient_turns"],
                "expected_consult_tokens_per_1000": final_costs[sid][action]["consult_tokens"],
                "expected_defer_tokens_per_1000": final_costs[sid][action]["defer_tokens"],
                "rollouts": int(item["n"]),
                "missing_cost_rollouts_patient_turns": item["cost_missing_n"]["patient_turns"],
                "missing_cost_rollouts_consult_tokens": item["cost_missing_n"]["consult_tokens"],
                "missing_cost_rollouts_defer_tokens": item["cost_missing_n"]["defer_tokens"],
            })
            for component in COST_COMPONENTS:
                missing_cost_component_observations[component] += int(item["cost_missing_n"][component])
    with (args.output_dir / "state_action_targets.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader(); writer.writerows(output_rows)

    # Global fixed-cost baseline: one action for all cases, selected on Pilot data.
    global_means = {a: _mean_utility(targets, all_ids, a, REFERENCE_COST) for a in ACTIONS}
    fixed_action = max(ACTIONS, key=lambda a: (global_means[a], -ACTIONS.index(a)))
    write_json(args.output_dir / "training_summary.json", {
        "status": "router_trained",
        "pilot_trajectory_sha256": file_sha256(args.pilot),
        "pilot_records": sum(r.get("record_type") == "trajectory" for r in records),
        "pilot_cases": len(case_ids),
        "pilot_states": len(states),
        "rollouts_per_state_action": {"min": min(int(x["n"]) for v in targets.values() for x in v.values()), "max": max(int(x["n"]) for v in targets.values() for x in v.values())},
        "invalid_rollouts_retained_as_zero_outcome": n_invalid,
        "missing_cost_component_observations_imputed_with_starting_action_train_mean": missing_cost_component_observations,
        "imputed_action_cost_means": imputed_means,
        "fixed_cost_baseline_action": fixed_action,
        "fixed_cost_training_mean_utility": global_means,
        "reference_cost": REFERENCE_COST,
        "cost_components": {"ASK": "all patient simulator turns in branch", "CONSULT": "all Expert CONSULT prompt+completion tokens / 1000 in branch", "DEFER": "all Expert DEFER prompt+completion tokens / 1000 in branch"},
        "split": {k: len(v) for k, v in split.items()},
        "train_only_fit_summary": train_fit_summary,
        "final_fit_summary": final_fit_summary,
        "no_reliability_fit_summary": no_rel_summary,
        "threshold_baseline": threshold_config,
        "decision_input_fields": ["public_state messages with non-system user/assistant text", "derived text and dialogue structure"],
        "explicitly_excluded_input_fields": ["gold", "ground_truth", "atomic_facts", "state_construction_trace", "rollout outcomes", "future patient response", "expert advice"],
    })


if __name__ == "__main__":
    main()
