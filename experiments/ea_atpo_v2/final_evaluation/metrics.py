"""Case-level policy metrics, paired bootstrap, Oracle upper bound, and risk coverage."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from . import ACTION_ORDER, BASELINE_POLICIES, DEFAULT_COSTS, DEPLOYABLE_POLICIES, LEARNED_POLICIES

METRIC_NAMES = (
    "accuracy",
    "utility",
    "ask_rate",
    "defer_rate",
    "mean_cost",
    "expert_correction_rate",
    "correct_answer_degradation_rate",
)
PREDICTION_FORBIDDEN_KEYS = {
    "gold", "ground_truth", "final_correct", "base_correct", "expert_correct",
    "correctness", "correct", "outcome", "outcomes", "atomic_facts", "hidden_facts",
}


def _is_bool(value: Any) -> bool:
    return isinstance(value, bool)


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _action_name(value: Any) -> str:
    action = str(value).upper()
    if action not in ACTION_ORDER:
        raise ValueError(f"unsupported action {action!r}; expected {ACTION_ORDER}")
    return action


def _walk_prediction_keys(value: Any, prefix: str = "") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).lower()
            if normalized in PREDICTION_FORBIDDEN_KEYS or normalized.endswith("_correct"):
                raise ValueError(f"prediction payload contains outcome/Gold field {prefix}{key}")
            _walk_prediction_keys(child, prefix + str(key) + ".")
    elif isinstance(value, list):
        for child in value:
            _walk_prediction_keys(child, prefix)


def validate_costs(costs: dict[str, float] | None = None) -> dict[str, float]:
    selected = dict(DEFAULT_COSTS if costs is None else costs)
    if set(selected) != set(ACTION_ORDER):
        raise ValueError(f"cost keys must exactly match action order {ACTION_ORDER}")
    result = {action: float(selected[action]) for action in ACTION_ORDER}
    if any(not math.isfinite(value) or value < 0 for value in result.values()):
        raise ValueError("action costs must be finite and nonnegative")
    if result["ANSWER"] != 0.0:
        raise ValueError("ANSWER cost is fixed at zero for the registered V2 evaluation")
    return result


def validate_outcomes(rows: list[dict[str, Any]], expected_case_ids: list[str]) -> dict[str, dict[str, Any]]:
    expected = set(map(str, expected_case_ids))
    outcomes: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = str(row.get("case_id", ""))
        if not case_id or case_id in outcomes:
            raise ValueError(f"missing or duplicate outcome case_id: {case_id!r}")
        if case_id not in expected:
            raise ValueError(f"outcome case is not in frozen test manifest: {case_id}")
        if not _is_bool(row.get("base_correct")) or not _is_bool(row.get("expert_correct")):
            raise ValueError(f"{case_id}: base_correct and expert_correct must be post-hoc booleans")
        action_outcomes = row.get("action_outcomes")
        if not isinstance(action_outcomes, dict) or set(action_outcomes) != set(ACTION_ORDER):
            raise ValueError(f"{case_id}: require complete ASK/ANSWER/DEFER counterfactual outcomes")
        for action in ACTION_ORDER:
            result = action_outcomes[action]
            if not isinstance(result, dict) or not _is_bool(result.get("final_correct")):
                raise ValueError(f"{case_id}/{action}: final_correct must be a boolean")
            if not _is_bool(result.get("valid")):
                raise ValueError(f"{case_id}/{action}: valid must be recorded explicitly")
            for count_key in ("ask_count", "defer_count"):
                count = result.get(count_key)
                if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                    raise ValueError(f"{case_id}/{action}: {count_key} must be a nonnegative integer")
            if action == "ANSWER" and (result["ask_count"] or result["defer_count"]):
                raise ValueError(f"{case_id}: ANSWER branch cannot incur ASK/DEFER calls")
            if action == "ASK" and result["ask_count"] < 1:
                raise ValueError(f"{case_id}: ASK branch must record at least one ASK call")
            if action == "DEFER" and result["defer_count"] < 1:
                raise ValueError(f"{case_id}: DEFER branch must record at least one DEFER call")
        if action_outcomes["ANSWER"]["final_correct"] != row["base_correct"]:
            raise ValueError(f"{case_id}: ANSWER outcome must equal Base correctness")
        if action_outcomes["DEFER"]["final_correct"] != row["expert_correct"]:
            raise ValueError(f"{case_id}: DEFER outcome must equal Expert correctness")
        outcomes[case_id] = row
    if set(outcomes) != expected:
        missing = len(expected - set(outcomes))
        extra = len(set(outcomes) - expected)
        raise ValueError(f"outcome coverage differs from frozen test manifest: missing={missing}, extra={extra}")
    return outcomes


def validate_predictions(rows: list[dict[str, Any]], expected_case_ids: list[str]) -> dict[str, dict[str, dict[str, Any]]]:
    expected = set(map(str, expected_case_ids))
    predictions: dict[str, dict[str, dict[str, Any]]] = {name: {} for name in LEARNED_POLICIES}
    for row in rows:
        _walk_prediction_keys(row)
        case_id = str(row.get("case_id", ""))
        policy = str(row.get("policy", ""))
        if policy not in LEARNED_POLICIES:
            raise ValueError(f"prediction file may contain only learned policies; got {policy!r}")
        if case_id not in expected:
            raise ValueError(f"prediction case is not in frozen test manifest: {case_id}")
        if case_id in predictions[policy]:
            raise ValueError(f"duplicate prediction for {policy}/{case_id}")
        action = _action_name(row.get("selected_action"))
        score_map = row.get("action_scores")
        if score_map is not None:
            if not isinstance(score_map, dict) or set(score_map) != set(ACTION_ORDER):
                raise ValueError(f"{policy}/{case_id}: action_scores must contain ASK/ANSWER/DEFER")
            if any(not _finite_number(score_map[name]) for name in ACTION_ORDER):
                raise ValueError(f"{policy}/{case_id}: action scores must be finite")
        defer_score = row.get("defer_score")
        if defer_score is None and score_map is not None:
            defer_score = float(score_map["DEFER"]) - max(float(score_map["ASK"]), float(score_map["ANSWER"]))
        if not _finite_number(defer_score):
            raise ValueError(f"{policy}/{case_id}: require a finite prediction-only defer_score")
        fallback = row.get("non_defer_action")
        if fallback is None and score_map is not None:
            fallback = max(("ASK", "ANSWER"), key=lambda name: float(score_map[name]))
        if fallback is None:
            raise ValueError(f"{policy}/{case_id}: require non_defer_action for risk-coverage analysis")
        fallback = _action_name(fallback)
        if fallback == "DEFER":
            raise ValueError(f"{policy}/{case_id}: non_defer_action cannot be DEFER")
        predictions[policy][case_id] = {
            "selected_action": action,
            "defer_score": float(defer_score),
            "non_defer_action": fallback,
            "action_scores": score_map,
            "model_revision": row.get("model_revision"),
            "prediction_config_sha256": row.get("prediction_config_sha256"),
        }
    for policy in LEARNED_POLICIES:
        if set(predictions[policy]) != expected:
            missing = len(expected - set(predictions[policy]))
            extra = len(set(predictions[policy]) - expected)
            raise ValueError(f"prediction coverage differs for {policy}: missing={missing}, extra={extra}")
    return predictions


def _case_metric_components(case: dict[str, Any], action: str, costs: dict[str, float]) -> dict[str, Any]:
    outcome = case["action_outcomes"][action]
    ask_count = int(outcome["ask_count"])
    defer_count = int(outcome["defer_count"])
    cost = costs["ASK"] * ask_count + costs["DEFER"] * defer_count
    final_correct = bool(outcome["final_correct"])
    return {
        "action": action,
        "final_correct": final_correct,
        "utility": float(final_correct) - cost,
        "ask_called": float(ask_count > 0),
        "defer_called": float(defer_count > 0),
        "mean_cost": cost,
        "ask_count": ask_count,
        "defer_count": defer_count,
        "correction_numerator": float(action == "DEFER" and not case["base_correct"] and case["expert_correct"]),
        "correction_denominator": float(action == "DEFER" and not case["base_correct"]),
        "degradation_numerator": float(action != "ANSWER" and case["base_correct"] and not final_correct),
        "degradation_denominator": float(action != "ANSWER" and case["base_correct"]),
        "valid": bool(outcome["valid"]),
    }


def _metric_from_components(components: list[dict[str, Any]], indices: np.ndarray | None = None) -> dict[str, float | None]:
    selected = components if indices is None else [components[int(i)] for i in indices]
    n = len(selected)
    if not n:
        raise ValueError("cannot evaluate an empty case sample")
    def mean(key: str) -> float:
        return float(np.mean([row[key] for row in selected]))
    correction_den = sum(row["correction_denominator"] for row in selected)
    degradation_den = sum(row["degradation_denominator"] for row in selected)
    return {
        "accuracy": mean("final_correct"),
        "utility": mean("utility"),
        "ask_rate": mean("ask_called"),
        "defer_rate": mean("defer_called"),
        "mean_cost": mean("mean_cost"),
        "expert_correction_rate": (sum(row["correction_numerator"] for row in selected) / correction_den
                                   if correction_den else None),
        "correct_answer_degradation_rate": (sum(row["degradation_numerator"] for row in selected) / degradation_den
                                            if degradation_den else None),
    }


def policy_case_components(case_ids: list[str], outcomes: dict[str, dict[str, Any]],
                           predictions: dict[str, dict[str, dict[str, Any]]],
                           costs: dict[str, float] | None = None) -> dict[str, list[dict[str, Any]]]:
    registered = validate_costs(costs)
    components: dict[str, list[dict[str, Any]]] = {name: [] for name in DEPLOYABLE_POLICIES}
    fixed = {"always_answer": "ANSWER", "always_ask": "ASK", "always_defer": "DEFER"}
    for case_id in case_ids:
        case = outcomes[case_id]
        for policy in BASELINE_POLICIES:
            components[policy].append(_case_metric_components(case, fixed[policy], registered))
        for policy in LEARNED_POLICIES:
            action = predictions[policy][case_id]["selected_action"]
            components[policy].append(_case_metric_components(case, action, registered))
    return components


def oracle_upper_bound_components(case_ids: list[str], outcomes: dict[str, dict[str, Any]],
                                  costs: dict[str, float] | None = None) -> list[dict[str, Any]]:
    """Post-hoc Gold-based upper bound; this function is not a prediction policy."""
    registered = validate_costs(costs)
    result = []
    for case_id in case_ids:
        case = outcomes[case_id]
        scored = []
        for action in ACTION_ORDER:
            components = _case_metric_components(case, action, registered)
            scored.append((components["utility"], action, components))
        # Fixed action order resolves exact utility ties deterministically.
        best = max(scored, key=lambda item: item[0])
        result.append(best[2])
    return result


def paired_bootstrap(case_components: dict[str, list[dict[str, Any]]], *, replicates: int = 10000,
                     seed: int = 20261120) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, dict[str, dict[str, Any]]]]:
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    policies = list(case_components)
    if "always_answer" not in case_components:
        raise ValueError("paired reference policy always_answer is required")
    n_values = {len(rows) for rows in case_components.values()}
    if len(n_values) != 1 or not n_values or next(iter(n_values)) == 0:
        raise ValueError("all policies must share the same nonempty case-level cohort")
    n = next(iter(n_values))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(replicates, n))
    bootstrap_values: dict[str, dict[str, np.ndarray]] = {}
    point_values: dict[str, dict[str, float | None]] = {}
    for policy in policies:
        rows = case_components[policy]
        columns = {name: np.asarray([row[name] for row in rows], dtype=np.float64) for name in (
            "final_correct", "utility", "ask_called", "defer_called", "mean_cost",
            "correction_numerator", "correction_denominator",
            "degradation_numerator", "degradation_denominator",
        )}
        point_values[policy] = _metric_from_components(rows)
        sampled = {name: values[draws] for name, values in columns.items()}
        bootstrap_values[policy] = {
            "accuracy": sampled["final_correct"].mean(axis=1),
            "utility": sampled["utility"].mean(axis=1),
            "ask_rate": sampled["ask_called"].mean(axis=1),
            "defer_rate": sampled["defer_called"].mean(axis=1),
            "mean_cost": sampled["mean_cost"].mean(axis=1),
        }
        correction_den = sampled["correction_denominator"].sum(axis=1)
        degradation_den = sampled["degradation_denominator"].sum(axis=1)
        correction = np.full(replicates, np.nan, dtype=np.float64)
        degradation = np.full(replicates, np.nan, dtype=np.float64)
        np.divide(sampled["correction_numerator"].sum(axis=1), correction_den,
                  out=correction, where=correction_den > 0)
        np.divide(sampled["degradation_numerator"].sum(axis=1), degradation_den,
                  out=degradation, where=degradation_den > 0)
        bootstrap_values[policy]["expert_correction_rate"] = correction
        bootstrap_values[policy]["correct_answer_degradation_rate"] = degradation
    metric_ci: dict[str, dict[str, dict[str, Any]]] = {}
    paired_delta_ci: dict[str, dict[str, dict[str, Any]]] = {}
    for policy in policies:
        metric_ci[policy] = {}
        paired_delta_ci[policy] = {}
        for metric in METRIC_NAMES:
            samples = np.asarray(bootstrap_values[policy][metric], dtype=np.float64)
            samples = samples[np.isfinite(samples)]
            estimate = point_values[policy][metric]
            metric_ci[policy][metric] = {
                "estimate": estimate,
                "ci95_low": float(np.quantile(samples, 0.025)) if len(samples) else None,
                "ci95_high": float(np.quantile(samples, 0.975)) if len(samples) else None,
                "valid_bootstrap_replicates": int(len(samples)),
            }
            if policy == "always_answer":
                delta_samples = np.zeros(replicates, dtype=np.float64)
                delta_estimate: float | None = 0.0
            else:
                pol_samples = bootstrap_values[policy][metric]
                ref_samples = bootstrap_values["always_answer"][metric]
                valid = np.isfinite(pol_samples) & np.isfinite(ref_samples)
                delta_samples = (pol_samples - ref_samples)[valid]
                pol_point = point_values[policy][metric]
                ref_point = point_values["always_answer"][metric]
                delta_estimate = float(pol_point - ref_point) if pol_point is not None and ref_point is not None else None
            paired_delta_ci[policy][metric] = {
                "estimate": delta_estimate,
                "ci95_low": float(np.quantile(delta_samples, 0.025)) if len(delta_samples) else None,
                "ci95_high": float(np.quantile(delta_samples, 0.975)) if len(delta_samples) else None,
                "valid_bootstrap_replicates": int(len(delta_samples)),
            }
    return metric_ci, paired_delta_ci


def risk_coverage_rows(case_ids: list[str], outcomes: dict[str, dict[str, Any]],
                       predictions: dict[str, dict[str, dict[str, Any]]],
                       costs: dict[str, float] | None = None) -> list[dict[str, Any]]:
    registered = validate_costs(costs)
    result: list[dict[str, Any]] = []

    def emit(policy: str, mode: str, threshold: float | str | None, actions: list[str]) -> None:
        selected = [_case_metric_components(outcomes[cid], action, registered)
                    for cid, action in zip(case_ids, actions)]
        kept = [row for row in selected if row["action"] != "DEFER"]
        coverage = len(kept) / len(selected)
        accuracy = float(np.mean([row["final_correct"] for row in selected]))
        covered_accuracy = float(np.mean([row["final_correct"] for row in kept])) if kept else None
        metrics = _metric_from_components(selected)
        result.append({
            "policy": policy,
            "curve_kind": mode,
            "defer_threshold": threshold,
            "coverage": coverage,
            "risk": (1.0 - covered_accuracy) if covered_accuracy is not None else None,
            "covered_accuracy": covered_accuracy,
            "all_case_accuracy": accuracy,
            "utility": metrics["utility"],
            "ask_rate": metrics["ask_rate"],
            "defer_rate": metrics["defer_rate"],
            "mean_cost": metrics["mean_cost"],
            "case_count": len(selected),
        })

    fixed = {"always_answer": "ANSWER", "always_ask": "ASK", "always_defer": "DEFER"}
    for policy, action in fixed.items():
        emit(policy, "fixed_baseline", None, [action] * len(case_ids))
    for policy in LEARNED_POLICIES:
        by_case = predictions[policy]
        emit(policy, "observed_policy", None, [by_case[cid]["selected_action"] for cid in case_ids])
        scores = [float(by_case[cid]["defer_score"]) for cid in case_ids]
        thresholds: list[float] = [math.inf, *sorted(set(scores), reverse=True), -math.inf]
        fallback = [by_case[cid]["non_defer_action"] for cid in case_ids]
        for threshold in thresholds:
            actions = ["DEFER" if score >= threshold else non_defer
                       for score, non_defer in zip(scores, fallback)]
            emit(policy, "score_threshold", threshold, actions)
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summary_rows(metric_ci: dict[str, dict[str, dict[str, Any]]], *, case_count: int) -> list[dict[str, Any]]:
    rows = []
    for policy, metrics in metric_ci.items():
        for metric, value in metrics.items():
            rows.append({"policy": policy, "metric": metric, **value, "case_count": case_count,
                         "oracle_only_upper_bound": policy == "oracle_upper_bound"})
    return rows


def paired_delta_rows(delta_ci: dict[str, dict[str, dict[str, Any]]], *, case_count: int) -> list[dict[str, Any]]:
    rows = []
    for policy, metrics in delta_ci.items():
        for metric, value in metrics.items():
            rows.append({"policy": policy, "reference": "always_answer", "metric": metric,
                         **value, "case_count": case_count,
                         "oracle_only_upper_bound": policy == "oracle_upper_bound"})
    return rows
