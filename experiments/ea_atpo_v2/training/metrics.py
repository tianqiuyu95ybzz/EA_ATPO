"""Case-clustered metrics for the V2 three-action offline policy comparison."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping

import numpy as np

from . import ACTION_NAMES


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and np.isfinite(float(value))


def utility_from_raw(correctness: float, ask_count: float, defer_count: float,
                     lambda_defer: float, ask_cost: float = 0.05) -> float:
    """Recompute utility from raw reward/cost components; never re-charge stored net return."""
    values = (correctness, ask_count, defer_count, lambda_defer, ask_cost)
    if not all(np.isfinite(float(value)) for value in values):
        raise ValueError("utility components must be finite")
    if ask_count < 0 or defer_count < 0 or lambda_defer < 0 or ask_cost < 0:
        raise ValueError("costs and action counts cannot be negative")
    return float(correctness) - float(ask_cost) * float(ask_count) - float(lambda_defer) * float(defer_count)


def aggregate_rollouts(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    grouped = defaultdict(lambda: {"actions": defaultdict(list), "public_state_digest": None})
    for row in rows:
        if row.get("record_type", "v2_counterfactual_rollout") != "v2_counterfactual_rollout":
            raise ValueError("trajectory file contains a non-V2 counterfactual record")
        action = str(row.get("action", ""))
        if action not in ACTION_NAMES:
            raise ValueError(f"unsupported action {action!r}; V2 uses {ACTION_NAMES}")
        key = (str(row.get("case_id", "")), str(row.get("state_id", "")))
        if not all(key):
            raise ValueError("trajectory row missing case_id/state_id")
        item = grouped[key]
        digest = row.get("public_state_digest")
        if item["public_state_digest"] not in (None, digest):
            raise ValueError(f"inconsistent public-state digest for {key}")
        item["public_state_digest"] = digest
        item["actions"][action].append(row)

    result = {}
    for key, item in grouped.items():
        actions = {}
        for action in ACTION_NAMES:
            attempts = item["actions"].get(action, [])
            if not attempts:
                actions[action] = {"present": False, "legal": False}
                continue
            illegal = all(row.get("failure_reason") == "illegal_action_masked_before_model_call" for row in attempts)
            accuracies = [float(row["final_accuracy"]) if _finite(row.get("final_accuracy")) and row.get("valid", False) else 0.0 for row in attempts]
            valid_rows = [row for row in attempts if row.get("valid", False) and _finite(row.get("final_accuracy"))]
            q_targets = []
            for row in valid_rows:
                cost = row.get("cost_components") or {}
                ask_count = float(row.get("ask_count", 0))
                defer_count = float(row.get("defer_count", 0))
                ask_cost = float(cost.get("ask_cost", 0.05 * ask_count))
                defer_cost = float(cost.get("defer_cost", 0.60 * defer_count))
                # Rebuild net target from untouched correctness and costs, never re-charge a stored return.
                q_targets.append(float(row["final_accuracy"]) - ask_cost - defer_cost)
            count_fields = [(float(row.get("ask_count", 0)), float(row.get("defer_count", 0))) for row in attempts]
            if len(set(count_fields)) != 1:
                raise ValueError(f"action counts vary across repeated rollouts for {key}/{action}")
            first_cost = attempts[0].get("cost_components") or {}
            actions[action] = {
                "present": True,
                "legal": not illegal,
                "outcome_accuracy": float(np.mean(accuracies)),
                "valid_rate": float(np.mean([bool(row.get("valid", False)) for row in attempts])),
                "q_target_reference_net": float(np.mean(q_targets)) if q_targets else None,
                "q_target_valid": bool(q_targets),
                "attempt_count": len(attempts),
                "ask_count": count_fields[0][0],
                "defer_count": count_fields[0][1],
                "cost_components": {key: float(value) for key, value in first_cost.items() if _finite(value)},
                "raw_final_accuracy": [row.get("final_accuracy") for row in attempts],
            }
        result[key] = {"case_id": key[0], "state_id": key[1], "actions": actions,
                       "public_state_digest": item["public_state_digest"]}
    return result


def safe_auc(binary_labels: np.ndarray, scores: np.ndarray) -> float | None:
    y = np.asarray(binary_labels, dtype=bool)
    s = np.asarray(scores, dtype=np.float64)
    positives, negatives = int(y.sum()), int((~y).sum())
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and s[order[j]] == s[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2
        i = j
    return float((ranks[y].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def regression_calibration(predicted: np.ndarray, target: np.ndarray, action: np.ndarray, bins: int = 10) -> dict[str, Any]:
    result = {}
    for j, name in enumerate(ACTION_NAMES):
        use = (action == j) & np.isfinite(predicted) & np.isfinite(target)
        if not use.any():
            result[name] = {"n": 0, "mae": None, "calibration_mae": None, "bins": []}
            continue
        p, y = predicted[use], target[use]
        groups = np.array_split(np.argsort(p, kind="mergesort"), min(int(bins), len(p)))
        detail = []
        gap = 0.0
        for group in groups:
            if len(group) == 0:
                continue
            local_gap = abs(float(p[group].mean() - y[group].mean()))
            gap += len(group) / len(p) * local_gap
            detail.append({"n": len(group), "mean_predicted": float(p[group].mean()), "mean_observed": float(y[group].mean())})
        result[name] = {"n": len(p), "mae": float(np.mean(np.abs(p - y))), "calibration_mae": gap, "bins": detail}
    return result
