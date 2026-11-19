"""Build paired action-value supervision from valid V2 counterfactual rollouts."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any, Mapping, Sequence


ACTION_NAMES = ("ASK", "ANSWER", "DEFER")


def _sample_summary(values: Sequence[float]) -> tuple[float | None, float | None, int]:
    if not values:
        return None, None, 0
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) >= 2 else None
    return float(mean), float(std) if std is not None else None, len(values)


def q_dataset_for_state(state: Mapping[str, Any], rollouts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return Q_dataset(state) with per-action targets and an explicit validity mask.

    Cost has already been applied to each rollout's return. Consumers must not deduct
    action cost again when fitting or ranking these Q targets.
    """
    matches = [
        row for row in rollouts
        if str(row.get("case_id")) == str(state["case_id"])
        and str(row.get("state_id")) == str(state["state_id"])
    ]
    policies = {str(row.get("continuation_policy_id", "")) for row in matches}
    if len(policies) > 1:
        raise ValueError("cannot aggregate Q targets collected under mixed continuation policies")
    values_by_action: dict[str, list[float]] = defaultdict(list)
    attempts_by_action: dict[str, int] = defaultdict(int)
    seeds_by_action: dict[str, set[int]] = defaultdict(set)
    for row in matches:
        action = str(row.get("action", ""))
        if action not in ACTION_NAMES:
            raise ValueError(f"unsupported V2 action in rollout: {action}")
        attempts_by_action[action] += int(row.get("sample_count", 1))
        seed = row.get("rollout_seed")
        if seed is not None:
            seed = int(seed)
            if seed in seeds_by_action[action]:
                raise ValueError(f"duplicate rollout seed for {action} at state {state['state_id']}")
            seeds_by_action[action].add(seed)
        if not bool(row.get("valid")):
            continue
        value = row.get("return")
        if value is None or not math.isfinite(float(value)):
            raise ValueError(f"valid rollout has no finite return for {action}")
        values_by_action[action].append(float(value))

    q_values: dict[str, float | None] = {}
    return_stds: dict[str, float | None] = {}
    valid_mask: dict[str, bool] = {}
    valid_sample_count: dict[str, int] = {}
    attempted_sample_count: dict[str, int] = {}
    for action in ACTION_NAMES:
        mean, std, valid_n = _sample_summary(values_by_action[action])
        q_values[f"Q_{action}"] = mean
        return_stds[f"Q_{action}"] = std
        valid_mask[action] = valid_n > 0
        valid_sample_count[action] = valid_n
        attempted_sample_count[action] = attempts_by_action[action]

    first = matches[0] if matches else {}
    metadata = {
        "case_id": str(state["case_id"]),
        "state_id": str(state["state_id"]),
        "state_index": int(state.get("state_index", 0)),
        "public_state": state.get("public_state"),
        "continuation_policy_id": next(iter(policies), None),
        "costs_included_in_targets": True,
        "feature_boundary": "public_state_before_action_only",
        "gold_or_future_fields_in_features": False,
        "source_rollout_count": len(matches),
    }
    return {
        **q_values,
        "valid_mask": valid_mask,
        "return_std": return_stds,
        "valid_sample_count": valid_sample_count,
        "attempted_sample_count": attempted_sample_count,
        "metadata": metadata,
    }


def build_q_dataset(rollouts: Sequence[Mapping[str, Any]], train_case_ids: set[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    state_rows: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rollouts:
        case_id = str(row["case_id"])
        if case_id not in train_case_ids:
            raise ValueError(f"non-training case entered Q regression data: {case_id}")
        key = (case_id, str(row["state_id"]))
        grouped[key].append(row)
        if key in state_rows:
            previous = state_rows[key]
            if previous.get("public_state_digest") != row.get("public_state_digest"):
                raise ValueError("one state_id is associated with inconsistent public-state content")
        else:
            state_rows[key] = row
    return [
        q_dataset_for_state(state_rows[key], grouped[key])
        for key in sorted(grouped)
    ]
