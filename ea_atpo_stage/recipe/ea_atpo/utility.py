"""Outcome-value and cost/harm-aware action selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .protocol import Action


@dataclass(frozen=True)
class UtilityConfig:
    resource_cost: Mapping[Action, float] = field(default_factory=lambda: {
        Action.ANSWER: 0.0,
        Action.ASK: 0.05,
        Action.CONSULT: 0.20,
        Action.DEFER: 0.60,
    })
    harm: Mapping[Action, float] = field(default_factory=lambda: {a: 0.0 for a in Action})
    lambda_cost: float = 1.0
    lambda_harm: float = 1.0


def compute_utilities(q_out: Mapping[Action, float], config: UtilityConfig) -> dict[Action, float]:
    missing = set(Action) - set(q_out)
    if missing:
        raise ValueError(f"missing outcome values for: {sorted(a.value for a in missing)}")
    return {
        action: float(q_out[action])
        - config.lambda_cost * float(config.resource_cost.get(action, 0.0))
        - config.lambda_harm * float(config.harm.get(action, 0.0))
        for action in Action
    }


def select_action(q_out: Mapping[Action, float], config: UtilityConfig) -> tuple[Action, dict[Action, float]]:
    utilities = compute_utilities(q_out, config)
    # Stable tie-breaking deliberately prefers less external intervention.
    priority = {Action.ANSWER: 3, Action.ASK: 2, Action.CONSULT: 1, Action.DEFER: 0}
    chosen = max(Action, key=lambda a: (utilities[a], priority[a]))
    return chosen, utilities
