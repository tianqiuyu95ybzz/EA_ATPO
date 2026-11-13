"""EA-ATPO scalar reward used for score logging and validation.

The tree runner supplies per-edge costs to the ATPO backup. This post-hoc
function computes the same episode-level sum for reward reporting; it is not
fed back into the precomputed Actor advantage or Critic return a second time.
"""
from __future__ import annotations

import re
from typing import Any, Mapping


def _gold_letter(ground_truth: Any) -> str:
    if isinstance(ground_truth, Mapping):
        ground_truth = ground_truth.get("answer", "")
    return str(ground_truth).strip().upper()


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict | None = None,
    *,
    correctness_reward: float = 1.0,
    ask_cost: float = 0.05,
    consult_cost: float = 0.20,
    defer_cost: float = 0.60,
    lambda_ask: float = 1.0,
    lambda_consult: float = 1.0,
    lambda_defer: float = 1.0,
) -> dict[str, float | int | str]:
    """Return terminal correctness minus each action's configured cost once."""
    text = str(solution_str)
    answers = re.findall(r"(?:Final Answer|Expert Final Answer):\s*([A-D])\b", text, re.IGNORECASE)
    final_answer = answers[-1].upper() if answers else ""
    gold = _gold_letter(ground_truth)
    correct = int(bool(final_answer) and final_answer == gold)
    asks = len(re.findall(r"(?m)^\s*Question:\s*", text))
    consults = len(re.findall(r"(?m)^\s*Consult Expert:\s*", text))
    defers = len(re.findall(r"(?m)^\s*Defer to Expert\s*$", text))
    cost = (
        asks * float(lambda_ask) * float(ask_cost)
        + consults * float(lambda_consult) * float(consult_cost)
        + defers * float(lambda_defer) * float(defer_cost)
    )
    outcome = float(correctness_reward) * correct
    return {
        "score": outcome - cost,
        "terminal_correctness_reward": outcome,
        "ask_count": asks,
        "consult_count": consults,
        "defer_count": defers,
        "resource_cost": cost,
        "correct": correct,
        "final_answer": final_answer,
    }
