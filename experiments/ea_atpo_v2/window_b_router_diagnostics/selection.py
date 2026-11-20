"""Conservative selection over the existing costed relative-value head."""
from __future__ import annotations

from typing import Sequence

import numpy as np

ACTION_ORDER = ("ASK", "ANSWER", "DEFER")


def select_relative_action(
    predicted_delta: Sequence[float],
    valid_mask: Sequence[bool],
    *,
    margin: float = 0.0,
    ask_budget_remaining: int = 1,
    after_ask_without_new_information: bool = False,
    costs_included_in_values: bool = True,
) -> tuple[int, dict[str, object]]:
    """Select an action from [Q_ASK-Q_ANSWER, Q_DEFER-Q_ANSWER].

    The deltas must already be net of registered ASK/DEFER costs. A non-ANSWER
    action is eligible only when its predicted net advantage is strictly above
    ``margin``. If ANSWER is masked, the highest-scoring remaining legal action
    is selected because the ANSWER anchor is unavailable.
    """
    delta = np.asarray(predicted_delta, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=bool)
    if delta.shape != (2,) or mask.shape != (3,):
        raise ValueError("expected two relative values and a three-action valid_mask")
    if not np.isfinite(delta).all():
        raise ValueError("relative predictions must be finite")
    if not np.isfinite(margin) or margin < 0:
        raise ValueError("conservative margin must be finite and nonnegative")
    if not isinstance(ask_budget_remaining, int) or ask_budget_remaining < 0:
        raise ValueError("ask_budget_remaining must be a nonnegative integer")
    if not costs_included_in_values:
        raise ValueError("relative values must already include registered action costs")

    legal = mask.copy()
    if ask_budget_remaining == 0 or after_ask_without_new_information:
        legal[0] = False
    if not legal.any():
        raise ValueError("no valid action remains after action and ASK-budget masks")

    scores = np.asarray([delta[0], 0.0, delta[1]], dtype=np.float64)
    if legal[1]:
        # ANSWER is the conservative default; ties never trigger ASK or DEFER.
        eligible = legal.copy()
        eligible[0] &= delta[0] > margin
        eligible[2] &= delta[1] > margin
        eligible[1] = True
        masked_scores = np.where(eligible, scores, -np.inf)
        action = int(np.argmax(masked_scores))
    else:
        # ANSWER is not available, so compare only the legal nonterminal choices.
        masked_scores = np.where(legal, scores, -np.inf)
        action = int(np.argmax(masked_scores))

    audit = {
        "margin": float(margin),
        "ask_budget_remaining": ask_budget_remaining,
        "after_ask_without_new_information": bool(after_ask_without_new_information),
        "legal_mask_after_budget": legal.tolist(),
        "selected_action": ACTION_ORDER[action],
        "costs_subtracted_by_selector": False,
    }
    return action, audit


def advance_after_ask(ask_budget_remaining: int, *, information_received: bool) -> tuple[int, bool]:
    """Consume one ASK and mark a no-information follow-up state explicitly."""
    if not isinstance(ask_budget_remaining, int) or ask_budget_remaining < 1:
        raise ValueError("an ASK transition requires at least one remaining ASK")
    return ask_budget_remaining - 1, not bool(information_received)


def select_distilled_action(logits: Sequence[float], valid_mask: Sequence[bool]) -> int:
    """Apply the original distillation policy with the same action mask."""
    values = np.asarray(logits, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=bool)
    if values.shape != (3,) or mask.shape != (3,):
        raise ValueError("expected three logits and a three-action valid_mask")
    if not np.isfinite(values).all() or not mask.any():
        raise ValueError("distillation logits must be finite and at least one action valid")
    return int(np.argmax(np.where(mask, values, -np.inf)))
