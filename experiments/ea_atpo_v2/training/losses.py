"""Masked Q regression and Q-target policy distillation objectives."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_regression_loss(prediction: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor,
                           *, loss: str = "huber", sample_count: torch.Tensor | None = None) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.shape != valid_mask.shape:
        raise ValueError("prediction, target, and valid_mask must have identical shapes")
    mask = valid_mask.bool()
    if not torch.isfinite(target[mask]).all():
        raise ValueError("valid Q targets must be finite")
    if not bool(mask.any()):
        raise ValueError("masked regression batch has no valid action targets")
    pred = prediction[mask]
    truth = target[mask]
    if loss == "huber":
        element = F.smooth_l1_loss(pred, truth, reduction="none")
    elif loss == "mse":
        element = (pred - truth).square()
    else:
        raise ValueError("loss must be 'huber' or 'mse'")
    if sample_count is None:
        return element.mean()
    if sample_count.shape != prediction.shape:
        raise ValueError("sample_count must match prediction shape")
    weights = sample_count[mask].to(dtype=element.dtype).clamp_min(1.0)
    return (element * weights).sum() / weights.sum()


def masked_distillation_loss(logits: torch.Tensor, q_target: torch.Tensor, valid_mask: torch.Tensor,
                             *, temperature: float = 1.0, minimum_comparable_actions: int = 2,
                             allow_empty: bool = False) -> tuple[torch.Tensor | None, int]:
    """Cross-entropy from masked softmax(Q/T); rows with <2 actions are skipped."""
    if logits.shape != q_target.shape or logits.shape != valid_mask.shape:
        raise ValueError("logits, q_target, and valid_mask must have identical shapes")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    mask = valid_mask.bool() & torch.isfinite(q_target)
    eligible = mask.sum(dim=1) >= int(minimum_comparable_actions)
    if not bool(eligible.any()):
        if allow_empty:
            return None, 0
        raise ValueError("no state has enough observed actions for comparative distillation")
    selected_logits = logits[eligible]
    selected_targets = q_target[eligible]
    selected_mask = mask[eligible]
    # Mask by boolean indexing per row; invalid values never enter softmax or log-softmax.
    row_losses = []
    for row_logits, row_targets, row_mask in zip(selected_logits, selected_targets, selected_mask):
        log_pi = F.log_softmax(row_logits[row_mask], dim=0)
        p_target = F.softmax(row_targets[row_mask] / float(temperature), dim=0)
        row_losses.append(-(p_target * log_pi).sum())
    return torch.stack(row_losses).mean(), int(eligible.sum().item())
