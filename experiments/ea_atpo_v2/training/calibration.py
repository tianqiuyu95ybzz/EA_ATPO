"""Action-wise affine calibration for cost-free expected correctness."""
from __future__ import annotations

import numpy as np


class ActionValueCalibrator:
    """Fit on a case-disjoint calibration partition, then subtract costs once."""

    def __init__(self, action_count: int = 3):
        self.scale = np.ones(action_count, dtype=np.float64)
        self.bias = np.zeros(action_count, dtype=np.float64)

    def fit(self, predicted_net: np.ndarray, target_net: np.ndarray, valid_mask: np.ndarray,
            reference_cost: np.ndarray) -> "ActionValueCalibrator":
        pred = np.asarray(predicted_net, dtype=np.float64)
        target = np.asarray(target_net, dtype=np.float64)
        mask = np.asarray(valid_mask, dtype=bool) & np.isfinite(pred) & np.isfinite(target)
        costs = np.asarray(reference_cost, dtype=np.float64)
        if pred.shape != target.shape or pred.shape != mask.shape or pred.ndim != 2 or costs.shape not in {(pred.shape[1],), pred.shape}:
            raise ValueError("calibration arrays have incompatible shapes")
        if costs.ndim == 1:
            costs = np.broadcast_to(costs, pred.shape)
        for action in range(pred.shape[1]):
            use = mask[:, action]
            if int(use.sum()) < 2:
                continue
            x = pred[use, action] + costs[use, action]
            y = target[use, action] + costs[use, action]
            if np.ptp(x) < 1e-8:
                self.scale[action] = 1.0
                self.bias[action] = float(np.mean(y - x))
                continue
            slope, intercept = np.polyfit(x, y, 1)
            self.scale[action] = float(np.clip(slope, 0.0, 5.0))
            self.bias[action] = float(np.mean(y - self.scale[action] * x))
        return self

    def calibrate_gross(self, predicted_net: np.ndarray, reference_cost: np.ndarray) -> np.ndarray:
        pred = np.asarray(predicted_net, dtype=np.float64)
        costs = np.asarray(reference_cost, dtype=np.float64)
        if costs.ndim == 1:
            costs = np.broadcast_to(costs, pred.shape)
        if pred.ndim != 2 or pred.shape[1] != self.scale.size or costs.shape != pred.shape:
            raise ValueError("predictions/reference_cost have incompatible shapes")
        return np.clip(self.scale[None, :] * (pred + costs) + self.bias[None, :], 0.0, 1.0)

    def to_json(self) -> dict:
        return {"scale": self.scale.tolist(), "bias": self.bias.tolist(), "kind": "per_action_affine_on_gross_correctness"}

    @classmethod
    def from_json(cls, payload: dict) -> "ActionValueCalibrator":
        result = cls(len(payload["scale"]))
        result.scale = np.asarray(payload["scale"], dtype=np.float64)
        result.bias = np.asarray(payload["bias"], dtype=np.float64)
        return result
