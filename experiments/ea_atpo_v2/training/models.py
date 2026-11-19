"""Small two-layer value and action-logit heads."""
from __future__ import annotations

import torch
from torch import nn
import numpy as np

from . import ACTION_NAMES


class ConstantMeanQRegressor(nn.Module):
    """Per-action masked, sample-count-weighted training mean baseline."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.means = nn.Parameter(torch.zeros(len(ACTION_NAMES), dtype=torch.float32), requires_grad=False)

    def fit_closed_form(self, rows) -> None:
        targets = np.stack([row.q_target for row in rows])
        masks = np.stack([row.valid_mask for row in rows])
        counts = np.stack([row.sample_count for row in rows])
        values = []
        for action_index in range(len(ACTION_NAMES)):
            valid = masks[:, action_index]
            if not valid.any():
                raise ValueError(f"constant-mean baseline has no observed target for {ACTION_NAMES[action_index]}")
            weights = counts[valid, action_index]
            values.append(float(np.average(targets[valid, action_index], weights=weights)))
        self.means.data.copy_(torch.as_tensor(values, dtype=self.means.dtype))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.means.unsqueeze(0).expand(features.shape[0], -1)


class RidgeQRegressor(nn.Module):
    """Per-action weighted ridge regression fitted by a stable dual solve."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.linear = nn.Linear(self.input_dim, len(ACTION_NAMES), bias=True)

    def fit_closed_form(self, features: np.ndarray, rows, *, alpha: float) -> None:
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("ridge alpha must be a finite positive value")
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.input_dim or not np.isfinite(x).all():
            raise ValueError("ridge features have invalid shape or values")
        y = np.stack([row.q_target for row in rows]).astype(np.float64)
        mask = np.stack([row.valid_mask for row in rows]).astype(bool)
        counts = np.stack([row.sample_count for row in rows]).astype(np.float64)
        weights = np.zeros((len(ACTION_NAMES), self.input_dim), dtype=np.float64)
        biases = np.zeros(len(ACTION_NAMES), dtype=np.float64)
        for action_index, action in enumerate(ACTION_NAMES):
            valid = mask[:, action_index]
            if not valid.any():
                raise ValueError(f"ridge baseline has no observed target for {action}")
            xv = x[valid]
            yv = y[valid, action_index]
            sample_weights = counts[valid, action_index]
            root_weights = np.sqrt(sample_weights)
            x_mean = np.average(xv, axis=0, weights=sample_weights)
            y_mean = float(np.average(yv, weights=sample_weights))
            centered_x = (xv - x_mean) * root_weights[:, None]
            centered_y = (yv - y_mean) * root_weights
            gram = centered_x @ centered_x.T
            dual = np.linalg.solve(gram + float(alpha) * np.eye(len(gram)), centered_y)
            weights[action_index] = centered_x.T @ dual
            biases[action_index] = y_mean - float(x_mean @ weights[action_index])
        with torch.no_grad():
            self.linear.weight.copy_(torch.as_tensor(weights, dtype=self.linear.weight.dtype))
            self.linear.bias.copy_(torch.as_tensor(biases, dtype=self.linear.bias.dtype))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)


class QRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, len(ACTION_NAMES)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class PolicyHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, len(ACTION_NAMES)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)
