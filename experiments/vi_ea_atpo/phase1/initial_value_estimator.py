"""CPU-only, train-fitted models for initial Expert relative value."""
from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


class TrainPreprocessor:
    def fit(self, x: np.ndarray) -> "TrainPreprocessor":
        self.median = np.nanmedian(x, axis=0)
        self.median = np.where(np.isfinite(self.median), self.median, 0.0)
        filled = np.where(np.isfinite(x), x, self.median)
        self.mean = filled.mean(axis=0)
        self.scale = filled.std(axis=0)
        self.scale = np.where(self.scale > 1e-8, self.scale, 1.0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        filled = np.where(np.isfinite(x), x, self.median)
        return (filled - self.mean) / self.scale


class LogisticBinary:
    def __init__(self, l2: float = 0.1, balanced: bool = False):
        self.l2 = float(l2)
        self.balanced = bool(balanced)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LogisticBinary":
        y = np.asarray(y, dtype=float)
        if len(np.unique(y)) < 2:
            raise ValueError("binary target must contain both classes")
        design = np.column_stack([np.ones(len(x)), x])
        weights = np.ones(len(y), dtype=float)
        if self.balanced:
            positive = max(1, int(y.sum()))
            negative = max(1, len(y) - positive)
            weights[y == 1] = len(y) / (2.0 * positive)
            weights[y == 0] = len(y) / (2.0 * negative)
        beta = np.zeros(design.shape[1], dtype=float)
        penalty = np.eye(design.shape[1]) * self.l2
        penalty[0, 0] = 0.0
        for _ in range(120):
            p = sigmoid(design @ beta)
            grad = design.T @ (weights * (p - y)) / len(y) + penalty @ beta
            curvature = weights * p * (1.0 - p)
            hessian = (design.T * curvature) @ design / len(y) + penalty
            hessian.flat[:: hessian.shape[0] + 1] += 1e-8
            step = np.linalg.solve(hessian, grad)
            beta -= step
            if float(np.max(np.abs(step))) < 1e-8:
                break
        self.coef = beta
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return sigmoid(np.column_stack([np.ones(len(x)), x]) @ self.coef)


class RidgeValue:
    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RidgeValue":
        design = np.column_stack([np.ones(len(x)), x])
        penalty = np.eye(design.shape[1]) * self.alpha
        penalty[0, 0] = 0.0
        self.coef = np.linalg.solve(design.T @ design + penalty, design.T @ y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.column_stack([np.ones(len(x)), x]) @ self.coef


class InitialDeferralValueEstimator:
    """Estimates helpful/harmful probabilities from decision-time features.

    The expected relative gain is P(helpful|x) - P(harmful|x). Bootstrap spread
    is an approximate train-sample model-uncertainty diagnostic, not a calibrated
    confidence bound.
    """

    def __init__(self, l2: float = 0.1, balanced: bool = False):
        self.l2 = float(l2)
        self.balanced = bool(balanced)

    def fit(self, x: np.ndarray, helpful: np.ndarray, harmful: np.ndarray) -> "InitialDeferralValueEstimator":
        self.helpful_model = LogisticBinary(self.l2, self.balanced).fit(x, helpful)
        self.harmful_model = LogisticBinary(self.l2, self.balanced).fit(x, harmful)
        return self

    def predict(self, x: np.ndarray) -> dict[str, np.ndarray]:
        p_help = self.helpful_model.predict_proba(x)
        p_harm = self.harmful_model.predict_proba(x)
        return {
            "p_helpful": p_help,
            "p_harmful": p_harm,
            "expected_relative_gain": p_help - p_harm,
        }
