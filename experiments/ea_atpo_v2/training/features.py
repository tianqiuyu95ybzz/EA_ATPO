"""Leakage-safe public-text features and train-fitted feature normalization."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Sequence

import numpy as np

from .data import QExample


class ExistingPublicStateEncoder:
    """Adapter around the audited V1 public-text TF-IDF vectorizer.

    It reads only user/assistant messages and fits vocabulary/statistics on train cases.
    The implementation and V1 router files remain untouched.
    """

    def __init__(self, max_features: int = 3000):
        self.max_features = int(max_features)
        self.vectorizer = None

    @staticmethod
    def _load_class():
        repo_root = Path(__file__).resolve().parents[3]
        source = repo_root / "experiments/ea_atpo_value_router_20261105/router_core.py"
        spec = importlib.util.spec_from_file_location("ea_atpo_v1_public_vectorizer", source)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load audited public text features from {source}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.PublicStateVectorizer

    def fit(self, examples: Sequence[QExample]) -> "ExistingPublicStateEncoder":
        if not examples:
            raise ValueError("cannot fit public-state vectorizer on an empty train split")
        cls = self._load_class()
        self.vectorizer = cls(self.max_features).fit([{"public_state": list(row.public_state)} for row in examples])
        return self

    def transform(self, examples: Sequence[QExample]) -> np.ndarray:
        if self.vectorizer is None:
            raise RuntimeError("public-state vectorizer has not been fitted")
        states = [{"public_state": list(row.public_state)} for row in examples]
        values = np.asarray(self.vectorizer.transform(states), dtype=np.float32)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError("public-state vectorizer produced invalid features")
        return values

    def to_json(self) -> dict:
        if self.vectorizer is None:
            raise RuntimeError("public-state vectorizer has not been fitted")
        return {"kind": "audited_v1_public_state_tfidf", **self.vectorizer.to_json()}

    @classmethod
    def from_json(cls, payload: dict) -> "ExistingPublicStateEncoder":
        encoder = cls(int(payload["max_features"]))
        vectorizer_class = cls._load_class()
        encoder.vectorizer = vectorizer_class.from_json(payload)
        return encoder


class Standardizer:
    def __init__(self):
        self.mean = None
        self.scale = None

    def fit(self, values: np.ndarray) -> "Standardizer":
        x = np.asarray(values, dtype=np.float64)
        if x.ndim != 2 or len(x) == 0 or not np.isfinite(x).all():
            raise ValueError("standardizer needs a finite non-empty train feature matrix")
        self.mean = x.mean(axis=0)
        self.scale = x.std(axis=0)
        self.scale[self.scale < 1e-8] = 1.0
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean is None or self.scale is None:
            raise RuntimeError("standardizer has not been fitted")
        x = np.asarray(values, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.mean.size or not np.isfinite(x).all():
            raise ValueError("feature matrix has invalid shape or values")
        return ((x - self.mean) / self.scale).astype(np.float32)

    def to_json(self) -> dict:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_json(cls, payload: dict) -> "Standardizer":
        result = cls()
        result.mean = np.asarray(payload["mean"], dtype=np.float64)
        result.scale = np.asarray(payload["scale"], dtype=np.float64)
        return result


def feature_matrix(examples: Sequence[QExample], *, encoder: ExistingPublicStateEncoder | None = None,
                   external_vectors: dict[tuple[str, str], np.ndarray] | None = None) -> tuple[np.ndarray, ExistingPublicStateEncoder | None]:
    if external_vectors is not None:
        missing = [(row.case_id, row.state_id) for row in examples if (row.case_id, row.state_id) not in external_vectors]
        if missing:
            raise ValueError(f"missing public-state features for {len(missing)} states; first={missing[0]}")
        matrix = np.stack([external_vectors[(row.case_id, row.state_id)] for row in examples]).astype(np.float32)
        return matrix, None
    encoder = encoder or ExistingPublicStateEncoder()
    if encoder.vectorizer is None:
        encoder.fit(examples)
    return encoder.transform(examples), encoder
