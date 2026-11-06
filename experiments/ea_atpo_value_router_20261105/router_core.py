"""Leakage-controlled small value router for the EA-ATPO feasibility data.

Only public_state text is featurized.  Outcome, gold, hidden facts, advice, and
future turns are labels/evaluation data and are never passed to a policy call.
The implementation uses NumPy only so its runtime dependencies stay small.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


ACTIONS = ("ANSWER", "ASK", "CONSULT", "DEFER")
TOKEN_RE = re.compile(r"[a-zA-Z]+|\d+(?:\.\d+)?")


def public_text(messages: Sequence[dict[str, Any]]) -> str:
    """Render only public, non-system dialogue for the pre-action feature set."""
    pieces = []
    for message in messages:
        role = str(message.get("role", ""))
        if role == "system":
            continue
        if role in ("user", "assistant"):
            pieces.append(role + ": " + str(message.get("content", "")))
    return "\n".join(pieces)


def _terms(text: str) -> list[str]:
    words = [x.lower() for x in TOKEN_RE.findall(text)]
    grams = words + [words[i] + "__" + words[i + 1] for i in range(len(words) - 1)]
    return grams


def _numeric(text: str, messages: Sequence[dict[str, Any]]) -> np.ndarray:
    words = TOKEN_RE.findall(text)
    roles = Counter(str(m.get("role", "")) for m in messages if m.get("role") != "system")
    option_marks = len(re.findall(r"(?:^|[\s{,])['\"]?[A-D]['\"]?\s*[:.)]", text))
    uncertainty = sum(
        text.lower().count(term)
        for term in ("uncertain", "unknown", "unclear", "not sure", "cannot", "insufficient")
    )
    return np.asarray(
        [
            math.log1p(len(text)),
            math.log1p(len(words)),
            math.log1p(roles.get("user", 0)),
            math.log1p(roles.get("assistant", 0)),
            math.log1p(text.count("?")),
            math.log1p(option_marks),
            math.log1p(sum(ch.isdigit() for ch in text)),
            math.log1p(uncertainty),
            len(words) / max(1, len(text)),
        ],
        dtype=np.float64,
    )


class PublicStateVectorizer:
    def __init__(self, max_features: int = 3000):
        self.max_features = int(max_features)
        self.vocab: dict[str, int] = {}
        self.idf = np.zeros(0, dtype=np.float64)
        self.numeric_mean = np.zeros(0, dtype=np.float64)
        self.numeric_std = np.ones(0, dtype=np.float64)

    def fit(self, states: Sequence[dict[str, Any]]) -> "PublicStateVectorizer":
        documents = [public_text(s["public_state"]) for s in states]
        df: Counter[str] = Counter()
        for text in documents:
            df.update(set(_terms(text)))
        ordered = sorted(df, key=lambda term: (-df[term], term))[: self.max_features]
        self.vocab = {term: i for i, term in enumerate(ordered)}
        n = max(1, len(documents))
        self.idf = np.asarray([math.log((1 + n) / (1 + df[t])) + 1.0 for t in ordered])
        nums = np.vstack([_numeric(t, s["public_state"]) for t, s in zip(documents, states)])
        self.numeric_mean = nums.mean(axis=0)
        self.numeric_std = nums.std(axis=0)
        self.numeric_std[self.numeric_std < 1e-8] = 1.0
        return self

    def transform(self, states: Sequence[dict[str, Any]]) -> np.ndarray:
        text_docs = [public_text(s["public_state"]) for s in states]
        n_docs = len(states)
        tfidf = np.zeros((n_docs, len(self.vocab)), dtype=np.float64)
        for i, text in enumerate(text_docs):
            counts = Counter(self.vocab[t] for t in _terms(text) if t in self.vocab)
            for index, count in counts.items():
                tfidf[i, index] = (1.0 + math.log(count)) * self.idf[index]
            norm = np.linalg.norm(tfidf[i])
            if norm > 0:
                tfidf[i] /= norm
        nums = np.vstack([_numeric(t, s["public_state"]) for t, s in zip(text_docs, states)])
        nums = (nums - self.numeric_mean) / self.numeric_std
        return np.column_stack([tfidf, nums, np.ones(n_docs, dtype=np.float64)])

    def to_json(self) -> dict[str, Any]:
        return {
            "max_features": self.max_features,
            "vocab": self.vocab,
            "idf": self.idf.tolist(),
            "numeric_mean": self.numeric_mean.tolist(),
            "numeric_std": self.numeric_std.tolist(),
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "PublicStateVectorizer":
        result = cls(int(payload["max_features"]))
        result.vocab = {str(k): int(v) for k, v in payload["vocab"].items()}
        result.idf = np.asarray(payload["idf"], dtype=np.float64)
        result.numeric_mean = np.asarray(payload["numeric_mean"], dtype=np.float64)
        result.numeric_std = np.asarray(payload["numeric_std"], dtype=np.float64)
        return result


def fit_kernel_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Return dual coefficients for a linear kernel ridge model."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    gram = x @ x.T
    return np.linalg.solve(gram + float(alpha) * np.eye(len(x)), y)


def predict_kernel_ridge(x_new: np.ndarray, x_train: np.ndarray, dual: np.ndarray) -> np.ndarray:
    pred = np.asarray(x_new, dtype=np.float64) @ np.asarray(x_train, dtype=np.float64).T @ np.asarray(dual, dtype=np.float64)
    return pred[:, 0] if pred.shape[1] == 1 else pred


def fit_reliability_oof(
    x: np.ndarray,
    y: np.ndarray,
    case_ids: Sequence[str],
    alpha: float,
    folds: int = 5,
    seed: int = 20261105,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cross-fit DEFER correctness using case-disjoint folds."""
    cases = sorted(set(str(c) for c in case_ids))
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(cases, dtype=object)
    rng.shuffle(shuffled)
    fold_for = {case: i % folds for i, case in enumerate(shuffled.tolist())}
    oof = np.full(len(x), np.nan, dtype=np.float64)
    for fold in range(folds):
        test_idx = np.asarray([i for i, c in enumerate(case_ids) if fold_for[str(c)] == fold], dtype=int)
        train_idx = np.asarray([i for i, c in enumerate(case_ids) if fold_for[str(c)] != fold], dtype=int)
        if not len(test_idx) or not len(train_idx):
            continue
        dual = fit_kernel_ridge(x[train_idx], y[train_idx], alpha)
        oof[test_idx] = predict_kernel_ridge(x[test_idx], x[train_idx], dual)
    missing = np.isnan(oof)
    if missing.any():
        oof[missing] = float(np.mean(y))
    oof = np.clip(oof, 0.0, 1.0)
    full_dual = fit_kernel_ridge(x, y, alpha)
    return oof, x.copy(), full_dual


def stable_split(case_ids: Iterable[str], seed: int = 20261105) -> dict[str, list[str]]:
    cases = sorted(set(str(c) for c in case_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(cases)
    n = len(cases)
    n_train = int(round(n * 0.70))
    n_val = int(round(n * 0.15))
    train = sorted(cases[:n_train])
    val = sorted(cases[n_train : n_train + n_val])
    test = sorted(cases[n_train + n_val :])
    assert not set(train) & set(val) and not set(train) & set(test) and not set(val) & set(test)
    return {"train": train, "validation": val, "internal_test": test}


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
