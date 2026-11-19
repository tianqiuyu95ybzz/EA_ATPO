#!/usr/bin/env python3
"""Frozen, CPU-only audit of Expert relative-gain predictability.

This script fits preprocessing, models, and cost-specific thresholds using only
the frozen train/validation splits. It writes policy_lock.json before opening
paired_test.jsonl. The test split is used once for final descriptive analysis.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent
RUN = AUDIT / "outputs/qwen32_router_generalization_20261118_attempt04"
ROLLOUTS = RUN / "paired_rollouts"
SELECTION = RUN / "case_selection.json"
SOURCE_TRAIN = Path("/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/experiments/real_case_pilot_20261103/assets/train.pilot.parquet")
OUT = HERE
COSTS = [0.00, 0.05, 0.10, 0.20, 0.40, 0.60]
CALL_FRACTIONS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 1.00]
SEED = 20261118
BOOTSTRAP_SEED = 20261119
BOOTSTRAP_N = 10000
AUC_BOOTSTRAP_N = 2000
RANDOM_SEEDS = list(range(20261118, 20261138))
LOGISTIC_L2 = [0.001, 0.01, 0.1, 1.0]
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]
MLP_DECAYS = [0.0001, 0.001, 0.01]
EXPECTED = {"train": 1000, "validation": 300, "test": 400}

PUBLIC_FIELDS = [
    "log_public_chars", "log_public_words", "question_marks", "digit_count",
    "clinical_lexicon_count",
]
OPTION_FIELDS = ["option_count", "mean_option_chars"]
FEATURE_NAMES = PUBLIC_FIELDS + OPTION_FIELDS + [
    "base_confidence", "base_entropy_normalized", "base_top1_top2_margin",
    "base_uncertainty_missing", "base_valid",
    "base_predicted_A", "base_predicted_B", "base_predicted_C", "base_predicted_D",
    "base_probability_A", "base_probability_B", "base_probability_C", "base_probability_D",
    "log_base_prompt_tokens", "log_base_completion_tokens",
]
CLINICAL_TERMS = (
    "pain", "fever", "cough", "blood", "dose", "drug", "symptom", "diagnosis",
    "treatment", "history", "onset", "patient", "exam", "test", "surgery",
    "infection", "pressure", "heart", "lung", "kidney", "pregnan",
)
METHODS = [
    "base_uncertainty", "logistic_base_error", "logistic_relative_gain",
    "ridge_relative_gain", "mlp_relative_gain",
]
METHOD_LABELS = {
    "always_answer": "Always Answer",
    "always_defer": "Always DEFER",
    "base_uncertainty": "Base Uncertainty",
    "logistic_base_error": "Logistic Base Error",
    "logistic_relative_gain": "Logistic Relative Gain",
    "ridge_relative_gain": "Ridge Relative Gain",
    "mlp_relative_gain": "MLP Relative Gain",
    "oracle_selective": "Oracle Selective DEFER (hindsight upper bound)",
}


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def feature_row(row: dict[str, Any]) -> dict[str, float | None]:
    public = row.get("public_features") or {}
    values: dict[str, float | None] = {}
    for name in PUBLIC_FIELDS:
        values[name] = safe_float(public.get(name))

    # Existing option features are the output of a regex fallback that writes
    # zero when no option line is matched. They are deliberately unavailable.
    values["option_count"] = None
    values["mean_option_chars"] = None

    probs = row.get("base_answer_option_probabilities") or {}
    prob_values = [safe_float(probs.get(letter)) for letter in "ABCD"]
    uncertainty_valid = bool(row.get("base_uncertainty_valid")) and all(v is not None for v in prob_values)
    if uncertainty_valid:
        p = np.asarray(prob_values, dtype=float)
        uncertainty_valid = bool(np.all(p >= 0.0) and np.isclose(p.sum(), 1.0, atol=1e-5))
    if uncertainty_valid:
        p = np.asarray(prob_values, dtype=float)
        ordered = np.sort(p)
        confidence = float(ordered[-1])
        entropy = -float(np.sum(np.where(p > 0, p * np.log(np.maximum(p, 1e-300)), 0.0))) / math.log(4.0)
        margin = float(ordered[-1] - ordered[-2])
        values["base_confidence"] = confidence
        values["base_entropy_normalized"] = safe_float(row.get("base_answer_entropy_normalized"))
        values["base_top1_top2_margin"] = safe_float(row.get("base_answer_probability_margin"))
        # Formula values are separately verified; use the cached fields after that check.
        for letter, probability in zip("ABCD", prob_values):
            values[f"base_probability_{letter}"] = probability
    else:
        entropy = None
        margin = None
        values["base_confidence"] = None
        values["base_entropy_normalized"] = None
        values["base_top1_top2_margin"] = None
        for letter in "ABCD":
            values[f"base_probability_{letter}"] = None
    values["base_uncertainty_missing"] = float(not uncertainty_valid)
    values["base_valid"] = float(bool(row.get("base_valid")))
    predicted = str(row.get("base_predicted") or "").upper()
    for letter in "ABCD":
        values[f"base_predicted_{letter}"] = float(predicted == letter) if predicted in "ABCD" else None
    for name, raw in (("log_base_prompt_tokens", row.get("base_prompt_tokens")),
                      ("log_base_completion_tokens", row.get("base_completion_tokens"))):
        number = safe_float(raw)
        values[name] = math.log1p(number) if number is not None and number >= 0 else None
    return values


def labels(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    base = np.asarray([bool(r["base_correct"]) for r in rows], dtype=bool)
    expert = np.asarray([bool(r["expert_correct"]) for r in rows], dtype=bool)
    delta = expert.astype(float) - base.astype(float)
    return {
        "base": base, "expert": expert, "delta": delta,
        "base_error": ~base, "helpful": delta == 1.0, "harmful": delta == -1.0,
    }


def verify_uncertainty(rows: list[dict[str, Any]]) -> dict[str, int]:
    checked = 0
    invalid_flagged = 0
    mismatch_entropy = 0
    mismatch_margin = 0
    probability_sum_failures = 0
    for row in rows:
        probs = row.get("base_answer_option_probabilities") or {}
        vals = [safe_float(probs.get(letter)) for letter in "ABCD"]
        valid = bool(row.get("base_uncertainty_valid"))
        if not valid:
            invalid_flagged += 1
            continue
        if any(value is None for value in vals):
            probability_sum_failures += 1
            continue
        p = np.asarray(vals, dtype=float)
        if np.any(p < 0) or not np.isclose(p.sum(), 1.0, atol=1e-5):
            probability_sum_failures += 1
            continue
        entropy = -float(np.sum(np.where(p > 0, p * np.log(np.maximum(p, 1e-300)), 0.0))) / math.log(4.0)
        sorted_p = np.sort(p)
        margin = float(sorted_p[-1] - sorted_p[-2])
        cached_entropy = safe_float(row.get("base_answer_entropy_normalized"))
        cached_margin = safe_float(row.get("base_answer_probability_margin"))
        if cached_entropy is None or not math.isclose(entropy, cached_entropy, rel_tol=1e-5, abs_tol=1e-6):
            mismatch_entropy += 1
        if cached_margin is None or not math.isclose(margin, cached_margin, rel_tol=1e-5, abs_tol=1e-6):
            mismatch_margin += 1
        checked += 1
    return {"valid_probability_rows": checked, "uncertainty_invalid_rows": invalid_flagged,
            "probability_sum_failures": probability_sum_failures, "entropy_formula_mismatches": mismatch_entropy,
            "margin_formula_mismatches": mismatch_margin}


def fit_preprocessor(train_rows: list[dict[str, Any]]) -> tuple[np.ndarray, dict[str, Any]]:
    train_features = [feature_row(r) for r in train_rows]
    kept: list[str] = []
    dropped: dict[str, str] = {}
    medians: dict[str, float] = {}
    train_imputed: dict[str, np.ndarray] = {}
    missing_columns: dict[str, np.ndarray] = {}
    for name in FEATURE_NAMES:
        raw = np.asarray([np.nan if f.get(name) is None else float(f[name]) for f in train_features], dtype=float)
        observed = np.isfinite(raw)
        if not np.any(observed):
            dropped[name] = "unavailable on all train cases"
            continue
        med = float(np.median(raw[observed]))
        filled = np.where(observed, raw, med)
        if float(np.std(filled)) < 1e-10:
            dropped[name] = "constant on train after train-median imputation"
            continue
        kept.append(name)
        medians[name] = med
        train_imputed[name] = filled
        if np.any(~observed):
            missing_name = f"{name}__missing"
            missing_values = (~observed).astype(float)
            if float(np.std(missing_values)) >= 1e-10:
                kept.append(missing_name)
                missing_columns[missing_name] = missing_values
    raw_columns = {name: train_imputed[name] for name in kept if not name.endswith("__missing")}
    raw_columns.update(missing_columns)
    means: dict[str, float] = {}
    scales: dict[str, float] = {}
    for name in kept:
        col = raw_columns[name]
        means[name] = float(np.mean(col))
        scale = float(np.std(col))
        scales[name] = scale if scale >= 1e-10 else 1.0
    x_train = np.column_stack([(raw_columns[name] - means[name]) / scales[name] for name in kept])
    state = {"feature_order": kept, "source_feature_order": FEATURE_NAMES, "train_medians": medians,
             "train_means": means, "train_scales": scales, "dropped_train_only": dropped,
             "fit_split": "train", "fit_case_count": len(train_rows)}
    return x_train, state


def transform_rows(rows: list[dict[str, Any]], state: dict[str, Any]) -> np.ndarray:
    features = [feature_row(r) for r in rows]
    cols: dict[str, np.ndarray] = {}
    for name in state["source_feature_order"]:
        if name in state["train_medians"]:
            raw = np.asarray([np.nan if f.get(name) is None else float(f[name]) for f in features], dtype=float)
            observed = np.isfinite(raw)
            cols[name] = np.where(observed, raw, float(state["train_medians"][name]))
            missing_name = f"{name}__missing"
            if missing_name in state["feature_order"]:
                cols[missing_name] = (~observed).astype(float)
    return np.column_stack([(cols[name] - state["train_means"][name]) / state["train_scales"][name]
                           for name in state["feature_order"]])


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))


def logloss_binary(prob: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(prob, 1e-9, 1.0 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def class_sample_weights(y: np.ndarray, balanced: bool) -> np.ndarray:
    if not balanced:
        return np.ones(len(y), dtype=float)
    counts = np.bincount(y.astype(int), minlength=2).astype(float)
    weights = np.asarray([len(y) / (2.0 * max(counts[int(value)], 1.0)) for value in y], dtype=float)
    return weights / float(np.mean(weights))


def fit_logistic_binary(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray,
                        y_val: np.ndarray, l2: float, balanced: bool) -> dict[str, Any]:
    n, d = x_train.shape
    design = np.column_stack([np.ones(n), x_train])
    weights = class_sample_weights(y_train.astype(int), balanced)
    penalty = np.eye(d + 1, dtype=float) * l2
    penalty[0, 0] = 0.0
    beta = np.zeros(d + 1, dtype=float)
    for iteration in range(300):
        p = sigmoid(design @ beta)
        gradient = design.T @ (weights * (p - y_train)) / n + penalty @ beta
        curvature = weights * p * (1.0 - p)
        hessian = (design.T * curvature) @ design / n + penalty + np.eye(d + 1) * 1e-9
        step = np.linalg.solve(hessian, gradient)
        norm = float(np.linalg.norm(step))
        if norm > 5.0:
            step *= 5.0 / norm
        beta -= step
        if float(np.linalg.norm(step)) < 1e-8:
            break
    val_prob = sigmoid(np.column_stack([np.ones(len(x_val)), x_val]) @ beta)
    return {"intercept": float(beta[0]), "coef": beta[1:].tolist(), "l2": l2,
            "class_weight": "balanced" if balanced else "none", "iterations": iteration + 1,
            "validation_logloss": logloss_binary(val_prob, y_val), "validation_probabilities": val_prob}


def fit_selected_logistic(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray,
                          y_val: np.ndarray) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates = []
    for balanced in (False, True):
        for l2 in LOGISTIC_L2:
            item = fit_logistic_binary(x_train, y_train.astype(float), x_val, y_val.astype(float), l2, balanced)
            candidates.append(item)
    best = min(candidates, key=lambda item: (item["validation_logloss"], item["l2"], item["class_weight"] != "none"))
    return best, [{k: v for k, v in item.items() if k != "validation_probabilities"} for item in candidates]


def predict_logistic(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    return sigmoid(x @ np.asarray(model["coef"], dtype=float) + float(model["intercept"]))


def fit_ridge(x_train: np.ndarray, delta_train: np.ndarray, x_val: np.ndarray,
              delta_val: np.ndarray) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    y_mean = float(np.mean(delta_train))
    candidates = []
    for alpha in RIDGE_ALPHAS:
        coef = np.linalg.solve(x_train.T @ x_train + alpha * np.eye(x_train.shape[1]),
                               x_train.T @ (delta_train - y_mean))
        prediction = y_mean + x_val @ coef
        candidates.append({"alpha": alpha, "intercept": y_mean, "coef": coef.tolist(),
                           "validation_mse": float(np.mean((prediction - delta_val) ** 2)),
                           "validation_prediction": prediction})
    best = min(candidates, key=lambda item: (item["validation_mse"], -item["alpha"]))
    return best, [{k: v for k, v in item.items() if k != "validation_prediction"} for item in candidates]


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=1, keepdims=True)
    expz = np.exp(np.clip(z, -50.0, 50.0))
    return expz / np.sum(expz, axis=1, keepdims=True)


def fit_mlp_candidate(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray,
                      decay: float, balanced: bool, seed: int) -> dict[str, Any]:
    n, d = x_train.shape
    hidden = 16
    rng = np.random.default_rng(seed)
    w1 = rng.normal(0.0, math.sqrt(2.0 / d), size=(d, hidden))
    b1 = np.zeros(hidden)
    w2 = rng.normal(0.0, math.sqrt(2.0 / hidden), size=(hidden, 3))
    b2 = np.zeros(3)
    moments = [np.zeros_like(p) for p in (w1, b1, w2, b2)]
    variances = [np.zeros_like(p) for p in (w1, b1, w2, b2)]
    class_counts = np.bincount(y_train.astype(int), minlength=3).astype(float)
    cw = np.ones(3) if not balanced else len(y_train) / (3.0 * np.maximum(class_counts, 1.0))
    sw = cw[y_train.astype(int)]
    sw /= float(np.mean(sw))
    y_onehot = np.eye(3)[y_train.astype(int)]
    best_loss = math.inf
    best_state: tuple[np.ndarray, ...] | None = None
    best_epoch = 0
    stale = 0
    beta1, beta2, eps, lr = 0.9, 0.999, 1e-8, 0.01
    for epoch in range(1, 501):
        hidden_raw = x_train @ w1 + b1
        hidden_act = np.maximum(hidden_raw, 0.0)
        probs = softmax(hidden_act @ w2 + b2)
        grad_logits = (probs - y_onehot) * sw[:, None] / n
        grad_w2 = hidden_act.T @ grad_logits + decay * w2
        grad_b2 = np.sum(grad_logits, axis=0)
        grad_hidden = grad_logits @ w2.T
        grad_hidden[hidden_raw <= 0.0] = 0.0
        grad_w1 = x_train.T @ grad_hidden + decay * w1
        grad_b1 = np.sum(grad_hidden, axis=0)
        grads = (grad_w1, grad_b1, grad_w2, grad_b2)
        params = [w1, b1, w2, b2]
        for i, (param, grad) in enumerate(zip(params, grads)):
            moments[i] = beta1 * moments[i] + (1.0 - beta1) * grad
            variances[i] = beta2 * variances[i] + (1.0 - beta2) * grad * grad
            mhat = moments[i] / (1.0 - beta1 ** epoch)
            vhat = variances[i] / (1.0 - beta2 ** epoch)
            param -= lr * mhat / (np.sqrt(vhat) + eps)
        val_h = np.maximum(x_val @ w1 + b1, 0.0)
        val_prob = softmax(val_h @ w2 + b2)
        loss = float(-np.mean(np.log(np.maximum(val_prob[np.arange(len(y_val)), y_val.astype(int)], 1e-12))))
        if loss < best_loss - 1e-6:
            best_loss = loss
            best_epoch = epoch
            best_state = tuple(p.copy() for p in (w1, b1, w2, b2))
            stale = 0
        else:
            stale += 1
            if stale >= 30:
                break
    assert best_state is not None
    return {"w1": best_state[0].tolist(), "b1": best_state[1].tolist(),
            "w2": best_state[2].tolist(), "b2": best_state[3].tolist(),
            "weight_decay": decay, "class_weight": "balanced" if balanced else "none",
            "best_epoch": best_epoch, "validation_logloss": best_loss}


def predict_mlp(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    w1 = np.asarray(model["w1"], dtype=float)
    b1 = np.asarray(model["b1"], dtype=float)
    w2 = np.asarray(model["w2"], dtype=float)
    b2 = np.asarray(model["b2"], dtype=float)
    return softmax(np.maximum(x @ w1 + b1, 0.0) @ w2 + b2)


def fit_selected_mlp(x_train: np.ndarray, delta_train: np.ndarray, x_val: np.ndarray,
                     delta_val: np.ndarray) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    y_train = (delta_train + 1).astype(int)
    y_val = (delta_val + 1).astype(int)
    candidates = []
    run_ix = 0
    for balanced in (False, True):
        for decay in MLP_DECAYS:
            run_ix += 1
            candidates.append(fit_mlp_candidate(x_train, y_train, x_val, y_val, decay, balanced, SEED + run_ix))
    best = min(candidates, key=lambda item: (item["validation_logloss"], item["weight_decay"], item["class_weight"] != "none"))
    summary = [{k: v for k, v in item.items() if k not in {"w1", "b1", "w2", "b2"}} for item in candidates]
    return best, summary


def select_threshold(scores: np.ndarray, rows: list[dict[str, Any]], cost: float) -> dict[str, Any]:
    outcome = labels(rows)
    candidates = sorted(set(float(v) for v in scores))
    candidates.append(float(np.nextafter(max(candidates), math.inf)))
    best: dict[str, Any] | None = None
    for threshold in candidates:
        calls = scores >= threshold
        correctness = outcome["base"].astype(float) + calls * outcome["delta"]
        rate = float(np.mean(calls))
        utility = float(np.mean(correctness) - cost * rate)
        item = {"threshold": float(threshold), "validation_utility": utility,
                "validation_accuracy": float(np.mean(correctness)), "validation_call_rate": rate,
                "validation_call_count": int(np.sum(calls))}
        if best is None or (item["validation_utility"], -item["validation_call_rate"], item["threshold"]) > (
                best["validation_utility"], -best["validation_call_rate"], best["threshold"]):
            best = item
    assert best is not None
    return best


def auc(scores: np.ndarray, target: np.ndarray) -> float:
    y = np.asarray(target, dtype=bool)
    pos, neg = int(np.sum(y)), int(np.sum(~y))
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return float((np.sum(ranks[y]) - pos * (pos + 1) / 2.0) / (pos * neg))


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def spearman(scores: np.ndarray, target: np.ndarray) -> float:
    a, b = average_ranks(np.asarray(scores, dtype=float)), average_ranks(np.asarray(target, dtype=float))
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def bootstrap_auc_ci(scores: np.ndarray, target: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    n = len(target)
    values = []
    for _ in range(AUC_BOOTSTRAP_N):
        ix = rng.integers(0, n, n)
        value = auc(scores[ix], target[ix])
        if math.isfinite(value):
            values.append(value)
    if not values:
        return float("nan"), float("nan")
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def bootstrap_mean_ci(values: np.ndarray, indices: np.ndarray) -> tuple[float, float]:
    sample_means = np.mean(values[indices], axis=1)
    return float(np.quantile(sample_means, 0.025)), float(np.quantile(sample_means, 0.975))


def bootstrap_ratio_ci(numerator: np.ndarray, denominator: np.ndarray,
                       indices: np.ndarray) -> tuple[float, float]:
    nums = np.sum(numerator[indices], axis=1)
    dens = np.sum(denominator[indices], axis=1)
    result = np.divide(nums, dens, out=np.full(len(nums), np.nan), where=dens > 0)
    return float(np.nanquantile(result, 0.025)), float(np.nanquantile(result, 0.975))


def policy_metrics(rows: list[dict[str, Any]], calls: np.ndarray, cost: float,
                   boot_indices: np.ndarray, method: str, validation_rate: float | None = None) -> dict[str, Any]:
    outcome = labels(rows)
    call = np.asarray(calls, dtype=float)
    base = outcome["base"].astype(float)
    delta = outcome["delta"]
    correctness = base + call * delta
    utility_values = correctness - cost * call
    lift_values = call * (delta - cost)
    accuracy_delta = call * delta
    acc_ci = bootstrap_mean_ci(correctness, boot_indices)
    util_ci = bootstrap_mean_ci(utility_values, boot_indices)
    call_ci = bootstrap_mean_ci(call, boot_indices)
    lift_ci = bootstrap_mean_ci(lift_values, boot_indices)
    acc_delta_ci = bootstrap_mean_ci(accuracy_delta, boot_indices)
    correction_capture = float(np.sum(call * outcome["helpful"]) / max(1, np.sum(outcome["helpful"])))
    correction_ci = bootstrap_ratio_ci(call * outcome["helpful"], outcome["helpful"].astype(float), boot_indices)
    regression_rate = float(np.sum(call * outcome["harmful"]) / max(1, np.sum(outcome["base"])))
    regression_ci = bootstrap_ratio_ci(call * outcome["harmful"], outcome["base"].astype(float), boot_indices)
    return {
        "policy": method, "cost": cost, "validation_call_rate": validation_rate,
        "test_n": len(rows), "expert_call_count": float(np.sum(call)), "defer_rate": float(np.mean(call)),
        "defer_rate_ci_low": call_ci[0], "defer_rate_ci_high": call_ci[1],
        "accuracy": float(np.mean(correctness)), "accuracy_ci_low": acc_ci[0], "accuracy_ci_high": acc_ci[1],
        "utility": float(np.mean(utility_values)), "utility_ci_low": util_ci[0], "utility_ci_high": util_ci[1],
        "average_cost": float(cost * np.mean(call)),
        "utility_gain_vs_always_answer": float(np.mean(lift_values)),
        "utility_gain_ci_low": lift_ci[0], "utility_gain_ci_high": lift_ci[1],
        "accuracy_gain_vs_always_answer": float(np.mean(accuracy_delta)),
        "accuracy_gain_ci_low": acc_delta_ci[0], "accuracy_gain_ci_high": acc_delta_ci[1],
        "expert_correction_capture_rate": correction_capture,
        "expert_correction_capture_ci_low": correction_ci[0], "expert_correction_capture_ci_high": correction_ci[1],
        "base_correct_regression_rate": regression_rate,
        "base_correct_regression_ci_low": regression_ci[0], "base_correct_regression_ci_high": regression_ci[1],
    }


def random_call_matrix(n: int, k: int) -> np.ndarray:
    calls = np.zeros((len(RANDOM_SEEDS), n), dtype=bool)
    k = max(0, min(int(k), n))
    for i, seed in enumerate(RANDOM_SEEDS):
        rng = np.random.default_rng(seed)
        selected = rng.choice(n, size=k, replace=False) if k else np.asarray([], dtype=int)
        calls[i, selected] = True
    return calls


def validate_pair_rows(split: str, rows: list[dict[str, Any]]) -> dict[str, int]:
    if len(rows) != EXPECTED[split]:
        raise RuntimeError(f"{split} row count mismatch: {len(rows)} != {EXPECTED[split]}")
    if any(r.get("split") != split for r in rows):
        raise RuntimeError(f"Incorrect split labels in {split}")
    report = {}
    for key in ("case_id", "case_hash", "prompt_hash"):
        values = [r.get(key) for r in rows]
        if len(set(values)) != len(values):
            raise RuntimeError(f"Duplicate {key} in {split}")
        report[f"unique_{key}"] = len(set(values))
    return report


def validate_trajectory_alignment(split: str, rows: list[dict[str, Any]],
                                  base: dict[str, dict[str, Any]], expert: dict[str, dict[str, Any]]) -> dict[str, int]:
    mismatches = {"missing_base": 0, "missing_expert": 0, "identity_mismatch": 0,
                  "base_label_mismatch": 0, "expert_label_mismatch": 0}
    for row in rows:
        key = row["case_hash"]
        b, e = base.get(key), expert.get(key)
        if b is None:
            mismatches["missing_base"] += 1
            continue
        if e is None:
            mismatches["missing_expert"] += 1
            continue
        for trace in (b, e):
            if any(trace.get(name) != row.get(name) for name in ("case_id", "case_hash", "prompt_hash", "split")):
                mismatches["identity_mismatch"] += 1
        if bool(b.get("correct")) != bool(row["base_correct"]):
            mismatches["base_label_mismatch"] += 1
        if bool(e.get("correct")) != bool(row["expert_correct"]):
            mismatches["expert_label_mismatch"] += 1
    if any(mismatches.values()):
        raise RuntimeError(f"Trajectory/paired alignment failure in {split}: {mismatches}")
    return mismatches


def format_number(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def read_csv_artifact(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as stream:
        for original in csv.DictReader(stream):
            row: dict[str, Any] = {}
            for key, value in original.items():
                if value in ("True", "False"):
                    row[key] = value == "True"
                elif value == "":
                    row[key] = ""
                else:
                    try:
                        number = float(value)
                        row[key] = int(number) if number.is_integer() and key in {"n", "test_n", "expert_call_count", "positive_count", "helpful_total", "harmful_total", "target_call_count"} else number
                    except (TypeError, ValueError):
                        row[key] = value
            rows.append(row)
    return rows


def finalize_existing_outputs() -> None:
    """Assemble report from already-written frozen metrics; never refit or retune."""
    required = ["policy_lock.json", "config.json", "integrity_audit.json", "split_summary.csv",
                "cost_sensitivity.csv", "router_metrics.csv", "predictive_targets.csv", "feature_audit.csv"]
    missing = [name for name in required if not (OUT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot finalize; missing existing analysis outputs: {missing}")
    integrity = read_json(OUT / "integrity_audit.json")
    uncertainty = integrity.get("uncertainty_audit", {})
    if "valid_probability_rows" not in uncertainty:
        by_split = uncertainty
        keys = ("valid_probability_rows", "uncertainty_invalid_rows", "probability_sum_failures",
                "entropy_formula_mismatches", "margin_formula_mismatches")
        integrity["uncertainty_audit_by_split"] = by_split
        integrity["uncertainty_audit"] = {key: sum(int(row.get(key, 0)) for row in by_split.values()) for key in keys}
        write_json(OUT / "integrity_audit.json", integrity)

    summary_rows = read_csv_artifact(OUT / "split_summary.csv")
    summary = {str(row["split"]): row for row in summary_rows}
    cost_rows = read_csv_artifact(OUT / "cost_sensitivity.csv")
    router_rows = read_csv_artifact(OUT / "router_metrics.csv")
    target_rows = read_csv_artifact(OUT / "predictive_targets.csv")
    feature_rows = read_csv_artifact(OUT / "feature_audit.csv")
    selection_doc = read_json(SELECTION)
    run_manifest = read_json(ROLLOUTS / "run_manifest.json")
    model_lock = read_json(OUT / "policy_lock.json")
    config = read_json(OUT / "config.json")
    config["finalize_existing_command"] = (
        "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/.venv-real-pilot/bin/python "
        "experiments/real_case_pilot_20261103/audit/relative_gain_predictability_20261118/run_relative_gain_predictability.py --finalize-existing"
    )
    config.setdefault("random_baseline", {})["seed_interval"] = (
        "2.5-97.5 percentile interval across 20 seed-level outcomes; paired-case bootstrap CI for the seed-average policy outcome also reported"
    )
    write_json(OUT / "config.json", config)

    rows_by_split = {split: read_jsonl(ROLLOUTS / f"paired_{split}.jsonl") for split in EXPECTED}
    expert_trace = {r["case_hash"]: r for r in read_jsonl(ROLLOUTS / "qwen32_defer_trajectories.jsonl")}
    parse_summary: dict[str, Any] = {}
    for split, rows in rows_by_split.items():
        invalid = [r for r in rows if not bool(r.get("expert_valid"))]
        base_invalid = [r for r in rows if not bool(r.get("base_valid"))]
        finish_counts: dict[str, int] = {}
        for row in invalid:
            finish = str(expert_trace[row["case_hash"]].get("finish_reason") or "missing")
            finish_counts[finish] = finish_counts.get(finish, 0) + 1
        parse_summary[split] = {"n": len(rows), "invalid": len(invalid),
                                "invalid_counted_wrong": sum(not bool(r.get("expert_correct")) for r in invalid),
                                "base_invalid": len(base_invalid), "expert_invalid_finish_reasons": finish_counts}
    parse_summary["total_invalid"] = sum(parse_summary[s]["invalid"] for s in EXPECTED)
    plots = [name for name in config.get("artifacts", []) if name.endswith(".png")]
    report, attribution = build_report(summary, cost_rows, router_rows, target_rows, integrity,
                                       feature_rows, integrity["source_prompt_audit"], selection_doc,
                                       run_manifest, model_lock, parse_summary, plots)
    (OUT / "RELATIVE_GAIN_PREDICTABILITY_REPORT.md").write_text(report, encoding="utf-8")
    (OUT / "failure_attribution.md").write_text(attribution, encoding="utf-8")
    print(json.dumps({"finalized_from_existing_metrics": True,
                      "report": str(OUT / "RELATIVE_GAIN_PREDICTABILITY_REPORT.md"),
                      "failure_attribution": str(OUT / "failure_attribution.md"),
                      "metrics_refit": False, "thresholds_changed": False}, ensure_ascii=False, indent=2))


def score_bundle(rows: list[dict[str, Any]], x: np.ndarray, models: dict[str, Any]) -> dict[str, dict[str, np.ndarray | None]]:
    base_error_prob = predict_logistic(models["logistic_base_error"], x)
    helpful_prob = predict_logistic(models["logistic_helpful"], x)
    harmful_prob = predict_logistic(models["logistic_harmful"], x)
    ridge_delta = float(models["ridge_relative_gain"]["intercept"]) + x @ np.asarray(models["ridge_relative_gain"]["coef"], dtype=float)
    mlp_prob = predict_mlp(models["mlp_relative_gain"], x)
    entropy = np.asarray([
        float(r["base_answer_entropy_normalized"])
        if r.get("base_uncertainty_valid") and safe_float(r.get("base_answer_entropy_normalized")) is not None
        else 1.0 for r in rows
    ], dtype=float)
    logistic_net = helpful_prob - harmful_prob
    mlp_net = mlp_prob[:, 2] - mlp_prob[:, 0]
    return {
        "base_uncertainty": {"route": entropy, "base_error": entropy, "helpful": entropy, "harmful": entropy, "relative_gain": entropy},
        "logistic_base_error": {"route": base_error_prob, "base_error": base_error_prob,
                                "helpful": base_error_prob, "harmful": base_error_prob, "relative_gain": base_error_prob},
        "logistic_relative_gain": {"route": logistic_net, "base_error": None,
                                   "helpful": helpful_prob, "harmful": harmful_prob, "relative_gain": logistic_net},
        "ridge_relative_gain": {"route": ridge_delta, "base_error": None,
                                "helpful": ridge_delta, "harmful": -ridge_delta, "relative_gain": ridge_delta},
        "mlp_relative_gain": {"route": mlp_net, "base_error": None,
                              "helpful": mlp_prob[:, 2], "harmful": mlp_prob[:, 0], "relative_gain": mlp_net},
    }


def audit_source_prompt_features(rows_by_split: dict[str, list[dict[str, Any]]],
                                 selection: dict[str, Any]) -> dict[str, Any]:
    """Recompute text-only public features without retaining or printing prompts."""
    import pyarrow.parquet as pq

    if not SOURCE_TRAIN.is_file():
        raise FileNotFoundError(f"Source prompt parquet is missing: {SOURCE_TRAIN}")
    table = pq.read_table(SOURCE_TRAIN, columns=["prompt"])
    prompt_col = table.column("prompt")
    spec_by_hash = {
        spec["case_hash"]: spec
        for split in ("train", "validation", "test")
        for spec in selection["rows"][split]
    }
    expected_field_checks = {name: {"match": 0, "n": 0} for name in PUBLIC_FIELDS}
    option_checks: dict[str, dict[str, int]] = {}
    patterns = {
        "legacy_ascii_line": re.compile(r"(?m)^\s*[A-D]\s*[).:]\s*(.*?)\s*$", re.IGNORECASE),
        "unicode_punctuation_line": re.compile(r"(?m)^\s*[A-D]\s*[).:：、．。]\s*(.*?)\s*$", re.IGNORECASE),
        "wrapped_label_line": re.compile(r"(?m)^\s*[\(\[]([A-D])[\)\]]\s*(.*?)\s*$", re.IGNORECASE),
        "wrapped_or_plain_line": re.compile(r"(?m)^\s*[\(\[]?[A-D][\)\]]?\s*[).:：、．。]\s*(.*?)\s*$", re.IGNORECASE),
    }
    for split, rows in rows_by_split.items():
        old_match_rows = 0
        expanded_match_rows = 0
        any_pattern_line_hits = 0
        all_four_label_rows = 0
        for row in rows:
            spec = spec_by_hash[row["case_hash"]]
            messages = prompt_col[int(spec["source_row_index"])].as_py()
            text = "\n".join(str(m.get("content", "")) for m in messages if m.get("role") != "system")
            words = re.findall(r"\b\w+\b", text, re.UNICODE)
            lower = text.lower()
            observed = {
                "log_public_chars": math.log1p(len(text)),
                "log_public_words": math.log1p(len(words)),
                "question_marks": float(text.count("?") + text.count("？")),
                "digit_count": float(sum(ch.isdigit() for ch in text)),
                "clinical_lexicon_count": float(sum(lower.count(term) for term in CLINICAL_TERMS)),
            }
            stored = row.get("public_features") or {}
            for name, value in observed.items():
                expected_field_checks[name]["n"] += 1
                if safe_float(stored.get(name)) is not None and math.isclose(float(stored[name]), value, rel_tol=1e-10, abs_tol=1e-10):
                    expected_field_checks[name]["match"] += 1
            lines_by_pattern = {name: pattern.findall(text) for name, pattern in patterns.items()}
            old_match_rows += int(bool(lines_by_pattern["legacy_ascii_line"]))
            expanded = any(bool(lines_by_pattern[name]) for name in ("unicode_punctuation_line", "wrapped_label_line", "wrapped_or_plain_line"))
            expanded_match_rows += int(expanded)
            any_pattern_line_hits += int(any(bool(v) for v in lines_by_pattern.values()))
            labels_found = set()
            for line in text.splitlines():
                for pattern in patterns.values():
                    match = pattern.match(line)
                    if match:
                        label = next((g for g in match.groups() if g and len(g) == 1 and g.upper() in "ABCD"), None)
                        if label:
                            labels_found.add(label.upper())
            all_four_label_rows += int(labels_found == set("ABCD"))
        option_checks[split] = {
            "legacy_ascii_rows_with_matches": old_match_rows,
            "expanded_pattern_rows_with_matches": expanded_match_rows,
            "any_pattern_rows_with_matches": any_pattern_line_hits,
            "rows_with_all_four_labels": all_four_label_rows,
            "n": len(rows),
        }
    for name, counts in expected_field_checks.items():
        if counts["match"] != counts["n"]:
            raise RuntimeError(f"Cached public feature reconstruction mismatch for {name}: {counts}")
    return {
        "source_path": str(SOURCE_TRAIN), "source_sha256": sha_file(SOURCE_TRAIN),
        "source_row_count": table.num_rows,
        "public_feature_reconstruction": expected_field_checks,
        "option_line_pattern_audit": option_checks,
        "prompt_text_saved_or_printed": False,
        "test_prompts_used_for_model_or_threshold_selection": False,
    }


def build_feature_audit(rows_by_split: dict[str, list[dict[str, Any]]],
                        preprocessor: dict[str, Any], prompt_audit: dict[str, Any]) -> list[dict[str, Any]]:
    final = set(preprocessor["feature_order"])
    output = []
    n_by_split = {split: len(rows) for split, rows in rows_by_split.items()}
    for name in FEATURE_NAMES + OPTION_FIELDS:
        for split, rows in rows_by_split.items():
            values = []
            for row in rows:
                value = feature_row(row).get(name)
                values.append(np.nan if value is None else float(value))
            arr = np.asarray(values, dtype=float)
            available = np.isfinite(arr)
            if name in OPTION_FIELDS:
                status = "unavailable_parse_not_found"
                notes = "Legacy extractor stored fallback zero; source prompt audit found no option-line matches. Treat as unavailable, not a true zero."
                used = False
                values_for_stats = np.asarray([], dtype=float)
            else:
                status = "available" if np.all(available) else "partially_missing"
                notes = "Decision-time Base/public-prompt feature; transform/imputation statistics fit on train only."
                used = name in final or f"{name}__missing" in final
                values_for_stats = arr[available]
            output.append({
                "feature": name, "split": split, "n": n_by_split[split],
                "available_n": int(np.sum(available)), "missing_rate": float(1.0 - np.mean(available)),
                "unique_value_count": int(len(np.unique(values_for_stats))) if len(values_for_stats) else 0,
                "mean": float(np.mean(values_for_stats)) if len(values_for_stats) else "",
                "standard_deviation": float(np.std(values_for_stats)) if len(values_for_stats) else "",
                "extraction_status": status, "usable_for_routing": used,
                "used_in_final_model": used, "decision_time": "before optional Expert call",
                "notes": notes,
            })
    for old_name, source_name in (("legacy_stored_option_count", "option_count"),
                                  ("legacy_stored_mean_option_chars", "mean_option_chars")):
        for split, rows in rows_by_split.items():
            legacy = np.asarray([float((r.get("public_features") or {}).get(source_name, 0.0) or 0.0) for r in rows])
            output.append({
                "feature": old_name, "split": split, "n": len(rows), "available_n": len(rows),
                "missing_rate": 0.0, "unique_value_count": int(len(np.unique(legacy))),
                "mean": float(np.mean(legacy)), "standard_deviation": float(np.std(legacy)),
                "extraction_status": "fallback_zero_not_semantic_value", "usable_for_routing": False,
                "used_in_final_model": False, "decision_time": "before optional Expert call",
                "notes": "Stored 0 on every row because the legacy regex returned no matches; source audit does not establish true absence of choices.",
            })
    return output


def write_paired_outcomes(rows_by_split: dict[str, list[dict[str, Any]]], path: Path) -> None:
    result = []
    for split, rows in rows_by_split.items():
        out = labels(rows)
        for index, row in enumerate(rows, 1):
            features = feature_row(row)
            result.append({
                "split": split, "record_ordinal": index,
                "base_correct": int(out["base"][index - 1]), "expert_correct": int(out["expert"][index - 1]),
                "relative_gain": int(out["delta"][index - 1]), "expert_helpful": int(out["helpful"][index - 1]),
                "expert_harmful": int(out["harmful"][index - 1]), "base_error": int(out["base_error"][index - 1]),
                "base_valid": int(bool(row.get("base_valid"))), "expert_valid": int(bool(row.get("expert_valid"))),
                "expert_parse_failure": int(not bool(row.get("expert_valid"))),
                "base_confidence": features["base_confidence"],
                "base_entropy_normalized": features["base_entropy_normalized"],
                "base_top1_top2_margin": features["base_top1_top2_margin"],
            })
    write_csv(path, result)


def split_outcome_summary(split: str, rows: list[dict[str, Any]], boot_indices: np.ndarray) -> dict[str, Any]:
    out = labels(rows)
    base, expert, delta = out["base"], out["expert"], out["delta"]
    base_ci = bootstrap_mean_ci(base.astype(float), boot_indices)
    expert_ci = bootstrap_mean_ci(expert.astype(float), boot_indices)
    pair_ci = bootstrap_mean_ci(delta, boot_indices)
    return {
        "split": split, "n": len(rows), "base_accuracy": float(np.mean(base)),
        "base_accuracy_ci_low": base_ci[0], "base_accuracy_ci_high": base_ci[1],
        "expert_accuracy": float(np.mean(expert)), "expert_accuracy_ci_low": expert_ci[0],
        "expert_accuracy_ci_high": expert_ci[1], "paired_accuracy_gain": float(np.mean(delta)),
        "paired_gain_ci_low": pair_ci[0], "paired_gain_ci_high": pair_ci[1],
        "base_correct_expert_correct": int(np.sum(base & expert)),
        "base_correct_expert_wrong": int(np.sum(out["harmful"])),
        "base_wrong_expert_correct": int(np.sum(out["helpful"])),
        "base_wrong_expert_wrong": int(np.sum((~base) & (~expert))),
        "relative_gain_minus1": int(np.sum(delta == -1)), "relative_gain_0": int(np.sum(delta == 0)),
        "relative_gain_plus1": int(np.sum(delta == 1)),
        "base_invalid": int(sum(not bool(r.get("base_valid")) for r in rows)),
        "expert_invalid": int(sum(not bool(r.get("expert_valid")) for r in rows)),
        "expert_invalid_counted_wrong": int(sum((not bool(r.get("expert_valid"))) and not bool(r.get("expert_correct")) for r in rows)),
    }


def build_predictive_target_rows(rows_by_split: dict[str, list[dict[str, Any]]],
                                 scores_by_split: dict[str, dict[str, dict[str, np.ndarray | None]]]) -> list[dict[str, Any]]:
    rows_out = []
    rng = np.random.default_rng(SEED + 100)
    for split, rows in rows_by_split.items():
        outcome = labels(rows)
        for method in METHODS:
            bundle = scores_by_split[split][method]
            for target, score_key in (("base_error", "base_error"), ("helpful", "helpful"), ("harmful", "harmful")):
                score = bundle[score_key]
                if score is None:
                    continue
                value = auc(np.asarray(score, dtype=float), outcome[target])
                lo, hi = (bootstrap_auc_ci(np.asarray(score, dtype=float), outcome[target], rng)
                          if split == "test" else (float("nan"), float("nan")))
                rows_out.append({"split": split, "model_or_score": method, "target": target,
                                 "score_direction": "higher means event more likely", "n": len(rows),
                                 "positive_count": int(np.sum(outcome[target])), "auroc": value,
                                 "auc_ci_low": lo, "auc_ci_high": hi, "relative_gain_spearman": ""})
            rel = bundle["relative_gain"]
            if rel is not None:
                rel = np.asarray(rel, dtype=float)
                rows_out.append({"split": split, "model_or_score": method, "target": "relative_gain",
                                 "score_direction": "higher means more positive expert-minus-base gain", "n": len(rows),
                                 "positive_count": "", "auroc": "", "auc_ci_low": "", "auc_ci_high": "",
                                 "relative_gain_spearman": spearman(rel, outcome["delta"])})
    return rows_out


def add_random_seed_intervals(row: dict[str, Any], rows: list[dict[str, Any]],
                              calls_by_seed: np.ndarray, cost: float) -> None:
    outcome = labels(rows)
    base = outcome["base"].astype(float)
    delta = outcome["delta"]
    accuracies, utilities, lifts, rates, counts = [], [], [], [], []
    for calls in calls_by_seed:
        correct = base + calls.astype(float) * delta
        rate = float(np.mean(calls))
        utilities.append(float(np.mean(correct - cost * calls)))
        accuracies.append(float(np.mean(correct)))
        lifts.append(float(np.mean(calls * (delta - cost))))
        rates.append(rate)
        counts.append(int(np.sum(calls)))
    def ci(values: list[float]) -> tuple[float, float]:
        vals = np.asarray(values, dtype=float)
        return float(np.quantile(vals, .025)), float(np.quantile(vals, .975))
    row.update({
        "random_seed_count": len(calls_by_seed),
        "random_accuracy_seed_ci_low": ci(accuracies)[0], "random_accuracy_seed_ci_high": ci(accuracies)[1],
        "random_utility_seed_ci_low": ci(utilities)[0], "random_utility_seed_ci_high": ci(utilities)[1],
        "random_utility_gain_seed_ci_low": ci(lifts)[0], "random_utility_gain_seed_ci_high": ci(lifts)[1],
        "random_call_count_seed_ci_low": ci(counts)[0], "random_call_count_seed_ci_high": ci(counts)[1],
    })


def add_paired_difference_vs_uncertainty(row: dict[str, Any], rows: list[dict[str, Any]],
                                         calls: np.ndarray, uncertainty_calls: np.ndarray,
                                         cost: float, boot_indices: np.ndarray) -> None:
    outcome = labels(rows)
    delta = outcome["delta"]
    difference = (np.asarray(calls, dtype=float) - np.asarray(uncertainty_calls, dtype=float)) * (delta - cost)
    ci = bootstrap_mean_ci(difference, boot_indices)
    row["utility_difference_vs_base_uncertainty"] = float(np.mean(difference))
    row["utility_difference_vs_uncertainty_ci_low"] = ci[0]
    row["utility_difference_vs_uncertainty_ci_high"] = ci[1]


def make_risk_coverage(rows_by_split: dict[str, list[dict[str, Any]]],
                       test_score_bundles: dict[str, dict[str, np.ndarray | None]]) -> list[dict[str, Any]]:
    test = rows_by_split["test"]
    outcome = labels(test)
    n = len(test)
    methods_to_rank = METHODS + ["oracle_selective"]
    predictions = {method: np.asarray(test_score_bundles[method]["route"], dtype=float) for method in METHODS}
    predictions["oracle_selective"] = outcome["delta"].astype(float)
    rows_out = []
    random_cache = {int(round(frac * n)): random_call_matrix(n, int(round(frac * n))) for frac in CALL_FRACTIONS}
    for method in methods_to_rank + ["random"]:
        for frac in CALL_FRACTIONS:
            k = int(round(frac * n))
            if method == "random":
                calls_matrix = random_cache[k]
                calls = np.mean(calls_matrix, axis=0)
                validation_rate = ""
                score_used = "uniform random; 20 fixed seeds"
            else:
                score = predictions[method]
                if method == "oracle_selective":
                    tie_key = np.asarray([hashlib.sha256(f"oracle:{r['case_hash']}".encode()).hexdigest() for r in test])
                else:
                    tie_key = np.asarray([hashlib.sha256(f"{method}:{r['case_hash']}".encode()).hexdigest() for r in test])
                order = np.lexsort((tie_key, -score))
                calls = np.zeros(n, dtype=float)
                calls[order[:k]] = 1.0
                calls_matrix = None
                validation_rate = ""
                score_used = "true relative_gain (hindsight)" if method == "oracle_selective" else method
            selected_helpful = float(np.sum(calls * outcome["helpful"]))
            selected_harmful = float(np.sum(calls * outcome["harmful"]))
            net = selected_helpful - selected_harmful
            random_expected_helpful = k * float(np.mean(outcome["helpful"]))
            random_expected_harmful = k * float(np.mean(outcome["harmful"]))
            random_expected_net = k * float(np.mean(outcome["delta"]))
            base_correct = float(np.mean(outcome["base"])) + net / n
            for cost in COSTS:
                utility = base_correct - cost * (k / n)
                row = {
                    "router": method, "score_used": score_used, "target_call_rate": frac,
                    "target_call_count": k, "actual_call_rate": float(np.mean(calls)), "cost": cost,
                    "accuracy": base_correct, "utility": utility, "helpful_selected_count": selected_helpful,
                    "helpful_total": int(np.sum(outcome["helpful"])),
                    "helpful_capture_rate": selected_helpful / max(1, int(np.sum(outcome["helpful"]))),
                    "harmful_selected_count": selected_harmful,
                    "harmful_total": int(np.sum(outcome["harmful"])),
                    "harmful_fraction_of_base_correct": selected_harmful / max(1, int(np.sum(outcome["base"]))),
                    "net_correctness_change": net,
                    "random_expected_helpful_at_same_call_count": random_expected_helpful,
                    "random_expected_harmful_at_same_call_count": random_expected_harmful,
                    "random_expected_net_change_at_same_call_count": random_expected_net,
                    "net_gain_vs_random_at_same_call_count": net - random_expected_net,
                    "helpful_capture_lift_vs_random": selected_helpful - random_expected_helpful,
                    "validation_call_rate": validation_rate,
                }
                if method == "random":
                    seed_net = np.sum(calls_matrix * outcome["delta"][None, :], axis=1)
                    seed_helpful = np.sum(calls_matrix * outcome["helpful"][None, :], axis=1)
                    seed_harmful = np.sum(calls_matrix * outcome["harmful"][None, :], axis=1)
                    def seed_interval(values: np.ndarray) -> tuple[float, float]:
                        return float(np.quantile(values, .025)), float(np.quantile(values, .975))
                    net_ci = seed_interval(seed_net - random_expected_net)
                    helpful_ci = seed_interval(seed_helpful)
                    harmful_ci = seed_interval(seed_harmful)
                    row["net_gain_vs_random_seed_ci_low"] = net_ci[0]
                    row["net_gain_vs_random_seed_ci_high"] = net_ci[1]
                    row["helpful_selected_seed_ci_low"] = helpful_ci[0]
                    row["helpful_selected_seed_ci_high"] = helpful_ci[1]
                    row["harmful_selected_seed_ci_low"] = harmful_ci[0]
                    row["harmful_selected_seed_ci_high"] = harmful_ci[1]
                else:
                    row["net_gain_vs_random_seed_ci_low"] = ""
                    row["net_gain_vs_random_seed_ci_high"] = ""
                    row["helpful_selected_seed_ci_low"] = ""
                    row["helpful_selected_seed_ci_high"] = ""
                    row["harmful_selected_seed_ci_low"] = ""
                    row["harmful_selected_seed_ci_high"] = ""
                rows_out.append(row)
    return rows_out


def create_plots(risk_rows: list[dict[str, Any]], base_accuracy: float) -> list[str]:
    selected_methods = ["base_uncertainty", "logistic_relative_gain", "ridge_relative_gain", "mlp_relative_gain", "oracle_selective", "random"]
    labels_for_plot = {
        "base_uncertainty": "Base uncertainty", "logistic_relative_gain": "Logistic relative gain",
        "ridge_relative_gain": "Ridge relative gain", "mlp_relative_gain": "MLP relative gain",
        "oracle_selective": "Oracle hindsight", "random": "Random (20 seeds)",
    }
    accuracy_rows = [r for r in risk_rows if r["cost"] == 0.0 and r["router"] in selected_methods]
    fig, ax = plt.subplots(figsize=(8.2, 5.3))
    for method in selected_methods:
        group = sorted((r for r in accuracy_rows if r["router"] == method), key=lambda r: r["actual_call_rate"])
        ax.plot([r["actual_call_rate"] * 100 for r in group], [r["accuracy"] for r in group], marker="o", label=labels_for_plot[method])
    ax.axhline(base_accuracy, color="black", linestyle="--", linewidth=1, label="Always Answer (test)")
    ax.set(xlabel="Expert call rate (%)", ylabel="End-to-end accuracy", title="Risk–coverage: accuracy vs Expert calls (n=400; 1% ≈ 4 cases)")
    ax.grid(alpha=.25); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(OUT / "risk_coverage_accuracy.png", dpi=160); plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True, sharey=False)
    for ax, cost in zip(axes.flat, COSTS):
        subset = [r for r in risk_rows if r["cost"] == cost]
        for method in selected_methods:
            group = sorted((r for r in subset if r["router"] == method), key=lambda r: r["actual_call_rate"])
            ax.plot([r["actual_call_rate"] * 100 for r in group], [r["utility"] for r in group], marker="o", label=labels_for_plot[method])
        ax.axhline(base_accuracy, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"Cost = {cost:.2f}"); ax.set_xlabel("Expert call rate (%)"); ax.set_ylabel("Utility")
        ax.grid(alpha=.25)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("Risk–coverage: utility = accuracy − cost × call rate")
    fig.tight_layout(); fig.savefig(OUT / "risk_coverage_utility_by_cost.png", dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for method in selected_methods:
        group = sorted((r for r in risk_rows if r["cost"] == 0.0 and r["router"] == method), key=lambda r: r["actual_call_rate"])
        axes[0].plot([r["actual_call_rate"] * 100 for r in group], [r["helpful_capture_rate"] * 100 for r in group], marker="o", label=labels_for_plot[method])
        axes[1].plot([r["actual_call_rate"] * 100 for r in group], [r["harmful_selected_count"] for r in group], marker="o", label=labels_for_plot[method])
    axes[0].plot([0, 100], [0, 100], linestyle=":", color="gray", label="Random expected capture")
    axes[0].set(xlabel="Expert call rate (%)", ylabel="Helpful cases captured (%)", title="Top-K helpful capture")
    axes[1].set(xlabel="Expert call rate (%)", ylabel="Harmful cases selected (count)", title="Harmful cases called")
    for ax in axes: ax.grid(alpha=.25)
    axes[0].legend(fontsize=7); fig.tight_layout()
    fig.savefig(OUT / "topk_helpful_harmful.png", dpi=160); plt.close(fig)
    return ["risk_coverage_accuracy.png", "risk_coverage_utility_by_cost.png", "topk_helpful_harmful.png"]


def build_report(summary: dict[str, dict[str, Any]], cost_rows: list[dict[str, Any]],
                 router_rows: list[dict[str, Any]], target_rows: list[dict[str, Any]],
                 integrity: dict[str, Any], feature_audit_rows: list[dict[str, Any]],
                 prompt_audit: dict[str, Any], selection: dict[str, Any],
                 run_manifest: dict[str, Any], model_lock: dict[str, Any],
                 parse_summary: dict[str, Any], plots: list[str]) -> tuple[str, str]:
    test_sum = summary["test"]
    by_cost = {(r["policy"], float(r["cost"])): r for r in cost_rows}
    learned_names = ["base_uncertainty", "logistic_base_error", "logistic_relative_gain", "ridge_relative_gain", "mlp_relative_gain"]
    main_rows = [by_cost[(name, 0.60)] for name in learned_names]
    zero_cost_utility = {name: by_cost[(name, 0.00)]["utility"] for name in learned_names}
    target_test = [r for r in target_rows if r["split"] == "test"]
    target_map = {(r["model_or_score"], r["target"]): r for r in target_test}
    router_map = {r["policy"]: r for r in router_rows}
    old_base_accuracy = float(test_sum["base_accuracy"])
    old_delta = float(test_sum["paired_accuracy_gain"])

    utility_rows = []
    for name in ["always_answer", "always_defer"] + learned_names + ["oracle_selective"]:
        values = [by_cost[(name, c)]["utility"] for c in COSTS]
        utility_rows.append("| " + METHOD_LABELS.get(name, name) + " | " + " | ".join(f"{v:.3f}" for v in values) + " |")
    for method in METHODS:
        values = [by_cost[(f"random_matched_to_{method}", c)]["utility"] for c in COSTS]
        utility_rows.append("| Random matched to " + METHOD_LABELS[method] + " | " + " | ".join(f"{v:.3f}" for v in values) + " |")
    utility_table = "\n".join([
        "| Router | 0.00 | 0.05 | 0.10 | 0.20 | 0.40 | 0.60 |",
        "|---|---:|---:|---:|---:|---:|---:|", *utility_rows,
    ])
    main_table_rows = []
    for row in [by_cost[(n, 0.60)] for n in ["always_answer", "always_defer"] + learned_names + ["oracle_selective"]]:
        main_table_rows.append(
            f"| {METHOD_LABELS.get(row['policy'], row['policy'])} | {row['accuracy']:.3f} | {row['utility']:.3f} "
            f"| {row['defer_rate']:.3f} | {row['expert_call_count']:.1f} | {row['utility_gain_vs_always_answer']:+.3f} "
            f"[{row['utility_gain_ci_low']:+.3f}, {row['utility_gain_ci_high']:+.3f}] "
            f"| {row.get('utility_difference_vs_base_uncertainty', 0.0):+.3f} "
            f"[{row.get('utility_difference_vs_uncertainty_ci_low', 0.0):+.3f}, {row.get('utility_difference_vs_uncertainty_ci_high', 0.0):+.3f}] "
            f"| {row['expert_correction_capture_rate']:.3f} | {row['base_correct_regression_rate']:.3f} |"
        )
    main_table = "\n".join([
        "| Router | Accuracy | Utility | DEFER rate | Expert calls | Utility lift vs Base (paired 95% CI) | Utility diff vs uncertainty (paired 95% CI) | Helpful capture | Base-correct regression |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|", *main_table_rows,
    ])
    predictive_names = ["base_uncertainty", "logistic_base_error", "logistic_relative_gain", "ridge_relative_gain", "mlp_relative_gain"]
    target_table_rows = []
    for name in predictive_names:
        base_auc = target_map.get((name, "base_error"), {}).get("auroc", "")
        help_auc = target_map.get((name, "helpful"), {}).get("auroc", "")
        harm_auc = target_map.get((name, "harmful"), {}).get("auroc", "")
        rel = next((r.get("relative_gain_spearman") for r in target_test if r["model_or_score"] == name and r["target"] == "relative_gain"), "")
        def auc_cell(model: str, target: str) -> str:
            row = target_map.get((model, target))
            if not row: return "—"
            return f"{float(row['auroc']):.3f} [{float(row['auc_ci_low']):.3f}, {float(row['auc_ci_high']):.3f}]"
        target_table_rows.append(f"| {METHOD_LABELS[name]} | {auc_cell(name, 'base_error')} | {auc_cell(name, 'helpful')} | {auc_cell(name, 'harmful')} | {format_number(rel)} |")
    target_table = "\n".join([
        "| Score/model | Base Error AUROC | Helpful AUROC | Harmful AUROC | Relative-gain Spearman |",
        "|---|---:|---:|---:|---:|", *target_table_rows,
    ])
    feature_rows_by_name = {}
    for row in feature_audit_rows:
        feature_rows_by_name.setdefault(row["feature"], {})[row["split"]] = row
    feature_table_lines = []
    for name in ["log_public_chars", "log_public_words", "option_count", "mean_option_chars",
                 "base_confidence", "base_entropy_normalized", "base_top1_top2_margin",
                 "base_uncertainty_missing", "base_valid", "log_base_prompt_tokens", "log_base_completion_tokens"]:
        fields = feature_rows_by_name.get(name, {})
        train_f = fields.get("train", {})
        val_f = fields.get("validation", {})
        test_f = fields.get("test", {})
        feature_table_lines.append(
            f"| {name} | {train_f.get('missing_rate', float('nan')):.3f} / {train_f.get('unique_value_count', 0)} "
            f"| {val_f.get('missing_rate', float('nan')):.3f} / {val_f.get('unique_value_count', 0)} "
            f"| {test_f.get('missing_rate', float('nan')):.3f} / {test_f.get('unique_value_count', 0)} "
            f"| {'yes' if train_f.get('used_in_final_model') else 'no'} |"
        )
    feature_table = "\n".join([
        "| Feature | Train missing / unique | Validation missing / unique | Test missing / unique | Used |",
        "|---|---:|---:|---:|---:|", *feature_table_lines,
    ])

    utility_gain_at_06 = [r for r in router_rows if r["policy"] in learned_names and float(r["cost"]) == .6]
    stable_low = [r for r in cost_rows if r["policy"] in learned_names and float(r["cost"]) in (.1, .2) and float(r["utility_gain_ci_low"]) > 0]
    helpful_candidates = [target_map.get((name, "helpful"), {}) for name in ("logistic_relative_gain", "ridge_relative_gain", "mlp_relative_gain")]
    harmful_candidates = [target_map.get((name, "harmful"), {}) for name in ("logistic_relative_gain", "ridge_relative_gain", "mlp_relative_gain")]
    reliable_relative = any(
        h and d and float(h.get("auc_ci_low", 0)) > .5 and float(d.get("auc_ci_low", 0)) > .5
        for h, d in zip(helpful_candidates, harmful_candidates)
    )
    uncertainty_06 = by_cost[("base_uncertainty", .60)]
    stable_06 = [r for r in utility_gain_at_06
                 if r["policy"] in {"logistic_relative_gain", "ridge_relative_gain", "mlp_relative_gain"}
                 and float(r["utility_gain_ci_low"]) > 0.0
                 and float(r["utility_difference_vs_uncertainty_ci_low"]) > 0.0
                 and float(r["utility"]) > float(uncertainty_06["utility"])]
    if stable_06 and reliable_relative:
        recommendation = "A. 建议进入受控的 PPO/ATPO RL 阶段，但先固定 Router 特征、阈值与成本并增加独立病例验证。"
        recommendation_code = "A"
    elif stable_low or reliable_relative:
        recommendation = "B. 暂不进入 RL；优先改进监督 Router、校准与决策前特征，并在新病例 holdout 上复验。"
        recommendation_code = "B"
    else:
        recommendation = "C. 当前 Expert 相对收益缺乏可靠可预测性；暂不进入 RL，优先重新设计 DEFER 机制或 Expert。"
        recommendation_code = "C"

    def predictive_value(split: str, method: str, target: str, field: str) -> Any:
        row = next((r for r in target_rows if r["split"] == split and r["model_or_score"] == method and r["target"] == target), {})
        return row.get(field, "")
    mlp_train_help = predictive_value("train", "mlp_relative_gain", "helpful", "auroc")
    mlp_train_harm = predictive_value("train", "mlp_relative_gain", "harmful", "auroc")
    mlp_val_help = predictive_value("validation", "mlp_relative_gain", "helpful", "auroc")
    mlp_val_harm = predictive_value("validation", "mlp_relative_gain", "harmful", "auroc")
    mlp_test_help = predictive_value("test", "mlp_relative_gain", "helpful", "auroc")
    mlp_test_harm = predictive_value("test", "mlp_relative_gain", "harmful", "auroc")
    mlp_train_spearman = predictive_value("train", "mlp_relative_gain", "relative_gain", "relative_gain_spearman")
    mlp_val_spearman = predictive_value("validation", "mlp_relative_gain", "relative_gain", "relative_gain_spearman")
    mlp_test_spearman = predictive_value("test", "mlp_relative_gain", "relative_gain", "relative_gain_spearman")
    unc_010 = by_cost[("base_uncertainty", .10)]
    unc_020 = by_cost[("base_uncertainty", .20)]
    lbase_010 = by_cost[("logistic_base_error", .10)]
    lbase_020 = by_cost[("logistic_base_error", .20)]
    uncertainty_help = target_map.get(("base_uncertainty", "helpful"), {})
    uncertainty_error = target_map.get(("base_uncertainty", "base_error"), {})
    logistic_help = target_map.get(("logistic_relative_gain", "helpful"), {})
    logistic_harm = target_map.get(("logistic_relative_gain", "harmful"), {})
    old_commit = str(run_manifest.get("code_commit", "unknown"))
    report = f"""# Expert 相对收益可预测性与 Router 修复报告

本报告对应 Attempt04 冻结的 1,700 例配对队列，分析输出独立保存于本目录。没有重跑 Base/Expert 推理，没有使用 GPU，也没有启动 PPO、GRPO 或 ATPO。

## 1. 实验设置和数据审计

- 模型：Base `{run_manifest.get('models', {}).get('base', {}).get('repo_id', 'Qwen3-8B')}`；Expert `{run_manifest.get('models', {}).get('expert', {}).get('repo_id', 'Qwen3-32B-AWQ')}`。
- 固定 split：train {summary['train']['n']}、validation {summary['validation']['n']}、test {summary['test']['n']}。Test 是从 ATPO train split 抽取的 400 例 holdout，不是官方独立 test split。
- 病例级检查：每个 split 内 `case_id`、`case_hash`、`prompt_hash` 唯一；训练、验证、测试互不重叠； paired rows 与两路 trajectory 的身份、预测、correct 标签逐条一致。完整计数见 `integrity_audit.json`。
- 同一 Gold：原推理脚本在 `build_case` 中为每例构造一次 Gold，并在 Base 与 Expert 两次 `predicted == case['gold']` 比较中复用；配对文件也逐条匹配两路轨迹。Gold 没有进入特征或新产物。
- 重复采样：三 split 共 1,700 个唯一 case/prompt hash，没有发现重复病例被当作独立记录。
- Test 配对结果：Base {test_sum['base_accuracy']:.3f} ({int(round(test_sum['base_accuracy'] * test_sum['n']))}/{test_sum['n']}); Expert {test_sum['expert_accuracy']:.4f} ({int(round(test_sum['expert_accuracy'] * test_sum['n']))}/{test_sum['n']}); paired gain {test_sum['paired_accuracy_gain']:+.4f}（+{100 * test_sum['paired_accuracy_gain']:.2f} pp），bootstrap 95% CI [{test_sum['paired_gain_ci_low']:+.3f}, {test_sum['paired_gain_ci_high']:+.3f}]。helpful {int(test_sum['base_wrong_expert_correct'])} 例，harmful {int(test_sum['base_correct_expert_wrong'])} 例。
- Expert parser failures：train/validation/test 分别 {parse_summary['train']['invalid']}/{parse_summary['validation']['invalid']}/{parse_summary['test']['invalid']}，共 {parse_summary['total_invalid']}；这些记录全部按错误计分。Base parser failures 分别 {parse_summary['train']['base_invalid']}/{parse_summary['validation']['base_invalid']}/{parse_summary['test']['base_invalid']}。
- 正确性、Utility 与 AUROC 的定义见配置。Bootstrap 使用 test 病例配对重采样 10,000 次；AUC 区间使用 2,000 次。成本按 `accuracy - cost × DEFER_rate` 一次扣除。
- 泄漏状态：训练/预处理只使用 train；模型和每个成本阈值仅由 train/validation 决定；policy lock 在读取 test 前写出。旧报告已对同一 test holdout 做过一次冻结后评估，因此本次是重复使用的 holdout 复核，不能宣称为独立新泛化验证。没有发现旧报告按 test 调阈值的记录，但重复查看结果带来研究者自由度风险。
- 执行记录：Attempt02 在 policy lock 后读取并转换 test 特征，随后因 train/validation bootstrap 索引长度 bug 在计算 split summary 前中止；Attempt03 使用相同确定性 train/validation fit 与阈值完成 test 指标，报告组装失败后由 `--finalize-existing` 读取已保存结果组装本文。修复期间没有用 test 结果改特征、模型或阈值；完整日志保留在 `logs/`。
- 提交版本：原始 run manifest 记录 `{old_commit}`；本地仓库 HEAD 与用户给出的 `2aedfa8` 不同（本次审计环境中 `2aedfa8` 不存在，HEAD 为 `2aebfa8`）。原始数据和报告均未覆盖。

## 2. 特征审计与修复

旧 `public_features()` 对选项行使用行首正则，未匹配时把 `option_count`、`mean_option_chars` 写成数值 0。源 prompt 重建审计确认，所有三 split 的旧与扩展选项行模式都没有匹配；这不能证明临床题目本身没有选项，所以本轮将两列标为 unavailable，而不是“真实 0”。模型特征纳入/排除只按 train 定：这两列在 train 全部不可用，故从各模型统一特征集排除；validation/test 的检查仅作为字段审计。Prompt 字符/词数、问号、数字和词典计数在 1,700 例上均与冻结 prompt 重算完全一致。

Base uncertainty 定义为对 A/B/C/D 归一化概率的 Shannon entropy 除以 `ln(4)`；confidence 是最大选项概率；margin 是 top-1 减 top-2 概率。有效行的概率和、重算熵和 margin 均通过检查：{integrity['uncertainty_audit']['valid_probability_rows']} 条有效，熵不匹配 {integrity['uncertainty_audit']['entropy_formula_mismatches']}，margin 不匹配 {integrity['uncertainty_audit']['margin_formula_mismatches']}。缺失通过 mask 保留，不填成 0；插补与标准化仅在 train 拟合。

Base prompt token 数、Base completion token 数以及回答 A-D 的置信度/概率、解析有效标志均在可选 Expert 调用前可见。没有使用 Gold、Expert 文本/正确率、Oracle 或事后结果。各 split 的缺失率、唯一值数、均值、标准差和可用性见 [`feature_audit.csv`](feature_audit.csv)。

{feature_table}

## 3. 可预测性比较

标签严格区分 `base_error`、`expert_helpful`、`expert_harmful` 与三值 `relative_gain = expert_correct - base_correct`。下面是 test AUROC（95% bootstrap CI）和 relative gain Spearman；Base Error AUROC 不再替代 Helpful/Relative Gain 指标。

{target_table}

完整 train/validation/test 目标指标见 [`predictive_targets.csv`](predictive_targets.csv)。用于决策的 Ridge 直接回归 relative gain；Logistic Relative Gain 是独立拟合 helpful/harmful 两个 logistic 概率后取 `P(helpful)-P(harmful)`；MLP 是三类 `{-1,0,+1}` softmax。

## 4. 成本敏感性

阈值只在 validation 上按相同 Utility 规则逐成本选择；test 只应用冻结阈值。表中为 test Utility，含 Always Answer/DEFER、匹配调用预算的 Random、学习路由、不确定性路由以及仅作理论上界的 Oracle。

{utility_table}

原主成本 0.60 的详细指标：

{main_table}

`cost_sensitivity.csv` 含每个 Router × 成本的 Accuracy、Utility、DEFER rate、平均成本、调用数、相对 Always Answer 的配对 95% CI、与 uncertainty 的配对差值、Expert correction capture 与 Base-correct regression。Random 以 20 个固定 seed 匹配每个 Router 的 validation 调用率；Utility/Accuracy 是 20 seed 均值，其 paired 95% CI 按病例重采样，另报告 20 个 seed 单次结果的 2.5–97.5 百分位区间。成本 0 的 Oracle 按 `relative_gain > 0` 调用；其他成本按净收益条件 `relative_gain > cost`，没有把所有 helpful 都无条件调用。

其中 correction capture = 被调用且 Expert 修复 Base 错误的病例 / 全部 helpful 病例；Base-correct regression = 被调用且 Expert 把 Base 正确变错的病例 / 全部 Base 正确病例。

## 5. 风险—覆盖率与收益排序

每个有预测分数的 Router 按 test 预测收益降序评估 `[1%, 2%, 5%, 10%, 20%, 30%, 50%, 100%]` 调用率。`risk_coverage.csv` 报告 helpful 捕获、harmful 调用、净正确变化以及相同调用数下相对随机期望的差异；这些曲线是冻结模型的 test 诊断，不用于改阈值。

![Risk–coverage accuracy](risk_coverage_accuracy.png)

![Risk–coverage utility by cost](risk_coverage_utility_by_cost.png)

![Helpful capture and harmful selections](topk_helpful_harmful.png)

Test 只有 400 例，1% 调用约 4 例，2% 约 8 例；这些小覆盖点的计数和曲线不稳定，不能据此推断部署表现。

## 6. 失败归因

详见 [`failure_attribution.md`](failure_attribution.md)。核心判断基于新输出：

1. 旧 Logistic 的特征问题部分得到证实：两项选项特征是解析 fallback 零值且被旧模型纳入；普通 prompt 词面特征则复现正确。移除不可用选项列后，新 Logistic Relative Gain test Helpful AUROC 为 {float(logistic_help.get('auroc')):.3f}、Harmful 为 {float(logistic_harm.get('auroc')):.3f}，net Spearman 为 {float(predictive_value('test', 'logistic_relative_gain', 'relative_gain', 'relative_gain_spearman')):+.3f}；因此特征修复没有单独解决净收益排序。
2. MLP Helpful/Harmful AUROC 从 train {float(mlp_train_help):.3f}/{float(mlp_train_harm):.3f}、validation {float(mlp_val_help):.3f}/{float(mlp_val_harm):.3f} 到 test {float(mlp_test_help):.3f}/{float(mlp_test_harm):.3f}；relative-gain Spearman 分别 {float(mlp_train_spearman):+.3f}/{float(mlp_val_spearman):+.3f}/{float(mlp_test_spearman):+.3f}。这支持对 harmful 区分有泛化衰减/过拟合风险，但不足以单凭 400 例区分样本不足与抽样噪声。
3. Base uncertainty 对 Base Error 与 Expert Helpful/Harmful 的能力分别报告。旧结果中的 AUROC 0.710 对应 helpful 标签，而不是 Base Error；旧 0.573/约 0.504 也不能被称作净相对收益 AUROC，旧分析把 `base_error_corrected` 和 `positive_relative_advantage` 映射成同一个 helpful 标签。
4. 若成本 0.60 下 validation 阈值为 0% DEFER，这是验证集估计的增量正确率不足以跨过预设成本的决策；Oracle 的 hindsight 结果不能证明可预测策略可达到。
5. 可部署收益以配对 Utility lift CI 与相对收益排序能力判断，不按 Accuracy 单独判定。

## 7. 局限性

- 同一 holdout 已在旧报告中查看，且只有 400 例；新结果不是独立复验。尤其低调用率处每个病例会明显改变估计。
- 这些是固定 Base/Expert、固定 parser 和一次性调用条件下的离线配对结果，不是临床效果或真实用户效用证据。
- Expert 格式失败被端到端计为错误，与实际严格 parser 的部署结果一致；若以后改 parser，必须另行预注册复验，不能事后修补本次标签。
- 输入特征主要是 prompt 词面、Base 概率与 Base 输出长度；没有检验新患者信息、检索、专家解释或跨模型族外部泛化。
- 本次阈值与模型选择复用同一 validation；小样本下阈值估计仍有选择噪声。

## 8. RL 决策

**{recommendation}**

结论编码：**{recommendation_code}**。在独立病例上稳定复现监督路由的相对收益和目标成本下的 Utility 前，不应把当前结果当作启动 RL 的充分证据。
"""

    high_cost = [by_cost[(name, .60)] for name in learned_names]
    best_high = max(high_cost, key=lambda r: r["utility_gain_vs_always_answer"])
    uncertainty_help = target_map.get(("base_uncertainty", "helpful"), {})
    uncertainty_error = target_map.get(("base_uncertainty", "base_error"), {})
    logistic_help = target_map.get(("logistic_relative_gain", "helpful"), {})
    logistic_harm = target_map.get(("logistic_relative_gain", "harmful"), {})
    fee_answer = by_cost[("always_answer", .60)]["utility"]
    learned_high_calls = {r["policy"]: r["defer_rate"] for r in high_cost}
    attribution = f"""# 失败归因

## 1. 旧 Logistic AUROC 是否来自特征问题？

部分是。`option_count` 和 `mean_option_chars` 在旧提取器中“未匹配就填 0”，train/validation/test 的存储值都只有一个唯一值 0。源 prompt 审计中，原选项行正则和扩展标点/括号模式对各 split 都未识别到选项行，因此不能把 0 解释为题目确实没有选项；本轮将其标为解析不可用并排除。其他 prompt 文本特征在 1,700 例上重算完全匹配。修复前后相对收益性能应看新 test helpful/harmful AUROC 和 Spearman；旧 .573 主要是 Helpful 目标结果，不是净收益 AUROC。特征错误是混杂因素，但不足以单独归因全部失败。

## 2. MLP 接近随机是否是样本不足或过拟合？

MLP Helpful/Harmful AUROC 从 train {float(mlp_train_help):.3f}/{float(mlp_train_harm):.3f}，到 validation {float(mlp_val_help):.3f}/{float(mlp_val_harm):.3f}，再到 test {float(mlp_test_help):.3f}/{float(mlp_test_harm):.3f}；relative-gain Spearman 为 {float(mlp_train_spearman):+.3f}/{float(mlp_val_spearman):+.3f}/{float(mlp_test_spearman):+.3f}。Helpful 排序在 test 高于随机，但 Harmful 从 train 的 {float(mlp_train_harm):.3f} 降至 test {float(mlp_test_harm):.3f}；net gain 排序在 validation/test 都接近 0。证据支持 harmful 类泛化不稳/过拟合风险，400 例不足以把它与抽样噪声完全区分。MLP 只在 1,000 条 train 上拟合，hidden=16；weight decay/class weighting 由 300 条 validation 三分类 log loss 选择。参数：`{json.dumps(model_lock['model_selection']['mlp_relative_gain']['selected'], ensure_ascii=False)}`。

## 3. Base uncertainty 是否同时预测 Expert 相对收益？

Base Error AUROC = {format_number(uncertainty_error.get('auroc'))} [{format_number(uncertainty_error.get('auc_ci_low'))}, {format_number(uncertainty_error.get('auc_ci_high'))}]；Helpful AUROC = {format_number(uncertainty_help.get('auroc'))} [{format_number(uncertainty_help.get('auc_ci_low'))}, {format_number(uncertainty_help.get('auc_ci_high'))}]。这两个目标不等价。净相对收益更应看 relative gain 的 Spearman，以及分别识别 helpful 和 harmful 的分数。

## 4. 成本 0.60 下 0% DEFER 是否合理？

各预测 Router 在 test 的调用率：`{json.dumps(learned_high_calls, ensure_ascii=False)}`；Always Answer utility = {fee_answer:.3f}。是否 defer 的阈值只从 validation 选出。对于成本 0.60，一次调用只有在预期 `relative_gain > 0.60` 时才提升期望 Utility；若验证集没有模型能把这类病例排到成本线上方，0% 是合理的保守决策。Oracle 在 test 可看见真实标签，仅是事后上界，不能在线实现。

## 5. 是否存在可预测 Expert 优势？

test 的 Logistic Relative Gain Helpful AUROC = {format_number(logistic_help.get('auroc'))} [{format_number(logistic_help.get('auc_ci_low'))}, {format_number(logistic_help.get('auc_ci_high'))}]；Harmful AUROC = {format_number(logistic_harm.get('auroc'))} [{format_number(logistic_harm.get('auc_ci_low'))}, {format_number(logistic_harm.get('auc_ci_high'))}]。其他模型见主报告目标表。预测能力需要同时考虑 helpful 捕获和 harmful 回避；只预测 Base 会错不能说明 Expert 能修复，也不能排除 Expert 会退化正确答案。

## 6. Learned Router 是否优于简单 uncertainty？

没有稳定优于。Cost .10 时 uncertainty 的 Utility lift 为 {unc_010['utility_gain_vs_always_answer']:+.4f} [{unc_010['utility_gain_ci_low']:+.4f}, {unc_010['utility_gain_ci_high']:+.4f}]；Logistic Base Error 为 {lbase_010['utility_gain_vs_always_answer']:+.4f}，但相对 uncertainty 的 paired 差值 {lbase_010['utility_difference_vs_base_uncertainty']:+.4f} [{lbase_010['utility_difference_vs_uncertainty_ci_low']:+.4f}, {lbase_010['utility_difference_vs_uncertainty_ci_high']:+.4f}] 跨过 0。Cost .20 时 uncertainty lift {unc_020['utility_gain_vs_always_answer']:+.4f} [{unc_020['utility_gain_ci_low']:+.4f}, {unc_020['utility_gain_ci_high']:+.4f}]，Logistic Base Error 则为 {lbase_020['utility_gain_vs_always_answer']:+.4f}。Cost .60 时没有 Learned Relative Gain Router 获得稳定的正收益。调用率匹配随机基线及各成本 paired 区间见 `cost_sensitivity.csv`。结论：{recommendation}

## 审计边界

没有重新推理，没有改变 frozen split，没有使用 test 调参或选阈值；但相同 400 例 test 已在先前报告中看过，所以这次不是独立 holdout。Gold 仅用于离线标签，所有 Router 特征均来自可见 prompt 或 Base 完成后的状态。
"""
    return report, attribution


def main() -> None:
    if "--finalize-existing" in sys.argv[1:]:
        finalize_existing_outputs()
        return
    OUT.mkdir(parents=True, exist_ok=True)
    lock_path = OUT / "policy_lock.json"
    if lock_path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing policy lock: {lock_path}")
    for path in (RUN / "paired_rollouts/run_manifest.json", SELECTION):
        if not path.is_file():
            raise FileNotFoundError(path)
    run_manifest = read_json(RUN / "paired_rollouts/run_manifest.json")
    selection_doc = read_json(SELECTION)
    train = read_jsonl(ROLLOUTS / "paired_train.jsonl")
    validation = read_jsonl(ROLLOUTS / "paired_validation.jsonl")
    train_integrity = validate_pair_rows("train", train)
    validation_integrity = validate_pair_rows("validation", validation)
    for key in ("case_id", "case_hash", "prompt_hash"):
        train_values = {r[key] for r in train}
        val_values = {r[key] for r in validation}
        if train_values & val_values:
            raise RuntimeError(f"Train/validation overlap found in {key}")
    selection_hashes = {
        split: {row["case_hash"] for row in selection_doc["rows"][split]}
        for split in ("train", "validation", "test")
    }
    if selection_hashes["train"] != {r["case_hash"] for r in train} or selection_hashes["validation"] != {r["case_hash"] for r in validation}:
        raise RuntimeError("Frozen paired train/validation cases differ from case_selection.json")
    if selection_hashes["train"] & selection_hashes["validation"] or selection_hashes["train"] & selection_hashes["test"] or selection_hashes["validation"] & selection_hashes["test"]:
        raise RuntimeError("Frozen selection has cross-split case_hash overlap")

    train_uncertainty_audit = verify_uncertainty(train)
    validation_uncertainty_audit = verify_uncertainty(validation)
    if train_uncertainty_audit["probability_sum_failures"] or train_uncertainty_audit["entropy_formula_mismatches"] or train_uncertainty_audit["margin_formula_mismatches"]:
        raise RuntimeError(f"Base uncertainty formulas failed on train: {train_uncertainty_audit}")
    if validation_uncertainty_audit["probability_sum_failures"] or validation_uncertainty_audit["entropy_formula_mismatches"] or validation_uncertainty_audit["margin_formula_mismatches"]:
        raise RuntimeError(f"Base uncertainty formulas failed on validation: {validation_uncertainty_audit}")

    x_train, preprocessor = fit_preprocessor(train)
    x_validation = transform_rows(validation, preprocessor)
    train_y, val_y = labels(train), labels(validation)
    if set(np.unique(train_y["delta"]).tolist()) != {-1.0, 0.0, 1.0}:
        raise RuntimeError("Training split must contain all three relative-gain labels")

    model_selection: dict[str, Any] = {}
    models: dict[str, Any] = {}
    for model_name, target_name in (("logistic_base_error", "base_error"),
                                    ("logistic_helpful", "helpful"),
                                    ("logistic_harmful", "harmful")):
        best, candidates = fit_selected_logistic(x_train, train_y[target_name].astype(int),
                                                 x_validation, val_y[target_name].astype(int))
        models[model_name] = {k: v for k, v in best.items() if k not in {"validation_logloss", "validation_probabilities"}}
        model_selection[model_name] = {"selected": {k: v for k, v in best.items() if k not in {"validation_probabilities"}},
                                       "candidates": candidates}
    ridge, ridge_candidates = fit_ridge(x_train, train_y["delta"], x_validation, val_y["delta"])
    models["ridge_relative_gain"] = {k: v for k, v in ridge.items() if k != "validation_prediction"}
    model_selection["ridge_relative_gain"] = {
        "selected": {k: v for k, v in ridge.items() if k != "validation_prediction"}, "candidates": ridge_candidates,
    }
    mlp, mlp_candidates = fit_selected_mlp(x_train, train_y["delta"], x_validation, val_y["delta"])
    models["mlp_relative_gain"] = mlp
    model_selection["mlp_relative_gain"] = {"selected": {k: v for k, v in mlp.items() if k not in {"w1", "b1", "w2", "b2"}},
                                              "candidates": mlp_candidates}

    scores_train = score_bundle(train, x_train, models)
    scores_validation = score_bundle(validation, x_validation, models)
    thresholds: dict[str, dict[str, Any]] = {name: {} for name in METHODS}
    for cost in COSTS:
        key = f"{cost:.2f}"
        for method in METHODS:
            thresholds[method][key] = select_threshold(np.asarray(scores_validation[method]["route"], dtype=float), validation, cost)

    model_lock = {
        "status": "locked_before_test_open",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED, "cost_grid": COSTS, "primary_cost": 0.60,
        "train_count": len(train), "validation_count": len(validation),
        "train_sha256": sha_file(ROLLOUTS / "paired_train.jsonl"),
        "validation_sha256": sha_file(ROLLOUTS / "paired_validation.jsonl"),
        "test_opened_by_this_fitter": False,
        "feature_preprocessing": preprocessor,
        "models": models, "model_selection": model_selection,
        "thresholds_selected_on_validation": thresholds,
        "policy_rule": "call Expert when route score >= validation-selected threshold; cost-specific utility maximized on validation; ties choose fewer calls",
        "cost_definition": "accuracy - cost * Expert_call_rate; Base is always answered first",
        "oracle_used_for_fitting": False,
        "test_results_or_labels_in_lock": False,
    }
    write_json(lock_path, model_lock)

    # First read of the frozen test outcomes occurs only after policy_lock.json is written.
    test = read_jsonl(ROLLOUTS / "paired_test.jsonl")
    test_integrity = validate_pair_rows("test", test)
    for key in ("case_id", "case_hash", "prompt_hash"):
        known = {r[key] for r in train + validation}
        if known & {r[key] for r in test}:
            raise RuntimeError(f"Train/validation/test overlap found in {key}")
    if selection_hashes["test"] != {r["case_hash"] for r in test}:
        raise RuntimeError("Frozen test cases differ from case_selection.json")

    # Check the two cached trajectories join one-to-one with every paired case.
    base_trace_rows = read_jsonl(ROLLOUTS / "base_answer_trajectories.jsonl")
    expert_trace_rows = read_jsonl(ROLLOUTS / "qwen32_defer_trajectories.jsonl")
    base_trace = {r["case_hash"]: r for r in base_trace_rows}
    expert_trace = {r["case_hash"]: r for r in expert_trace_rows}
    if len(base_trace) != len(base_trace_rows) or len(expert_trace) != len(expert_trace_rows):
        raise RuntimeError("Duplicate case_hash in a model trajectory file")
    rows_by_split = {"train": train, "validation": validation, "test": test}
    trajectory_audit = {}
    for split, rows in rows_by_split.items():
        trajectory_audit[split] = validate_trajectory_alignment(split, rows, base_trace, expert_trace)
    if len(base_trace_rows) != 1700 or len(expert_trace_rows) != 1700:
        raise RuntimeError(f"Expected 1700 raw trajectories per model; got {len(base_trace_rows)}/{len(expert_trace_rows)}")
    if any(not bool(r.get("expert_valid")) and bool(r.get("expert_correct")) for r in train + validation + test):
        raise RuntimeError("An Expert parse failure was counted as correct")

    all_uncertainty = {split: verify_uncertainty(rows) for split, rows in rows_by_split.items()}
    for split, audit in all_uncertainty.items():
        if audit["probability_sum_failures"] or audit["entropy_formula_mismatches"] or audit["margin_formula_mismatches"]:
            raise RuntimeError(f"Base uncertainty formula mismatch in {split}: {audit}")
    x_test = transform_rows(test, preprocessor)
    scores_test = score_bundle(test, x_test, models)
    scores_by_split = {"train": scores_train, "validation": scores_validation, "test": scores_test}

    n_test = len(test)
    boot_indices = np.random.default_rng(BOOTSTRAP_SEED).integers(0, n_test, size=(BOOTSTRAP_N, n_test))
    paired_summary = []
    for split_index, (split, rows) in enumerate(rows_by_split.items()):
        split_boot = (boot_indices if split == "test" else
                      np.random.default_rng(BOOTSTRAP_SEED + 20 + split_index).integers(0, len(rows), size=(BOOTSTRAP_N, len(rows))))
        paired_summary.append(split_outcome_summary(split, rows, split_boot))
    summary = {row["split"]: row for row in paired_summary}
    risk_rows = []
    all_cost_rows = []
    test_method_predictive: dict[str, dict[str, float]] = {}
    target_rows = build_predictive_target_rows(rows_by_split, scores_by_split)
    for method in METHODS:
        test_method_predictive[method] = {
            "base_error_auroc": auc(np.asarray(scores_test[method]["base_error"], dtype=float), labels(test)["base_error"])
            if scores_test[method]["base_error"] is not None else float("nan"),
            "helpful_auroc": auc(np.asarray(scores_test[method]["helpful"], dtype=float), labels(test)["helpful"]),
            "harmful_auroc": auc(np.asarray(scores_test[method]["harmful"], dtype=float), labels(test)["harmful"]),
            "relative_gain_spearman": spearman(np.asarray(scores_test[method]["relative_gain"], dtype=float), labels(test)["delta"]),
        }

    validation_rates: dict[tuple[str, float], float] = {}
    test_calls_by_method_cost: dict[tuple[str, float], np.ndarray] = {}
    for cost in COSTS:
        key = f"{cost:.2f}"
        call_map: dict[str, np.ndarray] = {
            "always_answer": np.zeros(n_test, dtype=bool),
            "always_defer": np.ones(n_test, dtype=bool),
            "oracle_selective": labels(test)["delta"] > cost,
        }
        for method in METHODS:
            threshold_info = thresholds[method][key]
            score = np.asarray(scores_test[method]["route"], dtype=float)
            call_map[method] = score >= float(threshold_info["threshold"])
            validation_rates[(method, cost)] = float(threshold_info["validation_call_rate"])
        for method, calls in call_map.items():
            test_calls_by_method_cost[(method, cost)] = calls
            v_rate = thresholds[method][key]["validation_call_rate"] if method in METHODS else (0.0 if method == "always_answer" else 1.0 if method == "always_defer" else None)
            row = policy_metrics(test, calls, cost, boot_indices, method, v_rate)
            add_paired_difference_vs_uncertainty(row, test, calls, call_map["base_uncertainty"], cost, boot_indices)
            row["validation_threshold"] = thresholds[method][key]["threshold"] if method in METHODS else ""
            row["random_seed_count"] = 0
            for field in ("random_accuracy_seed_ci_low", "random_accuracy_seed_ci_high", "random_utility_seed_ci_low",
                          "random_utility_seed_ci_high", "random_utility_gain_seed_ci_low", "random_utility_gain_seed_ci_high",
                          "random_call_count_seed_ci_low", "random_call_count_seed_ci_high"):
                row[field] = ""
            all_cost_rows.append(row)
        for matched_method in METHODS:
            v_rate = validation_rates[(matched_method, cost)]
            matched_calls = int(round(v_rate * n_test))
            calls_matrix = random_call_matrix(n_test, matched_calls)
            avg_calls = np.mean(calls_matrix, axis=0)
            random_name = f"random_matched_to_{matched_method}"
            random_row = policy_metrics(test, avg_calls, cost, boot_indices, random_name, v_rate)
            add_paired_difference_vs_uncertainty(random_row, test, avg_calls, call_map["base_uncertainty"], cost, boot_indices)
            random_row["validation_threshold"] = thresholds[matched_method][key]["threshold"]
            add_random_seed_intervals(random_row, test, calls_matrix, cost)
            all_cost_rows.append(random_row)

    risk_rows = make_risk_coverage(rows_by_split, scores_test)
    feature_audit_rows = []
    prompt_audit = audit_source_prompt_features(rows_by_split, selection_doc)
    feature_audit_rows = build_feature_audit(rows_by_split, preprocessor, prompt_audit)
    write_csv(OUT / "feature_audit.csv", feature_audit_rows)
    write_paired_outcomes(rows_by_split, OUT / "paired_outcomes.csv")
    write_csv(OUT / "split_summary.csv", paired_summary)
    write_csv(OUT / "predictive_targets.csv", target_rows)
    write_csv(OUT / "cost_sensitivity.csv", all_cost_rows)
    write_csv(OUT / "risk_coverage.csv", risk_rows)

    # One row per Router at the original primary cost (0.60), with predictive diagnostics.
    primary = [r for r in all_cost_rows if math.isclose(float(r["cost"]), .60)]
    router_rows = []
    for row in primary:
        method = row["policy"]
        out = dict(row)
        matched = method.startswith("random_matched_to_")
        source_method = method.removeprefix("random_matched_to_") if matched else method
        p = test_method_predictive.get(source_method)
        if p is None or matched or method in {"always_answer", "always_defer", "oracle_selective"}:
            out.update({"base_error_auroc": "", "helpful_auroc": "", "harmful_auroc": "", "relative_gain_spearman": ""})
        else:
            out.update({k: v if math.isfinite(v) else "" for k, v in p.items()})
        out["method_label"] = f"Random matched to {METHOD_LABELS.get(source_method, source_method)}" if matched else METHOD_LABELS.get(source_method, method)
        out["random_match_target"] = source_method if matched else ""
        router_rows.append(out)
    write_csv(OUT / "router_metrics.csv", router_rows)

    parser_summary: dict[str, Any] = {}
    for split, rows in rows_by_split.items():
        invalid_expert = [r for r in rows if not bool(r.get("expert_valid"))]
        invalid_base = [r for r in rows if not bool(r.get("base_valid"))]
        finish_counts: dict[str, int] = {}
        for row in invalid_expert:
            finish = str(expert_trace[row["case_hash"]].get("finish_reason") or "missing")
            finish_counts[finish] = finish_counts.get(finish, 0) + 1
        parser_summary[split] = {"n": len(rows), "invalid": len(invalid_expert),
                                 "invalid_counted_wrong": sum(not bool(r.get("expert_correct")) for r in invalid_expert),
                                 "base_invalid": len(invalid_base), "expert_invalid_finish_reasons": finish_counts}
    parser_summary["total_invalid"] = sum(parser_summary[s]["invalid"] for s in rows_by_split)

    integrity = {
        "split_counts": EXPECTED,
        "train_unique": train_integrity, "validation_unique": validation_integrity,
        "test_unique": test_integrity,
        "all_case_hashes_unique": True, "all_prompt_hashes_unique": True,
        "train_validation_test_overlap": False,
        "paired_rows": sum(len(r) for r in rows_by_split.values()),
        "base_trajectory_rows": len(base_trace_rows), "expert_trajectory_rows": len(expert_trace_rows),
        "trajectory_pairing": trajectory_audit,
        "uncertainty_audit": {key: sum(int(audit.get(key, 0)) for audit in all_uncertainty.values())
                              for key in ("valid_probability_rows", "uncertainty_invalid_rows", "probability_sum_failures",
                                          "entropy_formula_mismatches", "margin_formula_mismatches")},
        "uncertainty_audit_by_split": {split: audit for split, audit in all_uncertainty.items()},
        "policy_lock_written_before_test_read": True,
        "test_opened_by_this_script_after_lock": True,
        "base_and_expert_use_same_gold": "verified in frozen producer source: one Gold per built case reused for both correct comparisons",
        "source_prompt_audit": prompt_audit,
        "prior_test_reuse": "same frozen 400-case holdout was evaluated in the prior report; this is not a new independent holdout",
    }
    write_json(OUT / "integrity_audit.json", integrity)

    plots = create_plots(risk_rows, summary["test"]["base_accuracy"])
    config = {
        "experiment": "EA-ATPO Expert relative-gain predictability audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_run": str(RUN), "input_paired_sha256": {
            split: sha_file(ROLLOUTS / f"paired_{split}.jsonl") for split in EXPECTED
        },
        "source_prompt_asset": {"path": prompt_audit["source_path"], "sha256": prompt_audit["source_sha256"]},
        "split_counts": EXPECTED, "test_is_official_test": False,
        "cost_grid": COSTS, "utility": "mean(final correctness - cost * Expert call indicator)",
        "base_answer_is_always_generated": True,
        "labels": {"base_error": "1(base_correct == 0)", "helpful": "1(expert_correct - base_correct == +1)",
                   "harmful": "1(expert_correct - base_correct == -1)", "relative_gain": "expert_correct - base_correct"},
        "decision_time_features": "prompt-only fields and Base completion state before optional Expert call; no Expert outputs or correctness labels",
        "preprocessor_fit_split": "train only", "hyperparameter_and_threshold_selection_split": "validation only",
        "test_policy_lock_before_read": True, "test_used_for_tuning_or_feature_selection": False,
        "logistic_candidates": {"l2": LOGISTIC_L2, "class_weight": ["none", "balanced"], "selection": "validation unweighted log loss"},
        "ridge_candidates": {"alpha": RIDGE_ALPHAS, "selection": "validation MSE"},
        "mlp_candidates": {"hidden_units": 16, "weight_decay": MLP_DECAYS, "class_weight": ["none", "balanced"],
                           "max_epochs": 500, "early_stopping_patience": 30, "selection": "validation unweighted three-class log loss"},
        "random_baseline": {"seeds": RANDOM_SEEDS, "n_seeds": len(RANDOM_SEEDS), "matching": "validation-selected call rate per method and cost; fixed call count in each test randomization", "seed_interval": "2.5-97.5 percentile interval across 20 seed-level outcomes; paired-case bootstrap CI for the seed-average policy outcome also reported"},
        "bootstrap": {"paired_case_replicates": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED,
                      "auc_replicates": AUC_BOOTSTRAP_N, "ci": "percentile 95%"},
        "option_features": "unavailable due zero parser matches; legacy zeros excluded",
        "artifacts": ["feature_audit.csv", "paired_outcomes.csv", "split_summary.csv", "predictive_targets.csv",
                      "router_metrics.csv", "cost_sensitivity.csv", "risk_coverage.csv", "failure_attribution.md",
                      "RELATIVE_GAIN_PREDICTABILITY_REPORT.md", "integrity_audit.json", "policy_lock.json", *plots],
        "commands": [
            "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/.venv-real-pilot/bin/python experiments/real_case_pilot_20261103/audit/relative_gain_predictability_20261118/run_relative_gain_predictability.py"
        ],
    }
    write_json(OUT / "config.json", config)
    report, attribution = build_report(summary, all_cost_rows, router_rows, target_rows, integrity,
                                       feature_audit_rows, prompt_audit, selection_doc, run_manifest,
                                       model_lock, parser_summary, plots)
    (OUT / "RELATIVE_GAIN_PREDICTABILITY_REPORT.md").write_text(report, encoding="utf-8")
    (OUT / "failure_attribution.md").write_text(attribution, encoding="utf-8")
    print(json.dumps({
        "output_dir": str(OUT), "test_base_accuracy": summary["test"]["base_accuracy"],
        "test_expert_accuracy": summary["test"]["expert_accuracy"],
        "test_paired_gain": summary["test"]["paired_accuracy_gain"],
        "test_helpful": summary["test"]["base_wrong_expert_correct"],
        "test_harmful": summary["test"]["base_correct_expert_wrong"],
        "test_invalid_expert": parser_summary["test"]["invalid"],
        "primary_cost_utility": {r["policy"]: r["utility"] for r in router_rows if r["policy"] in METHODS + ["always_answer", "always_defer", "oracle_selective"]},
        "policy_lock_before_test": integrity["policy_lock_written_before_test_read"],
        "plots": plots,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


