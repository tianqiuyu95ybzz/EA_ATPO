#!/usr/bin/env python3
"""Fit and lock Attempt04 routers using only paired train/validation cases.

The independent test file is intentionally not opened by this script.
Gold/answers are represented only by post-hoc paired correctness labels and
never enter the feature matrix.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
RUN = HERE / "outputs/qwen32_router_generalization_20261118_attempt04"
ROLLOUTS = RUN / "paired_rollouts"
OUT = RUN / "router_generalization"
HISTORICAL_POLICY = HERE / "outputs/qwen32_defer_validation_20261119_attempt01/router/policy_lock.json"
COSTS = [0.00, 0.05, 0.10, 0.20, 0.40, 0.60]
SEED = 20261128
BOOTSTRAP_SEED = 20261130
LEXICAL = ["log_public_chars", "log_public_words", "option_count", "mean_option_chars", "question_marks", "digit_count", "clinical_lexicon_count"]
FEATURES = LEXICAL + [
    "base_answer_entropy_normalized", "base_answer_probability_margin", "base_uncertainty_missing",
    "base_valid", "base_answer_A", "base_answer_B", "base_answer_C", "base_answer_D",
]
DELTA_CLASSES = [-1, 0, 1]
REGULARIZATION = [0.001, 0.01, 0.1, 1.0]


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_split(name: str) -> list[dict[str, Any]]:
    path = ROLLOUTS / f"paired_{name}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def feature_row(row: dict[str, Any], entropy_median: float, margin_median: float) -> list[float]:
    public = row.get("public_features") or {}
    entropy, margin = row.get("base_answer_entropy_normalized"), row.get("base_answer_probability_margin")
    missing = not (row.get("base_uncertainty_valid") and isinstance(entropy, (int, float)) and isinstance(margin, (int, float)))
    if missing:
        entropy, margin = entropy_median, margin_median
    predicted = str(row.get("base_predicted") or "").upper()
    values = [*[float(public.get(name, 0.0) or 0.0) for name in LEXICAL], float(entropy), float(margin), float(missing), float(bool(row.get("base_valid"))), *[float(predicted == letter) for letter in "ABCD"]]
    if len(values) != len(FEATURES) or not np.isfinite(values).all():
        raise ValueError("Invalid predecision feature row")
    return values


def targets(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    base = np.asarray([bool(row["base_correct"]) for row in rows], dtype=np.float64)
    expert = np.asarray([bool(row["expert_correct"]) for row in rows], dtype=np.float64)
    return expert - base, ((~base.astype(bool)) & expert.astype(bool)).astype(np.int64)


def assert_split_integrity(train: list[dict[str, Any]], val: list[dict[str, Any]]) -> None:
    if len(train) != 1000 or len(val) != 300:
        raise RuntimeError(f"Frozen cohort must contain 1000 train / 300 validation, got {len(train)}/{len(val)}")
    for name, rows in (("train", train), ("validation", val)):
        if any(row.get("split") != name for row in rows):
            raise RuntimeError(f"Incorrect split labels in {name}")
        for key in ("case_id", "case_hash", "prompt_hash"):
            if len({row[key] for row in rows}) != len(rows):
                raise RuntimeError(f"Duplicate {key} in {name}")
    for key in ("case_id", "case_hash", "prompt_hash"):
        if {row[key] for row in train} & {row[key] for row in val}:
            raise RuntimeError(f"Train/validation overlap in {key}")


def standardized_features(train: list[dict[str, Any]], val: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    valid = [r for r in train if r.get("base_uncertainty_valid") and isinstance(r.get("base_answer_entropy_normalized"), (int, float)) and isinstance(r.get("base_answer_probability_margin"), (int, float))]
    entropy_med = float(np.median([r["base_answer_entropy_normalized"] for r in valid])) if valid else 0.5
    margin_med = float(np.median([r["base_answer_probability_margin"] for r in valid])) if valid else 0.0
    x_train = np.asarray([feature_row(r, entropy_med, margin_med) for r in train], dtype=np.float64)
    x_val = np.asarray([feature_row(r, entropy_med, margin_med) for r in val], dtype=np.float64)
    mean, scale = x_train.mean(axis=0), x_train.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return (x_train - mean) / scale, (x_val - mean) / scale, {"names": FEATURES, "mean": mean.tolist(), "scale": scale.tolist(), "entropy_median_from_train": entropy_med, "margin_median_from_train": margin_med}


def fit_torch_linear(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, multiclass: bool, reg: float) -> dict[str, Any]:
    xt = torch.tensor(x_train, dtype=torch.float64)
    xv = torch.tensor(x_val, dtype=torch.float64)
    classes = len(DELTA_CLASSES) if multiclass else 1
    model = nn.Linear(x_train.shape[1], classes, dtype=torch.float64)
    torch.manual_seed(SEED)
    with torch.no_grad():
        model.weight.zero_(); model.bias.zero_()
    if multiclass:
        counts = np.bincount(y_train.astype(int), minlength=3).astype(np.float64)
        weights = len(y_train) / np.maximum(3.0 * counts, 1.0)
        weights = np.minimum(weights, 10.0)
        yt = torch.tensor(y_train, dtype=torch.long)
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float64))
    else:
        positive = float(np.sum(y_train == 1)); negative = float(np.sum(y_train == 0))
        pos_weight = min(10.0, negative / max(positive, 1.0))
        yt = torch.tensor(y_train, dtype=torch.float64).reshape(-1, 1)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float64))
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=300, tolerance_grad=1e-10, tolerance_change=1e-12, line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad()
        logits = model(xt)
        loss = loss_fn(logits, yt) + 0.5 * reg * model.weight.square().sum()
        loss.backward()
        return loss
    optimizer.step(closure)
    with torch.no_grad():
        val_logits = model(xv)
        if multiclass:
            val_probs = torch.softmax(val_logits, dim=1).numpy()
            val_loss = float(nn.functional.cross_entropy(val_logits, torch.tensor(y_val, dtype=torch.long)))
        else:
            val_probs = torch.sigmoid(val_logits).numpy().reshape(-1)
            val_loss = float(nn.functional.binary_cross_entropy_with_logits(val_logits, torch.tensor(y_val, dtype=torch.float64).reshape(-1, 1)))
    return {"model": model, "val_loss": val_loss, "val_scores": val_probs, "regularization": reg}


def fit_ridge(x_train: np.ndarray, delta_train: np.ndarray, x_val: np.ndarray, delta_val: np.ndarray) -> dict[str, Any]:
    ymean = float(delta_train.mean())
    best = None
    for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
        coef = np.linalg.solve(x_train.T @ x_train + alpha * np.eye(x_train.shape[1]), x_train.T @ (delta_train - ymean))
        prediction = ymean + x_val @ coef
        mse = float(np.mean((prediction - delta_val) ** 2))
        item = {"alpha": alpha, "coef": coef, "intercept": ymean, "val_mse": mse, "val_prediction": prediction}
        if best is None or (mse, -alpha) < (best["val_mse"], -best["alpha"]):
            best = item
    assert best is not None
    return best


def fit_mlp(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> dict[str, Any]:
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(1)
    xt, xv = torch.tensor(x_train, dtype=torch.float32), torch.tensor(x_val, dtype=torch.float32)
    yt, yv = torch.tensor(y_train, dtype=torch.long), torch.tensor(y_val, dtype=torch.long)
    counts = np.bincount(y_train, minlength=3).astype(np.float64)
    class_weights = np.minimum(len(y_train) / np.maximum(3.0 * counts, 1.0), 10.0)
    candidates = []
    for weight_decay in (1e-3, 1e-2, 1e-1):
        torch.manual_seed(SEED)
        model = nn.Sequential(nn.Linear(x_train.shape[1], 16), nn.ReLU(), nn.Linear(16, 3))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=weight_decay)
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))
        best_state, best_val, best_epoch, stale = None, math.inf, 0, 0
        for epoch in range(1, 501):
            model.train(); optimizer.zero_grad(); train_loss = loss_fn(model(xt), yt); train_loss.backward(); optimizer.step()
            model.eval()
            with torch.no_grad():
                val_logits = model(xv)
                val_loss = float(nn.functional.cross_entropy(val_logits, yv))
            if val_loss < best_val - 1e-5:
                best_val, best_epoch, stale = val_loss, epoch, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= 30:
                    break
        assert best_state is not None
        model.load_state_dict(best_state); model.eval()
        with torch.no_grad():
            probs = torch.softmax(model(xv), dim=1).numpy()
        candidates.append({"model": model, "val_loss": best_val, "val_probs": probs, "weight_decay": weight_decay, "best_epoch": best_epoch})
    candidates.sort(key=lambda item: (item["val_loss"], item["weight_decay"]))
    return candidates[0]


def model_state(model: nn.Module) -> dict[str, Any]:
    return {key: value.detach().cpu().numpy().tolist() for key, value in model.state_dict().items()}


def scores_from_rows(rows: list[dict[str, Any]], feats: np.ndarray, stats: dict[str, Any]) -> dict[str, np.ndarray]:
    entropy = np.asarray([float(r["base_answer_entropy_normalized"]) if r.get("base_uncertainty_valid") and isinstance(r.get("base_answer_entropy_normalized"), (int, float)) else 1.0 for r in rows])
    # This helper is used only on validation rows during fitting.
    valid = np.asarray([bool(r.get("base_uncertainty_valid")) for r in rows])
    med = float(stats["entropy_median_from_train"])
    entropy = np.where(valid, entropy, 1.0)
    return {"uncertainty_threshold": entropy}


def select_threshold(scores: np.ndarray, rows: list[dict[str, Any]], cost: float) -> dict[str, Any]:
    values = sorted(set(map(float, scores)))
    candidates = [float(np.nextafter(max(values) if values else 0.0, math.inf)), *values]
    base = np.asarray([bool(r["base_correct"]) for r in rows], dtype=float)
    expert = np.asarray([bool(r["expert_correct"]) for r in rows], dtype=float)
    best = None
    for threshold in candidates:
        calls = scores >= threshold
        acc = np.where(calls, expert, base)
        utility = float(np.mean(acc) - cost * np.mean(calls))
        item = {"threshold": threshold, "validation_utility": utility, "validation_accuracy": float(np.mean(acc)), "validation_call_rate": float(np.mean(calls))}
        if best is None or (item["validation_utility"], -item["validation_call_rate"], -item["threshold"]) > (best["validation_utility"], -best["validation_call_rate"], -best["threshold"]):
            best = item
    assert best is not None
    return best


def main() -> None:
    if (OUT / "policy_lock.json").exists():
        raise FileExistsError(f"Policy already locked; refusing overwrite: {OUT / 'policy_lock.json'}")
    train, val = read_split("train"), read_split("validation")
    assert_split_integrity(train, val)
    if not HISTORICAL_POLICY.is_file():
        raise FileNotFoundError(f"Frozen historical Learned Router missing: {HISTORICAL_POLICY}")
    x_train, x_val, feature_stats = standardized_features(train, val)
    delta_train, correction_train = targets(train)
    delta_val, correction_val = targets(val)
    if set(np.unique(delta_train).tolist()) != {-1.0, 0.0, 1.0}:
        raise RuntimeError("Training split does not contain all three paired relative-value classes")
    y_train_class = (delta_train + 1).astype(np.int64)
    y_val_class = (delta_val + 1).astype(np.int64)
    torch.set_num_threads(1)

    # Logistic event target: whether Expert corrects a Base error.
    correction_candidates = [fit_torch_linear(x_train, correction_train, x_val, correction_val, False, reg) for reg in REGULARIZATION]
    correction_best = min(correction_candidates, key=lambda item: (item["val_loss"], item["regularization"]))
    # Logistic relative-value target: class probabilities for Δ ∈ {-1,0,+1}.
    delta_candidates = [fit_torch_linear(x_train, y_train_class, x_val, y_val_class, True, reg) for reg in REGULARIZATION]
    delta_logistic_best = min(delta_candidates, key=lambda item: (item["val_loss"], item["regularization"]))
    logistic_delta_val = delta_logistic_best["val_scores"][:, 2] - delta_logistic_best["val_scores"][:, 0]
    ridge_best = fit_ridge(x_train, delta_train, x_val, delta_val)
    mlp_best = fit_mlp(x_train, y_train_class, x_val, y_val_class)
    mlp_delta_val = mlp_best["val_probs"][:, 2] - mlp_best["val_probs"][:, 0]

    historical = json.loads(HISTORICAL_POLICY.read_text(encoding="utf-8"))
    if historical.get("method") != "ridge regression of paired accuracy advantage (expert_correct - base_correct)":
        raise RuntimeError("Historical Learned Router format or target changed; refuse silent substitution")
    hist_features = np.asarray([[float((r.get("public_features") or {}).get(name, 0.0) or 0.0) for name in LEXICAL] + [float(r["base_answer_entropy_normalized"]) if r.get("base_uncertainty_valid") else historical["entropy_imputation_median_from_train"], float(r["base_answer_probability_margin"]) if r.get("base_uncertainty_valid") else historical["margin_imputation_median_from_train"], float(not r.get("base_uncertainty_valid")), float(bool(r.get("base_valid")))] + [float(str(r.get("base_predicted") or "").upper() == letter) for letter in "ABCD"] for r in val], dtype=float)
    hist_mean, hist_scale = np.asarray(historical["feature_mean"]), np.asarray(historical["feature_scale"])
    historical_delta_val = float(historical["target_mean"]) + ((hist_features - hist_mean) / hist_scale) @ np.asarray(historical["coefficients"])

    scores_val = {
        "uncertainty_threshold": scores_from_rows(val, x_val, feature_stats)["uncertainty_threshold"],
        "logistic_correction": correction_best["val_scores"],
        "logistic_relative_value": logistic_delta_val,
        "ridge_relative_value": ridge_best["val_prediction"],
        "mlp_relative_value": mlp_delta_val,
        "historical_learned_router": historical_delta_val,
    }
    thresholds = {name: {f"{cost:.2f}": select_threshold(score, val, cost) for cost in COSTS} for name, score in scores_val.items()}
    thresholds["metadata"] = {"selection_data": "validation only", "utility": "accuracy - cost * expert_call_rate", "costs": COSTS}

    OUT.mkdir(parents=True, exist_ok=True)
    policy = {
        "status": "locked_before_test_analysis",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "primary_router": "logistic_relative_value",
        "primary_supervision": "three-class paired Δ = expert_correct - base_correct; score = P(Δ=+1)-P(Δ=-1)",
        "decision_cost_rule": "net action value = predicted Δ - lambda_defer; cost deducted exactly once",
        "cost_grid": COSTS,
        "feature_schema": feature_stats,
        "models": {
            "logistic_correction": {"type": "binary weighted logistic regression", "regularization": correction_best["regularization"], "validation_nll": correction_best["val_loss"], "weight": correction_best["model"].weight.detach().numpy().reshape(-1).tolist(), "bias": float(correction_best["model"].bias.detach().numpy()[0])},
            "logistic_relative_value": {"type": "three-class weighted logistic regression", "class_order": DELTA_CLASSES, "regularization": delta_logistic_best["regularization"], "validation_nll": delta_logistic_best["val_loss"], "state_dict": model_state(delta_logistic_best["model"])},
            "ridge_relative_value": {"type": "ridge regression", "alpha": ridge_best["alpha"], "validation_mse": ridge_best["val_mse"], "weight": ridge_best["coef"].tolist(), "intercept": ridge_best["intercept"]},
            "mlp_relative_value": {"type": "one-hidden-layer MLP, 16 hidden units", "class_order": DELTA_CLASSES, "weight_decay": mlp_best["weight_decay"], "best_epoch": mlp_best["best_epoch"], "validation_nll": mlp_best["val_loss"], "state_dict": model_state(mlp_best["model"])},
            "historical_learned_router": {"type": "frozen Attempt01 Ridge router, no refit", "path": str(HISTORICAL_POLICY), "sha256": sha_file(HISTORICAL_POLICY), "thresholds_selected_on_attempt04_validation": True},
        },
        "thresholds_selected_on_validation": thresholds,
        "training_label_distribution": {str(int(v)): int(np.sum(delta_train == v)) for v in DELTA_CLASSES},
        "training_correction_counts": {"corrects_base_error": int(correction_train.sum()), "all_cases": len(train), "base_errors": int(np.sum([not bool(r["base_correct"]) for r in train]))},
        "split_integrity": {"train_n": len(train), "validation_n": len(val), "train_case_id_sha256": sha_text("\n".join(sorted(r["case_id"] for r in train))), "validation_case_id_sha256": sha_text("\n".join(sorted(r["case_id"] for r in val))), "overlap_case_id_hash_prompt": 0, "test_split_opened_by_fitter": False},
        "train_sha256": sha_file(ROLLOUTS / "paired_train.jsonl"), "validation_sha256": sha_file(ROLLOUTS / "paired_validation.jsonl"),
        "fitter_sha256": sha_file(Path(__file__).resolve()), "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "seed": SEED,
        "gold_or_expert_answer_in_features": False,
    }
    path = OUT / "policy_lock.json"
    path.write_text(json.dumps(policy, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": policy["status"], "train": len(train), "validation": len(val), "primary_router": policy["primary_router"], "test_split_opened": False, "policy_lock": str(path)}, indent=2))


if __name__ == "__main__":
    main()
