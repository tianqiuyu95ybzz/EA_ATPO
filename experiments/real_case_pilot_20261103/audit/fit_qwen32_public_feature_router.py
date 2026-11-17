#!/usr/bin/env python3
"""Fit and lock a small predecision Qwen3-32B DEFER router on train/validation only."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
RUN = HERE / "outputs/qwen32_defer_validation_20261119_attempt01"
ROLLOUTS = RUN / "paired_rollouts"
OUT = RUN / "router"
COSTS = [0.0, 0.05, 0.10, 0.20, 0.40, 0.60]
SEED = 20261119
LEXICAL = ["log_public_chars", "log_public_words", "option_count", "mean_option_chars", "question_marks", "digit_count", "clinical_lexicon_count"]
FEATURES = LEXICAL + [
    "base_answer_entropy_normalized", "base_answer_probability_margin", "base_uncertainty_missing",
    "base_valid", "base_answer_A", "base_answer_B", "base_answer_C", "base_answer_D",
]


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_split(name: str) -> list[dict[str, Any]]:
    path = ROLLOUTS / f"paired_{name}.jsonl"
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def feature_row(row: dict[str, Any], entropy_median: float, margin_median: float) -> list[float]:
    public = row.get("public_features") or {}
    entropy = row.get("base_answer_entropy_normalized")
    margin = row.get("base_answer_probability_margin")
    missing = not (row.get("base_uncertainty_valid") and isinstance(entropy, (int, float)) and isinstance(margin, (int, float)))
    if missing:
        entropy, margin = entropy_median, margin_median
    predicted = str(row.get("base_predicted") or "").upper()
    return [
        *[float(public.get(name, 0.0) or 0.0) for name in LEXICAL],
        float(entropy), float(margin), float(missing), float(bool(row.get("base_valid"))),
        *[float(predicted == letter) for letter in "ABCD"],
    ]


def utility(rows: list[dict[str, Any]], calls: list[bool], cost: float) -> float:
    correctness = [bool(r["expert_correct"] if call else r["base_correct"]) for r, call in zip(rows, calls)]
    return float(np.mean(correctness) - cost * np.mean(calls))


def select_threshold(scores: np.ndarray, rows: list[dict[str, Any]], cost: float) -> dict[str, Any]:
    values = sorted(set(map(float, scores)), reverse=True)
    candidates = [float(np.nextafter(max(values) if values else 0.0, math.inf)), *values]
    best = None
    for threshold in candidates:
        calls = [float(s) >= threshold for s in scores]
        u = utility(rows, calls, cost)
        row = {"threshold": threshold, "validation_utility": u, "validation_accuracy": float(np.mean([r["expert_correct"] if call else r["base_correct"] for r, call in zip(rows, calls)])), "validation_call_rate": float(np.mean(calls))}
        if best is None or (row["validation_utility"], -row["validation_call_rate"]) > (best["validation_utility"], -best["validation_call_rate"]):
            best = row
    assert best is not None
    return best


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    if OUT.exists() and (OUT / "policy_lock.json").exists():
        raise FileExistsError(f"Router policy already locked: {OUT / 'policy_lock.json'}")
    train, val = read_split("train"), read_split("validation")
    if len(train) != 200 or len(val) != 100:
        raise RuntimeError(f"Expected 200 train/100 validation paired cases, got {len(train)}/{len(val)}")
    train_ids, val_ids = {r["case_id"] for r in train}, {r["case_id"] for r in val}
    train_hash, val_hash = {r["case_hash"] for r in train}, {r["case_hash"] for r in val}
    if train_ids & val_ids or train_hash & val_hash:
        raise RuntimeError("Case-level leakage between training and validation")
    if any(r.get("split") != "train" for r in train) or any(r.get("split") != "validation" for r in val):
        raise RuntimeError("Split label mismatch")

    valid_train = [r for r in train if r.get("base_uncertainty_valid")]
    entropy_median = float(np.median([r["base_answer_entropy_normalized"] for r in valid_train])) if valid_train else 0.5
    margin_median = float(np.median([r["base_answer_probability_margin"] for r in valid_train])) if valid_train else 0.0
    x_train = np.asarray([feature_row(r, entropy_median, margin_median) for r in train], dtype=float)
    x_val = np.asarray([feature_row(r, entropy_median, margin_median) for r in val], dtype=float)
    y_train = np.asarray([float(r["expert_correct"]) - float(r["base_correct"]) for r in train])
    y_val = np.asarray([float(r["expert_correct"]) - float(r["base_correct"]) for r in val])
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-8] = 1.0
    z_train, z_val = (x_train - mean) / scale, (x_val - mean) / scale
    y_mean = float(y_train.mean())
    best = None
    for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
        coef = np.linalg.solve(z_train.T @ z_train + alpha * np.eye(z_train.shape[1]), z_train.T @ (y_train - y_mean))
        pred_val = y_mean + z_val @ coef
        mse = float(np.mean((pred_val - y_val) ** 2))
        candidate = {"alpha": alpha, "coef": coef, "mse": mse, "pred_val": pred_val}
        if best is None or (mse, -alpha) < (best["mse"], -best["alpha"]):
            best = candidate
    assert best is not None
    alpha = float(best["alpha"])
    coef = best["coef"]
    pred_train = y_mean + z_train @ coef
    pred_val = best["pred_val"]

    # Uncertainty baseline: normalized entropy over the available answer-letter
    # logprobs. Missing logprob rows are treated as maximally uncertain.
    uncertainty_train = np.asarray([float(r.get("base_answer_entropy_normalized")) if r.get("base_uncertainty_valid") else 1.0 for r in train])
    uncertainty_val = np.asarray([float(r.get("base_answer_entropy_normalized")) if r.get("base_uncertainty_valid") else 1.0 for r in val])
    learned_thresholds, uncertainty_thresholds, validation_rows = {}, {}, []
    learned_sweep = sorted(set(map(float, pred_val)), reverse=True)
    uncertainty_sweep = sorted(set(map(float, uncertainty_val)), reverse=True)
    for cost in COSTS:
        key = f"{cost:.2f}"
        learned = select_threshold(pred_val, val, cost)
        uncertain = select_threshold(uncertainty_val, val, cost)
        learned_thresholds[key] = learned
        uncertainty_thresholds[key] = uncertain
        validation_rows.extend([
            {"cost": cost, "policy": "learned_value_router", **learned},
            {"cost": cost, "policy": "uncertainty_defer", **uncertain},
        ])

    OUT.mkdir(parents=True, exist_ok=True)
    policy = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method": "ridge regression of paired accuracy advantage (expert_correct - base_correct)",
        "cost_grid": COSTS, "primary_cost": 0.60,
        "utility_definition": "accuracy - cost * expert_call_rate",
        "selected_alpha_by_validation_mse": alpha,
        "feature_names": FEATURES, "feature_mean": mean.tolist(), "feature_scale": scale.tolist(),
        "target_mean": y_mean, "coefficients": coef.tolist(),
        "entropy_imputation_median_from_train": entropy_median, "margin_imputation_median_from_train": margin_median,
        "uncertainty_definition": "normalized entropy over A-D answer token logprobs; missing confidence assigned 1.0",
        "learned_thresholds_selected_on_validation": learned_thresholds,
        "uncertainty_thresholds_selected_on_validation": uncertainty_thresholds,
        "learned_risk_coverage_thresholds_from_validation": learned_sweep,
        "uncertainty_risk_coverage_thresholds_from_validation": uncertainty_sweep,
        "split_integrity": {
            "train_n": len(train), "validation_n": len(val), "train_case_ids_sha256": sha_text("\n".join(sorted(train_ids))),
            "validation_case_ids_sha256": sha_text("\n".join(sorted(val_ids))), "case_id_overlap": 0, "case_hash_overlap": 0,
            "test_split_opened_by_fitter": False,
        },
        "training_value_diagnostics": {"advantage_mean": float(y_train.mean()), "predicted_train_mse": float(np.mean((pred_train - y_train) ** 2)), "train_pearson": pearson(pred_train, y_train)},
        "validation_value_diagnostics": {"advantage_mean": float(y_val.mean()), "predicted_validation_mse": float(best["mse"]), "validation_pearson": pearson(pred_val, y_val)},
        "seed": SEED,
        "train_data_sha256": sha_file(ROLLOUTS / "paired_train.jsonl"), "validation_data_sha256": sha_file(ROLLOUTS / "paired_validation.jsonl"),
        "fit_script_sha256": sha_file(Path(__file__).resolve()),
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "oracle_used_for_fitting": False,
    }
    lock_path = OUT / "policy_lock.json"
    lock_path.write_text(json.dumps(policy, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (OUT / "validation_policy_selection.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["cost", "policy", "threshold", "validation_utility", "validation_accuracy", "validation_call_rate"])
        writer.writeheader(); writer.writerows(validation_rows)
    print(json.dumps({"status": "policy_locked_before_test_analysis", "alpha": alpha, "train_mse": policy["training_value_diagnostics"]["predicted_train_mse"], "validation_mse": best["mse"], "train_pearson": policy["training_value_diagnostics"]["train_pearson"], "validation_pearson": policy["validation_value_diagnostics"]["validation_pearson"], "policy_lock": str(lock_path), "test_split_opened": False}, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
