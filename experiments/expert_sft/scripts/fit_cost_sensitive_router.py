#!/usr/bin/env python3
"""Fit lightweight relative-gain routers using train cases; calibrate thresholds on validation only."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/expert_sft"
COSTS = [0.0, 0.05, 0.10, 0.20, 0.40, 0.60]
SEED = 20261119
FEATURES = [
    "log_public_chars", "log_public_words", "option_count", "mean_option_chars",
    "question_marks", "digit_count", "clinical_lexicon_count",
    "base_entropy_normalized", "base_probability_margin", "base_uncertainty_missing",
    "base_valid", "base_answer_A", "base_answer_B", "base_answer_C", "base_answer_D",
]
HISTORICAL_POLICY = ROOT / "experiments/real_case_pilot_20261103/audit/outputs/qwen32_defer_validation_20261119_attempt01/router/policy_lock.json"


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    if not rows:
        raise ValueError(f"Empty paired dataset: {path}")
    return rows


def assert_split(rows: list[dict[str, Any]], split: str, expected: int) -> None:
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} {split} cases, received {len(rows)}")
    if any(row.get("split") != split for row in rows):
        raise ValueError(f"Wrong split tag inside {split} file")
    for key in ("case_id", "case_hash", "prompt_hash"):
        values = [str(row.get(key, "")) for row in rows]
        if any(not x for x in values) or len(set(values)) != len(values):
            raise ValueError(f"Missing or duplicate {key} in {split}")


def row_features(row: dict[str, Any], entropy_median: float, margin_median: float) -> list[float]:
    public = row.get("public_features") or {}
    entropy = row.get("base_entropy_normalized")
    margin = row.get("base_probability_margin")
    missing = not (isinstance(entropy, (int, float)) and math.isfinite(float(entropy)) and isinstance(margin, (int, float)) and math.isfinite(float(margin)))
    if missing:
        entropy, margin = entropy_median, margin_median
    predicted = str(row.get("base_predicted") or "").upper()
    values = [
        *[float(public.get(k, 0.0) or 0.0) for k in FEATURES[:7]],
        float(entropy), float(margin), float(missing), float(bool(row.get("base_valid"))),
        *[float(predicted == letter) for letter in "ABCD"],
    ]
    if len(values) != len(FEATURES) or not np.isfinite(values).all():
        raise ValueError(f"Non-finite/malformed decision-time feature row: {row.get('case_id')}")
    return values


def matrix_for(rows: list[dict[str, Any]], medians: tuple[float, float] | None = None):
    if medians is None:
        valid_e = [float(r["base_entropy_normalized"]) for r in rows if isinstance(r.get("base_entropy_normalized"), (int, float)) and math.isfinite(float(r["base_entropy_normalized"]))]
        valid_m = [float(r["base_probability_margin"]) for r in rows if isinstance(r.get("base_probability_margin"), (int, float)) and math.isfinite(float(r["base_probability_margin"]))]
        medians = (float(np.median(valid_e)) if valid_e else 0.5, float(np.median(valid_m)) if valid_m else 0.0)
    return np.asarray([row_features(r, *medians) for r in rows], dtype=np.float64), medians


def labels(rows: list[dict[str, Any]]) -> np.ndarray:
    y = np.asarray([int(bool(r["expert_correct"])) - int(bool(r["base_correct"])) for r in rows], dtype=np.float64)
    if not np.isin(y, [-1.0, 0.0, 1.0]).all():
        raise ValueError("Relative-gain labels must be -1, 0 or +1")
    return y


def state_dict_json(model: nn.Module) -> dict[str, Any]:
    return {k: v.detach().cpu().double().tolist() for k, v in model.state_dict().items()}


def fit_logistic(x: np.ndarray, y: np.ndarray, reg: float, seed: int):
    torch.manual_seed(seed)
    tx = torch.tensor(x, dtype=torch.float64)
    ty = torch.tensor((y + 1).astype(np.int64), dtype=torch.long)
    model = nn.Linear(x.shape[1], 3, dtype=torch.float64)
    optimizer = torch.optim.LBFGS(model.parameters(), lr=0.5, max_iter=300, line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad()
        logits = model(tx)
        loss = nn.functional.cross_entropy(logits, ty) + reg * model.weight.square().sum() / (2 * len(x))
        loss.backward()
        return loss
    optimizer.step(closure)
    with torch.no_grad():
        logp = torch.log_softmax(model(tx), dim=1)
        nll = float(nn.functional.nll_loss(logp, ty).item())
        probs = torch.softmax(model(tx), dim=1).numpy()
    return model, probs, nll


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float):
    x1 = np.column_stack([np.ones(len(x)), x])
    penalty = np.eye(x1.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(x1.T @ x1 + penalty, x1.T @ y)
    return coef[0] + x @ coef[1:], coef


def fit_mlp(x: np.ndarray, y: np.ndarray, xv: np.ndarray, yv: np.ndarray, reg: float, seed: int):
    torch.manual_seed(seed)
    tx = torch.tensor(x, dtype=torch.float32)
    ty = torch.tensor(y[:, None], dtype=torch.float32)
    tvx = torch.tensor(xv, dtype=torch.float32)
    tvy = torch.tensor(yv[:, None], dtype=torch.float32)
    model = nn.Sequential(nn.Linear(x.shape[1], 16), nn.Tanh(), nn.Linear(16, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=reg)
    best, best_loss, best_epoch, stale = None, float("inf"), 0, 0
    for epoch in range(1, 1001):
        model.train()
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(tx), ty)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(nn.functional.mse_loss(model(tvx), tvy).item())
        if val_loss < best_loss - 1e-8:
            best_loss, best_epoch, best, stale = val_loss, epoch, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= 80:
                break
    model.load_state_dict(best)
    model.eval()
    with torch.no_grad():
        pred = model(tx).squeeze(1).numpy().astype(float)
    return model, pred, best_loss, best_epoch


def select_threshold(scores: np.ndarray, rows: list[dict[str, Any]], cost: float) -> dict[str, Any]:
    unique = sorted(set(float(x) for x in scores))
    # The first candidate explicitly means ANSWER on all cases.
    candidates = [(float(np.nextafter(max(unique), math.inf)) if unique else 0.0), *unique]
    base = np.asarray([bool(r["base_correct"]) for r in rows], dtype=float)
    delta = labels(rows)
    best = None
    for threshold in candidates:
        calls = scores >= threshold
        accuracy = float(np.mean(base + calls * delta))
        rate = float(np.mean(calls))
        utility = accuracy - cost * rate
        item = {"threshold": float(threshold), "validation_accuracy": accuracy, "validation_defer_rate": rate, "validation_average_cost": cost * rate, "validation_utility": utility}
        if best is None or (item["validation_utility"], -item["validation_defer_rate"], item["threshold"]) > (best["validation_utility"], -best["validation_defer_rate"], best["threshold"]):
            best = item
    return best or {"threshold": float("inf"), "validation_accuracy": float(np.mean([r["base_correct"] for r in rows])), "validation_defer_rate": 0.0, "validation_average_cost": 0.0, "validation_utility": float(np.mean([r["base_correct"] for r in rows]))}


def distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    y = labels(rows)
    return {
        "n": len(rows), "base_accuracy": float(np.mean([r["base_correct"] for r in rows])),
        "expert_accuracy": float(np.mean([r["expert_correct"] for r in rows])),
        "relative_gain_mean": float(y.mean()),
        "relative_gain_counts": {str(v): int(np.sum(y == v)) for v in (-1.0, 0.0, 1.0)},
        "helpful_rate": float(np.mean(y == 1)), "harmful_rate": float(np.mean(y == -1)),
        "base_parse_failure_rate": float(np.mean([not r["base_valid"] for r in rows])),
        "expert_parse_failure_rate": float(np.mean([not r["expert_valid"] for r in rows])),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--validation", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--expected-train", type=int, default=1000)
    ap.add_argument("--expected-validation", type=int, default=300)
    args = ap.parse_args()
    out_lock = args.output_dir / "router_policy_lock.json"
    if out_lock.exists():
        raise FileExistsError(f"Router already frozen: {out_lock}")
    torch.set_num_threads(1)
    train, val = read_rows(args.train), read_rows(args.validation)
    assert_split(train, "train", args.expected_train)
    assert_split(val, "validation", args.expected_validation)
    for key in ("case_id", "case_hash", "prompt_hash"):
        if {str(x[key]) for x in train} & {str(x[key]) for x in val}:
            raise ValueError(f"Train/validation leakage on {key}")
    x_train_raw, medians = matrix_for(train)
    x_val_raw, _ = matrix_for(val, medians)
    mean, scale = x_train_raw.mean(axis=0), x_train_raw.std(axis=0)
    scale[scale < 1e-8] = 1.0
    x_train = (x_train_raw - mean) / scale
    x_val = (x_val_raw - mean) / scale
    y_train, y_val = labels(train), labels(val)

    logistic_candidates = []
    for reg in (0.0001, 0.001, 0.01, 0.1, 1.0):
        model, train_probs, train_nll = fit_logistic(x_train, y_train, reg, SEED)
        with torch.no_grad():
            probs_val = torch.softmax(model(torch.tensor(x_val, dtype=torch.float64)), dim=1).numpy()
            targets = torch.tensor((y_val + 1).astype(np.int64), dtype=torch.long)
            val_nll = float(nn.functional.nll_loss(torch.log(torch.tensor(probs_val).clamp_min(1e-12)), targets).item())
        logistic_candidates.append((val_nll, reg, model, probs_val, train_probs))
    _, logistic_reg, logistic_model, logistic_probs_val, logistic_probs_train = min(logistic_candidates, key=lambda z: (z[0], z[1]))
    logistic_scores_val = logistic_probs_val[:, 2] - logistic_probs_val[:, 0]
    logistic_scores_train = logistic_probs_train[:, 2] - logistic_probs_train[:, 0]

    ridge_candidates = []
    for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
        pred_train, coef = fit_ridge(x_train, y_train, alpha)
        pred_val = coef[0] + x_val @ coef[1:]
        ridge_candidates.append((float(np.mean((pred_val - y_val) ** 2)), alpha, pred_train, pred_val, coef))
    ridge_mse, ridge_alpha, ridge_scores_train, ridge_scores_val, ridge_coef = min(ridge_candidates, key=lambda z: (z[0], z[1]))

    mlp_candidates = []
    for reg in (0.001, 0.01, 0.1):
        model, pred_train, val_mse, best_epoch = fit_mlp(x_train, y_train, x_val, y_val, reg, SEED)
        with torch.no_grad():
            pred_val = model(torch.tensor(x_val, dtype=torch.float32)).squeeze(1).numpy().astype(float)
        val_mse = float(np.mean((pred_val - y_val) ** 2))
        mlp_candidates.append((val_mse, reg, model, pred_train, pred_val, best_epoch, float(np.mean((pred_train - y_train) ** 2))))
    mlp_mse, mlp_reg, mlp_model, mlp_scores_train, mlp_scores_val, mlp_epoch, mlp_train_mse = min(mlp_candidates, key=lambda z: (z[0], z[1]))

    # Existing Attempt01 Learned Router is carried forward without refitting.
    if not HISTORICAL_POLICY.is_file():
        raise FileNotFoundError(f"Frozen historical Learned Router missing: {HISTORICAL_POLICY}")
    old = json.loads(HISTORICAL_POLICY.read_text(encoding="utf-8"))
    if old.get("method") != "ridge regression of paired accuracy advantage (expert_correct - base_correct)":
        raise ValueError("Unexpected historical router method; refusing silent substitution")
    old_features = []
    for row in (train + val):
        pub = row.get("public_features") or {}
        entropy, margin = row.get("base_entropy_normalized"), row.get("base_probability_margin")
        missing = not (isinstance(entropy, (int, float)) and isinstance(margin, (int, float)))
        entropy = float(entropy) if not missing else float(old["entropy_imputation_median_from_train"])
        margin = float(margin) if not missing else float(old["margin_imputation_median_from_train"])
        pred = str(row.get("base_predicted") or "").upper()
        raw = [*[float(pub.get(k, 0.0) or 0.0) for k in FEATURES[:7]], entropy, margin, float(missing), float(bool(row["base_valid"])), *[float(pred == letter) for letter in "ABCD"]]
        old_features.append(raw)
    old_x = np.asarray(old_features, dtype=float)
    old_score = float(old["target_mean"]) + ((old_x - np.asarray(old["feature_mean"])) / np.asarray(old["feature_scale"])) @ np.asarray(old["coefficients"])
    hist_train_score, hist_val_score = old_score[:len(train)], old_score[len(train):]

    scores_val = {
        "uncertainty_threshold": np.asarray([float(r["base_entropy_normalized"]) if isinstance(r.get("base_entropy_normalized"), (int, float)) else medians[0] for r in val]),
        "logistic_relative_gain": logistic_scores_val,
        "ridge_relative_gain": ridge_scores_val,
        "mlp_relative_gain": mlp_scores_val,
        "historical_learned_router": hist_val_score,
    }
    scores_train = {"logistic_relative_gain": logistic_scores_train, "ridge_relative_gain": ridge_scores_train, "mlp_relative_gain": mlp_scores_train, "historical_learned_router": hist_train_score}
    policies = {}
    for name, scores in scores_val.items():
        policies[name] = {f"{cost:.2f}": select_threshold(scores, val, cost) for cost in COSTS}
    uncertainty_calls = {cost: policies["uncertainty_threshold"][f"{cost:.2f}"]["validation_defer_rate"] for cost in COSTS}
    random_rates = {f"{cost:.2f}": uncertainty_calls[cost] for cost in COSTS}

    def model_blob(name: str):
        if name == "logistic_relative_gain":
            return {"kind": "multinomial_logistic", "class_order": [-1, 0, 1], "regularization": logistic_reg, "state_dict": state_dict_json(logistic_model)}
        if name == "ridge_relative_gain":
            return {"kind": "ridge_regression", "alpha": ridge_alpha, "coefficient": ridge_coef[1:].tolist(), "intercept": float(ridge_coef[0])}
        if name == "mlp_relative_gain":
            return {"kind": "mlp_regression", "hidden_units": 16, "weight_decay": mlp_reg, "best_epoch": mlp_epoch, "state_dict": state_dict_json(mlp_model)}
        if name == "historical_learned_router":
            return {"kind": "frozen_historical_attempt01_ridge", "path": str(HISTORICAL_POLICY.relative_to(ROOT)), "sha256": file_sha(HISTORICAL_POLICY), "refit": False}
        return {"kind": "normalized_base_entropy", "model": "fixed_uncertainty_score"}

    policy = {
        "status": "frozen_train_fit_validation_thresholds_before_fresh_test_inference",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED, "cost_grid": COSTS, "primary_cost": 0.60,
        "utility": "end_to_end_accuracy - cost * defer_rate",
        "decision_rule": "defer iff predicted_relative_gain >= validation_locked_threshold; costs are applied once in utility",
        "feature_schema": {"names": FEATURES, "train_mean": mean.tolist(), "train_scale": scale.tolist(), "missing_entropy_median_train": medians[0], "missing_margin_median_train": medians[1], "test_time_allowed": True},
        "models": {name: model_blob(name) for name in scores_val},
        "thresholds_selected_on_validation": policies,
        "random_defer_baseline": {"call_rate_matched_to_uncertainty_threshold_validation_rate": random_rates, "seed": 20261122},
        "train_metrics": distribution(train), "validation_metrics": distribution(val),
        "label_distribution": {"train": distribution(train)["relative_gain_counts"], "validation": distribution(val)["relative_gain_counts"]},
        "train_validation_feature_distribution": {
            name: {"train_mean": float(x_train_raw[:, i].mean()), "train_sd": float(x_train_raw[:, i].std()), "validation_mean": float(x_val_raw[:, i].mean()), "validation_sd": float(x_val_raw[:, i].std())}
            for i, name in enumerate(FEATURES)
        },
        "validation_model_diagnostics": {
            "logistic_relative_gain": {"nll_selected_by_validation": min(logistic_candidates, key=lambda z:(z[0],z[1]))[0]},
            "ridge_relative_gain": {"mse_selected_by_validation": ridge_mse, "train_mse": float(np.mean((ridge_scores_train-y_train)**2))},
            "mlp_relative_gain": {"mse_selected_by_validation": mlp_mse, "train_mse": float(np.mean((mlp_scores_train-y_train)**2)), "best_epoch": mlp_epoch},
        },
        "split_integrity": {
            "train_n": len(train), "validation_n": len(val), "case_overlap": 0,
            "train_case_ids_sha256": sha_text("\n".join(sorted(str(r["case_id"]) for r in train))),
            "validation_case_ids_sha256": sha_text("\n".join(sorted(str(r["case_id"]) for r in val))),
            "train_case_hashes": sorted(str(r["case_hash"]) for r in train),
            "train_prompt_hashes": sorted(str(r["prompt_hash"]) for r in train),
            "validation_case_hashes": sorted(str(r["case_hash"]) for r in val),
            "validation_prompt_hashes": sorted(str(r["prompt_hash"]) for r in val),
            "test_data_opened": False,
        },
        "paired_input_sha256": {"train": file_sha(args.train), "validation": file_sha(args.validation)},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "allowed_features_only": True,
        "gold_expert_output_correctness_oracle_features_used_for_prediction": False,
        "old_historical_router_not_refit": True,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_lock.write_text(json.dumps(policy, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report = {"status": "validation_complete_test_unopened", "policies": policies, "train": distribution(train), "validation": distribution(val), "router_validation_diagnostics": policy["validation_model_diagnostics"], "test_opened": False}
    (args.output_dir / "router_fit_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": policy["status"], "train_n": len(train), "validation_n": len(val), "models": list(scores_val), "test_opened": False, "policy_lock": str(out_lock)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
