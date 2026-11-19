#!/usr/bin/env python3
"""Fit a small, train-only VOI predictor after a real, completed Phase-2 rollout."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

COSTS = [0.00, 0.05, 0.10, 0.20, 0.40, 0.60]
REQUIRED = {"DIRECT_ANSWER", "DIRECT_DEFER", "ASK_DECISION", "CONSULT_DECISION"}
FEATURES = ["log_public_chars", "log_public_words", "question_marks", "base_confidence", "base_entropy", "base_margin", "uncertainty_missing"]
SEED = 20261125


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def feature(row: dict[str, Any]) -> list[float]:
    f = row.get("pre_action_features") or {}
    values = [
        math.log1p(max(0.0, float(f.get("public_chars", 0) or 0))),
        math.log1p(max(0.0, float(f.get("public_words", 0) or 0))),
        float(f.get("question_marks", 0) or 0),
    ]
    missing = False
    for key in ("base_confidence", "base_entropy", "base_top1_top2_margin"):
        value = f.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            values.append(0.0)
            missing = True
        else:
            values.append(float(value))
    values.append(float(missing or f.get("uncertainty_status") == "unavailable_from_current_completion_api"))
    if not np.isfinite(values).all():
        raise ValueError(f"Invalid pre-action feature row for {row.get('case_id')}")
    return values


def utility(row: dict[str, Any], cost: float) -> float:
    counts = row.get("action_counts") or {}
    return float(bool(row.get("terminal_correct"))) - 0.05 * int(counts.get("ask", 0)) - 0.20 * int(counts.get("consult", 0)) - cost * int(counts.get("defer", 0))


def group_cases(rows: list[dict[str, Any]], split: str, expected: int) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row.get("split") != split:
            raise ValueError(f"Unexpected split tag in {split} rollout file")
        out.setdefault(str(row["case_id"]), {})[str(row["intervention"])] = row
    if len(out) != expected:
        raise ValueError(f"Expected {expected} unique {split} cases, got {len(out)}")
    for case, branches in out.items():
        if set(branches) != REQUIRED:
            raise ValueError(f"{split} case {case}: missing/extra direct or information branch")
        # Each case is retained only when the fixed protocol completed every arm.
        for name, row in branches.items():
            if row.get("error"):
                raise ValueError(f"{split} case {case} branch {name} has protocol/API error; do not silently drop it")
        signatures = {json.dumps(branches[name].get("pre_action_features") or {}, sort_keys=True) for name in REQUIRED}
        if len(signatures) != 1:
            raise ValueError(f"{split} case {case}: pre-action state features differ across paired branches")
    return out


def ridge(x: np.ndarray, y: np.ndarray, alpha: float):
    x1 = np.column_stack([np.ones(len(x)), x])
    penalty = np.eye(x1.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(x1.T @ x1 + penalty, x1.T @ y)
    return coef


def auc(y: np.ndarray, score: np.ndarray) -> float | None:
    y = y.astype(bool)
    pos, neg = int(y.sum()), int((~y).sum())
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and score[order[j]] == score[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2
        i = j
    return float((ranks[y].sum() - pos * (pos + 1) / 2) / (pos * neg))


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    def rank(a):
        order = np.argsort(a, kind="mergesort")
        r = np.empty(len(a), dtype=float)
        i = 0
        while i < len(a):
            j = i + 1
            while j < len(a) and a[order[j]] == a[order[i]]:
                j += 1
            r[order[i:j]] = (i + 1 + j) / 2
            i = j
        return r
    a, b = rank(x), rank(y)
    return float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else None


def bootstrap_ci(values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    means = []
    for start in range(0, 5000, 250):
        n = min(250, 5000 - start)
        ids = rng.integers(0, len(values), size=(n, len(values)))
        means.extend(values[ids].mean(axis=1).tolist())
    return [float(np.quantile(means, .025)), float(np.quantile(means, .975))]


def calibration_bins(pred: np.ndarray, actual: np.ndarray, count: int = 5):
    groups = np.array_split(np.argsort(pred, kind="mergesort"), min(count, len(pred)))
    rows, error = [], 0.0
    for ids in groups:
        if not len(ids):
            continue
        p, y = float(pred[ids].mean()), float(actual[ids].mean())
        error += len(ids) / len(pred) * abs(p - y)
        rows.append({"n": len(ids), "mean_predicted_voi": p, "mean_observed_voi": y})
    return float(error), rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite VOI analysis: {args.output_dir}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or manifest.get("model_inference_started") is not True:
        raise ValueError("VOI fitting requires a completed real-model Phase-2 pilot")
    rows = load(args.rollouts)
    train_rows, val_rows = [r for r in rows if r.get("split") == "train"], [r for r in rows if r.get("split") == "validation"]
    train, val = group_cases(train_rows, "train", 100), group_cases(val_rows, "validation", 50)
    if set(train) & set(val):
        raise ValueError("Train/validation case leakage")
    train_ids, val_ids = sorted(train), sorted(val)
    xtr_raw = np.asarray([feature(train[k]["DIRECT_ANSWER"]) for k in train_ids], dtype=float)
    xva_raw = np.asarray([feature(val[k]["DIRECT_ANSWER"]) for k in val_ids], dtype=float)
    mean, scale = xtr_raw.mean(0), xtr_raw.std(0)
    scale[scale < 1e-8] = 1.0
    xtr, xva = (xtr_raw - mean) / scale, (xva_raw - mean) / scale
    rng = np.random.default_rng(SEED)
    all_rows, case_rows, model_locks = [], [], {}
    for cost in COSTS:
        train_u = {case: {name: utility(row, cost) for name, row in branches.items()} for case, branches in train.items()}
        val_u = {case: {name: utility(row, cost) for name, row in branches.items()} for case, branches in val.items()}
        mean_answer = float(np.mean([v["DIRECT_ANSWER"] for v in train_u.values()]))
        mean_defer = float(np.mean([v["DIRECT_DEFER"] for v in train_u.values()]))
        stop_name = "DIRECT_ANSWER" if mean_answer >= mean_defer else "DIRECT_DEFER"
        ytr = {action: np.asarray([train_u[c][f"{action}_DECISION"] - train_u[c][stop_name] for c in train_ids]) for action in ("ASK", "CONSULT")}
        yva = {action: np.asarray([val_u[c][f"{action}_DECISION"] - val_u[c][stop_name] for c in val_ids]) for action in ("ASK", "CONSULT")}
        action_models = {}
        val_pred, val_uncertainty = {}, {}
        for action in ("ASK", "CONSULT"):
            candidates = []
            for alpha in (0.1, 1.0, 10.0, 100.0):
                coef = ridge(xtr, ytr[action], alpha)
                pred = np.column_stack([np.ones(len(xva)), xva]) @ coef
                candidates.append((float(np.mean((pred - yva[action]) ** 2)), alpha, coef, pred))
            mse, alpha, coef, pred = min(candidates, key=lambda x: (x[0], x[1]))
            action_models[action] = {"alpha": alpha, "coef": coef.tolist(), "validation_mse": mse}
            val_pred[action] = pred
            # Case-level bootstrap spread is a model sensitivity measure, not a calibrated interval.
            samples = []
            for _ in range(200):
                ids = rng.integers(0, len(train_ids), size=len(train_ids))
                bcoef = ridge(xtr[ids], ytr[action][ids], alpha)
                samples.append(np.column_stack([np.ones(len(xva)), xva]) @ bcoef)
            val_uncertainty[action] = np.std(np.asarray(samples), axis=0, ddof=1)
            cal_mae, bins = calibration_bins(pred, yva[action])
            all_rows.append({
                "cost_defer": cost, "action": action, "n_train": len(train_ids), "n_validation": len(val_ids),
                "train_voi_mean": float(ytr[action].mean()), "validation_voi_mean": float(yva[action].mean()),
                "validation_mae": float(np.mean(np.abs(pred-yva[action]))), "validation_mse": mse,
                "validation_spearman": spearman(pred, yva[action]),
                "positive_voi_auroc": auc(yva[action] > 0, pred),
                "validation_calibration_mae": cal_mae,
                "mean_bootstrap_model_spread": float(val_uncertainty[action].mean()),
                "bootstrap_spread_is_calibrated_ci": False,
                "calibration_bins_json": json.dumps(bins, ensure_ascii=False),
                "validation_actual_voi_positive_rate": float(np.mean(yva[action] > 0)),
                "validation_action_always_utility": float(np.mean([val_u[c][f"{action}_DECISION"] for c in val_ids])),
                "validation_stop_policy": stop_name,
                "validation_stop_utility": float(np.mean([val_u[c][stop_name] for c in val_ids])),
            })
        # Validation chooses one confidence threshold for the pair of actions;
        # target VOI already includes acquisition and subsequent defer costs.
        best = None
        candidate_thresholds = [float(np.nextafter(max(np.max(val_pred[a]) for a in ("ASK", "CONSULT")), math.inf))]
        candidate_thresholds += sorted(set(float(v) for a in ("ASK", "CONSULT") for v in val_pred[a]))
        for threshold in candidate_thresholds:
            decisions = []
            utilities = []
            for i, case in enumerate(val_ids):
                candidates = [(val_pred[a][i], a) for a in ("ASK", "CONSULT")]
                score, action = max(candidates, key=lambda x: (x[0], x[1] == "ASK"))
                chosen = f"{action}_DECISION" if score >= threshold else stop_name
                decisions.append(chosen)
                utilities.append(val_u[case][chosen])
            value = float(np.mean(utilities))
            calls = float(np.mean([x != stop_name for x in decisions]))
            item = {"threshold": threshold, "utility": value, "action_rate": calls, "decisions": decisions, "utilities": utilities}
            if best is None or (value, -calls, threshold) > (best["utility"], -best["action_rate"], best["threshold"]):
                best = item
        stop_values = np.asarray([val_u[c][stop_name] for c in val_ids])
        selected_values = np.asarray(best["utilities"])
        model_locks[f"{cost:.2f}"] = {
            "train_selected_stop_policy": stop_name,
            "train_stop_means": {"DIRECT_ANSWER": mean_answer, "DIRECT_DEFER": mean_defer},
            "action_models": action_models,
            "validation_threshold": best["threshold"],
            "validation_policy_utility": best["utility"],
            "validation_policy_action_rate": best["action_rate"],
            "validation_stop_policy_utility": float(stop_values.mean()),
            "validation_utility_difference": float((selected_values-stop_values).mean()),
            "validation_utility_difference_ci95": bootstrap_ci(selected_values-stop_values, SEED + int(cost*100)),
        }
        for i, case in enumerate(val_ids):
            case_rows.append({
                "case_id": case, "cost_defer": cost, "train_stop_policy": stop_name,
                "voi_ask_target": float(yva["ASK"][i]), "voi_consult_target": float(yva["CONSULT"][i]),
                "voi_ask_pred": float(val_pred["ASK"][i]), "voi_consult_pred": float(val_pred["CONSULT"][i]),
                "voi_ask_bootstrap_spread": float(val_uncertainty["ASK"][i]), "voi_consult_bootstrap_spread": float(val_uncertainty["CONSULT"][i]),
                "selected_branch": best["decisions"][i], "selected_utility": float(best["utilities"][i]),
                "stop_policy_utility": float(stop_values[i]),
            })
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "voi_validation_diagnostics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0])); writer.writeheader(); writer.writerows(all_rows)
    with (args.output_dir / "voi_case_level_validation.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(case_rows[0])); writer.writeheader(); writer.writerows(case_rows)
    payload = {
        "status": "validation_only_no_search_integration",
        "rollout_manifest_sha256": sha_file(args.manifest), "rollouts_sha256": sha_file(args.rollouts),
        "split": {"train_cases": 100, "validation_cases": 50, "test_opened": False},
        "target": "single-rollout observed utility of ASK/CONSULT decision continuation minus train-selected static stop action; includes all action costs once",
        "continuation_policy": "terminal-answer-or-defer-v1",
        "features": FEATURES,
        "features_include_gold_expert_answer_future_reply_or_acquired_information": False,
        "bootstrap_spread_note": "Train-case bootstrap variation; not a calibrated confidence interval.",
        "cost_grid": COSTS, "models_by_cost": model_locks,
        "phase2_patient_simulator_limits_must_be_reported": True,
        "atpo_search_modified": False, "ppo_rl_started": False,
    }
    (args.output_dir / "voi_estimator_lock.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "train_cases": 100, "validation_cases": 50, "test_opened": False, "search_integrated": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
