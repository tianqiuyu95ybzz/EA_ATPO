#!/usr/bin/env python3
"""Rebuild leakage-separated paired records and fit CPU initial-value baselines."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from initial_value_estimator import (
    InitialDeferralValueEstimator,
    LogisticBinary,
    RidgeValue,
    TrainPreprocessor,
)

ROOT = Path(__file__).resolve().parents[3]
ATTEMPT = ROOT / "experiments/real_case_pilot_20261103/audit/outputs/qwen32_router_generalization_20261118_attempt04"
SOURCE = ATTEMPT / "case_selection.json"
PAIR_DIR = ATTEMPT / "paired_rollouts"
OLD_ANALYSIS = ROOT / "experiments/real_case_pilot_20261103/audit/relative_gain_predictability_20261118"
OUT_DEFAULT = ROOT / "experiments/vi_ea_atpo/phase1"
FEATURE_NAMES = (
    "log_public_chars",
    "log_public_words",
    "question_marks",
    "digit_count",
    "clinical_lexicon_count",
    "base_confidence",
    "base_entropy_normalized",
    "base_top1_top2_margin",
    "base_uncertainty_missing",
    "base_valid",
    "log_base_prompt_tokens",
    "log_base_completion_tokens",
)
COSTS = (0.00, 0.05, 0.10, 0.20, 0.40, 0.60)


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    positives = int(y.sum())
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    sorted_score = score[order]
    start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and sorted_score[end] == sorted_score[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return float((ranks[y == 1].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))


def ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    result = np.empty(len(x), dtype=float)
    sx = x[order]
    start = 0
    while start < len(x):
        end = start + 1
        while end < len(x) and sx[end] == sx[start]:
            end += 1
        result[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return result


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = ranks(np.asarray(x, float)), ranks(np.asarray(y, float))
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-8, 1.0 - 1e-8)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2))


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for i in range(bins):
        mask = (p >= edges[i]) & (p < edges[i + 1] if i + 1 < bins else p <= edges[i + 1])
        if mask.any():
            total += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return total


def bootstrap_ci(values: np.ndarray, seed: int, repeats: int = 2500) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(repeats, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def feature_vector(item: dict[str, Any]) -> dict[str, float]:
    public = item.get("public_features") or {}
    probs = item.get("base_answer_option_probabilities") or {}
    p = np.array([float(probs.get(letter, 0.0)) for letter in "ABCD"], dtype=float)
    confidence = float(np.max(p)) if item.get("base_uncertainty_valid") and len(p) else float("nan")

    def optional_float(value: Any) -> float:
        return float(value) if value is not None else float("nan")

    def optional_log1p(value: Any) -> float:
        return math.log1p(max(0.0, float(value))) if value is not None else float("nan")

    return {
        "log_public_chars": optional_float(public.get("log_public_chars")),
        "log_public_words": optional_float(public.get("log_public_words")),
        "question_marks": optional_float(public.get("question_marks")),
        "digit_count": optional_float(public.get("digit_count")),
        "clinical_lexicon_count": optional_float(public.get("clinical_lexicon_count")),
        "base_confidence": confidence,
        "base_entropy_normalized": optional_float(item.get("base_answer_entropy_normalized")),
        "base_top1_top2_margin": optional_float(item.get("base_answer_probability_margin")),
        "base_uncertainty_missing": float(not bool(item.get("base_uncertainty_valid"))),
        "base_valid": float(bool(item.get("base_valid"))),
        "log_base_prompt_tokens": optional_log1p(item.get("base_prompt_tokens")),
        "log_base_completion_tokens": optional_log1p(item.get("base_completion_tokens")),
    }


def source_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    raw = row.get("prompt")
    if raw is None:
        raw = row.get("extra_info", {}).get("prompt")
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if not isinstance(raw, list):
        raise ValueError("source row has no prompt message list")
    return [{"role": str(x.get("role", "user")), "content": str(x.get("content", ""))}
            for x in raw if str(x.get("role", "user")) != "system"]


def gold_letter(row: dict[str, Any]) -> str:
    value = row.get("ground_truth") or row.get("reward_model", {}).get("ground_truth")
    if isinstance(value, dict):
        value = value.get("answer", "")
    return str(value).strip().upper()


def recompute_correct(parsed: Any, valid: Any, gold: str) -> bool:
    answer = str(parsed).strip().upper() if parsed is not None else ""
    return bool(valid) and answer in {"A", "B", "C", "D"} and answer == gold


def load_records() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selection = json.loads(SOURCE.read_text(encoding="utf-8"))
    data_path = Path(selection["source_data"]["train_path"])
    frame = pd.read_parquet(data_path)
    if len(frame) != int(selection["source_data"]["train_rows"]):
        raise ValueError("source train Parquet row count changed")
    if selection["source_data"].get("train_sha256") and sha_file(data_path) != selection["source_data"]["train_sha256"]:
        raise ValueError("source train Parquet SHA256 differs from the frozen selection manifest")
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_prompt_hashes: set[str] = set()
    seen_source_indices: set[int] = set()
    for split in ("train", "validation", "test"):
        path = PAIR_DIR / f"paired_{split}.jsonl"
        selected = {str(row["case_id"]): row for row in selection["rows"][split]}
        split_count = 0
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                item = json.loads(line)
                original_id = str(item["case_id"])
                if original_id not in selected or item["split"] != split:
                    raise ValueError(f"paired case/split mismatch in {split}")
                selected_row = selected[original_id]
                source_ix = int(selected_row["source_row_index"])
                if str(item.get("case_hash")) != str(selected_row.get("case_hash")):
                    raise ValueError(f"paired case hash does not match frozen selection for {original_id}")
                if str(item.get("prompt_hash")) != str(selected_row.get("prompt_hash")):
                    raise ValueError(f"paired prompt hash does not match frozen selection for {original_id}")
                if source_ix < 0 or source_ix >= len(frame):
                    raise ValueError(f"invalid source row index: {source_ix}")
                case_id = str(item["case_hash"])
                if case_id in seen_ids:
                    raise ValueError("duplicate case hash across splits")
                if str(item["prompt_hash"]) in seen_prompt_hashes:
                    raise ValueError("duplicate public prompt hash across splits")
                if source_ix in seen_source_indices:
                    raise ValueError("duplicate source Parquet row across splits")
                seen_ids.add(case_id)
                seen_prompt_hashes.add(str(item["prompt_hash"]))
                seen_source_indices.add(source_ix)
                row = frame.iloc[source_ix].to_dict()
                gold = gold_letter(row)
                if gold not in {"A", "B", "C", "D"}:
                    raise ValueError(f"invalid shared Gold answer at source row {source_ix}: {gold!r}")
                base_correct = recompute_correct(item.get("base_predicted"), item.get("base_valid"), gold)
                expert_correct = recompute_correct(item.get("expert_predicted"), item.get("expert_valid"), gold)
                if base_correct != bool(item.get("base_correct")):
                    raise ValueError(f"Base correctness mismatch against shared Gold at {original_id}")
                if expert_correct != bool(item.get("expert_correct")):
                    raise ValueError(f"Expert correctness mismatch against shared Gold at {original_id}")
                public = source_messages(row)
                features = feature_vector(item)
                labels = {
                    "case_id": case_id,
                    "split": split,
                    "base_output": item.get("base_predicted"),
                    "expert_output": item.get("expert_predicted"),
                    "base_correct": int(base_correct),
                    "expert_correct": int(expert_correct),
                    "relative_gain": int(expert_correct) - int(base_correct),
                    "helpful": int(expert_correct and not base_correct),
                    "harmful": int(base_correct and not expert_correct),
                    "base_error": int(not base_correct),
                    "base_parse_status": "valid" if item.get("base_valid") else "parse_failure",
                    "expert_parse_status": "valid" if item.get("expert_valid") else "parse_failure",
                    "expert_parse_failure": int(not bool(item.get("expert_valid"))),
                }
                records.append({
                    "case_id": case_id,
                    "split": split,
                    "prompt": public,
                    "base_output": item.get("base_predicted"),
                    "observable_features": features,
                    "labels": labels,
                    "prompt_hash": item["prompt_hash"],
                })
                split_count += 1
        if split_count != len(selected):
            raise ValueError(f"paired row count mismatch for {split}: {split_count} vs {len(selected)}")
    expected = {"train": 1000, "validation": 300, "test": 400}
    counts = {split: sum(r["split"] == split for r in records) for split in expected}
    if counts != expected:
        raise ValueError(f"unexpected paired split counts: {counts}")
    return records, {"selection": selection, "data_path": str(data_path), "counts": counts}


def validation_threshold(score: np.ndarray, base: np.ndarray, expert: np.ndarray, cost: float) -> tuple[float, float, float]:
    candidates = [float("inf"), *sorted(set(float(x) for x in score), reverse=True)]
    best: tuple[float, float, float] | None = None
    for threshold in candidates:
        calls = score >= threshold
        utility = float(np.mean(np.where(calls, expert, base) - cost * calls))
        rate = float(calls.mean())
        candidate = (utility, -rate, threshold)
        if best is None or candidate > (best[0], -best[1], best[2]):
            best = (utility, rate, threshold)
    assert best is not None
    return best[2], best[0], best[1]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--bootstrap-repeats", type=int, default=2500)
    parser.add_argument("--model-bootstrap-repeats", type=int, default=24)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    generated_names = {
        "observable_cases.jsonl", "offline_labels.csv", "observable_features.csv",
        "feature_audit.csv", "initial_value_predictions.csv", "initial_value_analysis.csv",
        "risk_calibration.csv", "cost_policy_metrics.csv", "initial_value_estimator.json",
        "initial_value_metrics.json", "INITIAL_VALUE_REPORT.md",
    }
    existing_outputs = [out / name for name in generated_names if (out / name).exists()]
    if existing_outputs:
        raise FileExistsError(
            "refusing to overwrite generated Phase 1 outputs: "
            + ", ".join(str(path) for path in existing_outputs)
        )
    out.mkdir(parents=True, exist_ok=True)

    records, source_meta = load_records()
    observable_path = out / "observable_cases.jsonl"
    labels_path = out / "offline_labels.csv"
    with observable_path.open("w", encoding="utf-8") as stream:
        for r in records:
            payload = {
                "case_id": r["case_id"],
                "split": r["split"],
                "prompt": r["prompt"],
                "base_output": r["base_output"],
                "observable_features": r["observable_features"],
                "base_parse_status": r["labels"]["base_parse_status"],
                "prompt_hash": r["prompt_hash"],
            }
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
    label_fields = [
        "case_id", "split", "base_output", "expert_output", "base_correct",
        "expert_correct", "relative_gain", "helpful", "harmful", "base_error",
        "base_parse_status", "expert_parse_status", "expert_parse_failure",
    ]
    write_csv(labels_path, [r["labels"] for r in records], label_fields)
    feature_rows = []
    for r in records:
        feature_rows.append({"case_id": r["case_id"], "split": r["split"], **r["observable_features"],
                             "option_count": "", "mean_option_chars": ""})
    write_csv(out / "observable_features.csv", feature_rows)
    feature_audit = OLD_ANALYSIS / "feature_audit.csv"
    (out / "feature_audit.csv").write_bytes(feature_audit.read_bytes())

    split_ids = {split: [i for i, r in enumerate(records) if r["split"] == split]
                 for split in ("train", "validation", "test")}
    raw_x = np.array([[r["observable_features"][name] for name in FEATURE_NAMES] for r in records], dtype=float)
    y_help = np.array([r["labels"]["helpful"] for r in records], dtype=int)
    y_harm = np.array([r["labels"]["harmful"] for r in records], dtype=int)
    y_error = np.array([r["labels"]["base_error"] for r in records], dtype=int)
    y_gain = np.array([r["labels"]["relative_gain"] for r in records], dtype=float)
    y_base = np.array([r["labels"]["base_correct"] for r in records], dtype=float)
    y_expert = np.array([r["labels"]["expert_correct"] for r in records], dtype=float)
    pre = TrainPreprocessor().fit(raw_x[split_ids["train"]])
    x = pre.transform(raw_x)
    itr, iva = (split_ids[s] for s in ("train", "validation"))

    logistic_candidates = [(l2, balanced) for l2 in (0.001, 0.01, 0.1, 1.0, 10.0)
                           for balanced in (False, True)]
    best = None
    for l2, balanced in logistic_candidates:
        h = LogisticBinary(l2, balanced).fit(x[itr], y_help[itr])
        d = LogisticBinary(l2, balanced).fit(x[itr], y_harm[itr])
        val_loss = log_loss(y_help[iva], h.predict_proba(x[iva])) + log_loss(y_harm[iva], d.predict_proba(x[iva]))
        candidate = (val_loss, l2, balanced)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    _, selected_l2, selected_balanced = best
    estimator = InitialDeferralValueEstimator(selected_l2, selected_balanced).fit(
        x[itr], y_help[itr], y_harm[itr]
    )
    predictions = estimator.predict(x)

    error_candidates = []
    for l2, balanced in logistic_candidates:
        model = LogisticBinary(l2, balanced).fit(x[itr], y_error[itr])
        val_loss = log_loss(y_error[iva], model.predict_proba(x[iva]))
        error_candidates.append((val_loss, l2, balanced, model))
    _, error_l2, error_balanced, error_model = min(error_candidates, key=lambda row: row[:3])
    base_error_probability = error_model.predict_proba(x)

    ridge_candidates = []
    for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
        model = RidgeValue(alpha).fit(x[itr], y_gain[itr])
        mse = float(np.mean((model.predict(x[iva]) - y_gain[iva]) ** 2))
        ridge_candidates.append((mse, alpha, model))
    _, selected_alpha, ridge = min(ridge_candidates, key=lambda row: row[:2])
    ridge_prediction = ridge.predict(x)

    rng = np.random.default_rng(20261118)
    boot_gain = []
    for _ in range(max(0, args.model_bootstrap_repeats)):
        sample = rng.integers(0, len(itr), size=len(itr))
        boot_estimator = InitialDeferralValueEstimator(selected_l2, selected_balanced).fit(
            x[itr][sample], y_help[itr][sample], y_harm[itr][sample]
        )
        boot_gain.append(boot_estimator.predict(x)["expected_relative_gain"])
    estimation_uncertainty = (
        np.std(np.stack(boot_gain, axis=0), axis=0, ddof=1) if boot_gain
        else np.full(len(records), np.nan)
    )

    prediction_rows = []
    for i, r in enumerate(records):
        prediction_rows.append({
            "case_id": r["case_id"], "split": r["split"],
            "base_uncertainty_score": r["observable_features"]["base_entropy_normalized"],
            "base_error_probability": float(base_error_probability[i]),
            "p_helpful": float(predictions["p_helpful"][i]),
            "p_harmful": float(predictions["p_harmful"][i]),
            "predicted_relative_gain": float(predictions["expected_relative_gain"][i]),
            "ridge_relative_gain": float(ridge_prediction[i]),
            "estimation_uncertainty_bootstrap_sd": float(estimation_uncertainty[i]),
            "base_correct": int(y_base[i]), "expert_correct": int(y_expert[i]),
            "relative_gain": int(y_gain[i]), "helpful": int(y_help[i]),
            "harmful": int(y_harm[i]), "base_error": int(y_error[i]),
        })
    write_csv(out / "initial_value_predictions.csv", prediction_rows)

    analysis_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    policy_scores = {
        "InitialDeferralValueEstimator": predictions["expected_relative_gain"],
        "Base Uncertainty": np.array([r["observable_features"]["base_entropy_normalized"] for r in records], float),
        "Ridge Relative Gain": ridge_prediction,
    }
    for split in ("train", "validation", "test"):
        idx = np.array(split_ids[split], dtype=int)
        targets = {"base_error": y_error[idx], "helpful": y_help[idx], "harmful": y_harm[idx]}
        scores = {
            "Base uncertainty": np.array([r["observable_features"]["base_entropy_normalized"] for r in (records[i] for i in idx)], float),
            "Logistic Base Error": base_error_probability[idx],
            "Logistic Expected Relative Gain": predictions["expected_relative_gain"][idx],
            "Ridge Relative Gain": ridge_prediction[idx],
        }
        for model_name, score in scores.items():
            applicable = {
                "Base uncertainty": ("base_error", "helpful", "harmful"),
                "Logistic Base Error": ("base_error",),
                "Logistic Expected Relative Gain": ("helpful", "harmful"),
                "Ridge Relative Gain": ("helpful", "harmful"),
            }[model_name]
            for target_name in applicable:
                analysis_rows.append({
                    "split": split, "model": model_name, "metric": f"{target_name}_auroc",
                    "value": auc(targets[target_name], score), "n": len(idx),
                })
        gain_score = predictions["expected_relative_gain"][idx]
        for metric, value in (
            ("relative_gain_spearman", spearman(gain_score, y_gain[idx])),
            ("relative_gain_mae", float(np.mean(np.abs(gain_score - y_gain[idx])))),
            ("relative_gain_rmse", float(np.sqrt(np.mean((gain_score - y_gain[idx]) ** 2)))),
            ("helpful_brier", brier(y_help[idx], predictions["p_helpful"][idx])),
            ("helpful_ece_10bin", ece(y_help[idx], predictions["p_helpful"][idx])),
            ("harmful_brier", brier(y_harm[idx], predictions["p_harmful"][idx])),
            ("harmful_ece_10bin", ece(y_harm[idx], predictions["p_harmful"][idx])),
        ):
            analysis_rows.append({"split": split, "model": "InitialDeferralValueEstimator",
                                  "metric": metric, "value": value, "n": len(idx)})
        for target_name, probability, target in (
            ("helpful", predictions["p_helpful"][idx], y_help[idx]),
            ("harmful", predictions["p_harmful"][idx], y_harm[idx]),
        ):
            bin_ids = np.minimum((probability * 10).astype(int), 9)
            for bin_id in range(10):
                mask = bin_ids == bin_id
                if mask.any():
                    calibration_rows.append({
                        "split": split, "model": "InitialDeferralValueEstimator",
                        "target": target_name, "bin": bin_id,
                        "n": int(mask.sum()), "mean_predicted_probability": float(probability[mask].mean()),
                        "observed_rate": float(target[mask].mean()),
                    })

    # Select each policy threshold on validation only, then freeze it for all splits.
    policy_rows = []
    for cost in COSTS:
        thresholds = {}
        for policy, score in policy_scores.items():
            threshold, _, _ = validation_threshold(score[iva], y_base[iva], y_expert[iva], cost)
            thresholds[policy] = threshold
        for split in ("train", "validation", "test"):
            idx = np.array(split_ids[split], dtype=int)
            for policy, score in policy_scores.items():
                calls = score[idx] >= thresholds[policy]
                predicted_correct = np.where(calls, y_expert[idx], y_base[idx])
                utility = float(np.mean(predicted_correct - cost * calls))
                delta = predicted_correct - cost * calls - y_base[idx]
                low, high = bootstrap_ci(delta, 20261118 + int(cost * 1000) + len(idx), args.bootstrap_repeats)
                policy_rows.append({
                    "split": split, "router": policy, "cost_defer": cost,
                    "frozen_validation_threshold": thresholds[policy],
                    "accuracy": float(predicted_correct.mean()), "utility": utility,
                    "defer_rate": float(calls.mean()), "average_cost": float(cost * calls.mean()),
                    "utility_lift_vs_always_answer": float(delta.mean()),
                    "utility_lift_ci95_low": low, "utility_lift_ci95_high": high,
                    "expert_calls": int(calls.sum()),
                })
            if split == "test":
                oracle_calls = y_gain[idx] > cost
                oracle_correct = np.where(oracle_calls, y_expert[idx], y_base[idx])
                policy_rows.append({
                    "split": split, "router": "Hindsight Oracle (upper bound)", "cost_defer": cost,
                    "frozen_validation_threshold": "", "accuracy": float(oracle_correct.mean()),
                    "utility": float(np.mean(oracle_correct - cost * oracle_calls)),
                    "defer_rate": float(oracle_calls.mean()), "average_cost": float(cost * oracle_calls.mean()),
                    "utility_lift_vs_always_answer": float(np.mean(oracle_correct - cost * oracle_calls - y_base[idx])),
                    "utility_lift_ci95_low": "", "utility_lift_ci95_high": "",
                    "expert_calls": int(oracle_calls.sum()),
                })
    analysis_rows.extend({
        "split": r["split"], "model": r["router"], "metric": f"utility_cost_{r['cost_defer']:.2f}",
        "value": r["utility"], "n": len(split_ids[r["split"]]),
    } for r in policy_rows)
    write_csv(out / "initial_value_analysis.csv", analysis_rows,
              ["split", "model", "metric", "value", "n"])
    write_csv(out / "risk_calibration.csv", calibration_rows)
    write_csv(out / "cost_policy_metrics.csv", policy_rows)

    estimator_json = {
        "class": "InitialDeferralValueEstimator",
        "feature_names": FEATURE_NAMES,
        "preprocessor": {"median": pre.median.tolist(), "mean": pre.mean.tolist(), "scale": pre.scale.tolist()},
        "logistic": {
            "l2_selected_on_validation": selected_l2, "balanced_selected_on_validation": selected_balanced,
            "helpful_coefficients": estimator.helpful_model.coef.tolist(),
            "harmful_coefficients": estimator.harmful_model.coef.tolist(),
            "base_error_l2_selected_on_validation": error_l2,
            "base_error_balanced_selected_on_validation": error_balanced,
            "base_error_coefficients": error_model.coef.tolist(),
        },
        "ridge": {"alpha_selected_on_validation": selected_alpha, "coefficients": ridge.coef.tolist()},
        "estimation_uncertainty": "standard deviation across train-only bootstrap refits; approximate epistemic model spread, not a calibrated confidence interval",
    }
    (out / "initial_value_estimator.json").write_text(json.dumps(estimator_json, indent=2), encoding="utf-8")

    metadata = {
        "experiment": "VI-EA-ATPO InitialDeferralValueEstimator offline audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repo_head": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "input_source_run": str(ATTEMPT),
        "input_data_path": source_meta["data_path"],
        "input_data_sha256": sha_file(Path(source_meta["data_path"])),
        "paired_split_counts": source_meta["counts"],
        "paired_test_is_official_test": False,
        "paired_test_previously_examined": True,
        "router_tuning": "train-only preprocessing and fitting; validation-only model/cost threshold selection; test descriptive final scoring only",
        "predecision_features": list(FEATURE_NAMES),
        "option_features": "Unavailable; prior audit found parser fallback zero, excluded from all estimators.",
        "cost_grid": list(COSTS),
        "cost_utility": "mean(deferred? expert_correct : base_correct) - cost_defer * defer_rate",
        "selected_logistic": {"l2": selected_l2, "balanced": selected_balanced},
        "selected_ridge_alpha": selected_alpha,
        "bootstrap_replicates": args.bootstrap_repeats,
        "estimation_uncertainty_replicates": args.model_bootstrap_repeats,
        "no_gpu_or_inference": True,
        "observable_file_sha256": sha_file(observable_path),
        "offline_label_file_sha256": sha_file(labels_path),
    }
    (out / "initial_value_metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(out, metadata, policy_rows, analysis_rows)
    print(json.dumps({"status": "completed", "counts": source_meta["counts"],
                      "selected_logistic": metadata["selected_logistic"],
                      "selected_ridge_alpha": selected_alpha,
                      "output_dir": str(out)}, indent=2))


def write_report(out: Path, metadata: dict[str, Any], policies: list[dict[str, Any]],
                 analysis: list[dict[str, Any]]) -> None:
    def find(split: str, router: str, cost: float) -> dict[str, Any] | None:
        return next((r for r in policies if r["split"] == split and r["router"] == router
                     and float(r["cost_defer"]) == float(cost)), None)
    lines = [
        "# Initial Relative Value Analysis",
        "",
        "## 设计",
        "",
        "- 复用 Attempt04 固定的 1,000/300/400 paired case；没有任何重新推理。",
        "- 分开写入 observable_cases.jsonl 和 offline_labels.csv。Gold 不保存；correctness/relative_gain 标签不进入可见特征文件。",
        "- 12 个决策前字段中不含选项字段；option_count 与 mean_option_chars 在旧解析器中是假零值，因此标为 unavailable 并排除。",
        "- 插补和标准化只用 train 拟合；logistic 与 ridge 超参数、每个成本的阈值只用 validation 选择。",
        "- 400 例 test holdout 来自 ATPO train split，已有旧报告和新 Router 报告查看过。以下 test 指标是污染 holdout 复核，不是独立泛化验证。",
        "",
        "## 结果",
        "",
        "| Router | Cost | Test accuracy | Test utility | Defer rate | Utility lift vs Answer (paired 95% CI) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for router in ("InitialDeferralValueEstimator", "Base Uncertainty", "Ridge Relative Gain"):
        row = find("test", router, 0.60)
        if row:
            lines.append(f"| {router} | 0.60 | {row['accuracy']:.3f} | {row['utility']:.3f} | {row['defer_rate']:.3f} | {row['utility_lift_vs_always_answer']:+.3f} [{row['utility_lift_ci95_low']:+.3f}, {row['utility_lift_ci95_high']:+.3f}] |")
    lines.extend(["", "### Test 主要可预测性指标"])
    metric_names = ["base_error_auroc", "helpful_auroc", "harmful_auroc", "relative_gain_spearman",
                    "relative_gain_mae", "relative_gain_rmse", "helpful_brier", "helpful_ece_10bin",
                    "harmful_brier", "harmful_ece_10bin"]
    for metric_name in metric_names:
        row = next((r for r in analysis if r["split"] == "test" and r["model"] == "InitialDeferralValueEstimator"
                    and r["metric"] == metric_name), None)
        if row:
            lines.append(f"- {metric_name}: {float(row['value']):.4f}")
    lines.extend([
        "",
        "## 解释边界",
        "",
        "- InitialDeferralValueEstimator 输出 P(helpful)、P(harmful) 和两者之差。bootstrap refit SD 是训练样本扰动诊断，不是经过覆盖校准的置信界。",
        "- Utility 的成本是归一化 DEFER 成本，不是美元或临床资源成本。",
        "- 当前数据只涉及固定模型、一次 direct answer 与一次 Expert 最终答复；不包含 ASK/CONSULT 干预，所以不能回答主动获取信息的价值。",
        "- 详情见 initial_value_analysis.csv、risk_calibration.csv、cost_policy_metrics.csv、initial_value_predictions.csv。",
        "",
        "## 文件",
        "",
        "- observable_cases.jsonl：公开 prompt、Base 输出和可见特征。",
        "- offline_labels.csv：专家输出、正确性、相对收益和解析状态；与可见特征分开。",
        "- feature_audit.csv：按 split 的完整特征审计。",
        "- initial_value_estimator.json：冻结参数。",
    ])
    (out / "INITIAL_VALUE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
