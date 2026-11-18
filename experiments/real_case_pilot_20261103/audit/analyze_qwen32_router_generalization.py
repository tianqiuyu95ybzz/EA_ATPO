#!/usr/bin/env python3
"""One-time Attempt04 test analysis after validation policy lock exists."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE / "outputs/qwen32_router_generalization_20261118_attempt04"
ROLLOUTS = RUN / "paired_rollouts"
ROUTER = RUN / "router_generalization"
POLICY_PATH = ROUTER / "policy_lock.json"
OUT = RUN / "analysis"
COSTS = [0.00, 0.05, 0.10, 0.20, 0.40, 0.60]
BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 20261130
AUC_BOOTSTRAP_N = 2000
RANDOM_POLICY_SEED = 20261129
LEXICAL = ["log_public_chars", "log_public_words", "option_count", "mean_option_chars", "question_marks", "digit_count", "clinical_lexicon_count"]
FEATURES = LEXICAL + ["base_answer_entropy_normalized", "base_answer_probability_margin", "base_uncertainty_missing", "base_valid", "base_answer_A", "base_answer_B", "base_answer_C", "base_answer_D"]
METHODS = ["uncertainty_threshold", "logistic_correction", "logistic_relative_value", "ridge_relative_value", "mlp_relative_value", "historical_learned_router"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    pos, neg = int(labels.sum()), int((~labels).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return float((ranks[labels].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def mean_ci(values: np.ndarray, index: np.ndarray) -> tuple[float, float]:
    means = values[index].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def ratio_ci(num: np.ndarray, den: np.ndarray, index: np.ndarray) -> tuple[float, float]:
    nums, dens = num[index].sum(axis=1), den[index].sum(axis=1)
    vals = np.divide(nums, dens, out=np.full(nums.shape, np.nan, dtype=float), where=dens > 0)
    return float(np.nanquantile(vals, 0.025)), float(np.nanquantile(vals, 0.975))


def bootstrap_auc_ci(scores: np.ndarray, labels: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    n = len(labels); vals = []
    # AUC is undefined in a one-class resample; skip such draws.
    for _ in range(AUC_BOOTSTRAP_N):
        ix = rng.integers(0, n, n)
        value = auc(scores[ix], labels[ix])
        if math.isfinite(value):
            vals.append(value)
    if not vals:
        return float("nan"), float("nan")
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def assert_test_lock(rows: list[dict[str, Any]], policy: dict[str, Any]) -> None:
    if policy.get("status") != "locked_before_test_analysis" or policy.get("split_integrity", {}).get("test_split_opened_by_fitter") is not False:
        raise RuntimeError("The policy is not locked before test analysis")
    if len(rows) != 400 or any(row.get("split") != "test" for row in rows):
        raise RuntimeError(f"Expected the frozen 400-case test set; got {len(rows)} rows")
    for key in ("case_id", "case_hash", "prompt_hash"):
        if len({row[key] for row in rows}) != len(rows):
            raise RuntimeError(f"Duplicate test {key}")


def make_features(rows: list[dict[str, Any]], stats: dict[str, Any]) -> np.ndarray:
    med_e, med_m = float(stats["entropy_median_from_train"]), float(stats["margin_median_from_train"])
    result = []
    for row in rows:
        pub = row.get("public_features") or {}
        entropy, margin = row.get("base_answer_entropy_normalized"), row.get("base_answer_probability_margin")
        missing = not (row.get("base_uncertainty_valid") and isinstance(entropy, (int, float)) and isinstance(margin, (int, float)))
        if missing:
            entropy, margin = med_e, med_m
        predicted = str(row.get("base_predicted") or "").upper()
        result.append([*[float(pub.get(name, 0.0) or 0.0) for name in LEXICAL], float(entropy), float(margin), float(missing), float(bool(row.get("base_valid"))), *[float(predicted == letter) for letter in "ABCD"]])
    return np.asarray(result, dtype=np.float64)


def apply_models(rows: list[dict[str, Any]], raw_x: np.ndarray, policy: dict[str, Any]) -> dict[str, np.ndarray]:
    feat = policy["feature_schema"]
    z = (raw_x - np.asarray(feat["mean"])) / np.asarray(feat["scale"])
    models = policy["models"]
    corr = models["logistic_correction"]
    logits = z @ np.asarray(corr["weight"]) + float(corr["bias"])
    corr_prob = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
    delta_log = models["logistic_relative_value"]["state_dict"]
    logits3 = z @ np.asarray(delta_log["weight"]).T + np.asarray(delta_log["bias"])
    logits3 -= logits3.max(axis=1, keepdims=True)
    probs3 = np.exp(logits3); probs3 /= probs3.sum(axis=1, keepdims=True)
    ridge = models["ridge_relative_value"]
    ridge_delta = float(ridge["intercept"]) + z @ np.asarray(ridge["weight"])
    mlp = models["mlp_relative_value"]["state_dict"]
    hidden = np.maximum(0.0, z @ np.asarray(mlp["0.weight"]).T + np.asarray(mlp["0.bias"]))
    mlp_logits = hidden @ np.asarray(mlp["2.weight"]).T + np.asarray(mlp["2.bias"])
    mlp_logits -= mlp_logits.max(axis=1, keepdims=True)
    mlp_probs = np.exp(mlp_logits); mlp_probs /= mlp_probs.sum(axis=1, keepdims=True)
    historical = json.loads(Path(models["historical_learned_router"]["path"]).read_text(encoding="utf-8"))
    old_feats = []
    for row in rows:
        pub = row.get("public_features") or {}
        valid = bool(row.get("base_uncertainty_valid"))
        e = float(row["base_answer_entropy_normalized"]) if valid else float(historical["entropy_imputation_median_from_train"])
        m = float(row["base_answer_probability_margin"]) if valid else float(historical["margin_imputation_median_from_train"])
        predicted = str(row.get("base_predicted") or "").upper()
        old_feats.append([*[float(pub.get(name, 0.0) or 0.0) for name in LEXICAL], e, m, float(not valid), float(bool(row.get("base_valid"))), *[float(predicted == letter) for letter in "ABCD"]])
    old = np.asarray(old_feats, dtype=np.float64)
    old_z = (old - np.asarray(historical["feature_mean"])) / np.asarray(historical["feature_scale"])
    old_delta = float(historical["target_mean"]) + old_z @ np.asarray(historical["coefficients"])
    uncertainty = np.asarray([float(r["base_answer_entropy_normalized"]) if r.get("base_uncertainty_valid") and isinstance(r.get("base_answer_entropy_normalized"), (int, float)) else 1.0 for r in rows])
    return {
        "uncertainty_threshold": uncertainty,
        "logistic_correction": corr_prob,
        "logistic_relative_value": probs3[:, 2] - probs3[:, 0],
        "ridge_relative_value": ridge_delta,
        "mlp_relative_value": mlp_probs[:, 2] - mlp_probs[:, 0],
        "historical_learned_router": old_delta,
    }


def label_summary(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    base = np.asarray([bool(r["base_correct"]) for r in rows])
    exp = np.asarray([bool(r["expert_correct"]) for r in rows])
    delta = exp.astype(int) - base.astype(int)
    return {"split": name, "n": len(rows), "base_correct": int(base.sum()), "base_accuracy": float(base.mean()), "expert_correct": int(exp.sum()), "expert_accuracy": float(exp.mean()), "base_right_expert_right": int(np.sum(base & exp)), "base_right_expert_wrong": int(np.sum(base & ~exp)), "base_wrong_expert_right": int(np.sum(~base & exp)), "base_wrong_expert_wrong": int(np.sum(~base & ~exp)), "delta_minus_1": int(np.sum(delta == -1)), "delta_0": int(np.sum(delta == 0)), "delta_plus_1": int(np.sum(delta == 1)), "expert_fixes_base_errors": int(np.sum(~base & exp)), "base_error_count": int(np.sum(~base)), "expert_degrades_base_correct": int(np.sum(base & ~exp)), "base_correct_count": int(np.sum(base)), "invalid_base": int(sum(not bool(r.get("base_valid")) for r in rows)), "invalid_expert": int(sum(not bool(r.get("expert_valid")) for r in rows))}


def feature_distribution(train: list[dict[str, Any]], val: list[dict[str, Any]], test: list[dict[str, Any]], stats: dict[str, Any]) -> list[dict[str, Any]]:
    groups = {name: make_features(rows, stats) for name, rows in (("train", train), ("validation", val), ("test", test))}
    rows_out = []
    for j, name in enumerate(FEATURES):
        item = {"feature": name}
        for split, matrix in groups.items():
            item[f"{split}_mean"] = float(matrix[:, j].mean())
            item[f"{split}_sd"] = float(matrix[:, j].std())
        denom = math.sqrt((groups["train"][:, j].var() + groups["test"][:, j].var()) / 2.0)
        item["test_vs_train_standardized_mean_difference"] = float((groups["test"][:, j].mean() - groups["train"][:, j].mean()) / denom) if denom > 1e-12 else 0.0
        rows_out.append(item)
    return rows_out


def stable_random_calls(rows: list[dict[str, Any]], count: int) -> np.ndarray:
    count = max(0, min(int(count), len(rows)))
    order = sorted(range(len(rows)), key=lambda i: hashlib.sha256(f"{RANDOM_POLICY_SEED}:{rows[i]['case_id']}".encode()).hexdigest())
    calls = np.zeros(len(rows), dtype=bool); calls[order[:count]] = True
    return calls


def main() -> None:
    if (OUT / "analysis_manifest.json").exists():
        raise FileExistsError(f"Final test analysis already exists; refusing a second look: {OUT}")
    if not POLICY_PATH.is_file():
        raise FileNotFoundError("Validation policy lock is missing; test data remain unopened")
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    test = read_jsonl(ROLLOUTS / "paired_test.jsonl")
    assert_test_lock(test, policy)
    train, val = read_jsonl(ROLLOUTS / "paired_train.jsonl"), read_jsonl(ROLLOUTS / "paired_validation.jsonl")
    if {r["case_id"] for r in train} & {r["case_id"] for r in test} or {r["case_hash"] for r in val} & {r["case_hash"] for r in test}:
        raise RuntimeError("Train/validation/test case overlap detected")
    x_test = make_features(test, policy["feature_schema"])
    score_map = apply_models(test, x_test, policy)
    base = np.asarray([bool(r["base_correct"]) for r in test], dtype=bool)
    expert = np.asarray([bool(r["expert_correct"]) for r in test], dtype=bool)
    delta = expert.astype(float) - base.astype(float)
    correction_label = (~base & expert).astype(bool)
    positive_delta = (delta > 0)
    n = len(test)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, n, size=(BOOTSTRAP_N, n))
    rng_auc = np.random.default_rng(BOOTSTRAP_SEED + 1)

    summaries = [label_summary("train", train), label_summary("validation", val), label_summary("test", test)]
    feature_rows = feature_distribution(train, val, test, policy["feature_schema"])
    feature_metrics = []
    for name, scores in score_map.items():
        for target_name, target in (("base_error_corrected", correction_label), ("positive_relative_advantage", positive_delta)):
            a = auc(scores, target)
            lo, hi = bootstrap_auc_ci(scores, target, rng_auc)
            feature_metrics.append({"model_or_feature": name, "target": target_name, "test_auroc": a, "auc_ci_low": lo, "auc_ci_high": hi})
    uncert = score_map["uncertainty_threshold"]
    for label_name, label in (("base_error_corrected", correction_label), ("positive_relative_advantage", positive_delta)):
        a = auc(uncert, label); lo, hi = bootstrap_auc_ci(uncert, label, rng_auc)
        feature_metrics.append({"model_or_feature": "base_answer_uncertainty", "target": label_name, "test_auroc": a, "auc_ci_low": lo, "auc_ci_high": hi})

    # Regression errors use each frozen model's predicted relative value.
    delta_scores = {name: score_map[name] for name in ("logistic_relative_value", "ridge_relative_value", "mlp_relative_value", "historical_learned_router")}
    value_metrics = []
    for split_name, rows in (("train", train), ("validation", val), ("test", test)):
        xx = make_features(rows, policy["feature_schema"])
        scores = apply_models(rows, xx, policy)
        target = np.asarray([float(r["expert_correct"]) - float(r["base_correct"]) for r in rows])
        for model_name in delta_scores:
            pred = scores[model_name]
            corr = float(np.corrcoef(pred, target)[0, 1]) if np.std(pred) > 0 and np.std(target) > 0 else float("nan")
            value_metrics.append({"split": split_name, "model": model_name, "mae": float(np.mean(np.abs(pred - target))), "mse": float(np.mean((pred - target) ** 2)), "pearson": corr})

    # Frozen thresholds were optimized only on validation. Cost is subtracted once.
    policy_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    test_base_tokens = np.asarray([int(r.get("base_prompt_tokens", 0)) + int(r.get("base_completion_tokens", 0)) for r in test], dtype=float)
    test_expert_tokens = np.asarray([int(r.get("expert_prompt_tokens", 0)) + int(r.get("expert_completion_tokens", 0)) for r in test], dtype=float)
    test_base_latency = np.asarray([float(r.get("base_latency_s", 0.0)) for r in test], dtype=float)
    test_expert_latency = np.asarray([float(r.get("expert_latency_s", 0.0)) for r in test], dtype=float)
    for cost in COSTS:
        key = f"{cost:.2f}"
        calls_map: dict[str, np.ndarray] = {
            "always_answer": np.zeros(n, dtype=bool),
            "always_defer": np.ones(n, dtype=bool),
            "uncertainty_defer": score_map["uncertainty_threshold"] >= float(policy["thresholds_selected_on_validation"]["uncertainty_threshold"][key]["threshold"]),
            "logistic_correction": score_map["logistic_correction"] >= float(policy["thresholds_selected_on_validation"]["logistic_correction"][key]["threshold"]),
            "logistic_relative_value": score_map["logistic_relative_value"] >= float(policy["thresholds_selected_on_validation"]["logistic_relative_value"][key]["threshold"]),
            "ridge_relative_value": score_map["ridge_relative_value"] >= float(policy["thresholds_selected_on_validation"]["ridge_relative_value"][key]["threshold"]),
            "mlp_relative_value": score_map["mlp_relative_value"] >= float(policy["thresholds_selected_on_validation"]["mlp_relative_value"][key]["threshold"]),
            "historical_learned_router": score_map["historical_learned_router"] >= float(policy["thresholds_selected_on_validation"]["historical_learned_router"][key]["threshold"]),
            "oracle_selective_defer_upper_bound": delta > cost,
        }
        split_val_call_rate = float(policy["thresholds_selected_on_validation"]["ridge_relative_value"][key]["validation_call_rate"])
        matched_n = int(round(split_val_call_rate * n))
        calls_map["random_defer_budget_matched_realized"] = stable_random_calls(test, matched_n)
        calls_map["random_defer_budget_matched_expectation"] = np.full(n, matched_n / n, dtype=float)
        for policy_name, calls_raw in calls_map.items():
            calls = np.asarray(calls_raw, dtype=float)
            realized_bool = calls.astype(bool)
            if policy_name == "always_defer":
                correctness = expert.astype(float)
                tokens, latency = test_expert_tokens, test_expert_latency
            elif policy_name == "random_defer_budget_matched_expectation":
                correctness = (1.0 - calls) * base.astype(float) + calls * expert.astype(float)
                tokens = test_base_tokens + calls * test_expert_tokens
                latency = test_base_latency + calls * test_expert_latency
            else:
                correctness = np.where(realized_bool, expert, base).astype(float)
                tokens = test_base_tokens + calls * test_expert_tokens
                latency = test_base_latency + calls * test_expert_latency
            utility = correctness - cost * calls
            acc_ci = mean_ci(correctness, indices); util_ci = mean_ci(utility, indices); call_ci = mean_ci(calls, indices)
            answer_delta = correctness - base.astype(float)
            fix_weight = calls * (~base & expert).astype(float)
            degrade_weight = calls * (base & ~expert).astype(float)
            selected_wrong_weight = calls * (~base).astype(float)
            selected_right_weight = calls * base.astype(float)
            fix_ci = ratio_ci(fix_weight, (~base).astype(float), indices)
            degradation_ci = ratio_ci(degrade_weight, base.astype(float), indices)
            selected_fix_ci = ratio_ci(fix_weight, selected_wrong_weight, indices)
            selected_degrade_ci = ratio_ci(degrade_weight, selected_right_weight, indices)
            pair_ci = mean_ci(answer_delta, indices)
            row = {
                "cost": cost, "policy": policy_name, "accuracy": float(np.mean(correctness)), "accuracy_ci_low": acc_ci[0], "accuracy_ci_high": acc_ci[1],
                "utility": float(np.mean(utility)), "utility_ci_low": util_ci[0], "utility_ci_high": util_ci[1], "defer_call_rate": float(np.mean(calls)), "defer_call_rate_ci_low": call_ci[0], "defer_call_rate_ci_high": call_ci[1],
                "mean_cost": float(cost * np.mean(calls)), "mean_total_tokens": float(np.mean(tokens)), "mean_total_latency_s": float(np.mean(latency)),
                "random_defer_budget_from_validation": (split_val_call_rate if "random_defer_budget_matched" in policy_name else None),
                "expert_fixes_selected_or_expected": float(fix_weight.sum()), "expert_degradations_selected_or_expected": float(degrade_weight.sum()),
                "expert_correction_rate_given_base_error": float(fix_weight.sum() / max(1, int((~base).sum()))), "correction_rate_ci_low": fix_ci[0], "correction_rate_ci_high": fix_ci[1],
                "degradation_rate_given_base_correct": float(degrade_weight.sum() / max(1, int(base.sum()))), "degradation_rate_ci_low": degradation_ci[0], "degradation_rate_ci_high": degradation_ci[1],
                "fix_rate_among_selected_base_errors": float(fix_weight.sum() / max(1e-12, float(selected_wrong_weight.sum()))), "selected_fix_rate_ci_low": selected_fix_ci[0], "selected_fix_rate_ci_high": selected_fix_ci[1],
                "degradation_rate_among_selected_base_correct": float(degrade_weight.sum() / max(1e-12, float(selected_right_weight.sum()))), "selected_degradation_ci_low": selected_degrade_ci[0], "selected_degradation_ci_high": selected_degrade_ci[1],
                "paired_accuracy_delta_vs_always_answer": float(np.mean(answer_delta)), "paired_delta_ci_low": pair_ci[0], "paired_delta_ci_high": pair_ci[1],
                "validation_threshold": (policy["thresholds_selected_on_validation"].get(policy_name, {}).get(key, {}).get("threshold")),
                "oracle_is_posthoc_nondeployable": policy_name == "oracle_selective_defer_upper_bound",
            }
            policy_rows.append(row)
            if policy_name not in ("always_answer", "always_defer", "oracle_selective_defer_upper_bound"):
                baseline_utility = base.astype(float)
                for comparator in ("always_answer", "uncertainty_defer", "random_defer_budget_matched_realized", "random_defer_budget_matched_expectation"):
                    if comparator == "always_answer":
                        other_u = base.astype(float)
                    elif comparator == "uncertainty_defer":
                        other_calls = calls_map[comparator].astype(float)
                        other_u = np.where(other_calls.astype(bool), expert, base).astype(float) - cost * other_calls
                    elif comparator.endswith("expectation"):
                        other_calls = calls_map[comparator].astype(float)
                        other_u = (1.0 - other_calls) * base.astype(float) + other_calls * expert.astype(float) - cost * other_calls
                    else:
                        other_calls = calls_map[comparator].astype(float)
                        other_u = np.where(other_calls.astype(bool), expert, base).astype(float) - cost * other_calls
                    diff = utility - other_u
                    dci = mean_ci(diff, indices)
                    comparison_rows.append({"cost": cost, "policy": policy_name, "comparator": comparator, "paired_utility_difference": float(diff.mean()), "paired_ci_low": dci[0], "paired_ci_high": dci[1], "n_cases": n})

    for i, row in enumerate(test):
        item = {"case_id": row["case_id"], "case_hash": row["case_hash"], "prompt_hash": row["prompt_hash"], "split": "test", "base_predicted": row.get("base_predicted"), "expert_predicted": row.get("expert_predicted"), "base_valid": bool(row.get("base_valid")), "expert_valid": bool(row.get("expert_valid")), "expert_correct": bool(row["expert_correct"]), "base_correct": bool(row["base_correct"]), "relative_gain": float(delta[i]), "base_uncertainty_valid": bool(row.get("base_uncertainty_valid")), "base_answer_entropy_normalized": row.get("base_answer_entropy_normalized")}
        for model_name, scores in score_map.items():
            item[f"score_{model_name}"] = float(scores[i])
        for cost in COSTS:
            key = f"{cost:.2f}"
            item[f"expert_call_logistic_relative_value_cost_{key}"] = int(score_map["logistic_relative_value"][i] >= float(policy["thresholds_selected_on_validation"]["logistic_relative_value"][key]["threshold"]))
            item[f"expert_call_historical_router_cost_{key}"] = int(score_map["historical_learned_router"][i] >= float(policy["thresholds_selected_on_validation"]["historical_learned_router"][key]["threshold"]))
        case_rows.append(item)

    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "case_level_paired_results.csv", case_rows)
    write_csv(OUT / "policy_cost_sensitivity.csv", policy_rows)
    write_csv(OUT / "paired_policy_comparisons.csv", comparison_rows)
    write_csv(OUT / "train_validation_test_label_distribution.csv", summaries)
    write_csv(OUT / "feature_distribution.csv", feature_rows)
    write_csv(OUT / "router_predictive_diagnostics.csv", feature_metrics)
    write_csv(OUT / "relative_value_prediction_errors.csv", value_metrics)
    paired_expert = []
    for split_name, rows in (("train", train), ("validation", val), ("test", test)):
        b = np.asarray([bool(r["base_correct"]) for r in rows], dtype=float)
        e = np.asarray([bool(r["expert_correct"]) for r in rows], dtype=float)
        split_indices = np.random.default_rng(BOOTSTRAP_SEED + len(rows)).integers(0, len(rows), size=(BOOTSTRAP_N, len(rows)))
        bci, eci = mean_ci(b, split_indices), mean_ci(e, split_indices)
        dci = mean_ci(e - b, split_indices)
        corr_ci = ratio_ci(((b == 0) & (e == 1)).astype(float), (b == 0).astype(float), split_indices)
        deg_ci = ratio_ci(((b == 1) & (e == 0)).astype(float), (b == 1).astype(float), split_indices)
        paired_expert.append({"split": split_name, "n": len(rows), "base_accuracy": float(b.mean()), "base_accuracy_ci_low": bci[0], "base_accuracy_ci_high": bci[1], "expert_accuracy": float(e.mean()), "expert_accuracy_ci_low": eci[0], "expert_accuracy_ci_high": eci[1], "paired_accuracy_delta_expert_minus_base": float((e-b).mean()), "paired_delta_ci_low": dci[0], "paired_delta_ci_high": dci[1], "expert_correction_rate_given_base_error": float(np.sum((b == 0) & (e == 1)) / max(1, np.sum(b == 0))), "correction_ci_low": corr_ci[0], "correction_ci_high": corr_ci[1], "expert_degradation_rate_given_base_correct": float(np.sum((b == 1) & (e == 0)) / max(1, np.sum(b == 1))), "degradation_ci_low": deg_ci[0], "degradation_ci_high": deg_ci[1], "base_right_expert_right": int(np.sum((b == 1) & (e == 1))), "base_right_expert_wrong": int(np.sum((b == 1) & (e == 0))), "base_wrong_expert_right": int(np.sum((b == 0) & (e == 1))), "base_wrong_expert_wrong": int(np.sum((b == 0) & (e == 0)))})
    write_csv(OUT / "paired_expert_summary.csv", paired_expert)
    manifest = {"status": "completed_once", "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "test_rows": len(test), "test_opened_after_policy_lock": True, "policy_sha256": hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest(), "test_sha256": hashlib.sha256((ROLLOUTS / "paired_test.jsonl").read_bytes()).hexdigest(), "bootstrap_replicates_policy_metrics": BOOTSTRAP_N, "bootstrap_replicates_auc": AUC_BOOTSTRAP_N, "bootstrap_seed": BOOTSTRAP_SEED, "primary_router": policy["primary_router"], "costs": COSTS, "case_level_csv": str(OUT / "case_level_paired_results.csv")}
    (OUT / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    primary = [r for r in policy_rows if r["policy"] == "logistic_relative_value" and r["cost"] == 0.60][0]
    pairings = {(r["comparator"], r["cost"]): r for r in comparison_rows if r["policy"] == "logistic_relative_value"}
    comparable = [pairings[(name, 0.60)] for name in ("always_answer", "uncertainty_defer", "random_defer_budget_matched_realized")]
    # Build the lookup before the decision rule uses the primary Router's AUC.
    auc_by_model = {(m["model_or_feature"], m["target"]): m for m in feature_metrics}
    primary_auc = auc_by_model.get(("logistic_relative_value", "positive_relative_advantage"))
    reliable = all(float(r["paired_ci_low"]) > 0.0 for r in comparable) and primary_auc is not None and float(primary_auc["auc_ci_low"]) > 0.5
    decision = "Evidence supports a stable learned conditional advantage at primary cost 0.60; replication is still required before any RL." if reliable else "The primary Learned Router does not show reliable positive utility over every simple comparator at cost 0.60; do not start PPO/ATPO RL."
    report = [
        "# Qwen3-32B-AWQ Router 泛化最终分析报告", "", f"**测试集：** {len(test)} 个本阶段冻结的独立病例。**主 Router：** `logistic_relative_value`。**判定：** {decision}", "",
        "## 配对专家结果", "", "| Split | N | Base 正确 | Expert 正确 | Base错/Expert对 | Base对/Expert错 | Δ=-1/0/+1 |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        report.append(f"| {s['split']} | {s['n']} | {s['base_correct']} ({s['base_accuracy']:.3f}) | {s['expert_correct']} ({s['expert_accuracy']:.3f}) | {s['base_wrong_expert_right']} | {s['base_right_expert_wrong']} | {s['delta_minus_1']}/{s['delta_0']}/{s['delta_plus_1']} |")
    report += ["", "## 主成本 0.60 的策略结果", "", "| 策略 | Accuracy (95% CI) | Utility (95% CI) | DEFER率 | 平均成本 | AUROC (正相对收益) |", "|---|---:|---:|---:|---:|---:|"]
    auc_policy_map = {"uncertainty_defer": "uncertainty_threshold"}
    for r in policy_rows:
        if r["cost"] != 0.60:
            continue
        model_key = auc_policy_map.get(r["policy"], r["policy"])
        metric = auc_by_model.get((model_key, "positive_relative_advantage"))
        auc_text = "—" if metric is None else f"{metric['test_auroc']:.3f} [{metric['auc_ci_low']:.3f}, {metric['auc_ci_high']:.3f}]"
        report.append(f"| {r['policy']} | {r['accuracy']:.3f} [{r['accuracy_ci_low']:.3f}, {r['accuracy_ci_high']:.3f}] | {r['utility']:.3f} [{r['utility_ci_low']:.3f}, {r['utility_ci_high']:.3f}] | {r['defer_call_rate']:.3f} | {r['mean_cost']:.3f} | {auc_text} |")
    test_pair = paired_expert[-1]
    report += ["", "## 独立测试集专家互补性（病例级 95% CI）", "", f"Base accuracy {test_pair['base_accuracy']:.3f} [{test_pair['base_accuracy_ci_low']:.3f}, {test_pair['base_accuracy_ci_high']:.3f}]；Expert accuracy {test_pair['expert_accuracy']:.3f} [{test_pair['expert_accuracy_ci_low']:.3f}, {test_pair['expert_accuracy_ci_high']:.3f}]；配对差 {test_pair['paired_accuracy_delta_expert_minus_base']:.3f} [{test_pair['paired_delta_ci_low']:.3f}, {test_pair['paired_delta_ci_high']:.3f}]。", "", f"2×2：Base对/Expert对 {test_pair['base_right_expert_right']}；Base对/Expert错 {test_pair['base_right_expert_wrong']}；Base错/Expert对 {test_pair['base_wrong_expert_right']}；Base错/Expert错 {test_pair['base_wrong_expert_wrong']}。", "", f"条件纠错率 {test_pair['expert_correction_rate_given_base_error']:.3f} [{test_pair['correction_ci_low']:.3f}, {test_pair['correction_ci_high']:.3f}]；条件退化率 {test_pair['expert_degradation_rate_given_base_correct']:.3f} [{test_pair['degradation_ci_low']:.3f}, {test_pair['degradation_ci_high']:.3f}]。", "", "## 主 Router 配对 Utility 差异（成本 0.60）", "", "| 比较 | 差值 | 病例级 95% CI |", "|---|---:|---:|"]
    for r in comparable:
        report.append(f"| Learned − {r['comparator']} | {r['paired_utility_difference']:.3f} | [{r['paired_ci_low']:.3f}, {r['paired_ci_high']:.3f}] |")
    report += ["", "## 解释与结论", "", f"- 按预先确定的判据，本轮结论：{decision}", "- 纠错率、退化率、AUROC、相对价值 MAE/MSE/Pearson、类别不平衡、特征分布、完整成本网格和所有基线见本目录 CSV。", "- Oracle Selective DEFER 是使用事后配对正确性计算的理论上界，绝不可部署或作为 Router 成绩。", "- 测试标签只在验证阈值和模型锁定后读取一次；本脚本拒绝覆盖已存在的最终分析目录。", "- 即使主 Router 有优势，本阶段也只允许再做新的独立病例复现；本阶段不启动 PPO/ATPO RL。", ""]
    (OUT / "QWEN32_ROUTER_GENERALIZATION_FINAL_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"status": "completed_once", "test_n": len(test), "primary_cost_0_60": primary, "decision": decision, "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
