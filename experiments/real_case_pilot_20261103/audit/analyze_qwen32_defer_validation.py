#!/usr/bin/env python3
"""Analyze the frozen test cohort after the validation-only policy lock."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
RUN = HERE / "outputs/qwen32_defer_validation_20261119_attempt01"
ROLLOUTS = RUN / "paired_rollouts"
ROUTER = RUN / "router"
POLICY_PATH = ROUTER / "policy_lock.json"
OUT = RUN / "analysis_v4"
COSTS = [0.0, 0.05, 0.10, 0.20, 0.40, 0.60]
BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 20261121
RANDOM_POLICY_SEED = 20261122
sys.path.insert(0, str(HERE))
import fit_qwen32_public_feature_router as fit  # noqa: E402


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def ci(values: np.ndarray, indices: np.ndarray) -> tuple[float, float]:
    boot = values[indices].mean(axis=1)
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def ratio_ci(numerator: np.ndarray, denominator: np.ndarray, indices: np.ndarray) -> tuple[float, float]:
    nums = numerator[indices].sum(axis=1)
    dens = denominator[indices].sum(axis=1)
    values = np.divide(nums, dens, out=np.full_like(nums, np.nan, dtype=float), where=dens > 0)
    return float(np.nanquantile(values, 0.025)), float(np.nanquantile(values, 0.975))


def auc_binary(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(bool)
    pos, neg = scores[labels], scores[~labels]
    if not len(pos) or not len(neg):
        return float("nan")
    sorted_scores = np.sort(scores)
    ranks = np.searchsorted(sorted_scores, scores, side="left") + (np.searchsorted(sorted_scores, scores, side="right") - np.searchsorted(sorted_scores, scores, side="left") + 1) / 2.0
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def threshold_calls(scores: np.ndarray, threshold: float) -> np.ndarray:
    return scores >= threshold


def stable_random_calls(rows: list[dict[str, Any]], count: int) -> np.ndarray:
    count = max(0, min(int(count), len(rows)))
    order = sorted(range(len(rows)), key=lambda i: hashlib.sha256(f"{RANDOM_POLICY_SEED}:{rows[i]['case_id']}".encode()).hexdigest())
    calls = np.zeros(len(rows), dtype=bool)
    calls[order[:count]] = True
    return calls


def main() -> None:
    if (OUT / "analysis_manifest.json").exists():
        raise FileExistsError(f"Analysis version is already frozen: {OUT}; increment the analysis directory version to retain it")
    if not POLICY_PATH.is_file():
        raise FileNotFoundError("Validation-only router policy has not been locked")
    if not (ROLLOUTS / "paired_test.jsonl").is_file():
        raise FileNotFoundError("Fresh paired 200-case test output is missing")
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    if policy.get("split_integrity", {}).get("test_split_opened_by_fitter") is not False:
        raise RuntimeError("Router was not locked before independent test analysis")
    cases = read_jsonl(ROLLOUTS / "paired_test.jsonl")
    if len(cases) != 200 or any(r.get("split") != "test" for r in cases):
        raise RuntimeError(f"Expected 200 independent test cases, got {len(cases)}")
    if len({r["case_id"] for r in cases}) != len(cases) or len({r["case_hash"] for r in cases}) != len(cases):
        raise RuntimeError("Duplicate test cases detected")
    base = np.asarray([bool(r["base_correct"]) for r in cases], dtype=bool)
    expert = np.asarray([bool(r["expert_correct"]) for r in cases], dtype=bool)
    delta = expert.astype(float) - base.astype(float)
    base_tokens = np.asarray([int(r.get("base_prompt_tokens", 0)) + int(r.get("base_completion_tokens", 0)) for r in cases], dtype=float)
    expert_tokens = np.asarray([int(r.get("expert_prompt_tokens", 0)) + int(r.get("expert_completion_tokens", 0)) for r in cases], dtype=float)
    base_latency = np.asarray([float(r.get("base_latency_s", 0.0)) for r in cases], dtype=float)
    expert_latency = np.asarray([float(r.get("expert_latency_s", 0.0)) for r in cases], dtype=float)
    indices = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(cases), size=(BOOTSTRAP_N, len(cases)))

    x = np.asarray([fit.feature_row(r, policy["entropy_imputation_median_from_train"], policy["margin_imputation_median_from_train"]) for r in cases], dtype=float)
    mean, scale = np.asarray(policy["feature_mean"]), np.asarray(policy["feature_scale"])
    z = (x - mean) / scale
    predicted_delta = float(policy["target_mean"]) + z @ np.asarray(policy["coefficients"])
    uncertainty = np.asarray([float(r.get("base_answer_entropy_normalized")) if r.get("base_uncertainty_valid") else 1.0 for r in cases], dtype=float)

    methods: dict[str, dict[str, np.ndarray]] = {}
    random_selection_manifest = {}
    for cost in COSTS:
        key = f"{cost:.2f}"
        learned_cfg = policy["learned_thresholds_selected_on_validation"][key]
        uncertainty_cfg = policy["uncertainty_thresholds_selected_on_validation"][key]
        learned_calls = threshold_calls(predicted_delta, float(learned_cfg["threshold"]))
        uncertainty_calls = threshold_calls(uncertainty, float(uncertainty_cfg["threshold"]))
        calls_learned_budget = int(round(float(learned_cfg["validation_call_rate"]) * len(cases)))
        random_calls = stable_random_calls(cases, calls_learned_budget)
        random_selection_manifest[key] = {"target_calls_from_validation_router_budget": calls_learned_budget, "selected_case_ids": [cases[i]["case_id"] for i, selected in enumerate(random_calls) if selected]}
        oracle_calls = delta > cost
        calls_by_policy = {
            "always_answer": np.zeros(len(cases), dtype=bool),
            "always_defer": np.ones(len(cases), dtype=bool),
            "random_defer_budget_matched_realized": random_calls,
            "random_defer_budget_matched_expectation": random_calls,
            "uncertainty_defer": uncertainty_calls,
            "learned_value_router": learned_calls,
            "oracle_selective_defer_upper_bound": oracle_calls,
        }
        for name, calls in calls_by_policy.items():
            result = expert if name == "always_defer" else np.where(calls, expert, base).astype(float)
            if name == "random_defer_budget_matched_expectation":
                # Report both the fixed-seed realized random policy and its
                # exact randomized expectation; do not conflate the two.
                p = calls_learned_budget / len(cases)
                result = (1.0 - p) * base.astype(float) + p * expert.astype(float)
                calls_for_metrics = np.full(len(cases), p, dtype=float)
            else:
                calls_for_metrics = calls.astype(float)
            methods[f"{key}:{name}"] = {"calls": calls_for_metrics, "realized_calls": calls.astype(float), "correct": np.asarray(result, dtype=float), "cost": np.full(len(cases), cost)}

    rows = []
    comparison_rows = []
    baseline = base.astype(float)
    for cost in COSTS:
        key = f"{cost:.2f}"
        learned_vector = methods[f"{key}:learned_value_router"]["correct"] - cost * methods[f"{key}:learned_value_router"]["calls"]
        for name in ("always_answer", "always_defer", "random_defer_budget_matched_realized", "random_defer_budget_matched_expectation", "uncertainty_defer", "learned_value_router", "oracle_selective_defer_upper_bound"):
            m = methods[f"{key}:{name}"]
            correctness = m["correct"]
            calls = m["calls"]
            utility = correctness - cost * calls
            acc_lo, acc_hi = ci(correctness, indices)
            util_lo, util_hi = ci(utility, indices)
            call_lo, call_hi = ci(calls, indices)
            delta_answer = correctness - baseline
            delta_lo, delta_hi = ci(delta_answer, indices)
            # Policies based on Base's answer pay the Base inference for every
            # case; Always DEFER uses only the Expert generation.
            if name == "always_defer":
                tokens = expert_tokens
                latency = expert_latency
            else:
                tokens = base_tokens + calls * expert_tokens
                latency = base_latency + calls * expert_latency
            rows.append({
                "cost": cost, "policy": name, "accuracy": float(np.mean(correctness)), "accuracy_ci_low": acc_lo, "accuracy_ci_high": acc_hi,
                "utility": float(np.mean(utility)), "utility_ci_low": util_lo, "utility_ci_high": util_hi,
                "defer_call_rate": float(np.mean(calls)), "defer_call_rate_ci_low": call_lo, "defer_call_rate_ci_high": call_hi,
                "mean_cost": float(cost * np.mean(calls)), "mean_total_tokens": float(np.mean(tokens)), "mean_total_latency_s": float(np.mean(latency)),
                "paired_accuracy_delta_vs_always_answer": float(np.mean(delta_answer)), "paired_delta_ci_low": delta_lo, "paired_delta_ci_high": delta_hi,
                "validation_threshold": (policy["learned_thresholds_selected_on_validation"][key]["threshold"] if name == "learned_value_router" else policy["uncertainty_thresholds_selected_on_validation"][key]["threshold"] if name == "uncertainty_defer" else None),
                "random_matched_validation_call_rate": policy["learned_thresholds_selected_on_validation"][key]["validation_call_rate"] if name == "random_defer_budget_matched" else None,
                "oracle_is_posthoc_nondeployable": name == "oracle_selective_defer_upper_bound",
            })
        # Paired value differences are computed case-wise and bootstrapped.
        for other in ("uncertainty_defer", "random_defer_budget_matched_realized", "random_defer_budget_matched_expectation", "always_answer"):
            other_m = methods[f"{key}:{other}"]
            other_u = other_m["correct"] - cost * other_m["calls"]
            diff = learned_vector - other_u
            lo, hi = ci(diff, indices)
            comparison_rows.append({"cost": cost, "comparison": f"learned_value_router_minus_{other}", "utility_difference": float(np.mean(diff)), "paired_ci_low": lo, "paired_ci_high": hi, "n_cases": len(cases)})
        for method in ("always_defer", "random_defer_budget_matched_realized", "random_defer_budget_matched_expectation", "uncertainty_defer", "oracle_selective_defer_upper_bound"):
            method_m = methods[f"{key}:{method}"]
            method_u = method_m["correct"] - cost * method_m["calls"]
            diff = method_u - methods[f"{key}:always_answer"]["correct"]
            lo, hi = ci(diff, indices)
            comparison_rows.append({"cost": cost, "comparison": f"{method}_minus_always_answer", "utility_difference": float(np.mean(diff)), "paired_ci_low": lo, "paired_ci_high": hi, "n_cases": len(cases)})

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "cost_sensitivity.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (OUT / "paired_policy_comparisons.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison_rows[0])); writer.writeheader(); writer.writerows(comparison_rows)

    # Base/Expert 2x2 table and paired correction/degradation CIs.
    bb = int(np.sum(base & expert)); base_right_exp_wrong = int(np.sum(base & ~expert)); base_wrong_exp_right = int(np.sum(~base & expert)); ww = int(np.sum(~base & ~expert))
    n_base_wrong = int(np.sum(~base)); n_base_right = int(np.sum(base))
    correction = (~base & expert).astype(float); degradation = (base & ~expert).astype(float)
    fix_ci = ratio_ci(correction, (~base).astype(float), indices)
    degradation_ci = ratio_ci(degradation, base.astype(float), indices)
    delta_ci = ci(delta, indices)
    base_ci = ci(base.astype(float), indices); expert_ci = ci(expert.astype(float), indices)
    cells = [
        {"base_correct": True, "expert_correct": True, "count": bb},
        {"base_correct": True, "expert_correct": False, "count": base_right_exp_wrong},
        {"base_correct": False, "expert_correct": True, "count": base_wrong_exp_right},
        {"base_correct": False, "expert_correct": False, "count": ww},
    ]
    with (OUT / "base_expert_2x2.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(cells[0])); writer.writeheader(); writer.writerows(cells)

    # Risk/coverage points use only thresholds frozen from the validation scores.
    risk_rows = []
    for method, score_values, threshold_map in (
        ("learned_value_router", predicted_delta, policy["learned_risk_coverage_thresholds_from_validation"]),
        ("uncertainty_defer", uncertainty, policy["uncertainty_risk_coverage_thresholds_from_validation"]),
    ):
        for threshold in threshold_map:
            calls = score_values >= float(threshold)
            kept = ~calls
            final = np.where(calls, expert, base)
            coverage_ci = ci(kept.astype(float), indices)
            covered_risk_ci = ratio_ci((kept & ~base).astype(float), kept.astype(float), indices)
            final_risk_ci = ci((~final).astype(float), indices)
            kept_risk = float(np.mean(~base[kept])) if np.any(kept) else float("nan")
            final_risk = float(np.mean(~final))
            error_capture = float(np.mean(calls[~base])) if np.any(~base) else float("nan")
            error_capture_ci = ratio_ci((calls & ~base).astype(float), (~base).astype(float), indices)
            fixes = int(np.sum(calls & ~base & expert))
            degradations = int(np.sum(calls & base & ~expert))
            num_defer_correct = (calls & ~base & expert).astype(float)
            err_den = (~base).astype(float)
            fix_rate_ci = ratio_ci(num_defer_correct, err_den, indices)
            deferred_correct_break = (calls & base & ~expert).astype(float)
            base_right = base.astype(float)
            degrade_rate_ci = ratio_ci(deferred_correct_break, base_right, indices)
            risk_rows.append({
                "score_method": method, "threshold_from_validation": float(threshold),
                "base_coverage_non_deferred": float(np.mean(kept)), "base_risk_on_covered": kept_risk,
                "base_coverage_ci_low": coverage_ci[0], "base_coverage_ci_high": coverage_ci[1],
                "base_risk_on_covered_ci_low": covered_risk_ci[0], "base_risk_on_covered_ci_high": covered_risk_ci[1],
                "final_system_accuracy": float(np.mean(final)), "final_system_risk": final_risk,
                "final_system_risk_ci_low": final_risk_ci[0], "final_system_risk_ci_high": final_risk_ci[1],
                "base_error_capture_rate": error_capture, "expert_corrections_on_called_cases": fixes,
                "base_error_capture_rate_ci_low": error_capture_ci[0], "base_error_capture_rate_ci_high": error_capture_ci[1],
                "correct_base_answers_broken_on_called_cases": degradations,
                "correction_rate_among_base_errors_ci_low": fix_rate_ci[0], "correction_rate_among_base_errors_ci_high": fix_rate_ci[1],
                "degradation_rate_among_base_correct_ci_low": degrade_rate_ci[0], "degradation_rate_among_base_correct_ci_high": degrade_rate_ci[1],
                "defer_rate": float(np.mean(calls)),
            })
    with (OUT / "risk_coverage.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(risk_rows[0])); writer.writeheader(); writer.writerows(risk_rows)
    (OUT / "random_policy_selections.json").write_text(json.dumps({"seed": RANDOM_POLICY_SEED, "selection_by_cost": random_selection_manifest}, indent=2) + "\n", encoding="utf-8")

    # Frozen model's ability to predict positive expert advantage on the test.
    actual_fix = (~base & expert).astype(bool)
    actual_positive_delta = delta > 0
    boot_corr = []
    for ix in indices:
        s, y = predicted_delta[ix], delta[ix]
        boot_corr.append(float(np.corrcoef(s, y)[0, 1]) if np.std(s) and np.std(y) else float("nan"))
    boot_auc_fix, boot_auc_delta = [], []
    for ix in indices:
        boot_auc_fix.append(auc_binary(predicted_delta[ix], actual_fix[ix]))
        boot_auc_delta.append(auc_binary(predicted_delta[ix], actual_positive_delta[ix]))
    feature_predictability = {
        "test_n": len(cases), "predicted_advantage_mse": float(np.mean((predicted_delta - delta) ** 2)),
        "predicted_advantage_pearson": float(np.corrcoef(predicted_delta, delta)[0, 1]) if np.std(predicted_delta) and np.std(delta) else None,
        "predicted_advantage_pearson_ci_low": float(np.nanquantile(boot_corr, .025)), "predicted_advantage_pearson_ci_high": float(np.nanquantile(boot_corr, .975)),
        "auc_for_expert_fixing_base_error": auc_binary(predicted_delta, actual_fix),
        "auc_for_expert_fixing_base_error_ci_low": float(np.nanquantile(boot_auc_fix, .025)), "auc_for_expert_fixing_base_error_ci_high": float(np.nanquantile(boot_auc_fix, .975)),
        "auc_for_positive_net_accuracy_advantage": auc_binary(predicted_delta, actual_positive_delta),
        "auc_for_positive_net_accuracy_advantage_ci_low": float(np.nanquantile(boot_auc_delta, .025)), "auc_for_positive_net_accuracy_advantage_ci_high": float(np.nanquantile(boot_auc_delta, .975)),
        "predicted_advantage_mean": float(np.mean(predicted_delta)), "observed_advantage_mean": float(np.mean(delta)),
        "test_used_to_fit_or_choose_threshold": False,
    }
    (OUT / "expert_advantage_predictability.json").write_text(json.dumps(feature_predictability, indent=2, allow_nan=True) + "\n", encoding="utf-8")

    result_manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "test_n": len(cases),
        "test_data_sha256": json.loads((RUN / "case_selection.json").read_text())["source_data"]["test_sha256"],
        "test_case_hashes_sha256": hashlib.sha256("\n".join(r["case_hash"] for r in cases).encode()).hexdigest(),
        "policy_lock_sha256": sha_file(POLICY_PATH), "test_trajectory_sha256": sha_file(ROLLOUTS / "paired_test.jsonl"),
        "analysis_script_sha256": sha_file(Path(__file__).resolve()), "bootstrap_replicates": BOOTSTRAP_N,
        "bootstrap_seed": BOOTSTRAP_SEED, "random_policy_seed": RANDOM_POLICY_SEED,
        "oracle_used_only_for_posthoc_upper_bound": True, "cost_grid": COSTS, "primary_cost": 0.60,
        "outputs": {name: sha_file(OUT / name) for name in ("cost_sensitivity.csv", "paired_policy_comparisons.csv", "base_expert_2x2.csv", "risk_coverage.csv", "expert_advantage_predictability.json", "random_policy_selections.json")},
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }
    (OUT / "analysis_manifest.json").write_text(json.dumps(result_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = {
        "test_n": len(cases), "base_accuracy": float(np.mean(base)), "base_accuracy_ci": base_ci,
        "defer_expert_accuracy": float(np.mean(expert)), "defer_expert_accuracy_ci": expert_ci,
        "base_correct_expert_correct": bb, "base_correct_expert_wrong": base_right_exp_wrong,
        "base_wrong_expert_correct": base_wrong_exp_right, "base_wrong_expert_wrong": ww,
        "expert_correction_rate_given_base_wrong": float(base_wrong_exp_right / n_base_wrong) if n_base_wrong else None,
        "expert_correction_rate_ci": fix_ci, "correct_answer_degradation_rate_given_base_correct": float(base_right_exp_wrong / n_base_right) if n_base_right else None,
        "degradation_rate_ci": degradation_ci, "paired_accuracy_delta_expert_minus_base": float(np.mean(delta)),
        "paired_accuracy_delta_ci": delta_ci, "expert_advantage_predictability": feature_predictability,
        "cost_0_60_learned": next(r for r in rows if r["cost"] == 0.6 and r["policy"] == "learned_value_router"),
        "cost_0_60_uncertainty": next(r for r in rows if r["cost"] == 0.6 and r["policy"] == "uncertainty_defer"),
        "cost_0_60_oracle": next(r for r in rows if r["cost"] == 0.6 and r["policy"] == "oracle_selective_defer_upper_bound"),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "analyzed", "summary": summary, "output": str(OUT)}, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
