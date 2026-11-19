#!/usr/bin/env python3
"""One-shot evaluation of frozen routers on the fresh case-level test cohort."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/expert_sft"
COSTS = [0.0, 0.05, 0.10, 0.20, 0.40, 0.60]
FEATURES = [
    "log_public_chars", "log_public_words", "option_count", "mean_option_chars",
    "question_marks", "digit_count", "clinical_lexicon_count",
    "base_entropy_normalized", "base_probability_margin", "base_uncertainty_missing",
    "base_valid", "base_answer_A", "base_answer_B", "base_answer_C", "base_answer_D",
]


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    if not rows:
        raise ValueError(f"Empty test input: {path}")
    return rows


def features(rows: list[dict[str, Any]], lock: dict[str, Any]) -> np.ndarray:
    schema = lock["feature_schema"]
    med_e = float(schema["missing_entropy_median_train"])
    med_m = float(schema["missing_margin_median_train"])
    out = []
    for row in rows:
        pub = row.get("public_features") or {}
        entropy, margin = row.get("base_entropy_normalized"), row.get("base_probability_margin")
        missing = not (isinstance(entropy, (int, float)) and math.isfinite(float(entropy)) and isinstance(margin, (int, float)) and math.isfinite(float(margin)))
        if missing:
            entropy, margin = med_e, med_m
        predicted = str(row.get("base_predicted") or "").upper()
        vals = [
            *[float(pub.get(k, 0.0) or 0.0) for k in FEATURES[:7]],
            float(entropy), float(margin), float(missing), float(bool(row.get("base_valid"))),
            *[float(predicted == letter) for letter in "ABCD"],
        ]
        if not np.isfinite(vals).all():
            raise ValueError(f"Invalid feature in test case {row.get('case_id')}")
        out.append(vals)
    x = np.asarray(out, dtype=np.float64)
    mean = np.asarray(schema["train_mean"], dtype=np.float64)
    scale = np.asarray(schema["train_scale"], dtype=np.float64)
    if x.shape[1] != len(schema["names"]) or schema["names"] != FEATURES:
        raise ValueError("Frozen feature schema mismatch")
    return (x - mean) / scale


def model_scores(rows: list[dict[str, Any]], x: np.ndarray, lock: dict[str, Any]) -> dict[str, np.ndarray]:
    models = lock["models"]
    scores = {"uncertainty_threshold": np.asarray([float(r["base_entropy_normalized"]) if isinstance(r.get("base_entropy_normalized"), (int, float)) else float(lock["feature_schema"]["missing_entropy_median_train"]) for r in rows], dtype=float)}
    # Multinomial probabilities ordered [-1,0,+1].
    lg = models["logistic_relative_gain"]["state_dict"]
    tx64 = torch.tensor(x, dtype=torch.float64)
    logits = torch.nn.functional.linear(
        tx64,
        torch.tensor(lg["weight"], dtype=torch.float64),
        torch.tensor(lg["bias"], dtype=torch.float64),
    )
    probs = torch.softmax(logits, dim=1).numpy()
    scores["logistic_relative_gain"] = probs[:, 2] - probs[:, 0]
    rg = models["ridge_relative_gain"]
    scores["ridge_relative_gain"] = x @ np.asarray(rg["coefficient"], dtype=float) + float(rg["intercept"])
    mlp = models["mlp_relative_gain"]["state_dict"]
    tx32 = torch.tensor(x, dtype=torch.float32)
    hidden = torch.tanh(torch.nn.functional.linear(
        tx32, torch.tensor(mlp["0.weight"], dtype=torch.float32), torch.tensor(mlp["0.bias"], dtype=torch.float32),
    ))
    scores["mlp_relative_gain"] = torch.nn.functional.linear(
        hidden, torch.tensor(mlp["2.weight"], dtype=torch.float32), torch.tensor(mlp["2.bias"], dtype=torch.float32),
    ).squeeze(1).detach().cpu().numpy().astype(float)
    old_path = ROOT / models["historical_learned_router"]["path"]
    if sha_file(old_path) != models["historical_learned_router"]["sha256"]:
        raise ValueError("Frozen historical Learned Router changed since policy fitting")
    old = json.loads(old_path.read_text(encoding="utf-8"))
    old_x = []
    for row in rows:
        pub = row.get("public_features") or {}
        ent, margin = row.get("base_entropy_normalized"), row.get("base_probability_margin")
        missing = not (isinstance(ent, (int, float)) and isinstance(margin, (int, float)))
        ent = float(ent) if not missing else float(old["entropy_imputation_median_from_train"])
        margin = float(margin) if not missing else float(old["margin_imputation_median_from_train"])
        pred = str(row.get("base_predicted") or "").upper()
        old_x.append([*[float(pub.get(k, 0.0) or 0.0) for k in FEATURES[:7]], ent, margin, float(missing), float(bool(row["base_valid"])), *[float(pred == letter) for letter in "ABCD"]])
    old_x = np.asarray(old_x, dtype=float)
    scores["historical_learned_router"] = float(old["target_mean"]) + ((old_x - np.asarray(old["feature_mean"])) / np.asarray(old["feature_scale"])) @ np.asarray(old["coefficients"])
    return scores


def auc(y: np.ndarray, score: np.ndarray) -> float | None:
    y = y.astype(bool)
    positives, negatives = int(y.sum()), int((~y).sum())
    if not positives or not negatives:
        return None
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and score[order[j]] == score[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return float((ranks[y].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2
        i = j
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    ra, rb = rank(a), rank(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def calibration(predicted: np.ndarray, observed: np.ndarray, bins: int = 10) -> dict[str, Any]:
    order = np.argsort(predicted, kind="mergesort")
    groups = np.array_split(order, min(bins, len(order)))
    rows = []
    weighted_error = 0.0
    for ids in groups:
        if len(ids) == 0:
            continue
        p, y = float(predicted[ids].mean()), float(observed[ids].mean())
        weighted_error += len(ids) / len(predicted) * abs(p - y)
        rows.append({"n": int(len(ids)), "mean_predicted_relative_gain": p, "mean_observed_relative_gain": y})
    return {"expected_gain_calibration_mae": float(weighted_error), "bins": rows}


def bootstrap_diff(diff: np.ndarray, seed: int, reps: int = 5000) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = len(diff)
    means = np.empty(reps, dtype=float)
    # Chunk to keep memory modest.
    for lo in range(0, reps, 250):
        size = min(250, reps - lo)
        ix = rng.integers(0, n, size=(size, n))
        means[lo:lo + size] = diff[ix].mean(axis=1)
    return {"mean": float(diff.mean()), "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))], "bootstrap_replicates": reps, "unit": "case_id"}


def metric_vectors(base: np.ndarray, delta: np.ndarray, calls: np.ndarray, cost: float) -> tuple[np.ndarray, np.ndarray]:
    correctness = base + calls * delta
    utility = correctness - cost * calls
    return correctness, utility


def summarize_policy(name: str, cost: float, base: np.ndarray, expert: np.ndarray, delta: np.ndarray, calls: np.ndarray, expected_random: bool = False) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    if expected_random:
        p = float(np.mean(calls))
        corr = base + p * delta
        util = corr - cost * p
        defer_rate = p
        random_expected = True
    else:
        corr, util = metric_vectors(base, delta, calls, cost)
        defer_rate = float(np.mean(calls))
        random_expected = False
    attempted = calls.astype(bool)
    base_wrong = base == 0
    base_right = base == 1
    helpful = (base_wrong & (expert == 1)).astype(float)
    harmful = (base_right & (expert == 0)).astype(float)
    d = int(attempted.sum())
    summary = {
        "policy": name, "defer_cost": cost, "n": int(len(base)),
        "accuracy": float(np.mean(corr)), "utility": float(np.mean(util)),
        "defer_rate": defer_rate, "expert_calls": d,
        "average_cost": float(cost * defer_rate),
        "helpful_calls": int(np.sum(attempted & (helpful > 0))),
        "harmful_calls": int(np.sum(attempted & (harmful > 0))),
        "correction_rate_among_base_errors": float(np.sum(attempted * helpful) / max(1, int(base_wrong.sum()))),
        "regression_rate_among_base_correct": float(np.sum(attempted * harmful) / max(1, int(base_right.sum()))),
        "helpful_fraction_of_calls": float(np.sum(attempted * helpful) / d) if d else 0.0,
        "harmful_fraction_of_calls": float(np.sum(attempted * harmful) / d) if d else 0.0,
        "random_expected_value": random_expected,
    }
    return summary, corr, util


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--policy-lock", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--selection", type=Path, default=EXP / "data/fresh_test_selection.json")
    args = ap.parse_args()
    opened_marker = args.policy_lock.with_name("test_opened_once.json")
    if opened_marker.exists():
        raise FileExistsError(f"This frozen policy has already opened a test cohort: {opened_marker}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite test evaluation output: {args.output_dir}")
    lock = json.loads(args.policy_lock.read_text(encoding="utf-8"))
    if lock.get("status") != "frozen_train_fit_validation_thresholds_before_fresh_test_inference":
        raise ValueError("Router policy was not frozen before test evaluation")
    # Record the one-shot opening before reading outcomes. If a later step fails,
    # investigate and resume this same run; do not refit using the opened cohort.
    opened_marker.write_text(json.dumps({"test_sha256": sha_file(args.test), "opened_by": str(Path(__file__).resolve()), "status": "test_opened_once"}, indent=2) + "\n", encoding="utf-8")
    rows = read_rows(args.test)
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if selection.get("status") != "frozen_before_sft_checkpoint_selection":
        raise ValueError("Fresh test selection was not frozen before SFT checkpoint selection")
    if len(rows) != 400 or len(selection.get("rows", [])) != 400 or any(r.get("split") != "test" for r in rows):
        raise ValueError("Fresh final test must contain exactly 400 test cases")
    for key in ("case_id", "case_hash", "prompt_hash"):
        values = [str(r.get(key, "")) for r in rows]
        if len(set(values)) != len(rows):
            raise ValueError(f"Duplicate {key} in fresh test")
    selected = {str(r["case_id"]): (str(r["case_hash"]), str(r["prompt_hash"])) for r in selection["rows"]}
    tested = {str(r["case_id"]): (str(r["case_hash"]), str(r["prompt_hash"])) for r in rows}
    if len(selected) != 400 or selected != tested:
        raise ValueError("Final test rows do not exactly match the pre-frozen 400-case selection")
    integ = lock["split_integrity"]
    seen_case = set(integ["train_case_hashes"]) | set(integ["validation_case_hashes"])
    seen_prompt = set(integ["train_prompt_hashes"]) | set(integ["validation_prompt_hashes"])
    if any(r["case_hash"] in seen_case or r["prompt_hash"] in seen_prompt for r in rows):
        raise ValueError("Fresh test overlaps router train/validation case or prompt hashes")
    x = features(rows, lock)
    scores = model_scores(rows, x, lock)
    base = np.asarray([int(bool(r["base_correct"])) for r in rows], dtype=float)
    expert = np.asarray([int(bool(r["expert_correct"])) for r in rows], dtype=float)
    delta = expert - base
    if not np.isin(delta, [-1.0, 0.0, 1.0]).all():
        raise ValueError("Unexpected paired accuracy relative gain")
    rng = np.random.default_rng(20261122)
    n = len(rows)
    metrics, case_records = [], []
    score_diagnostics = {}
    for name, score in scores.items():
        helpful = delta > 0
        harmful = delta < 0
        score_diagnostics[name] = {
            "helpful_auroc": auc(helpful, score),
            "harmful_auroc": auc(harmful, -score),
            "relative_gain_spearman": spearman(score, delta),
            "score_is_calibrated_expected_gain": name != "uncertainty_threshold",
        }
        if name != "uncertainty_threshold":
            score_diagnostics[name]["relative_gain_mae"] = float(np.mean(np.abs(score - delta)))
            score_diagnostics[name]["relative_gain_mse"] = float(np.mean((score - delta) ** 2))
            score_diagnostics[name].update(calibration(score, delta))

    policies = ["always_answer", "always_defer", *scores.keys(), "oracle_selective_defer"]
    for ci, cost in enumerate(COSTS):
        calls_by_policy: dict[str, np.ndarray] = {"always_answer": np.zeros(n, dtype=bool), "always_defer": np.ones(n, dtype=bool)}
        for name in scores:
            if name == "uncertainty_threshold":
                policy_key = "uncertainty_threshold"
            else:
                policy_key = name
            threshold = float(lock["thresholds_selected_on_validation"][policy_key][f"{cost:.2f}"]["threshold"])
            calls_by_policy[name] = scores[name] >= threshold
        calls_by_policy["oracle_selective_defer"] = delta > cost
        # Random policies are matched to validation call rates and draw without
        # using any test feature or label.
        for name in ("uncertainty_threshold", "logistic_relative_gain", "ridge_relative_gain", "mlp_relative_gain", "historical_learned_router"):
            p = float(lock["thresholds_selected_on_validation"][name][f"{cost:.2f}"]["validation_defer_rate"])
            count = int(round(p * n))
            chosen = np.zeros(n, dtype=bool)
            chosen[rng.permutation(n)[:count]] = True
            calls_by_policy[f"random_matched_{name}"] = chosen
        for name, calls in calls_by_policy.items():
            summary, correctness, utility = summarize_policy(name, cost, base, expert, delta, calls, expected_random=False)
            aa_correct, aa_utility = metric_vectors(base, delta, np.zeros(n), cost)
            summary["accuracy_difference_vs_always_answer_ci95"] = bootstrap_diff(correctness - aa_correct, seed=SEED_FOR_COST(ci, 0))["ci95"]
            summary["accuracy_difference_vs_always_answer"] = float(np.mean(correctness - aa_correct))
            summary["utility_difference_vs_always_answer_ci95"] = bootstrap_diff(utility - aa_utility, seed=SEED_FOR_COST(ci, 1))["ci95"]
            summary["utility_difference_vs_always_answer"] = float(np.mean(utility - aa_utility))
            metrics.append(summary)
            if name in ("always_answer", "always_defer", "oracle_selective_defer", "uncertainty_threshold", "logistic_relative_gain", "ridge_relative_gain", "mlp_relative_gain", "historical_learned_router"):
                for i, row in enumerate(rows):
                    case_records.append({"case_id": row["case_id"], "case_hash": row["case_hash"], "defer_cost": cost, "policy": name, "base_correct": int(base[i]), "expert_correct": int(expert[i]), "relative_gain": int(delta[i]), "decision_defer": int(calls[i]), "final_correctness": float(correctness[i]), "utility": float(utility[i]), "score": float(scores[name][i]) if name in scores else None})

    # The random expected-value baselines are deterministic conditional expectations,
    # reported separately from one fixed-seed randomized realization above.
    for cost in COSTS:
        for name in ("uncertainty_threshold", "logistic_relative_gain", "ridge_relative_gain", "mlp_relative_gain", "historical_learned_router"):
            p = float(lock["thresholds_selected_on_validation"][name][f"{cost:.2f}"]["validation_defer_rate"])
            exp_calls = np.full(n, p, dtype=float)
            expected_correct = base + exp_calls * delta
            expected_utility = expected_correct - cost * exp_calls
            row = {
                "policy": f"random_expected_matched_{name}", "defer_cost": cost, "n": n,
                "accuracy": float(expected_correct.mean()), "utility": float(expected_utility.mean()),
                "defer_rate": p, "expert_calls": p * n, "average_cost": cost * p,
                "helpful_calls": p * int(np.sum(delta == 1)), "harmful_calls": p * int(np.sum(delta == -1)),
                "correction_rate_among_base_errors": p * float(np.sum((base == 0) & (expert == 1))) / max(1, int(np.sum(base == 0))),
                "regression_rate_among_base_correct": p * float(np.sum((base == 1) & (expert == 0))) / max(1, int(np.sum(base == 1))),
                "helpful_fraction_of_calls": float(np.mean(delta == 1)), "harmful_fraction_of_calls": float(np.mean(delta == -1)),
                "random_expected_value": True,
                "accuracy_difference_vs_always_answer": float(expected_correct.mean() - base.mean()),
                "utility_difference_vs_always_answer": float(expected_utility.mean() - base.mean()),
                "accuracy_difference_vs_always_answer_ci95": bootstrap_diff(expected_correct - base, seed=SEED_FOR_COST(int(cost * 100), 2))["ci95"],
                "utility_difference_vs_always_answer_ci95": bootstrap_diff(expected_utility - base, seed=SEED_FOR_COST(int(cost * 100), 3))["ci95"],
            }
            metrics.append(row)

    aa = float(base.mean())
    helpful_count, harmful_count = int(np.sum(delta == 1)), int(np.sum(delta == -1))
    metrics_meta = {
        "status": "completed_one_shot_fresh_case_test",
        "n": n, "base_accuracy": aa, "expert_accuracy": float(expert.mean()),
        "relative_gain": float(delta.mean()), "helpful_count": helpful_count, "harmful_count": harmful_count,
        "both_correct": int(np.sum((base == 1) & (expert == 1))), "both_wrong": int(np.sum((base == 0) & (expert == 0))),
        "expert_correction_rate": helpful_count / max(1, int(np.sum(base == 0))),
        "base_correct_regression_rate": harmful_count / max(1, int(np.sum(base == 1))),
        "base_expert_paired_accuracy_difference_ci95": bootstrap_diff(delta, seed=20261123),
        "model_score_diagnostics": score_diagnostics,
        "cost_grid": COSTS, "utility": "accuracy - defer_cost * defer_rate", "cost_deducted_once": True,
        "oracle_is_posthoc_upper_bound_only": True,
        "gold_expert_correctness_oracle_not_used_as_router_input": True,
        "router_train_validation_case_leakage": False,
        "test_selection_sha256": None,
        "test_predictions_sha256": None,
        "policy_lock_sha256": sha_file(args.policy_lock),
        "selection_manifest_sha256": sha_file(args.selection),
        "test_feature_distribution": {
            name: {"train_mean": float(lock["feature_schema"]["train_mean"][i]), "train_sd": float(lock["feature_schema"]["train_scale"][i]), "test_mean": float((x * np.asarray(lock["feature_schema"]["train_scale"]) + np.asarray(lock["feature_schema"]["train_mean"]))[:, i].mean()), "test_sd": float((x * np.asarray(lock["feature_schema"]["train_scale"]) + np.asarray(lock["feature_schema"]["train_mean"]))[:, i].std())}
            for i, name in enumerate(FEATURES)
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "action_value_analysis.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0]))
        writer.writeheader(); writer.writerows(metrics)
    with (args.output_dir / "case_level_router_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(case_records[0]))
        writer.writeheader(); writer.writerows(case_records)
    test_hash = sha_file(args.test)
    metrics_meta["test_selection_sha256"] = test_hash
    metrics_meta["test_case_results_sha256"] = sha_file(args.output_dir / "case_level_router_results.csv")
    metrics_meta["test_action_values_sha256"] = sha_file(args.output_dir / "action_value_analysis.csv")
    (args.output_dir / "test_summary.json").write_text(json.dumps(metrics_meta, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": metrics_meta["status"], "n": n, "base_accuracy": aa, "expert_accuracy": float(expert.mean()), "relative_gain": float(delta.mean()), "helpful": helpful_count, "harmful": harmful_count, "output_dir": str(args.output_dir)}, ensure_ascii=False))


def SEED_FOR_COST(cost_index: int, stream: int) -> int:
    return 20261124 + cost_index * 11 + stream


if __name__ == "__main__":
    main()
