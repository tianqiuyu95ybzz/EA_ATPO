#!/usr/bin/env python3
"""Evaluate V2 baselines from paired raw counterfactual outcomes and policy scores."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from . import ACTION_NAMES
from .data import read_jsonl
from .metrics import aggregate_rollouts, regression_calibration, safe_auc, utility_from_raw


DEFAULT_COST_GRID = (0.00, 0.05, 0.10, 0.20, 0.40, 0.60)
ASK_COST = 0.05


def validate_real_run_provenance(trajectories: list[dict[str, Any]], trajectories_path: Path,
                                manifest_path: Path) -> dict[str, Any]:
    """Bind a real evaluation to one immutable model/config/continuation-policy run."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if trajectories_path.resolve().parent != manifest_path.resolve().parent:
        raise ValueError("trajectories and run_manifest must come from the same run output directory")
    config_path_value = manifest.get("config_path")
    expected_config_hash = str(manifest.get("config_sha256", ""))
    if not config_path_value or len(expected_config_hash) != 64:
        raise ValueError("run manifest must include config_path and config_sha256")
    config_path = Path(config_path_value)
    if not config_path.is_file():
        raise ValueError(f"run config referenced by manifest is unavailable: {config_path}")
    config_bytes = config_path.read_bytes()
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    if config_hash != expected_config_hash:
        raise ValueError("run config has changed since the trajectories were collected")
    config = json.loads(config_bytes)
    models = manifest.get("models") or {}
    roles = config.get("roles") or {}
    identities = {}
    for role in ("base", "expert"):
        observed = models.get(role) or {}
        configured = roles.get(role) or {}
        model_id = str(observed.get("model") or "")
        revision = str(observed.get("model_revision") or "")
        if not model_id or not revision:
            raise ValueError(f"run manifest lacks pinned {role} model identity")
        if configured and (str(configured.get("model") or configured.get("model_id") or "") != model_id
                           or str(configured.get("model_revision") or "") != revision):
            raise ValueError(f"run manifest and config disagree on {role} model/revision")
        identities[role] = {"model": model_id, "model_revision": revision}

    policy = config.get("policy") or {}
    continuation_policy_id = str(policy.get("continuation_policy_id") or "")
    observed_policy_ids = {str(row.get("continuation_policy_id") or "") for row in trajectories}
    if not continuation_policy_id or observed_policy_ids != {continuation_policy_id}:
        raise ValueError("real evaluation trajectories must use exactly the run manifest continuation_policy_id")
    costs = config.get("costs") or {}
    configured_ask_cost = float(costs.get("ask", -1))
    configured_defer_cost = float(costs.get("defer", -1))
    if abs(configured_ask_cost - ASK_COST) > 1e-9 or abs(configured_defer_cost - 0.60) > 1e-9:
        raise ValueError("run manifest costs do not match the registered V2 evaluation costs")
    cost_errors = []
    for index, row in enumerate(trajectories):
        components = row.get("cost_components") or {}
        expected_ask = configured_ask_cost * int(row.get("ask_count", 0))
        expected_defer = configured_defer_cost * int(row.get("defer_count", 0))
        if abs(float(components.get("ask_cost", -100)) - expected_ask) > 1e-7:
            cost_errors.append(index)
        if abs(float(components.get("defer_cost", -100)) - expected_defer) > 1e-7:
            cost_errors.append(index)
        if row.get("valid") and row.get("return") is not None and row.get("final_accuracy") is not None:
            expected_return = float(row["final_accuracy"]) - expected_ask - expected_defer
            if abs(float(row["return"]) - expected_return) > 1e-7:
                cost_errors.append(index)
    if cost_errors:
        raise ValueError(f"raw trajectory cost/return accounting mismatch in {len(set(cost_errors))} rows")
    return {
        "run_manifest": str(manifest_path.resolve()),
        "config_sha256": config_hash,
        "models": identities,
        "continuation_policy_id": continuation_policy_id,
        "costs": {"ask": configured_ask_cost, "defer_reference": configured_defer_cost},
        "same_run_manifest_applied_to_all_states": True,
    }


def _score_map(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict) or any(name not in value for name in ACTION_NAMES):
        return None
    result = {name: float(value[name]) for name in ACTION_NAMES if value[name] is not None}
    if len(result) != len(ACTION_NAMES) or not all(math.isfinite(x) for x in result.values()):
        return None
    return result


def read_predictions(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    out = {}
    for row in read_jsonl(path):
        key = (str(row.get("case_id", "")), str(row.get("state_id", "")))
        if not all(key) or key in out:
            raise ValueError(f"missing/duplicate policy prediction state: {key}")
        out[key] = row
    if not out:
        raise ValueError("empty policy prediction file")
    return out


def legal_actions(state: dict[str, Any]) -> list[str]:
    available = [action for action in ACTION_NAMES if state["actions"].get(action, {}).get("present")]
    return [action for action in available if state["actions"][action].get("legal", False)]


def action_for(policy: str, prediction: dict[str, Any], legal: list[str], lambda_defer: float,
               uncertainty_threshold: float | None, ask_cost: float) -> str:
    if not legal:
        raise ValueError("state has no legal actions")
    if policy == "always_answer":
        return "ANSWER"
    if policy == "always_defer":
        return "DEFER"
    if policy == "static_l2d":
        q = _score_map(prediction.get("q_gross_values"))
        if q is None:
            raise ValueError("static_l2d requires calibrated q_gross_values")
        return "DEFER" if q["DEFER"] - q["ANSWER"] >= lambda_defer else "ANSWER"
    if policy == "uncertainty_threshold":
        value = prediction.get("uncertainty_score")
        if uncertainty_threshold is None or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError("uncertainty_threshold needs a finite uncertainty_score and a locked threshold")
        return "DEFER" if float(value) >= uncertainty_threshold else "ANSWER"
    if policy == "sft_only":
        action = str(prediction.get("sft_action", "")).upper()
        if action not in ACTION_NAMES:
            raise ValueError("sft_only needs an sft_action prediction in ASK/ANSWER/DEFER")
        return action
    if policy == "q_guided_distillation":
        logits = _score_map(prediction.get("policy_logits"))
        if logits is None:
            raise ValueError("q_guided_distillation needs action-ordered policy_logits")
        temperature = float(prediction.get("policy_temperature", 1.0))
        if temperature <= 0:
            raise ValueError("policy temperature must be positive")
        reference = prediction.get("policy_reference_costs") or {"ASK": 0.05, "ANSWER": 0.0, "DEFER": 0.60}
        adjusted = dict(logits)
        # The actor logits approximate Q_target / T. Shift them by the known cost delta once.
        adjusted["ASK"] += (float(reference["ASK"]) - ask_cost) / temperature
        adjusted["DEFER"] += (float(reference["DEFER"]) - lambda_defer) / temperature
        return max(legal, key=lambda action: adjusted[action])
    raise ValueError(f"unknown policy {policy}")


def available_policies(states: dict[tuple[str, str], dict[str, Any]], predictions: dict[tuple[str, str], dict[str, Any]],
                       uncertainty_threshold: float | None) -> tuple[list[str], dict[str, str]]:
    policies = ["always_answer", "always_defer"]
    missing = {}
    aligned = [predictions[key] for key in states]
    if all(_score_map(row.get("q_gross_values")) is not None for row in aligned):
        policies.append("static_l2d")
    else:
        missing["static_l2d"] = "calibrated Q gross-value predictions are absent"
    has_uncertainty = all(isinstance(row.get("uncertainty_score"), (int, float)) and math.isfinite(float(row["uncertainty_score"])) for row in aligned)
    if has_uncertainty and uncertainty_threshold is not None:
        policies.append("uncertainty_threshold")
    else:
        missing["uncertainty_threshold"] = "provide uncertainty_score and select/lock threshold on validation cases"
    if all(str(row.get("sft_action", "")).upper() in ACTION_NAMES for row in aligned):
        policies.append("sft_only")
    else:
        missing["sft_only"] = "external SFT-only action predictions are absent"
    if all(_score_map(row.get("policy_logits")) is not None for row in aligned):
        policies.append("q_guided_distillation")
    else:
        missing["q_guided_distillation"] = "distilled policy logits are absent"
    return policies, missing


def select_uncertainty_threshold(states, predictions, ask_cost: float, reference_lambda: float) -> float:
    scores = [float(predictions[key]["uncertainty_score"]) for key in states]
    candidates = [float(np.nextafter(max(scores), math.inf)), *sorted(set(scores))]
    best = None
    for threshold in candidates:
        by_case = {}
        for key, state in states.items():
            pred = predictions[key]
            action = "DEFER" if float(pred["uncertainty_score"]) >= threshold else "ANSWER"
            outcomes = state["actions"][action]
            if not outcomes["legal"]:
                continue
            accuracy = outcomes["outcome_accuracy"]
            ask_n, defer_n = outcomes["ask_count"], outcomes["defer_count"]
            utility = utility_from_raw(accuracy, ask_n, defer_n, reference_lambda, ask_cost)
            by_case.setdefault(key[0], []).append(utility)
        score = float(np.mean([np.mean(v) for v in by_case.values()]))
        defer_rate = float(np.mean([float(predictions[key]["uncertainty_score"]) >= threshold for key in states]))
        candidate = (score, -defer_rate, threshold)
        if best is None or candidate > best:
            best = candidate
    return float(best[2])


def case_level_metrics(states, predictions, policies, cost_grid, threshold, ask_cost, bootstrap_seed=20261120):
    summaries, paired, curves, normalized = [], [], [], []
    state_actions_by_policy_cost = {}
    for lambda_defer in cost_grid:
        for policy in policies:
            per_case = {}
            decisions = {}
            for key, state in states.items():
                pred = predictions[key]
                legal = legal_actions(state)
                selected = action_for(policy, pred, legal, lambda_defer, threshold, ask_cost)
                action_outcome = state["actions"].get(selected, {})
                selected_is_legal = bool(action_outcome.get("legal", False))
                accuracy = float(action_outcome["outcome_accuracy"]) if selected_is_legal else 0.0
                valid_rate = float(action_outcome.get("valid_rate", 0.0)) if selected_is_legal else 0.0
                ask_n = float(action_outcome.get("ask_count", 0.0)) if selected_is_legal else 0.0
                defer_n = float(action_outcome.get("defer_count", 0.0)) if selected_is_legal else 0.0
                utility = utility_from_raw(accuracy, ask_n, defer_n, lambda_defer, ask_cost)
                answer_acc = state["actions"].get("ANSWER", {}).get("outcome_accuracy")
                defer_acc = state["actions"].get("DEFER", {}).get("outcome_accuracy")
                correction = selected == "DEFER" and answer_acc is not None and defer_acc is not None and answer_acc <= 0.0 and defer_acc > 0.0
                spoilage = selected == "DEFER" and answer_acc is not None and defer_acc is not None and answer_acc > 0.0 and defer_acc <= 0.0
                decisions[key] = {
                    "action": selected, "action_legal": selected_is_legal, "accuracy": accuracy,
                    "utility": utility, "valid_rate": valid_rate, "ask_count": ask_n, "defer_count": defer_n,
                    "expert_correction": bool(correction), "expert_spoilage": bool(spoilage),
                }
                per_case.setdefault(key[0], []).append(decisions[key])
            state_actions_by_policy_cost[(policy, lambda_defer)] = decisions
            case_means = {}
            for case_id, rows in per_case.items():
                case_means[case_id] = {
                    keyname: float(np.mean([row[keyname] for row in rows]))
                    for keyname in ("accuracy", "utility", "valid_rate", "ask_count", "defer_count", "expert_correction", "expert_spoilage")
                }
            cases = sorted(case_means)
            accuracy_values = [case_means[c]["accuracy"] for c in cases]
            utility_values = [case_means[c]["utility"] for c in cases]
            acc_ci = bootstrap_ci(accuracy_values, seed=bootstrap_seed + int(lambda_defer * 1000) + policies.index(policy))
            util_ci = bootstrap_ci(utility_values, seed=bootstrap_seed + 10000 + int(lambda_defer * 1000) + policies.index(policy))
            selected_defer_n = sum(row["defer_count"] for rows in per_case.values() for row in rows)
            selected_ask_n = sum(row["ask_count"] for rows in per_case.values() for row in rows)
            selected_defer_decisions = sum(row["action"] == "DEFER" for row in decisions.values())
            corrections = sum(row["expert_correction"] for rows in per_case.values() for row in rows)
            spoils = sum(row["expert_spoilage"] for rows in per_case.values() for row in rows)
            summary = {
                "policy": policy, "lambda_defer": lambda_defer, "n_cases": len(cases), "n_states": len(decisions),
                "accuracy": float(np.mean(accuracy_values)), "accuracy_ci95_low": acc_ci[0], "accuracy_ci95_high": acc_ci[1],
                "utility": float(np.mean(utility_values)), "utility_ci95_low": util_ci[0], "utility_ci95_high": util_ci[1],
                "valid_rate": float(np.mean([case_means[c]["valid_rate"] for c in cases])),
                "defer_rate": selected_defer_decisions / max(1, len(decisions)),
                "ask_count_per_state": float(np.mean([case_means[c]["ask_count"] for c in cases])),
                "ask_count_total": float(selected_ask_n),
                "expert_correction_count": int(corrections), "expert_correction_rate_per_state": corrections / max(1, len(decisions)),
                "expert_spoilage_count": int(spoils), "expert_spoilage_rate_per_state": spoils / max(1, len(decisions)),
                "expert_correction_rate_given_defer": corrections / max(1, selected_defer_decisions),
                "expert_spoilage_rate_given_defer": spoils / max(1, selected_defer_decisions),
            }
            summaries.append(summary)
            curves.append({key: summary[key] for key in ("policy", "lambda_defer", "accuracy", "accuracy_ci95_low", "accuracy_ci95_high", "utility", "defer_rate", "ask_count_per_state")})
            for key, decision in decisions.items():
                normalized.append({
                    "case_id": key[0], "state_id": key[1], "policy": policy, "lambda_defer": lambda_defer,
                    "selected_action": decision["action"], "selected_action_legal": decision["action_legal"],
                    "selected_accuracy": decision["accuracy"], "selected_utility": decision["utility"],
                    "raw_action_correctness": {a: states[key]["actions"].get(a, {}).get("outcome_accuracy") for a in ACTION_NAMES},
                    "raw_action_q_target_reference_net": {a: states[key]["actions"].get(a, {}).get("q_target_reference_net") for a in ACTION_NAMES},
                    "raw_action_cost_components": {a: states[key]["actions"].get(a, {}).get("cost_components") for a in ACTION_NAMES},
                    "raw_ask_count": states[key]["actions"].get(decision["action"], {}).get("ask_count"),
                    "raw_defer_count": states[key]["actions"].get(decision["action"], {}).get("defer_count"),
                })
        reference = {row["case_id"]: row["accuracy"] for row in _per_case_from_decisions(states, state_actions_by_policy_cost[("always_answer", lambda_defer)]).values()}
        reference_utility = {row["case_id"]: row["utility"] for row in _per_case_from_decisions(states, state_actions_by_policy_cost[("always_answer", lambda_defer)]).values()}
        for policy in policies:
            decisions = state_actions_by_policy_cost[(policy, lambda_defer)]
            case_values = _per_case_from_decisions(states, decisions)
            diff_acc = [case_values[c]["accuracy"] - reference[c] for c in sorted(reference)]
            diff_util = [case_values[c]["utility"] - reference_utility[c] for c in sorted(reference)]
            acc_ci = bootstrap_ci(diff_acc, seed=bootstrap_seed + 20000 + int(lambda_defer * 1000) + policies.index(policy))
            util_ci = bootstrap_ci(diff_util, seed=bootstrap_seed + 30000 + int(lambda_defer * 1000) + policies.index(policy))
            paired.append({"policy": policy, "reference": "always_answer", "lambda_defer": lambda_defer,
                           "paired_accuracy_delta": float(np.mean(diff_acc)), "accuracy_delta_ci95_low": acc_ci[0],
                           "accuracy_delta_ci95_high": acc_ci[1], "paired_utility_delta": float(np.mean(diff_util)),
                           "utility_delta_ci95_low": util_ci[0], "utility_delta_ci95_high": util_ci[1], "n_cases": len(diff_acc)})
    return summaries, paired, curves, normalized


def _per_case_from_decisions(states, decisions):
    by_case = {}
    for key, decision in decisions.items():
        by_case.setdefault(key[0], []).append(decision)
    return {case: {"case_id": case,
                   "accuracy": float(np.mean([row["accuracy"] for row in rows])),
                   "utility": float(np.mean([row["utility"] for row in rows]))}
            for case, rows in by_case.items()}


def bootstrap_ci(values, *, seed: int, replications: int = 10000):
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        return None, None
    if len(data) == 1:
        return float(data[0]), float(data[0])
    rng = np.random.default_rng(seed)
    boot = np.empty(replications, dtype=np.float64)
    for start in range(0, replications, 250):
        size = min(250, replications - start)
        boot[start:start + size] = data[rng.integers(0, len(data), size=(size, len(data)))].mean(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975], method="linear")
    return float(low), float(high)


def q_diagnostics(states, predictions, reference_costs):
    predicted, targets, actions = [], [], []
    advantage_labels, advantage_scores = [], []
    for key, state in states.items():
        pred = predictions[key]
        qnet = _score_map(pred.get("q_values_reference_net"))
        qgross = _score_map(pred.get("q_gross_values"))
        if qnet:
            for j, action in enumerate(ACTION_NAMES):
                target = state["actions"].get(action, {}).get("q_target_reference_net")
                if target is not None:
                    predicted.append(qnet[action])
                    targets.append(float(target))
                    actions.append(j)
        answer = state["actions"].get("ANSWER", {}).get("outcome_accuracy")
        defer = state["actions"].get("DEFER", {}).get("outcome_accuracy")
        if qgross and answer is not None and defer is not None:
            if answer != defer:
                advantage_labels.append(defer > answer)
                advantage_scores.append(qgross["DEFER"] - qgross["ANSWER"])
    if not predicted:
        return {"q_mae": None, "calibration": None, "advantage_auroc": None,
                "note": "Q diagnostics need Q predictions and at least one valid observed action return"}
    pred = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    action = np.asarray(actions, dtype=int)
    return {
        "q_mae": float(np.mean(np.abs(pred - target))),
        "q_mae_by_action": {name: regression_calibration(pred, target, action)[name]["mae"] for name in ACTION_NAMES},
        "calibration": regression_calibration(pred, target, action),
        "advantage_auroc": safe_auc(np.asarray(advantage_labels, dtype=bool), np.asarray(advantage_scores, dtype=np.float64)) if advantage_labels else None,
        "advantage_auroc_n": len(advantage_labels),
        "advantage_target": "DEFER raw correctness greater than ANSWER raw correctness; ties excluded",
        "q_target_costs": "reference net return rebuilt from final_accuracy and raw cost components",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--case-split", type=Path, default=Path(__file__).resolve().parents[3] / "experiments/ea_atpo_value_router_20261105/artifacts/case_split.json")
    parser.add_argument("--run-manifest", type=Path, help="required for real data; verifies model revisions, continuation policy, and costs")
    parser.add_argument("--split", choices=("validation", "internal_test"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cost-grid", type=float, nargs="+", default=list(DEFAULT_COST_GRID))
    parser.add_argument("--ask-cost", type=float, default=ASK_COST)
    parser.add_argument("--uncertainty-threshold", type=float)
    parser.add_argument("--select-uncertainty-threshold", action="store_true", help="validation only; writes a locked threshold for later test evaluation")
    parser.add_argument("--bootstrap-seed", type=int, default=20261120)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite evaluation directory: {args.output_dir}")
    if args.select_uncertainty_threshold and args.split != "validation":
        raise ValueError("uncertainty threshold selection is allowed on validation only, never internal test")
    if len(set(args.cost_grid)) != len(args.cost_grid) or any(x < 0 for x in args.cost_grid):
        raise ValueError("cost grid must contain unique non-negative lambda_defer values")
    official = json.loads(args.case_split.read_text(encoding="utf-8"))
    if official.get("unit") != "case_id":
        raise ValueError("case split must be case_id based")
    allowed_cases = set(map(str, official[args.split]))
    rollout_rows = read_jsonl(args.trajectories)
    synthetic_flags = {bool(row.get("synthetic_data", False)) for row in rollout_rows}
    if len(synthetic_flags) > 1:
        raise ValueError("cannot mix synthetic and real trajectories in one evaluation")
    is_synthetic = synthetic_flags == {True}
    provenance = None
    if not is_synthetic:
        if args.run_manifest is None:
            raise ValueError("real evaluation requires --run-manifest to verify the same Base/Expert and continuation policy")
        provenance = validate_real_run_provenance(rollout_rows, args.trajectories, args.run_manifest)
    states = aggregate_rollouts(rollout_rows)
    observed_cases = {case for case, _ in states}
    if not observed_cases or not observed_cases.issubset(allowed_cases):
        raise ValueError(f"trajectory contains cases outside official {args.split} split")
    predictions = read_predictions(args.predictions)
    if provenance is not None:
        trained_policy_ids = {
            str(row["training_continuation_policy_id"])
            for row in predictions.values() if row.get("training_continuation_policy_id")
        }
        if trained_policy_ids and trained_policy_ids != {provenance["continuation_policy_id"]}:
            raise ValueError("training Q/Actor and evaluation rollouts use different continuation_policy_id values")
        trained_models = {
            json.dumps(row["training_model_identity"], sort_keys=True)
            for row in predictions.values() if isinstance(row.get("training_model_identity"), dict)
        }
        if trained_models and trained_models != {json.dumps(provenance["models"], sort_keys=True)}:
            raise ValueError("training Q/Actor and evaluation use different pinned Base/Expert model revisions")
    prediction_synthetic = {bool(row.get("synthetic_data", False)) for row in predictions.values()}
    if prediction_synthetic not in ({False}, {True}) or (prediction_synthetic == {True}) != is_synthetic:
        raise ValueError("prediction/trajectory synthetic-data provenance mismatch")
    if set(states) != set(predictions):
        raise ValueError(f"prediction/trajectory state mismatch: missing={len(set(states)-set(predictions))}, extra={len(set(predictions)-set(states))}")
    threshold = args.uncertainty_threshold
    if args.select_uncertainty_threshold:
        if not all(isinstance(row.get("uncertainty_score"), (int, float)) and math.isfinite(float(row["uncertainty_score"])) for row in predictions.values()):
            raise ValueError("cannot tune Uncertainty Threshold: predictions lack finite uncertainty_score values")
        threshold = select_uncertainty_threshold(states, predictions, args.ask_cost, 0.60)
    policies, unavailable = available_policies(states, predictions, threshold)
    summaries, paired, curves, normalized = case_level_metrics(
        states, predictions, policies, args.cost_grid, threshold, args.ask_cost, args.bootstrap_seed
    )
    q_metrics = q_diagnostics(states, predictions, np.asarray([0.05, 0.0, 0.60]))
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "policy_metrics.csv", summaries)
    write_csv(output / "paired_bootstrap_vs_always_answer.csv", paired)
    write_csv(output / "accuracy_cost_curve.csv", curves)
    with (output / "normalized_evaluation.jsonl").open("w", encoding="utf-8") as stream:
        for row in normalized:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    result = {
        "status": "completed", "synthetic_data": is_synthetic,
        "split": args.split, "case_ids": sorted(observed_cases), "n_cases": len(observed_cases), "n_states": len(states),
        "action_order": list(ACTION_NAMES), "policies_evaluated": policies, "policies_unavailable": unavailable,
        "cost_grid_lambda_defer": args.cost_grid, "ask_cost": args.ask_cost,
        "uncertainty_threshold": threshold,
        "uncertainty_threshold_selected_on_validation": bool(args.select_uncertainty_threshold),
        "frozen_holdout_opened": False,
        "metrics": {"accuracy": "macro mean of within-case state accuracy", "utility": "raw correctness - ask_cost * raw ask_count - lambda_defer * raw defer_count; stored net return is not charged again",
                    "expert_correction": "among selected DEFER states where ANSWER correctness is 0 and DEFER correctness is 1",
                    "expert_spoilage": "among selected DEFER states where ANSWER correctness is 1 and DEFER correctness is 0",
                    "bootstrap_unit": "case_id; paired by identical case set"},
        "q_diagnostics": q_metrics,
        "real_run_provenance": provenance,
        "outputs": {"policy_metrics": str((output / "policy_metrics.csv").resolve()),
                    "paired_bootstrap": str((output / "paired_bootstrap_vs_always_answer.csv").resolve()),
                    "accuracy_cost_curve": str((output / "accuracy_cost_curve.csv").resolve()),
                    "normalized_evaluation": str((output / "normalized_evaluation.jsonl").resolve())},
    }
    (output / "evaluation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.select_uncertainty_threshold:
        (output / "uncertainty_threshold_lock.json").write_text(json.dumps({
            "threshold": threshold, "selected_on_split": "validation", "cost_defer": 0.60,
            "case_ids": sorted(observed_cases), "selection_rule": "maximize case-macro utility; tie-break toward lower DEFER rate",
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "split": args.split, "n_cases": len(observed_cases), "n_states": len(states),
                      "policies_evaluated": policies, "policies_unavailable": unavailable,
                      "q_mae": q_metrics["q_mae"], "advantage_auroc": q_metrics["advantage_auroc"],
                      "output_dir": str(output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
