#!/usr/bin/env python3
"""Re-audit the existing Window A V2 Router results without training or GPU use.

This script reads the frozen counterfactual/Q inputs and saved development
predictions from an existing worktree. It never loads a model or rewrites source
results. Bootstrap draws are case_id clusters; every state and all three fixed
seed predictions for a sampled case move together.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ACTION_ORDER = ("ASK", "ANSWER", "DEFER")
METHODS = ("q_guided_distillation", "mlp_relative")
SEEDS = (17, 42, 2026)
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 20261120
TOLERANCE = 1e-8


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def key_case_state(row: dict[str, Any]) -> tuple[str, str]:
    meta = row.get("metadata") or {}
    case_id = str(row.get("case_id", meta.get("case_id", "")))
    state_id = str(row.get("state_id", meta.get("state_id", "")))
    if not case_id or not state_id:
        raise ValueError("each Q/state row must include case_id and state_id")
    return case_id, state_id


def canonical_state_digest(state: Any) -> str:
    payload = json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def pct(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("empty bootstrap distribution")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def bootstrap_ci(
    case_values: dict[str, float],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> list[float]:
    """Percentile CI over case means; never samples individual states/seeds."""
    cases = sorted(case_values)
    if not cases:
        raise ValueError("no case clusters for bootstrap")
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    n = len(cases)
    for _ in range(replicates):
        draw = rng.choice(cases, size=n, replace=True).tolist()
        samples.append(sum(case_values[c] for c in draw) / n)
    return [pct(samples, 0.025), pct(samples, 0.975)]


def replay_seed_paired_bootstrap(
    policy_cases_by_seed: list[dict[str, dict[str, float]]],
    baseline_cases_by_seed: list[dict[str, dict[str, float]]],
    metric: str,
    *,
    repeat_single_baseline: bool,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> list[float]:
    """Replay the training script's case bootstrap before/after the seed fix."""
    baselines = baseline_cases_by_seed
    if repeat_single_baseline and len(baselines) == 1 and len(policy_cases_by_seed) > 1:
        baselines = baselines * len(policy_cases_by_seed)
    if not baselines or (repeat_single_baseline and len(policy_cases_by_seed) != len(baselines)):
        raise ValueError("policy/baseline seed rows mismatch")
    case_ids = sorted(set.intersection(*(set(x) for x in policy_cases_by_seed),
                                       *(set(x) for x in baselines)))
    if not case_ids:
        raise ValueError("no paired case clusters")
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(replicates):
        drawn = rng.choice(case_ids, size=len(case_ids), replace=True)
        seed_deltas = [
            float(np.mean([policy[c][metric] - baseline[c][metric] for c in drawn]))
            for policy, baseline in zip(policy_cases_by_seed, baselines)
        ]
        samples.append(float(np.mean(seed_deltas)))
    return [float(np.quantile(samples, .025)), float(np.quantile(samples, .975))]


def mean_ci_by_case(
    case_values: dict[str, float], *, replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> list[float]:
    return bootstrap_ci(case_values, replicates=replicates, seed=seed)


def equal_case_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    cases: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cases[row["case_id"]].append(row)
    fields = ("accuracy", "utility", "ask_rate", "answer_rate", "defer_rate")
    per_case = {
        case: {field: statistics.mean(float(r[field]) for r in case_rows) for field in fields}
        for case, case_rows in cases.items()
    }
    return {field: statistics.mean(per_case[c][field] for c in per_case) for field in fields}


def state_weighted_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    n = len(rows)
    if not n:
        raise ValueError("no state rows")
    return {
        "accuracy": sum(float(r["accuracy"]) for r in rows) / n,
        "utility": sum(float(r["utility"]) for r in rows) / n,
        "ask_rate": sum(float(r["ask_rate"]) for r in rows) / n,
        "answer_rate": sum(float(r["answer_rate"]) for r in rows) / n,
        "defer_rate": sum(float(r["defer_rate"]) for r in rows) / n,
    }


def assert_close(a: float, b: float, label: str) -> None:
    if abs(float(a) - float(b)) > TOLERANCE:
        raise AssertionError(f"{label} mismatch: {a} vs {b}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True,
                        help="read-only source worktree containing the existing ignored outputs")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="new audit-only output path; must not overlap source outputs")
    args = parser.parse_args()
    source = args.source_root.resolve()
    output = args.output_dir.resolve()
    src_run = source / "experiments/ea_atpo_v2/router_training/outputs/real_q_router_20261120"
    raw_dir = source / "experiments/ea_atpo_v2/recovery_bundle_20261120"
    split_path = src_run / "case_split_35_train_15_dev.json"
    public_path = src_run / "public_states_only.jsonl"
    q_path = src_run / "q_dataset_cost_ask001_defer005.jsonl"
    embedding_path = src_run / "qwen3_8b_frozen_public_embeddings.jsonl"
    trajectories_path = raw_dir / "pilot_trajectories.jsonl"
    prediction_path = src_run / "training_final/dev_predictions.jsonl"
    results_path = src_run / "training_final/training_results.json"
    config_path = source / "experiments/ea_atpo_v2/router_training/configs/primary_20261120.json"
    frozen_path = source / "experiments/ea_atpo_v2/router_training/confirmation_20261120/configs/frozen_candidate_20261120.json"
    reload_path = src_run / "training_final/checkpoint_reload_audit.json"
    for path in (split_path, public_path, q_path, embedding_path, trajectories_path,
                 prediction_path, results_path, config_path, frozen_path, reload_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("audit output must be isolated from the read-only source worktree")

    split = read_json(split_path)
    if split.get("unit") != "case_id":
        raise AssertionError("split unit is not case_id")
    train_ids, dev_ids = set(map(str, split["train"])), set(map(str, split["dev"]))
    if len(train_ids) != 35 or len(dev_ids) != 15 or train_ids & dev_ids:
        raise AssertionError("expected disjoint 35/15 case_id split")

    q_rows = read_jsonl(q_path)
    public_rows = read_jsonl(public_path)
    embedding_rows = read_jsonl(embedding_path)
    trajectories = read_jsonl(trajectories_path)
    predictions = read_jsonl(prediction_path)
    old_results = read_json(results_path)
    costs = {"ASK": 0.01, "ANSWER": 0.0, "DEFER": 0.05}
    primary_config = read_json(config_path)
    config_costs = primary_config.get("costs", primary_config.get("action_costs"))
    if isinstance(config_costs, dict):
        for action in costs:
            if action in config_costs:
                costs[action] = float(config_costs[action])
    if costs != {"ASK": 0.01, "ANSWER": 0.0, "DEFER": 0.05}:
        raise AssertionError(f"primary costs differ from frozen audit expectation: {costs}")

    q_by_state: dict[tuple[str, str], dict[str, Any]] = {}
    q_duplicate_keys: list[tuple[str, str]] = []
    for row in q_rows:
        key = key_case_state(row)
        if key in q_by_state:
            q_duplicate_keys.append(key)
        q_by_state[key] = row
    if q_duplicate_keys:
        raise AssertionError("duplicate state row in Q dataset")

    state_rows = {(str(r["case_id"]), str(r["state_id"])): r for r in public_rows}
    if len(state_rows) != len(public_rows):
        raise AssertionError("duplicate case_id/state_id in public state file")
    q_state_keys = set(q_by_state)
    if q_state_keys != set(state_rows):
        raise AssertionError("Q rows and public-state rows do not cover the same states")
    cases = {case for case, _ in q_state_keys}
    if cases != train_ids | dev_ids:
        raise AssertionError("case split does not cover exactly the Q dataset cases")

    # Exact state-content duplicate audit, including duplicates crossing Train/Dev.
    digest_cases: dict[str, set[str]] = defaultdict(set)
    digest_keys: dict[str, list[tuple[str, str]]] = defaultdict(list)
    within_case_duplicate_states = 0
    digest_by_key: dict[tuple[str, str], str] = {}
    for key, row in state_rows.items():
        digest = canonical_state_digest(row["public_state"])
        digest_by_key[key] = digest
        digest_cases[digest].add(key[0])
        digest_keys[digest].append(key)
    for dig, keys in digest_keys.items():
        per_case = Counter(case for case, _ in keys)
        within_case_duplicate_states += sum(n - 1 for n in per_case.values() if n > 1)
    train_digests = {digest_by_key[k] for k in q_state_keys if k[0] in train_ids}
    dev_digests = {digest_by_key[k] for k in q_state_keys if k[0] in dev_ids}
    cross_split_duplicate_digests = train_digests & dev_digests

    # Ensure embeddings are keyed one-to-one to the same pre-action public states.
    embedding_keys = [(str(r["case_id"]), str(r["state_id"])) for r in embedding_rows]
    if len(embedding_keys) != len(set(embedding_keys)) or set(embedding_keys) != q_state_keys:
        raise AssertionError("embedding cache has duplicate/missing/unexpected state keys")

    # Check duplicate rollout keys and independently rebuild each net utility.
    outcome: dict[tuple[str, str, str], dict[str, Any]] = {}
    trajectory_duplicate_keys = []
    action_counts = Counter()
    invalid_trajectory_count = 0
    utility_label_mismatches = []
    for tr in trajectories:
        key = (str(tr["case_id"]), str(tr["state_id"]), str(tr["action"]))
        if key in outcome:
            trajectory_duplicate_keys.append(key)
        outcome[key] = tr
        action_counts[key[2]] += 1
        if tr.get("failure_reason") or not bool(tr.get("format_valid", True)):
            invalid_trajectory_count += 1
        if key[:2] not in q_by_state:
            raise AssertionError("trajectory state absent from Q dataset")
        qrow = q_by_state[(key[0], key[1])]
        mask = qrow.get("valid_mask", {})
        if not bool(mask.get(key[2], False)):
            raise AssertionError("trajectory action has no valid Q label")
        correct = float(tr["final_accuracy"])
        ask_count = float(tr.get("ask_count", 0))
        defer_count = float(tr.get("defer_count", 0))
        target = correct - costs["ASK"] * ask_count - costs["DEFER"] * defer_count
        q_target = float(qrow[f"Q_{key[2]}"])
        if abs(target - q_target) > 1e-7:
            utility_label_mismatches.append({"action": key[2], "abs_delta": abs(target-q_target)})
    if trajectory_duplicate_keys:
        raise AssertionError("duplicate state/action rollout in source trajectories")
    if set(action_counts) != set(ACTION_ORDER) or action_counts != Counter({a: 146 for a in ACTION_ORDER}):
        raise AssertionError(f"unexpected trajectory coverage: {dict(action_counts)}")
    if invalid_trajectory_count or utility_label_mismatches:
        raise AssertionError("invalid source trajectories or Q target/cost mismatch")

    # Every Dev case/state gets one prediction for each selected method and seed.
    prediction_map: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    prediction_duplicates = []
    for row in predictions:
        if row.get("feature") != "embedding" or row.get("model") not in METHODS:
            continue
        key = (str(row["case_id"]), str(row["state_id"]), str(row["model"]), int(row["seed"]))
        if key in prediction_map:
            prediction_duplicates.append(key)
        prediction_map[key] = row
    if prediction_duplicates:
        raise AssertionError("duplicate saved Dev prediction")
    expected_keys = {
        (case, state, method, seed)
        for case, state in q_state_keys if case in dev_ids
        for method in METHODS for seed in SEEDS
    }
    if set(prediction_map) != expected_keys:
        raise AssertionError(f"saved candidate predictions mismatch: expected {len(expected_keys)}, got {len(prediction_map)}")
    if any(case not in dev_ids for case, _, _, _ in prediction_map):
        raise AssertionError("Dev predictions include a non-Dev case")

    per_seed: dict[str, dict[int, dict[str, Any]]] = {m: {} for m in METHODS}
    per_case: dict[str, dict[int, dict[str, dict[str, float]]]] = {m: {} for m in METHODS}
    for method in METHODS:
        for seed in SEEDS:
            state_eval = []
            for case, state in sorted((c, s) for c, s in q_state_keys if c in dev_ids):
                pred = prediction_map[(case, state, method, seed)]
                action = str(pred["predicted_action"])
                if action not in ACTION_ORDER:
                    raise AssertionError(f"invalid predicted action: {action}")
                tr = outcome[(case, state, action)]
                selected_correct = float(tr["final_accuracy"])
                ask_count = float(tr.get("ask_count", 0))
                defer_count = float(tr.get("defer_count", 0))
                utility = selected_correct - costs["ASK"] * ask_count - costs["DEFER"] * defer_count
                assert_close(utility, float(q_by_state[(case, state)][f"Q_{action}"]),
                             f"selected action utility {method}/seed{seed}")
                state_eval.append({
                    "case_id": case, "state_id": state, "accuracy": selected_correct,
                    "utility": utility, "ask_rate": float(action == "ASK"),
                    "answer_rate": float(action == "ANSWER"), "defer_rate": float(action == "DEFER"),
                })
            equal_case = equal_case_metrics(state_eval)
            state_weighted = state_weighted_metrics(state_eval)
            per_seed[method][seed] = {
                "equal_case": equal_case,
                "state_weighted": state_weighted,
                "dev_state_count": len(state_eval),
                "dev_case_count": len({r["case_id"] for r in state_eval}),
            }
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in state_eval:
                grouped[row["case_id"]].append(row)
            per_case[method][seed] = {
                case: {field: statistics.mean(float(r[field]) for r in case_rows)
                       for field in ("accuracy", "utility", "ask_rate", "answer_rate", "defer_rate")}
                for case, case_rows in grouped.items()
            }

    # Check stored per-seed summaries against independent recomputation.
    old_result_rows = [r for r in old_results["results"]
                       if r.get("feature") == "embedding" and r.get("model") in METHODS]
    if len(old_result_rows) != len(METHODS) * len(SEEDS):
        raise AssertionError("stored training_results has unexpected candidate seed rows")
    old_summary_deltas = {}
    max_old_seed_metric_delta = 0.0
    for row in old_result_rows:
        method, seed = str(row["model"]), int(row["seed"])
        rec = per_seed[method][seed]["equal_case"]
        stored = row["dev_policy"]
        for new_name, old_name in (("accuracy", "accuracy"), ("utility", "utility"),
                                   ("ask_rate", "ask"), ("answer_rate", "answer"),
                                   ("defer_rate", "defer")):
            delta = abs(float(rec[new_name]) - float(stored[old_name]))
            max_old_seed_metric_delta = max(max_old_seed_metric_delta, delta)
    old_summaries = old_results["policy_summary_mean_sd_and_case_clustered_ci"]
    for method in METHODS:
        row = next(r for r in old_summaries if r.get("feature") == "embedding" and r.get("model") == method)
        for metric, old_field in (("accuracy", "accuracy_mean"), ("utility", "utility_mean"),
                                  ("ask_rate", "ask_mean"), ("answer_rate", "answer_mean"),
                                  ("defer_rate", "defer_mean")):
            new_mean = statistics.mean(per_seed[method][s]["equal_case"][metric] for s in SEEDS)
            old_summary_deltas[f"{method}.{metric}"] = new_mean - float(row[old_field])
            sd_field = {"accuracy":"accuracy_seed_sd","utility":"utility_seed_sd",
                        "ask_rate":"ask_seed_sd","answer_rate":"answer_seed_sd",
                        "defer_rate":"defer_seed_sd"}[metric]
            new_sd = statistics.stdev(per_seed[method][s]["equal_case"][metric] for s in SEEDS)
            old_summary_deltas[f"{method}.{metric}_sample_sd"] = new_sd - float(row[sd_field])

    # Cluster bootstrap on case_id. The paired contrast is computed within each
    # case first, averaging the three fixed seed outcomes before resampling.
    baseline_case: dict[str, dict[str, float]] = {}
    baseline_by_case = {}
    base_seed_rows = []
    for case in sorted(dev_ids):
        case_states = [s for c, s in q_state_keys if c == case]
        selected = [float(outcome[(case, state, "ANSWER")]["final_accuracy"]) for state in case_states]
        base_seed_rows.append({"case_id": case, "accuracy": statistics.mean(selected),
                               "utility": statistics.mean(selected)})
        baseline_by_case[case] = {"accuracy": statistics.mean(selected), "utility": statistics.mean(selected)}
    ci_results = {}
    for method in METHODS:
        all_case_metrics = {}
        for case in sorted(dev_ids):
            all_case_metrics[case] = {
                metric: statistics.mean(per_case[method][seed][case][metric] for seed in SEEDS)
                for metric in ("accuracy", "utility", "ask_rate", "defer_rate")
            }
        for metric in ("accuracy", "utility"):
            delta_by_case = {
                case: all_case_metrics[case][metric] - baseline_by_case[case][metric]
                for case in sorted(dev_ids)
            }
            ci_results[f"{method}.paired_delta_{metric}"] = {
                "point_estimate": statistics.mean(delta_by_case.values()),
                "ci95": mean_ci_by_case(delta_by_case),
            }
        for metric in ("ask_rate", "defer_rate"):
            values = {case: all_case_metrics[case][metric] for case in sorted(dev_ids)}
            ci_results[f"{method}.{metric}"] = {
                "point_estimate_equal_case": statistics.mean(values.values()),
                "ci95_equal_case": mean_ci_by_case(values),
            }

    # Replay the old bug exactly (zip(3 policy seeds, 1 shared baseline) only
    # evaluates seed 17), then compare with a fixed three-seed paired bootstrap.
    bootstrap_fix_audit = {}
    comparison_rows = []
    # Historical CIs are read-only comparisons; audit computation above does
    # not edit training_results.json or any prior report.
    for method in METHODS:
        row = next(r for r in old_summaries if r.get("feature") == "embedding" and r.get("model") == method)
        method_rows = [r for r in old_results["results"]
                       if r.get("feature") == "embedding" and r.get("model") == method]
        baseline_rows = [r for r in old_results["results"]
                         if r.get("feature") == "embedding" and r.get("model") == "always_answer"]
        policy_case_rows = [r["dev_case_metrics"] for r in method_rows]
        baseline_case_rows = [r["dev_case_metrics"] for r in baseline_rows]
        replay = {}
        for metric in ("accuracy", "utility"):
            ci_field = f"paired_vs_always_answer_delta_{metric}_ci95"
            legacy = replay_seed_paired_bootstrap(policy_case_rows, baseline_case_rows, metric,
                                                   repeat_single_baseline=False)
            corrected = replay_seed_paired_bootstrap(policy_case_rows, baseline_case_rows, metric,
                                                      repeat_single_baseline=True)
            replay[metric] = {"legacy_replay_ci95": legacy,
                              "legacy_replay_matches_saved_ci": all(abs(a-b) <= 1e-12 for a,b in zip(legacy,row[ci_field])),
                              "corrected_three_seed_ci95": corrected}
            ci_results[f"{method}.historical_{metric}_ci95"] = {
                "ci95": row[ci_field],
                "bootstrap_unit_recorded": old_results.get("bootstrap", {}).get("unit"),
                "replicates_recorded": old_results.get("bootstrap", {}).get("replicates"),
            }
            ci_results[f"{method}.corrected_{metric}_ci95"] = {
                "point_estimate": ci_results[f"{method}.paired_delta_{metric}"]["point_estimate"],
                "ci95": corrected,
            }
            comparison_rows.append({"method":method,"metric":metric,
                                    "legacy_low":row[ci_field][0],"legacy_high":row[ci_field][1],
                                    "corrected_low":corrected[0],"corrected_high":corrected[1],
                                    "policy_seed_rows":len(policy_case_rows),
                                    "baseline_rows_before_fix":len(baseline_case_rows),
                                    "seed_rows_used_before_fix":min(len(policy_case_rows),len(baseline_case_rows)),
                                    "seed_rows_used_after_fix":len(policy_case_rows),
                                    "legacy_replay_matches_saved":replay[metric]["legacy_replay_matches_saved_ci"]})
        bootstrap_fix_audit[method] = {
            "policy_seed_rows":len(policy_case_rows),
            "shared_baseline_rows_before_fix":len(baseline_case_rows),
            "zip_rows_used_before_fix":min(len(policy_case_rows),len(baseline_case_rows)),
            "policy_seed_rows_omitted_before_fix":max(0,len(policy_case_rows)-len(baseline_case_rows)),
            "shared_baseline_repeated_after_fix":len(policy_case_rows),
            "metrics":replay,
        }

    # Check all six frozen candidate checkpoints against the pinned hashes.
    frozen = read_json(frozen_path)
    candidate_checkpoints = []
    for family in frozen["candidate_families"]:
        if family["name"] not in METHODS:
            continue
        for item in family["checkpoints"]:
            path = source / item["path"]
            actual = sha256(path)
            candidate_checkpoints.append({"family": family["name"], "seed": int(item["seed"]),
                                          "sha256_matches": actual == item["sha256"],
                                          "bytes_matches": path.stat().st_size == int(item["bytes"])})
    if len(candidate_checkpoints) != 6 or not all(x["sha256_matches"] and x["bytes_matches"] for x in candidate_checkpoints):
        raise AssertionError("frozen candidate checkpoint/config integrity check failed")
    reload_audit = read_json(reload_path)
    if reload_audit.get("status") != "passed" or int(reload_audit.get("checkpoint_count", 0)) != 30:
        raise AssertionError("saved primary checkpoint reload audit is not clean")
    reload_max_diff = max((float(x.get("max_abs_prediction_difference", 0.0))
                           for x in reload_audit.get("comparisons", [])), default=0.0)
    if any(not result["legacy_replay_matches_saved_ci"]
           for method_result in bootstrap_fix_audit.values()
           for result in method_result["metrics"].values()):
        raise AssertionError("legacy bootstrap replay did not match the saved historical CI")
    for method in METHODS:
        for metric in ("accuracy", "utility"):
            replay_ci = bootstrap_fix_audit[method]["metrics"][metric]["corrected_three_seed_ci95"]
            audit_ci = ci_results[f"{method}.paired_delta_{metric}"]["ci95"]
            if any(abs(a-b) > 1e-12 for a,b in zip(replay_ci,audit_ci)):
                raise AssertionError("corrected replay and independent case bootstrap disagree")
    if max((abs(v) for k,v in old_summary_deltas.items() if k.endswith("_sample_sd")),default=0.0) > TOLERANCE:
        raise AssertionError("three-seed sample SD does not match the stored summary")

    output.mkdir(parents=True, exist_ok=True)
    seed_rows = []
    for method in METHODS:
        for seed in SEEDS:
            row = {"method": method, "seed": seed}
            row.update({f"case_weighted_{k}": v for k, v in per_seed[method][seed]["equal_case"].items()})
            row.update({f"state_weighted_{k}": v for k, v in per_seed[method][seed]["state_weighted"].items()})
            seed_rows.append(row)
    with (output / "seed_metrics_case_vs_state_weighted.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(seed_rows[0].keys()), lineterminator="\n")
        writer.writeheader(); writer.writerows(seed_rows)

    audit = {
        "audit_status": "passed",
        "source_commit": "ba698a8d8176ae2dfe34e8f654adc12ca6208530",
        "training_performed": False,
        "gpu_used": False,
        "source_results_modified": False,
        "data_integrity": {
            "train_cases": len(train_ids), "dev_cases": len(dev_ids),
            "train_dev_case_id_overlap": len(train_ids & dev_ids),
            "all_cases": len(cases), "train_states": sum(c in train_ids for c, _ in q_state_keys),
            "dev_states": sum(c in dev_ids for c, _ in q_state_keys),
            "q_states": len(q_rows), "public_state_rows": len(public_rows),
            "embedding_rows": len(embedding_rows), "trajectory_rows": len(trajectories),
            "trajectory_action_counts": dict(action_counts), "invalid_trajectories": invalid_trajectory_count,
            "duplicate_q_state_keys": len(q_duplicate_keys),
            "duplicate_trajectory_state_action_keys": len(trajectory_duplicate_keys),
            "duplicate_public_state_content_within_same_case": within_case_duplicate_states,
            "identical_public_state_digests_shared_across_train_dev": len(cross_split_duplicate_digests),
            "utility_q_label_mismatches": len(utility_label_mismatches),
            "prediction_rows_for_two_candidates": len(prediction_map),
            "prediction_duplicate_keys": len(prediction_duplicates),
        },
        "seed_aggregation": {
            "seed_ids": list(SEEDS), "aggregate": "equal-weight mean over three fixed seed-level policies",
            "seed_dispersion": "sample standard deviation across the three seed-level metrics",
            "case_metrics": "mean states within each case, then equal mean over case_id",
            "state_weighted_rates_also_reported": True,
            "max_abs_difference_vs_saved_seed_metrics": max_old_seed_metric_delta,
            "max_abs_difference_vs_saved_three_seed_means": max((abs(x) for k,x in old_summary_deltas.items() if not k.endswith("_sample_sd")), default=0.0),
            "max_abs_difference_vs_saved_seed_sample_sds": max((abs(x) for k,x in old_summary_deltas.items() if k.endswith("_sample_sd")), default=0.0),
        },
        "bootstrap": {
            "unit": "case_id cluster", "states_per_case_kept_together": True,
            "all_three_fixed_seed_predictions_kept_with_case": True,
            "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
            "rng": "NumPy default_rng / PCG64", "ci": "paired percentile 95%",
            "results": ci_results,
            "seed_truncation_bug_replay": bootstrap_fix_audit,
            "historical_method_record": old_results.get("bootstrap"),
        },
        "methods": {
            method: {
                "per_seed": {str(seed): per_seed[method][seed] for seed in SEEDS},
                "mean_over_seeds_equal_case": {
                    metric: statistics.mean(per_seed[method][seed]["equal_case"][metric] for seed in SEEDS)
                    for metric in ("accuracy", "utility", "ask_rate", "answer_rate", "defer_rate")
                },
                "sample_sd_over_seeds_equal_case": {
                    metric: statistics.stdev([per_seed[method][seed]["equal_case"][metric] for seed in SEEDS])
                    for metric in ("accuracy", "utility", "ask_rate", "answer_rate", "defer_rate")
                },
                "mean_over_seeds_state_weighted": {
                    metric: statistics.mean(per_seed[method][seed]["state_weighted"][metric] for seed in SEEDS)
                    for metric in ("accuracy", "utility", "ask_rate", "answer_rate", "defer_rate")
                },
                "difference_new_mean_minus_saved_mean": {
                    metric: old_summary_deltas[f"{method}.{metric}"]
                    for metric in ("accuracy", "utility", "ask_rate", "answer_rate", "defer_rate")
                },
            } for method in METHODS
        },
        "checkpoint_integrity": {
            "frozen_candidate_checkpoint_count": len(candidate_checkpoints),
            "all_six_sha256_and_size_match_config": True,
            "candidate_checks": candidate_checkpoints,
            "previous_reload_audit_status": reload_audit.get("status"),
            "previous_reload_audit_count": reload_audit.get("checkpoint_count"),
            "previous_reload_max_abs_prediction_difference": reload_max_diff,
            "checkpoint_loaded_or_modified_in_this_audit": False,
        },
        "source_hashes": {
            "case_split": sha256(split_path), "q_dataset": sha256(q_path),
            "public_states": sha256(public_path), "embeddings": sha256(embedding_path),
            "trajectories": sha256(trajectories_path), "predictions": sha256(prediction_path),
            "training_results": sha256(results_path), "primary_training_config": sha256(config_path),
            "frozen_candidate_config": sha256(frozen_path), "previous_reload_audit": sha256(reload_path),
        },
    }
    with (output / "router_result_audit.json").open("w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
        f.write("\n")

    # A compact aggregate-only CI table is safe to publish; it contains no IDs/text.
    ci_rows = []
    for key, value in ci_results.items():
        if "ci95" in value:
            interval = value["ci95"]
        elif "ci95_equal_case" in value:
            interval = value["ci95_equal_case"]
        else:
            continue
        ci_rows.append({"result": key, "point_estimate": value.get("point_estimate", value.get("point_estimate_equal_case")),
                        "ci95_low": interval[0], "ci95_high": interval[1]})
    with (output / "bootstrap_ci_case_clustered.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["result", "point_estimate", "ci95_low", "ci95_high"],
                                lineterminator="\n")
        writer.writeheader(); writer.writerows(ci_rows)
    with (output / "bootstrap_ci_before_after.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["method","metric","legacy_low","legacy_high","corrected_low","corrected_high",
                  "policy_seed_rows","baseline_rows_before_fix","seed_rows_used_before_fix",
                  "seed_rows_used_after_fix","legacy_replay_matches_saved"]
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(comparison_rows)
    print(json.dumps({
        "status": audit["audit_status"], "train_dev_overlap": audit["data_integrity"]["train_dev_case_id_overlap"],
        "train_dev_state_content_overlap": audit["data_integrity"]["identical_public_state_digests_shared_across_train_dev"],
        "train_states": audit["data_integrity"]["train_states"], "dev_states": audit["data_integrity"]["dev_states"],
        "old_seed_metric_max_delta": max_old_seed_metric_delta,
        "q_guided_distillation": audit["methods"]["q_guided_distillation"]["mean_over_seeds_equal_case"],
        "mlp_relative": audit["methods"]["mlp_relative"]["mean_over_seeds_equal_case"],
        "outputs": str(output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
