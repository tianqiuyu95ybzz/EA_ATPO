#!/usr/bin/env python3
"""Recompute integrity, paired outcomes, and action values from immutable Pilot JSONL."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ACTIONS = ("ANSWER", "ASK", "CONSULT", "DEFER")
TIE_PRIORITY = {"ANSWER": 3, "ASK": 2, "CONSULT": 1, "DEFER": 0}
PILOT_DIR = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ci(values: list[float], rng: random.Random, n_boot: int = 10000) -> tuple[float, float]:
    if len(values) <= 1:
        x = values[0] if values else float("nan")
        return x, x
    n = len(values)
    means = [sum(rng.choices(values, k=n)) / n for _ in range(n_boot)]
    means.sort()
    return means[int(.025 * n_boot)], means[min(n_boot - 1, int(.975 * n_boot))]


def percentile(values: list[float]) -> tuple[float, float]:
    values = sorted(values)
    if not values:
        return float("nan"), float("nan")
    return values[int(.025 * len(values))], values[min(len(values) - 1, int(.975 * len(values)))]


def action_stats(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    trajectories = [x for x in rows if x.get("record_type") == "trajectory"]
    statuses = [x for x in rows if x.get("record_type") == "run_status"]
    states_file = load_jsonl(PILOT_DIR / "outputs/full/pilot_public_states.jsonl")
    states_file = [x for x in states_file if x.get("record_type") == "public_state"]

    by_state: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_pair: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    state_serializations: dict[tuple[str, str], set[str]] = defaultdict(set)
    action_counts = Counter()
    invalid_actions = Counter()
    failure_reasons = Counter()
    failed_cases: dict[str, Counter] = defaultdict(Counter)
    error_types = Counter()
    expert_validation_failures = 0
    invalid_no_expert_usage = 0
    exact_message_hash_matches = 0
    for x in trajectories:
        action = str(x.get("action"))
        key = (str(x.get("case_id")), str(x.get("state_id")))
        by_state[key].append(x)
        by_pair[(key[0], key[1], int(x.get("rollout_index", -1)))][action] = x
        state_serializations[key].add(json.dumps(x.get("public_state"), sort_keys=True, ensure_ascii=False, separators=(",", ":")))
        action_counts[action] += 1
        if not x.get("valid"):
            invalid_actions[action] += 1
            reason = str(x.get("failure_reason") or x.get("error") or "unknown")
            failure_reasons[reason] += 1
            failed_cases[str(x.get("case_id"))][action] += 1
            error_types[reason.split(":", 1)[0]] += 1
            if "CONSULT response" in reason or "CONSULT leaked" in reason or "DEFER response" in reason:
                expert_validation_failures += 1
            # This counts malformed Expert responses for which no Expert call metrics
            # reached the saved trace; completed but rejected responses are censored.
            if not any(t.get("actor") == "expert" for t in x.get("trace", [])):
                invalid_no_expert_usage += 1
        public = x.get("public_state", [])
        sample_id = str(x.get("sample_id", ""))
        digest = hashlib.sha256(json.dumps(public, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
        if str(x.get("state_id", "")).endswith(digest) and str(x.get("state_id", "")).startswith(sample_id + ":"):
            exact_message_hash_matches += 1

    incomplete_pairs = sum(set(p) != set(ACTIONS) for p in by_pair.values())
    counts_per_state_action = Counter()
    state_action_sets_ok = True
    within_state_gold_mismatch = 0
    for key, group in by_state.items():
        local = Counter(str(x.get("action")) for x in group)
        counts_per_state_action.update(local.values())
        if set(local) != set(ACTIONS) or any(local[a] != 3 for a in ACTIONS):
            state_action_sets_ok = False
        if len(state_serializations[key]) != 1:
            state_action_sets_ok = False
        if len({str(x.get("gold")) for x in group}) != 1:
            within_state_gold_mismatch += 1

    by_case_action: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for x in trajectories:
        by_case_action[(str(x["case_id"]), str(x["action"]))].append(x)

    case_action_outcome: dict[str, dict[str, float]] = defaultdict(dict)
    case_action_cost: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for (case, action), group in by_case_action.items():
        # Invalid branches already carry outcome=0; retain every scheduled rollout.
        case_action_outcome[case][action] = statistics.mean(float(x.get("outcome") or 0.0) for x in group)
        case_action_cost[case][action] = {
            "expert_tokens": statistics.mean(float(x.get("expert_prompt_tokens", 0) or 0) + float(x.get("expert_completion_tokens", 0) or 0) for x in group),
            "patient_turns": statistics.mean(float(sum(t.get("actor") == "patient" for t in x.get("trace", []))) for x in group),
            "consult_tokens": statistics.mean(float(sum(int(t.get("prompt_tokens", 0) or 0) + int(t.get("completion_tokens", 0) or 0) for t in x.get("trace", []) if t.get("actor") == "expert" and t.get("action") == "CONSULT")) for x in group),
            "defer_tokens": statistics.mean(float(sum(int(t.get("prompt_tokens", 0) or 0) + int(t.get("completion_tokens", 0) or 0) for t in x.get("trace", []) if t.get("actor") == "expert" and t.get("action") == "DEFER")) for x in group),
            "total_tokens": statistics.mean(float(x.get("total_prompt_tokens", 0) or 0) + float(x.get("total_completion_tokens", 0) or 0) for x in group),
            "latency_s": statistics.mean(float(x.get("total_latency_s", 0) or 0) for x in group),
            "expert_latency_s": statistics.mean(float(x.get("expert_latency_s", 0) or 0) for x in group),
            "valid_rate": statistics.mean(float(bool(x.get("valid"))) for x in group),
        }

    paired_deltas: dict[str, list[float]] = defaultdict(list)
    correct_to_wrong = wrong_to_correct = answer_wrong = answer_correct = 0
    paired_state_count = 0
    for (case, _state, _rep), branches in by_pair.items():
        if not all(a in branches for a in ACTIONS):
            continue
        paired_state_count += 1
        a = float(branches["ANSWER"].get("outcome") or 0.0)
        c = float(branches["CONSULT"].get("outcome") or 0.0)
        d = float(branches["DEFER"].get("outcome") or 0.0)
        paired_deltas[case].append(c - a)
        if a:
            answer_correct += 1
            correct_to_wrong += int(not c)
        else:
            answer_wrong += 1
            wrong_to_correct += int(bool(c))

    cases = sorted(case_action_outcome)
    scenario_grid = [
        ("no_cost", 0.0, 0.0, 0.0),
        ("low_all", 0.01, 0.02, 0.02),
        ("consult_expensive", 0.01, 0.10, 0.02),
        ("defer_expensive", 0.01, 0.02, 0.10),
        ("high_all", 0.03, 0.10, 0.10),
    ]
    state_values: dict[str, dict[str, dict[str, float]]] = {}
    for (case, state), group in by_state.items():
        vals = {}
        for action in ACTIONS:
            chosen = [x for x in group if x.get("action") == action]
            vals[action] = {
                "q": statistics.mean(float(x.get("outcome") or 0.0) for x in chosen),
                "expert_tokens": statistics.mean(float(x.get("expert_prompt_tokens", 0) or 0) + float(x.get("expert_completion_tokens", 0) or 0) for x in chosen),
                "patient_turns": statistics.mean(float(sum(t.get("actor") == "patient" for t in x.get("trace", []))) for x in chosen),
                "consult_tokens": statistics.mean(float(sum(int(t.get("prompt_tokens", 0) or 0) + int(t.get("completion_tokens", 0) or 0) for t in x.get("trace", []) if t.get("actor") == "expert" and t.get("action") == "CONSULT")) for x in chosen),
                "defer_tokens": statistics.mean(float(sum(int(t.get("prompt_tokens", 0) or 0) + int(t.get("completion_tokens", 0) or 0) for t in x.get("trace", []) if t.get("actor") == "expert" and t.get("action") == "DEFER")) for x in chosen),
            }
        state_values[f"{case}\t{state}"] = vals

    case_states: dict[str, list[dict[str, dict[str, float]]]] = defaultdict(list)
    for key, values in state_values.items():
        case = key.split("\t", 1)[0]
        case_states[case].append(values)

    def calc_for_sample(sampled_cases: list[str], lam_ask: float, lam_consult: float, lam_defer: float) -> dict[str, float]:
        wins = Counter()
        strict_consult = strict_defer = 0
        nstates = 0
        gaps = []
        for case in sampled_cases:
            for values in case_states[case]:
                nstates += 1
                utility = {a: values[a]["q"] - lam_ask * values[a]["patient_turns"] - lam_consult * values[a]["consult_tokens"] / 1000.0 - lam_defer * values[a]["defer_tokens"] / 1000.0 for a in ACTIONS}
                ranked = sorted(ACTIONS, key=lambda a: (utility[a], -values[a]["expert_tokens"], -values[a]["patient_turns"], TIE_PRIORITY[a]), reverse=True)
                wins[ranked[0]] += 1
                gaps.append(utility[ranked[0]] - utility[ranked[1]])
                strict_consult += int(utility["CONSULT"] > max(utility[a] for a in ACTIONS if a != "CONSULT"))
                strict_defer += int(utility["DEFER"] > max(utility[a] for a in ACTIONS if a != "DEFER"))
        result = {f"share_{a}": wins[a] / nstates if nstates else 0.0 for a in ACTIONS}
        result["mean_best_second_gap"] = statistics.mean(gaps) if gaps else 0.0
        result["strict_consult_advantage_share"] = strict_consult / nstates if nstates else 0.0
        result["strict_defer_advantage_share"] = strict_defer / nstates if nstates else 0.0
        return result

    metrics_rows: list[dict[str, Any]] = []
    rng = random.Random(20261104)
    boot_samples = 4000
    for scenario, lam_ask, lam_consult, lam_defer in scenario_grid:
        point = calc_for_sample(cases, lam_ask, lam_consult, lam_defer)
        boots = [calc_for_sample(rng.choices(cases, k=len(cases)), lam_ask, lam_consult, lam_defer) for _ in range(boot_samples)]
        for metric, value in point.items():
            low, high = percentile([b[metric] for b in boots])
            metrics_rows.append({"scenario": scenario, "metric": metric, "action": metric.removeprefix("share_" ) if metric.startswith("share_") else "", "value": value, "ci95_low": low, "ci95_high": high, "n_cases": len(cases), "n_states": len(state_values), "notes": f"U=Q-lambdaASK*patient_turns-lambdaCONSULT*CONSULT_tokens/1k-lambdaDEFER*DEFER_tokens/1k; weights={lam_ask:g}/{lam_consult:g}/{lam_defer:g}. Q is raw correct=1, incorrect/invalid=0. Costs are subtracted once. Failed Expert-call usage is missing, so costed results are sensitivity estimates using observed lower-bound costs."})

    rng = random.Random(20261104)
    action_accuracy = {}
    for action in ACTIONS:
        case_values = [case_action_outcome[c][action] for c in cases]
        action_accuracy[action] = {"mean": statistics.mean(case_values), "ci95": ci(case_values, rng), "n_cases": len(case_values)}

    pair_case_means = {case: statistics.mean(vals) for case, vals in paired_deltas.items()}
    consult_answer_ci = ci(list(pair_case_means.values()), random.Random(20261105))

    # Cluster-bootstrap the correction/degradation ratios by case, retaining all
    # paired state/repeat observations inside each sampled case.
    per_case_pairs: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for (case, _state, rep), branches in by_pair.items():
        if set(branches) == set(ACTIONS):
            a = float(branches["ANSWER"].get("outcome") or 0.0)
            c = float(branches["CONSULT"].get("outcome") or 0.0)
            per_case_pairs[case].append((a, c))
    crng = random.Random(20261106)
    correction_boot, degradation_boot = [], []
    for _ in range(10000):
        sample = crng.choices(cases, k=len(cases))
        pairs = [p for case in sample for p in per_case_pairs[case]]
        aw = sum(a == 0 for a, _ in pairs)
        ac = sum(a == 1 for a, _ in pairs)
        correction_boot.append(sum(a == 0 and c == 1 for a, c in pairs) / aw if aw else 0.0)
        degradation_boot.append(sum(a == 1 and c == 0 for a, c in pairs) / ac if ac else 0.0)

    prefix_rates = []
    log_path = PILOT_DIR / "outputs/model_server.log"
    if log_path.exists():
        rx = re.compile(r"Prefix cache hit rate: ([0-9.]+)%")
        with log_path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                match = rx.search(line)
                if match:
                    prefix_rates.append(float(match.group(1)))

    by_source_index = defaultdict(list)
    for x in trajectories:
        if not x.get("valid"):
            by_source_index[int(x.get("source_row_index", -1))].append(x)

    evidence = {
        "input_hashes": {
            "pilot_trajectories.jsonl": sha256(PILOT_DIR / "outputs/full/pilot_trajectories.jsonl"),
            "pilot_public_states.jsonl": sha256(PILOT_DIR / "outputs/full/pilot_public_states.jsonl"),
            "summary.csv": sha256(PILOT_DIR / "outputs/full/summary.csv"),
            "action_value_analysis.csv": sha256(PILOT_DIR / "outputs/full/action_value_analysis.csv"),
            "run_manifest.json": sha256(PILOT_DIR / "outputs/full/run_manifest.json"),
        },
        "record_count_including_status": len(rows),
        "trajectory_count": len(trajectories),
        "run_status_rows": statuses,
        "case_count": len({str(x.get("case_id")) for x in trajectories}),
        "state_count": len(by_state),
        "action_counts": dict(action_counts),
        "paired_action_rollout_tuples": paired_state_count,
        "incomplete_action_pairs": incomplete_pairs,
        "all_states_have_four_actions_x_three_repeats": state_action_sets_ok,
        "state_serializations_per_case_state_max": max((len(v) for v in state_serializations.values()), default=0),
        "state_id_message_hash_matches": exact_message_hash_matches,
        "gold_mismatch_within_state_count": within_state_gold_mismatch,
        "invalid_count": sum(invalid_actions.values()),
        "invalid_by_action": dict(invalid_actions),
        "invalid_error_type_counts": dict(error_types),
        "invalid_error_details": dict(failure_reasons),
        "invalid_unique_cases": len(by_source_index),
        "invalid_cases": {str(k): {"count": len(v), "actions": dict(Counter(x.get("action") for x in v))} for k, v in sorted(by_source_index.items())},
        "expert_validation_failures": expert_validation_failures,
        "invalid_rows_without_saved_expert_usage_event": invalid_no_expert_usage,
        "case_clustered_accuracy": action_accuracy,
        "consult_minus_answer_case_paired_delta": {"mean": statistics.mean(pair_case_means.values()), "ci95": consult_answer_ci},
        "consult_corrected_answer_errors": {"count": wrong_to_correct, "denominator": answer_wrong, "rate": wrong_to_correct / answer_wrong if answer_wrong else None, "cluster_bootstrap_ci95": percentile(correction_boot)},
        "consult_degraded_answer_correct": {"count": correct_to_wrong, "denominator": answer_correct, "rate": correct_to_wrong / answer_correct if answer_correct else None, "cluster_bootstrap_ci95": percentile(degradation_boot)},
        "prefix_cache_hit_rate_log_percent": {"n_log_samples": len(prefix_rates), "min": min(prefix_rates) if prefix_rates else None, "mean": statistics.mean(prefix_rates) if prefix_rates else None, "max": max(prefix_rates) if prefix_rates else None},
        "cost_sensitivity_scenarios": len(scenario_grid),
        "bootstrap_case_clusters": boot_samples,
    }
    return evidence, metrics_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PILOT_DIR / "audit/outputs")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    raw_path = PILOT_DIR / "outputs/full/pilot_trajectories.jsonl"
    evidence, metric_rows = action_stats(load_jsonl(raw_path))
    (args.output / "pilot_audit_metrics.json").write_text(json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8")
    with (args.output / "action_value_reanalysis.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=("scenario", "metric", "action", "value", "ci95_low", "ci95_high", "n_cases", "n_states", "notes"))
        writer.writeheader()
        writer.writerows(metric_rows)
    print(json.dumps({k: evidence[k] for k in ("trajectory_count", "case_count", "state_count", "action_counts", "paired_action_rollout_tuples", "incomplete_action_pairs", "invalid_count", "invalid_by_action", "invalid_unique_cases", "invalid_rows_without_saved_expert_usage_event", "prefix_cache_hit_rate_log_percent")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
