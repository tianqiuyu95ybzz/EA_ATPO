#!/usr/bin/env python3
"""Read-only analysis of EA-ATPO V2 paired counterfactual data (Window B).

The script reads the merged Q/trajectory files and their manifests. It does not
start model services, access protected splits, or write into the source run.
All reported uncertainty intervals resample case_id clusters.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ACTION_ORDER = ("ASK", "ANSWER", "DEFER")
ANSWER_INDEX = ACTION_ORDER.index("ANSWER")
SEED = 20261120
BOOTSTRAP_REPLICATES = 10_000
MAIN_ASK_COST = 0.01
MAIN_DEFER_COST = 0.05
ASK_COST_GRID = (0.00, 0.01, 0.03, 0.05)
DEFER_COST_GRID = (0.00, 0.05, 0.10, 0.20, 0.40, 0.60)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: {exc}") from exc
    return rows


def canonical(row: dict[str, Any]) -> str:
    return json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def verify_data_version(data_dir: Path, q_rows: list[dict[str, Any]],
                        trajectories: list[dict[str, Any]], split_path: Path) -> dict[str, Any]:
    q_path = data_dir / "q_dataset.jsonl"
    trajectory_path = data_dir / "pilot_trajectories.jsonl"
    manifest_path = data_dir / "merged_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("status") != "complete" or manifest.get("frozen_holdout_opened") is not False:
        raise ValueError("source manifest is incomplete or indicates Frozen Holdout access")
    for key in ("all_states_paired", "all_rollouts_valid", "all_q_targets_valid",
                "costs_included_once_in_return", "case_ids_all_train_only"):
        if manifest.get(key) is not True:
            raise ValueError(f"source manifest does not certify {key}")
    if manifest.get("gold_or_hidden_facts_serialized") is not False:
        raise ValueError("source manifest indicates Gold/hidden facts were serialized")
    if Path(manifest["trajectory_output"]).resolve() != trajectory_path.resolve():
        raise ValueError("trajectory file path does not match merged manifest")
    if sha256_file(trajectory_path) != manifest.get("trajectory_sha256"):
        raise ValueError("trajectory SHA-256 differs from merged manifest")
    if manifest.get("q_dataset_sha256") and sha256_file(q_path) != manifest["q_dataset_sha256"]:
        raise ValueError("Q dataset SHA-256 differs from merged manifest")
    if sha256_file(split_path) != "b55d889a713b2b1d32afc25d89aaf9205891a1ee5fbaae470702ff31a9f81267":
        raise ValueError("official case split hash differs from the audited frozen split")

    config_path = Path(manifest["config"]).resolve()
    if sha256_file(config_path) != manifest.get("config_sha256"):
        raise ValueError("run config hash differs from merged manifest")
    config = read_json(config_path)
    original_ask_cost = float(config["costs"]["ask"])
    original_defer_cost = float(config["costs"]["defer"])
    policy_id = str(config["policy"]["continuation_policy_id"])

    split = read_json(split_path)
    if split.get("unit") != "case_id":
        raise ValueError("official split must be case_id based")
    train_cases = set(map(str, split["train"]))
    protected = set(map(str, split.get("validation", []))) | set(map(str, split.get("internal_test", [])))

    batch_q_counter: Counter[str] = Counter()
    batch_trajectory_counter: Counter[str] = Counter()
    batch_provenance = []
    for batch in manifest.get("batches", []):
        bq = Path(batch["q_dataset_path"]).resolve()
        bt = Path(batch["trajectory_path"]).resolve()
        bm = Path(batch["run_manifest"]).resolve()
        if sha256_file(bq) != batch["q_dataset_sha256"] or sha256_file(bt) != batch["trajectory_sha256"]:
            raise ValueError("a source batch changed after merge")
        if sha256_file(bm) != batch["run_manifest_sha256"]:
            raise ValueError("a source run manifest changed after merge")
        batch_q_counter.update(canonical(row) for row in read_jsonl(bq))
        batch_trajectory_counter.update(canonical(row) for row in read_jsonl(bt))
        batch_manifest = read_json(bm)
        batch_config = Path(batch_manifest["config_path"]).resolve()
        if (batch_manifest.get("config_sha256") != manifest.get("config_sha256")
                or sha256_file(batch_config) != manifest.get("config_sha256")):
            raise ValueError("source batch config differs from merged config")
        batch_provenance.append(batch_manifest)
    if batch_q_counter != Counter(canonical(row) for row in q_rows):
        raise ValueError("merged Q rows are not the exact source-batch union")
    if batch_trajectory_counter != Counter(canonical(row) for row in trajectories):
        raise ValueError("merged trajectory rows are not the exact source-batch union")

    q_by_key = {}
    for row in q_rows:
        md = row.get("metadata") or {}
        key = (str(md.get("case_id", "")), str(md.get("state_id", "")))
        if not all(key) or key in q_by_key:
            raise ValueError(f"missing/duplicate Q state identity: {key}")
        if list((row.get("valid_mask") or {}).keys()) != list(ACTION_ORDER):
            raise ValueError(f"{key}: Q valid_mask action order mismatch")
        if md.get("continuation_policy_id") != policy_id:
            raise ValueError(f"{key}: continuation policy differs from run config")
        if md.get("costs_included_in_targets") is not True:
            raise ValueError(f"{key}: Q target cost-inclusion marker missing")
        if md.get("feature_boundary") != "public_state_before_action_only" or md.get("gold_or_future_fields_in_features") is not False:
            raise ValueError(f"{key}: Q feature boundary is not public-state-only")
        q_by_key[key] = row

    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    raw_state_keys = set()
    for row in trajectories:
        case_id, state_id, action = str(row.get("case_id", "")), str(row.get("state_id", "")), str(row.get("action", ""))
        key = (case_id, state_id)
        if action not in ACTION_ORDER or not case_id or not state_id:
            raise ValueError("malformed trajectory identity/action")
        if action in grouped[key]:
            raise ValueError(f"duplicate state/action rollout: {key}/{action}")
        if row.get("continuation_policy_id") != policy_id:
            raise ValueError("trajectory continuation policy differs from manifest")
        if row.get("valid") is not True or row.get("final_accuracy") not in (0, 0.0, 1, 1.0):
            raise ValueError(f"invalid/non-binary rollout outcome: {key}/{action}")
        if row.get("return_std") is not None or int(row.get("sample_count", 1)) != 1:
            raise ValueError("expected one rollout per state/action in this data version")
        expected_ask = int(action == "ASK")
        expected_defer = int(action == "DEFER")
        components = row.get("cost_components") or {}
        if int(row.get("ask_count", -1)) != expected_ask or int(row.get("defer_count", -1)) != expected_defer:
            raise ValueError(f"unexpected action cost counts: {key}/{action}")
        ask_cost = float(components.get("ask_cost", math.nan))
        defer_cost = float(components.get("defer_cost", math.nan))
        if abs(ask_cost - original_ask_cost * expected_ask) > 1e-9 or abs(defer_cost - original_defer_cost * expected_defer) > 1e-9:
            raise ValueError(f"raw action costs differ from source config: {key}/{action}")
        expected_return = float(row["final_accuracy"]) - ask_cost - defer_cost
        if abs(float(row["return"]) - expected_return) > 1e-9:
            raise ValueError(f"raw return cost formula mismatch: {key}/{action}")
        qrow = q_by_key.get(key)
        if qrow is None:
            raise ValueError(f"raw state has no Q row: {key}")
        action_index = ACTION_ORDER.index(action)
        action_q = qrow[f"Q_{action}"]
        if qrow["valid_mask"][action] is not True or int(qrow["valid_sample_count"][action]) != 1:
            raise ValueError(f"expected one valid Q target: {key}/{action}")
        if abs(float(action_q) - expected_return) > 1e-9:
            raise ValueError(f"Q target is not the raw return: {key}/{action}")
        if qrow["return_std"].get(f"Q_{action}") is not None:
            raise ValueError(f"K=1 target unexpectedly has return_std: {key}/{action}")
        grouped[key][action] = row
        raw_state_keys.add(key)
    if set(q_by_key) != raw_state_keys:
        raise ValueError("Q/raw trajectory state keys differ")
    if any(set(actions) != set(ACTION_ORDER) for actions in grouped.values()):
        raise ValueError("a state is missing an action rollout")
    cases = {case for case, _ in grouped}
    if not cases.issubset(train_cases) or cases & protected:
        raise ValueError("dataset contains cases outside official training split")
    if len(cases) != int(manifest["case_count"]) or len(grouped) != int(manifest["state_count"]):
        raise ValueError("case/state counts differ from merged manifest")
    if len(trajectories) != int(manifest["rollout_count"]) or len(trajectories) != len(grouped) * 3:
        raise ValueError("rollout count differs from merged manifest or three-action pairing")
    if int(manifest["valid_q_targets"]) != len(trajectories):
        raise ValueError("valid Q target count differs from merged manifest")

    return {
        "manifest": manifest,
        "config": config,
        "q_path": q_path,
        "trajectory_path": trajectory_path,
        "q_sha256": sha256_file(q_path),
        "trajectory_sha256": sha256_file(trajectory_path),
        "config_sha256": manifest["config_sha256"],
        "case_split_sha256": sha256_file(split_path),
        "original_costs": {"ASK": original_ask_cost, "ANSWER": 0.0, "DEFER": original_defer_cost},
        "continuation_policy_id": policy_id,
        "q_by_key": q_by_key,
        "grouped": grouped,
        "cases": sorted(cases),
        "n_train_cases_official": len(train_cases),
        "n_total_train_states_from_manifest": int((batch_provenance[0].get("data_audit") or {}).get("train_public_state_rows", 0)),
        "n_source_batches": len(batch_provenance),
        "source_project_commits": sorted({str(item.get("project_commit") or "") for item in batch_provenance}),
        "source_project_branches": sorted({str(item.get("project_branch") or "") for item in batch_provenance}),
        "runner_source_hash_pinned": all(bool(item.get("runner_script_sha256") or item.get("source_script_sha256")) for item in batch_provenance),
    }


def bootstrap_ci(case_values: dict[str, float], seed: int, n_boot: int = BOOTSTRAP_REPLICATES) -> tuple[float, float]:
    values = np.asarray(list(case_values.values()), dtype=np.float64)
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[draws].mean(axis=1)
    return tuple(float(x) for x in np.quantile(means, [0.025, 0.975]))


def case_means(values: np.ndarray, case_ids: list[str], eligible: np.ndarray | None = None) -> dict[str, float]:
    result = {}
    for case_id in sorted(set(case_ids)):
        idx = np.asarray([c == case_id for c in case_ids], dtype=bool)
        if eligible is not None:
            idx &= eligible
        if idx.any():
            result[case_id] = float(np.mean(values[idx]))
    return result


def bootstrap_conditional(values: np.ndarray, case_ids: list[str], eligible: np.ndarray,
                          seed: int) -> tuple[float, tuple[float, float], int]:
    grouped = case_means(values, case_ids, eligible)
    if not grouped:
        return float("nan"), (float("nan"), float("nan")), 0
    return float(np.mean(list(grouped.values()))), bootstrap_ci(grouped, seed), len(grouped)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def state_arrays(grouped: dict[tuple[str, str], dict[str, dict[str, Any]]], q_by_key: dict) -> tuple[list[tuple[str, str]], list[str], np.ndarray, np.ndarray, list[dict[str, Any]]]:
    keys = sorted(grouped)
    case_ids = [case for case, _ in keys]
    accuracy = np.asarray([[float(grouped[key][action]["final_accuracy"]) for action in ACTION_ORDER] for key in keys], dtype=np.float64)
    q_values = np.asarray([[float(q_by_key[key][f"Q_{action}"]) for action in ACTION_ORDER] for key in keys], dtype=np.float64)
    asks = [grouped[key]["ASK"] for key in keys]
    return keys, case_ids, accuracy, q_values, asks


def utility_matrix(accuracy: np.ndarray, ask_cost: float, defer_cost: float) -> np.ndarray:
    costs = np.asarray([ask_cost, 0.0, defer_cost], dtype=np.float64)
    return accuracy - costs[None, :]


def make_policy_probabilities(utilities: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    n_states = utilities.shape[0]
    policies = {}
    for name, action_index in (("Always ANSWER", 1), ("Always ASK", 0), ("Always DEFER", 2)):
        probabilities = np.zeros((n_states, 3), dtype=np.float64)
        probabilities[:, action_index] = 1.0
        policies[name] = probabilities
    best_mask = np.isclose(utilities, utilities.max(axis=1, keepdims=True), atol=1e-12, rtol=0.0)
    tie_counts = best_mask.sum(axis=1)
    oracle_probabilities = best_mask / tie_counts[:, None]
    policies["Retrospective Oracle"] = oracle_probabilities
    return policies, best_mask, tie_counts


def summarize_cost_grid(keys: list[tuple[str, str]], case_ids: list[str], accuracy: np.ndarray,
                        output_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    table_rows = []
    oracle_distribution_rows = []
    all_oracle_best_sets: dict[tuple[str, str], set[tuple[str, ...]]] = defaultdict(set)
    seed_base = 1300
    scenario_index = 0
    for ask_cost in ASK_COST_GRID:
        for defer_cost in DEFER_COST_GRID:
            utility = utility_matrix(accuracy, ask_cost, defer_cost)
            policies, best_mask, tie_counts = make_policy_probabilities(utility)
            oracle_prob = policies["Retrospective Oracle"]
            oracle_acc = np.sum(oracle_prob * accuracy, axis=1)
            oracle_util = np.sum(oracle_prob * utility, axis=1)
            oracle_ask = oracle_prob[:, 0]
            oracle_defer = oracle_prob[:, 2]
            answer_acc = accuracy[:, 1]
            answer_util = utility[:, 1]

            for idx, key in enumerate(keys):
                best_set = tuple(ACTION_ORDER[j] for j in range(3) if best_mask[idx, j])
                all_oracle_best_sets[key].add(best_set)

            distributions = {
                "Always ANSWER": np.asarray([0.0, 1.0, 0.0]),
                "Always ASK": np.asarray([1.0, 0.0, 0.0]),
                "Always DEFER": np.asarray([0.0, 0.0, 1.0]),
                "Retrospective Oracle": oracle_prob,
            }
            for policy_name, probs in distributions.items():
                if probs.ndim == 1:
                    probs = np.tile(probs[None, :], (len(keys), 1))
                acc_values = np.sum(probs * accuracy, axis=1)
                util_values = np.sum(probs * utility, axis=1)
                ask_values, defer_values = probs[:, 0], probs[:, 2]
                by_case = {
                    "accuracy": case_means(acc_values, case_ids),
                    "utility": case_means(util_values, case_ids),
                    "ask_rate": case_means(ask_values, case_ids),
                    "defer_rate": case_means(defer_values, case_ids),
                }
                row = {
                    "ask_cost": ask_cost, "defer_cost": defer_cost, "policy": policy_name,
                    "n_cases": len(set(case_ids)), "n_states": len(keys),
                }
                pooled = {
                    "accuracy": float(acc_values.mean()), "utility": float(util_values.mean()),
                    "ask_rate": float(ask_values.mean()), "defer_rate": float(defer_values.mean()),
                }
                for metric, values in by_case.items():
                    point = float(np.mean(list(values.values())))
                    low, high = bootstrap_ci(values, seed_base + scenario_index * 7 + ("accuracy", "utility", "ask_rate", "defer_rate").index(metric))
                    row[metric] = point
                    row[f"{metric}_ci95_low"] = low
                    row[f"{metric}_ci95_high"] = high
                    row[f"{metric}_state_pooled"] = pooled[metric]

                if policy_name == "Retrospective Oracle":
                    acc_gain = oracle_acc - answer_acc
                    util_gain = oracle_util - answer_util
                    for metric, values in (("oracle_accuracy_gain_vs_answer", acc_gain),
                                           ("oracle_utility_gain_vs_answer", util_gain)):
                        by_case_gain = case_means(values, case_ids)
                        low, high = bootstrap_ci(by_case_gain, seed_base + scenario_index * 11 + 91 + (metric.endswith("utility_gain_vs_answer")))
                        row[metric] = float(np.mean(list(by_case_gain.values())))
                        row[f"{metric}_ci95_low"] = low
                        row[f"{metric}_ci95_high"] = high
                    row["oracle_unique_best_states"] = int(np.sum(tie_counts == 1))
                    row["oracle_tied_best_states"] = int(np.sum(tie_counts > 1))
                else:
                    row["oracle_accuracy_gain_vs_answer"] = 0.0
                    row["oracle_utility_gain_vs_answer"] = 0.0
                    row["oracle_unique_best_states"] = ""
                    row["oracle_tied_best_states"] = ""
                table_rows.append(row)

            oracle_distribution_rows.append({
                "ask_cost": ask_cost, "defer_cost": defer_cost,
                "best_ASK_state_equivalents": float(oracle_prob[:, 0].sum()),
                "best_ANSWER_state_equivalents": float(oracle_prob[:, 1].sum()),
                "best_DEFER_state_equivalents": float(oracle_prob[:, 2].sum()),
                "best_ASK_share": float(oracle_prob[:, 0].mean()),
                "best_ANSWER_share": float(oracle_prob[:, 1].mean()),
                "best_DEFER_share": float(oracle_prob[:, 2].mean()),
                "unique_best_states": int(np.sum(tie_counts == 1)),
                "tied_best_states": int(np.sum(tie_counts > 1)),
            })
            scenario_index += 1
    write_csv(output_dir / "cost_sensitivity.csv", table_rows)
    write_csv(output_dir / "oracle_action_distribution.csv", oracle_distribution_rows)
    switch_details = {
        "cost_combinations": len(ASK_COST_GRID) * len(DEFER_COST_GRID),
        "states_with_any_best_action_set_change": sum(len(sets) > 1 for sets in all_oracle_best_sets.values()),
        "states_with_unique_best_action_change": sum(
            len({action_set[0] for action_set in sets if len(action_set) == 1}) > 1
            for sets in all_oracle_best_sets.values()
        ),
        "states_with_same_best_action_set_all_costs": sum(len(sets) == 1 for sets in all_oracle_best_sets.values()),
        "tie_policy": "equal fractional mass over all exactly tied maximizing actions; tie states also counted separately",
    }
    (output_dir / "oracle_action_switches.json").write_text(json.dumps(switch_details, indent=2) + "\n", encoding="utf-8")
    return table_rows, oracle_distribution_rows, switch_details


def complementarity_table(case_ids: list[str], accuracy: np.ndarray, output_dir: Path) -> list[dict[str, Any]]:
    answer = accuracy[:, 1].astype(bool)
    defer = accuracy[:, 2].astype(bool)
    ask = accuracy[:, 0].astype(bool)
    specifications = [
        ("Expert vs Base", "ANSWER correct / DEFER correct", answer & defer),
        ("Expert vs Base", "ANSWER correct / DEFER wrong (Expert spoilage)", answer & ~defer),
        ("Expert vs Base", "ANSWER wrong / DEFER correct (Expert correction)", ~answer & defer),
        ("Expert vs Base", "ANSWER wrong / DEFER wrong", ~answer & ~defer),
        ("ASK vs ANSWER", "ANSWER correct / ASK correct (both correct)", answer & ask),
        ("ASK vs ANSWER", "ANSWER correct / ASK wrong (ASK degradation)", answer & ~ask),
        ("ASK vs ANSWER", "ANSWER wrong / ASK correct (ASK correction)", ~answer & ask),
        ("ASK vs ANSWER", "ANSWER wrong / ASK wrong (both wrong)", ~answer & ~ask),
        ("ASK vs ANSWER", "ASK unchanged (both correct or both wrong)", answer == ask),
        ("ASK vs ANSWER", "ASK changed (correction or degradation)", answer != ask),
    ]
    rows = []
    for index, (comparison, category, mask) in enumerate(specifications):
        mask_float = mask.astype(np.float64)
        per_case = case_means(mask_float, case_ids)
        low, high = bootstrap_ci(per_case, 3000 + index)
        cases_with = sum(any(mask[j] for j, case in enumerate(case_ids) if case == c) for c in set(case_ids))
        rows.append({
            "comparison": comparison, "category": category,
            "n_states": int(mask.sum()), "state_share": float(mask.mean()),
            "n_cases_with_at_least_one": int(cases_with), "n_cases": len(set(case_ids)),
            "case_macro_share": float(np.mean(list(per_case.values()))),
            "case_macro_share_ci95_low": low, "case_macro_share_ci95_high": high,
        })
    write_csv(output_dir / "action_complementarity.csv", rows)
    return rows


def ask_information_analysis(case_ids: list[str], accuracy: np.ndarray, asks: list[dict[str, Any]],
                             output_dir: Path) -> list[dict[str, Any]]:
    unknown = np.asarray([bool(row.get("ask_response_unknown")) for row in asks], dtype=bool)
    nonempty = np.asarray([bool(row.get("ask_response_nonempty")) for row in asks], dtype=bool)
    novel = np.asarray([bool(row.get("ask_response_novel_text")) for row in asks], dtype=bool)
    review_accepted = np.asarray([bool(row.get("ask_response_review_accepted")) for row in asks], dtype=bool)
    transition_valid = np.asarray([bool((row.get("ask_state_transition") or {}).get("structurally_valid")) for row in asks], dtype=bool)
    info_proxy = nonempty & ~unknown & novel
    groups = [
        ("All ASK states", np.ones(len(asks), dtype=bool)),
        ("Unknown after bounded review", unknown),
        ("Nonempty, non-Unknown, novel-text proxy", info_proxy),
        ("No information proxy", ~info_proxy),
        ("Patient review accepted", review_accepted),
    ]
    ask_delta_accuracy = accuracy[:, 0] - accuracy[:, 1]
    ask_delta_utility = ask_delta_accuracy - MAIN_ASK_COST
    corrections = (accuracy[:, 0] == 1) & (accuracy[:, 1] == 0)
    degradations = (accuracy[:, 0] == 0) & (accuracy[:, 1] == 1)
    unchanged = accuracy[:, 0] == accuracy[:, 1]
    output = []
    for index, (group_name, selected) in enumerate(groups):
        acc_values = accuracy[:, 0]
        answer_values = accuracy[:, 1]
        metrics = {
            "ask_accuracy": acc_values,
            "answer_accuracy": answer_values,
            "ask_minus_answer_accuracy": ask_delta_accuracy,
            "ask_minus_answer_utility_at_ask_cost_0_01": ask_delta_utility,
            "ask_correction_rate": corrections.astype(np.float64),
            "ask_degradation_rate": degradations.astype(np.float64),
            "ask_unchanged_rate": unchanged.astype(np.float64),
        }
        row: dict[str, Any] = {
            "group": group_name,
            "n_states": int(selected.sum()),
            "state_share_of_asks": float(selected.mean()),
            "n_cases_with_group": len({c for c, keep in zip(case_ids, selected) if keep}),
            "unknown_states": int(np.sum(selected & unknown)),
            "proxy_information_states": int(np.sum(selected & info_proxy)),
            "review_accepted_states": int(np.sum(selected & review_accepted)),
            "structurally_valid_ask_transitions": int(np.sum(selected & transition_valid)),
            "ask_corrections": int(np.sum(selected & corrections)),
            "ask_degradations": int(np.sum(selected & degradations)),
            "ask_unchanged": int(np.sum(selected & unchanged)),
        }
        for metric_name, values in metrics.items():
            estimate, ci, case_count = bootstrap_conditional(values, case_ids, selected, 7000 + index * 20 + len(row))
            row[metric_name + "_case_macro"] = estimate
            row[metric_name + "_ci95_low"] = ci[0]
            row[metric_name + "_ci95_high"] = ci[1]
            row[metric_name + "_cases_in_bootstrap"] = case_count
            row[metric_name + "_state_pooled"] = float(np.mean(values[selected])) if selected.any() else float("nan")
        output.append(row)
    write_csv(output_dir / "ask_information.csv", output)
    (output_dir / "ask_information_definition.json").write_text(json.dumps({
        "effective_information_proxy": "ask_response_nonempty AND NOT ask_response_unknown AND ask_response_novel_text",
        "novel_text_definition": "normalized final patient reply is not an exact normalized substring of the prior public state, per V2 runner; this is lexical novelty, not clinical validity",
        "main_ask_cost": MAIN_ASK_COST,
        "unknown_is_final_reply_after_bounded_review": True,
        "gold_used_to_generate_patient_reply": False,
    }, indent=2) + "\n", encoding="utf-8")
    return output


def advantage_analysis(keys: list[tuple[str, str]], case_ids: list[str], accuracy: np.ndarray,
                       q_values: np.ndarray, output_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    gross = {
        "ASK_minus_ANSWER_gross_accuracy": accuracy[:, 0] - accuracy[:, 1],
        "DEFER_minus_ANSWER_gross_accuracy": accuracy[:, 2] - accuracy[:, 1],
    }
    stored = {
        "stored_Q_ASK_minus_ANSWER_original_costs_0.05_0.60": q_values[:, 0] - q_values[:, 1],
        "stored_Q_DEFER_minus_ANSWER_original_costs_0.05_0.60": q_values[:, 2] - q_values[:, 1],
    }
    advantage_rows = []
    all_advantages = {**gross, **stored}
    for name, values in all_advantages.items():
        by_case = case_means(values, case_ids)
        low, high = bootstrap_ci(by_case, 8100 + len(advantage_rows))
        advantage_rows.append({
            "advantage": name, "n_states": len(values),
            "positive_states": int(np.sum(values > 1e-12)),
            "zero_states": int(np.sum(np.abs(values) <= 1e-12)),
            "negative_states": int(np.sum(values < -1e-12)),
            "positive_state_share": float(np.mean(values > 1e-12)),
            "mean_case_macro": float(np.mean(list(by_case.values()))),
            "mean_case_macro_ci95_low": low, "mean_case_macro_ci95_high": high,
            "mean_abs": float(np.mean(np.abs(values))), "median_abs": float(np.median(np.abs(values))),
            "min": float(np.min(values)), "max": float(np.max(values)),
        })
    write_csv(output_dir / "q_advantage_distribution.csv", advantage_rows)

    cost_rows = []
    for action, gross_values, cost_grid in (
        ("ASK", gross["ASK_minus_ANSWER_gross_accuracy"], ASK_COST_GRID),
        ("DEFER", gross["DEFER_minus_ANSWER_gross_accuracy"], DEFER_COST_GRID),
    ):
        for cost in cost_grid:
            net = gross_values - cost
            positive = (net > 1e-12).astype(np.float64)
            by_case = case_means(positive, case_ids)
            low, high = bootstrap_ci(by_case, 9100 + int(cost * 1000) + (0 if action == "ASK" else 100))
            cost_rows.append({
                "action": action, "action_cost": cost, "n_states_positive_net_advantage": int(np.sum(net > 1e-12)),
                "n_states_zero_net_advantage": int(np.sum(np.abs(net) <= 1e-12)),
                "n_states_negative_net_advantage": int(np.sum(net < -1e-12)),
                "positive_state_share": float(positive.mean()),
                "positive_case_macro_share": float(np.mean(list(by_case.values()))),
                "positive_case_macro_share_ci95_low": low, "positive_case_macro_share_ci95_high": high,
                "mean_net_advantage": float(net.mean()), "mean_abs_net_advantage": float(np.abs(net).mean()),
            })
    write_csv(output_dir / "advantage_by_cost.csv", cost_rows)

    primary_ask = gross["ASK_minus_ANSWER_gross_accuracy"] - MAIN_ASK_COST
    primary_defer = gross["DEFER_minus_ANSWER_gross_accuracy"] - MAIN_DEFER_COST
    all_gross_equal = (accuracy[:, 0] == accuracy[:, 1]) & (accuracy[:, 1] == accuracy[:, 2])
    # The action set is evaluated at every point in the user-specified cost grid.
    best_set_by_state: dict[tuple[str, str], set[tuple[str, ...]]] = defaultdict(set)
    for ask_cost in ASK_COST_GRID:
        for defer_cost in DEFER_COST_GRID:
            utility = utility_matrix(accuracy, ask_cost, defer_cost)
            mask = np.isclose(utility, utility.max(axis=1, keepdims=True), atol=1e-12, rtol=0.0)
            for index, key in enumerate(keys):
                best_set_by_state[key].add(tuple(ACTION_ORDER[j] for j in range(3) if mask[index, j]))
    switch_summary = {
        "n_states_with_all_three_actions_equal_on_gross_accuracy": int(np.sum(all_gross_equal)),
        "n_states_with_positive_ASK_advantage_at_main_cost_0.01": int(np.sum(primary_ask > 1e-12)),
        "n_states_with_positive_DEFER_advantage_at_main_cost_0.05": int(np.sum(primary_defer > 1e-12)),
        "n_states_with_any_oracle_best_set_change_over_cost_grid": sum(len(v) > 1 for v in best_set_by_state.values()),
        "n_states_with_unique_oracle_action_change_over_cost_grid": sum(
            len({s[0] for s in v if len(s) == 1}) > 1 for v in best_set_by_state.values()
        ),
        "n_states_with_stable_oracle_best_set_over_cost_grid": sum(len(v) == 1 for v in best_set_by_state.values()),
        "primary_costs": {"ASK": MAIN_ASK_COST, "ANSWER": 0.0, "DEFER": MAIN_DEFER_COST},
        "primary_ask_net_advantage_mean": float(primary_ask.mean()),
        "primary_defer_net_advantage_mean": float(primary_defer.mean()),
    }
    (output_dir / "advantage_summary.json").write_text(json.dumps(switch_summary, indent=2) + "\n", encoding="utf-8")
    return advantage_rows, cost_rows, switch_summary


TOKEN_RE = re.compile(r"[\w]+", flags=re.UNICODE)


def public_tokens(public_state: list[dict[str, Any]]) -> list[str]:
    tokens = []
    for message in public_state:
        role = str(message.get("role", ""))
        if role not in {"user", "assistant"}:
            raise ValueError(f"non-public role in state feature input: {role}")
        tokens.extend(TOKEN_RE.findall(str(message.get("content", "")).lower()))
    return tokens


def public_structural_features(public_state: list[dict[str, Any]]) -> np.ndarray:
    role_counts = Counter(str(message.get("role", "")) for message in public_state)
    texts = [str(message.get("content", "")) for message in public_state]
    all_text = " ".join(texts)
    tokens = TOKEN_RE.findall(all_text.lower())
    return np.asarray([
        float(len(public_state)), float(len(all_text)), float(len(tokens)),
        float(role_counts.get("user", 0)), float(role_counts.get("assistant", 0)),
        float(all_text.count("?")), float(len(re.findall(r"\d+", all_text))),
        float(np.mean([len(text) for text in texts])) if texts else 0.0,
    ], dtype=np.float64)


def make_tfidf(train_documents: list[list[str]], test_documents: list[list[str]], max_features: int = 3000) -> tuple[np.ndarray, np.ndarray]:
    document_frequency = Counter()
    for tokens in train_documents:
        terms = set(tokens)
        terms.update(f"{a}__{b}" for a, b in zip(tokens, tokens[1:]))
        document_frequency.update(terms)
    eligible = [(term, count) for term, count in document_frequency.items() if count >= 2]
    eligible.sort(key=lambda item: (-item[1], item[0]))
    vocabulary = {term: index for index, (term, _) in enumerate(eligible[:max_features])}
    if not vocabulary:
        return np.zeros((len(train_documents), 0)), np.zeros((len(test_documents), 0))
    df_by_index = np.zeros(len(vocabulary), dtype=np.float64)
    for term, index in vocabulary.items():
        df_by_index[index] = document_frequency[term]
    idf = np.log((1.0 + len(train_documents)) / (1.0 + df_by_index)) + 1.0

    def transform(documents: list[list[str]]) -> np.ndarray:
        matrix = np.zeros((len(documents), len(vocabulary)), dtype=np.float64)
        for row_index, tokens in enumerate(documents):
            counts = Counter(tokens)
            counts.update(f"{a}__{b}" for a, b in zip(tokens, tokens[1:]))
            for term, count in counts.items():
                column = vocabulary.get(term)
                if column is not None:
                    matrix[row_index, column] = (1.0 + math.log(count)) * idf[column]
            norm = np.linalg.norm(matrix[row_index])
            if norm > 0:
                matrix[row_index] /= norm
        return matrix

    return transform(train_documents), transform(test_documents)


def grouped_folds(case_ids: list[str], n_folds: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    case_sizes = Counter(case_ids)
    ordered = sorted(case_sizes, key=lambda case: hashlib.sha256(f"{SEED}:{case}".encode()).hexdigest())
    fold_cases: list[list[str]] = [[] for _ in range(n_folds)]
    fold_sizes = [0] * n_folds
    for case in ordered:
        fold = min(range(n_folds), key=lambda i: (fold_sizes[i], i))
        fold_cases[fold].append(case)
        fold_sizes[fold] += case_sizes[case]
    folds = []
    for cases in fold_cases:
        heldout = np.asarray([case in set(cases) for case in case_ids], dtype=bool)
        folds.append((np.where(~heldout)[0], np.where(heldout)[0]))
    return folds


def ridge_grouped_cv(keys: list[tuple[str, str]], case_ids: list[str], q_by_key: dict,
                     output_dir: Path) -> dict[str, Any]:
    public_states = [q_by_key[key]["metadata"]["public_state"] for key in keys]
    documents = [public_tokens(state) for state in public_states]
    structural = np.vstack([public_structural_features(state) for state in public_states])
    targets = np.column_stack([
        np.asarray([float(q_by_key[key]["Q_ASK"]) + 0.05 - float(q_by_key[key]["Q_ANSWER"]) for key in keys]),
        np.asarray([float(q_by_key[key]["Q_DEFER"]) + 0.60 - float(q_by_key[key]["Q_ANSWER"]) for key in keys]),
    ])
    # Remove the source collection costs from stored Q before diagnosing public-state signal.
    # This reconstructs observed gross accuracy advantages; it does not alter training labels.
    predictions = np.full_like(targets, np.nan)
    baseline_predictions = np.full_like(targets, np.nan)
    fold_sizes = []
    alpha = 1.0
    for train_idx, test_idx in grouped_folds(case_ids, n_folds=5):
        train_docs = [documents[i] for i in train_idx]
        test_docs = [documents[i] for i in test_idx]
        train_tfidf, test_tfidf = make_tfidf(train_docs, test_docs)
        structural_train = structural[train_idx]
        structural_test = structural[test_idx]
        struct_mean = structural_train.mean(axis=0)
        struct_std = structural_train.std(axis=0)
        struct_std[struct_std < 1e-8] = 1.0
        x_train = np.column_stack([train_tfidf, (structural_train - struct_mean) / struct_std])
        x_test = np.column_stack([test_tfidf, (structural_test - struct_mean) / struct_std])

        x_mean = x_train.mean(axis=0)
        y_mean = targets[train_idx].mean(axis=0)
        x_centered = x_train - x_mean
        y_centered = targets[train_idx] - y_mean
        gram = x_centered @ x_centered.T
        dual_weights = np.linalg.solve(gram + alpha * np.eye(len(train_idx)), y_centered)
        predictions[test_idx] = y_mean + (x_test - x_mean) @ x_centered.T @ dual_weights
        baseline_predictions[test_idx] = y_mean
        fold_sizes.append({"train_cases": len(set(case_ids[i] for i in train_idx)), "test_cases": len(set(case_ids[i] for i in test_idx))})

    results = {}
    for target_index, name in enumerate(("gross_ASK_minus_ANSWER", "gross_DEFER_minus_ANSWER")):
        model_error = np.abs(targets[:, target_index] - predictions[:, target_index])
        baseline_error = np.abs(targets[:, target_index] - baseline_predictions[:, target_index])
        model_by_case = case_means(model_error, case_ids)
        baseline_by_case = case_means(baseline_error, case_ids)
        paired_delta = {case: model_by_case[case] - baseline_by_case[case] for case in model_by_case}
        low, high = bootstrap_ci(paired_delta, 11000 + target_index)
        denominator = float(np.sum((targets[:, target_index] - targets[:, target_index].mean()) ** 2))
        r2 = 1.0 - float(np.sum((targets[:, target_index] - predictions[:, target_index]) ** 2)) / denominator if denominator > 1e-12 else float("nan")
        results[name] = {
            "states": len(keys), "cases": len(set(case_ids)), "folds": 5,
            "ridge_alpha_fixed_no_search": alpha,
            "model_case_macro_mae": float(np.mean(list(model_by_case.values()))),
            "constant_mean_case_macro_mae": float(np.mean(list(baseline_by_case.values()))),
            "model_minus_constant_case_mae": float(np.mean(list(paired_delta.values()))),
            "model_minus_constant_case_mae_ci95": [low, high],
            "cases_model_lower_mae": sum(value < 0 for value in paired_delta.values()),
            "cases_constant_lower_mae": sum(value > 0 for value in paired_delta.values()),
            "oof_state_r2": r2,
            "oof_state_mae": float(np.mean(model_error)),
            "target_mean": float(targets[:, target_index].mean()),
        }
    payload = {
        "purpose": "small diagnostic ridge CV; not the Window A Q/Router model and not a performance estimate",
        "feature_source": "public_state only: fold-fitted unigram/bigram TF-IDF plus turn/length/question/digit statistics",
        "case_split": "deterministic five-fold GroupKFold-style split; no case_id appears in both sides of a fold",
        "fold_case_counts": fold_sizes,
        "target": "gross accuracy differences reconstructed from raw correctness by adding back the original ASK=0.05 and DEFER=0.60 costs",
        "model": "dual-space Ridge, alpha=1.0, no hyperparameter search",
        "data_limitations": ["single rollout per state/action", "50 cases", "146 correlated states"],
        "results": results,
    }
    (output_dir / "feature_advantage_diagnostic.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return payload


def api_call_estimates(grouped: dict[tuple[str, str], dict[str, dict[str, Any]]],
                       case_ids: list[str], accuracy: np.ndarray, keys: list[tuple[str, str]],
                       total_train_states: int, output_dir: Path) -> dict[str, Any]:
    role_by_action: dict[str, Counter[str]] = {action: Counter() for action in ACTION_ORDER}
    case_role_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for key, actions in grouped.items():
        for action, row in actions.items():
            for role, count in (row.get("api_call_counts") or {}).items():
                role_by_action[action][role] += int(count)
                case_role_counts[key[0]][role] += int(count)
    observed_roles = sorted({role for counts in role_by_action.values() for role in counts})
    repeats_existing = {role: int(2 * sum(counts[role] for counts in role_by_action.values())) for role in observed_roles}
    additional_current_rollouts = len(keys) * len(ACTION_ORDER) * 2
    missing_states = max(0, total_train_states - len(keys))
    new_rollouts_by_action = {action: missing_states * 3 for action in ACTION_ORDER}
    new_calls_by_role = {}
    for role in observed_roles:
        expected = 0.0
        for action in ACTION_ORDER:
            observed_n = sum(1 for actions in grouped.values() if action in actions)
            mean_calls = role_by_action[action][role] / max(1, observed_n)
            expected += mean_calls * new_rollouts_by_action[action]
        new_calls_by_role[role] = expected
    full_train_extra_rollouts = additional_current_rollouts + sum(new_rollouts_by_action.values())
    full_train_calls_by_role = {role: repeats_existing.get(role, 0) + new_calls_by_role.get(role, 0.0) for role in observed_roles}

    # A deliberately small, stratified resampling design for diagnosis only:
    # select up to four cases from each prespecified signal stratum, then repeat
    # all three actions twice for every state in the union of selected cases.
    ask_delta = accuracy[:, 0] - accuracy[:, 1] - MAIN_ASK_COST
    defer_delta = accuracy[:, 2] - accuracy[:, 1] - MAIN_DEFER_COST
    answer_defer_disagreement = accuracy[:, 1] != accuracy[:, 2]
    asks = [grouped[key]["ASK"] for key in keys]
    unknown = np.asarray([bool(row.get("ask_response_unknown")) for row in asks])
    near_zero = (np.abs(ask_delta) <= 0.05 + 1e-12) | (np.abs(defer_delta) <= 0.05 + 1e-12)
    stratum_masks = {
        "near_zero_advantage": near_zero,
        "ASK_unknown": unknown,
        "Base_Expert_disagreement": answer_defer_disagreement,
    }
    selected_cases_by_stratum = {}
    selected_union: set[str] = set()
    for stratum, mask in stratum_masks.items():
        candidates = {case for case, keep in zip(case_ids, mask) if keep}
        ranked = sorted(candidates, key=lambda case: hashlib.sha256(f"{SEED}:{stratum}:{case}".encode()).hexdigest())
        selected = ranked[:4]
        selected_cases_by_stratum[stratum] = len(selected)
        selected_union.update(selected)
    subset_states = [key for key in keys if key[0] in selected_union]
    subset_calls = Counter()
    for key in subset_states:
        for action in ACTION_ORDER:
            for role, count in (grouped[key][action].get("api_call_counts") or {}).items():
                subset_calls[role] += int(count)
    targeted_plan = {
        "selection_rule": "up to four case_id clusters per prespecified stratum (near-zero primary-cost advantage, final ASK Unknown, Base/Expert correctness disagreement); stable hash sampling within stratum; deduplicate cases",
        "selected_case_count_union": len(selected_union),
        "selected_state_count": len(subset_states),
        "new_state_action_rollouts_if_K3": len(subset_states) * 3 * 2,
        "estimated_additional_api_calls_by_role_if_same_call_pattern": {role: int(2 * count) for role, count in subset_calls.items()},
        "this_subset_does_not_make_the_full_50_or_70_case_dataset_K3": True,
    }
    payload = {
        "observed_api_calls_by_action_and_role": {action: dict(counts) for action, counts in role_by_action.items()},
        "current_50_case_full_K3": {
            "cases_to_resample": len(set(case_ids)), "states": len(keys),
            "additional_state_action_rollouts": additional_current_rollouts,
            "estimated_additional_api_calls_by_role": repeats_existing,
            "estimated_total_api_calls": int(sum(repeats_existing.values())),
        },
        "full_70_case_train_pool_K3": {
            "additional_independent_cases": max(0, 70 - len(set(case_ids))),
            "missing_public_states_from_frozen_train_state_count": missing_states,
            "extra_rollouts_existing_states": additional_current_rollouts,
            "extra_rollouts_new_states": sum(new_rollouts_by_action.values()),
            "total_additional_valid_action_rollouts": full_train_extra_rollouts,
            "estimated_additional_api_calls_by_role": {role: round(value, 1) for role, value in full_train_calls_by_role.items()},
            "estimated_total_additional_api_calls": round(sum(full_train_calls_by_role.values())),
            "call_estimates_assume_same_mean_calls_per_action_as_current_data": True,
        },
        "targeted_diagnostic_subset": targeted_plan,
        "no_models_called_in_this_analysis": True,
    }
    (output_dir / "repeat_sampling_estimate.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def make_plots(cost_rows: list[dict[str, Any]], case_ids: list[str], accuracy: np.ndarray,
               output_dir: Path) -> None:
    curve_rows = [{k: row[k] for k in ("ask_cost", "defer_cost", "policy", "accuracy", "utility", "ask_rate", "defer_rate")}
                  for row in cost_rows]
    write_csv(output_dir / "accuracy_utility_curve.csv", curve_rows)
    colors = {"Always ANSWER": "#3b6fb6", "Always ASK": "#d9812b", "Always DEFER": "#5a9b65", "Retrospective Oracle": "#bd4a4a"}
    fig, ax = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    for policy in colors:
        subset = [row for row in cost_rows if row["policy"] == policy]
        ax.scatter([row["accuracy"] for row in subset], [row["utility"] for row in subset],
                   label=policy, s=34 if policy != "Retrospective Oracle" else 40,
                   color=colors[policy], alpha=0.78, edgecolors="none")
    primary = [row for row in cost_rows if row["ask_cost"] == MAIN_ASK_COST and row["defer_cost"] == MAIN_DEFER_COST]
    for row in primary:
        ax.scatter(row["accuracy"], row["utility"], marker="*", s=160, color=colors[row["policy"]], edgecolor="black", linewidth=0.35, zorder=5)
    ax.set_xlabel("Case-macro Accuracy")
    ax.set_ylabel("Case-macro Utility")
    ax.set_title("Cost sensitivity: Accuracy–Utility (stars = ASK 0.01, DEFER 0.05)")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.savefig(output_dir / "accuracy_utility_curve.svg")
    plt.close(fig)

    q_advantages = np.column_stack([accuracy[:, 0] - accuracy[:, 1], accuracy[:, 2] - accuracy[:, 1]])
    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    bins = np.arange(-1.1, 1.21, 0.2)
    ax.hist(q_advantages[:, 0], bins=bins, alpha=0.65, label="ASK − ANSWER (gross)", color="#d9812b")
    ax.hist(q_advantages[:, 1], bins=bins, alpha=0.60, label="DEFER − ANSWER (gross)", color="#5a9b65")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Observed correctness advantage (single rollout)")
    ax.set_ylabel("States")
    ax.set_title("One-shot gross advantage labels")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(output_dir / "gross_advantage_distribution.svg")
    plt.close(fig)


def summarize_ask_noise(asks: list[dict[str, Any]], trajectories: list[dict[str, Any]],
                        grouped: dict[tuple[str, str], dict[str, dict[str, Any]]]) -> dict[str, Any]:
    ask_unknown = [bool(row.get("ask_response_unknown")) for row in asks]
    ask_proxy = [bool(row.get("ask_response_nonempty")) and not bool(row.get("ask_response_unknown"))
                 and bool(row.get("ask_response_novel_text")) for row in asks]
    review_attempts = sum(int(row.get("ask_response_review_attempts", 0) or 0) for row in asks)
    review_accepted = sum(bool(row.get("ask_response_review_accepted")) for row in asks)
    role_calls = Counter()
    role_attempts = Counter()
    finish_reasons = Counter()
    format_valid = Counter()
    privacy = Counter()
    token_usage = Counter()
    for row in trajectories:
        for role, count in (row.get("api_call_counts") or {}).items():
            role_calls[role] += int(count)
        for role, count in (row.get("api_attempt_counts") or {}).items():
            role_attempts[role] += int(count)
        format_valid[str(bool(row.get("format_valid")))]+=1
        privacy[str(row.get("prompt_privacy_checks_passed"))]+=1
        token_usage[str(bool(row.get("token_usage_complete")))]+=1
        for reason in row.get("finish_reasons") or []:
            finish_reasons[str(reason)] += 1
    return {
        "state_action_sample_count_distribution": {"K=1": len(trajectories)},
        "return_std_missing_state_action_labels": len(trajectories),
        "cannot_estimate_within_state_action_return_std_from_K1": True,
        "ask_states": len(asks),
        "ask_final_unknown_states": int(sum(ask_unknown)),
        "ask_final_unknown_rate_state_pooled": float(np.mean(ask_unknown)),
        "ask_nonempty_nonunknown_novel_text_proxy_states": int(sum(ask_proxy)),
        "ask_information_proxy_rate_state_pooled": float(np.mean(ask_proxy)),
        "bounded_review_calls": review_attempts,
        "bounded_review_accepted": review_accepted,
        "ask_state_transition_structurally_valid": sum(bool((row.get("ask_state_transition") or {}).get("structurally_valid")) for row in asks),
        "format_valid_distribution": dict(format_valid),
        "prompt_privacy_check_distribution": dict(privacy),
        "token_usage_complete_distribution": dict(token_usage),
        "finish_reason_distribution": dict(finish_reasons),
        "api_calls_by_role": dict(role_calls),
        "api_attempts_by_role": dict(role_attempts),
        "api_retries_by_role_estimate": {role: max(0, int(role_attempts[role] - role_calls[role])) for role in set(role_calls) | set(role_attempts)},
        "continuation_policy_id": next(iter({row["continuation_policy_id"] for actions in grouped.values() for row in actions.values()})),
    }


def primary_and_oracle_summary(cost_rows: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    rows = [row for row in cost_rows if row["ask_cost"] == MAIN_ASK_COST and row["defer_cost"] == MAIN_DEFER_COST]
    lookup = {row["policy"]: row for row in rows}
    answer = lookup["Always ANSWER"]
    oracle = lookup["Retrospective Oracle"]
    summary = {
        "predeclared_primary_costs": {"ASK": MAIN_ASK_COST, "ANSWER": 0.0, "DEFER": MAIN_DEFER_COST},
        "always_answer": {key: answer[key] for key in ("accuracy", "utility", "ask_rate", "defer_rate", "accuracy_ci95_low", "accuracy_ci95_high", "utility_ci95_low", "utility_ci95_high")},
        "retrospective_oracle": {key: oracle[key] for key in ("accuracy", "utility", "ask_rate", "defer_rate", "oracle_accuracy_gain_vs_answer", "oracle_utility_gain_vs_answer", "oracle_accuracy_gain_vs_answer_ci95_low", "oracle_accuracy_gain_vs_answer_ci95_high", "oracle_utility_gain_vs_answer_ci95_low", "oracle_utility_gain_vs_answer_ci95_high", "oracle_unique_best_states", "oracle_tied_best_states")},
        "oracle_is_hindsight_only": True,
    }
    (output_dir / "oracle_upper_bound.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    gap_rows = []
    for ask_cost in ASK_COST_GRID:
        for defer_cost in DEFER_COST_GRID:
            by_policy = {row["policy"]: row for row in cost_rows if row["ask_cost"] == ask_cost and row["defer_cost"] == defer_cost}
            oracle_row = by_policy["Retrospective Oracle"]
            gap_rows.append({
                "ask_cost": ask_cost, "defer_cost": defer_cost,
                "always_answer_accuracy": by_policy["Always ANSWER"]["accuracy"],
                "oracle_accuracy": oracle_row["accuracy"],
                "oracle_accuracy_gain_vs_answer": oracle_row["oracle_accuracy_gain_vs_answer"],
                "oracle_accuracy_gain_ci95_low": oracle_row["oracle_accuracy_gain_vs_answer_ci95_low"],
                "oracle_accuracy_gain_ci95_high": oracle_row["oracle_accuracy_gain_vs_answer_ci95_high"],
                "always_answer_utility": by_policy["Always ANSWER"]["utility"],
                "oracle_utility": oracle_row["utility"],
                "oracle_utility_gain_vs_answer": oracle_row["oracle_utility_gain_vs_answer"],
                "oracle_utility_gain_ci95_low": oracle_row["oracle_utility_gain_vs_answer_ci95_low"],
                "oracle_utility_gain_ci95_high": oracle_row["oracle_utility_gain_vs_answer_ci95_high"],
                "oracle_ask_rate": oracle_row["ask_rate"],
                "oracle_defer_rate": oracle_row["defer_rate"],
            })
    write_csv(output_dir / "oracle_vs_answer.csv", gap_rows)
    return summary


def write_noise_summary(noise: dict[str, Any], output_dir: Path) -> None:
    (output_dir / "label_noise_diagnostics.json").write_text(json.dumps(noise, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="read-only merged run directory containing q_dataset.jsonl, pilot_trajectories.jsonl, merged_manifest.json")
    parser.add_argument("--case-split", type=Path, required=True, help="frozen official case_id split JSON")
    parser.add_argument("--output-dir", type=Path, required=True, help="new independent report directory; must not already exist")
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    split_path = args.case_split.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite analysis outputs: {output_dir}")
    output_dir.mkdir(parents=True)

    q_path = data_dir / "q_dataset.jsonl"
    trajectory_path = data_dir / "pilot_trajectories.jsonl"
    q_rows = read_jsonl(q_path)
    trajectories = read_jsonl(trajectory_path)
    audit = verify_data_version(data_dir, q_rows, trajectories, split_path)
    keys, case_ids, accuracy, q_values, asks = state_arrays(audit["grouped"], audit["q_by_key"])
    cost_rows, oracle_distribution, switch_summary = summarize_cost_grid(keys, case_ids, accuracy, output_dir)
    complementarity = complementarity_table(case_ids, accuracy, output_dir)
    ask_info = ask_information_analysis(case_ids, accuracy, asks, output_dir)
    advantage_rows, advantage_cost_rows, advantage_summary = advantage_analysis(keys, case_ids, accuracy, q_values, output_dir)
    feature_diagnostic = ridge_grouped_cv(keys, case_ids, audit["q_by_key"], output_dir)
    call_estimates = api_call_estimates(
        audit["grouped"], case_ids, accuracy, keys,
        audit["n_total_train_states_from_manifest"], output_dir,
    )
    noise = summarize_ask_noise(asks, trajectories, audit["grouped"])
    write_noise_summary(noise, output_dir)
    make_plots(cost_rows, case_ids, accuracy, output_dir)
    oracle_summary = primary_and_oracle_summary(cost_rows, output_dir)

    base_manifest = audit["manifest"]["batches"][0]
    source_run_manifest = read_json(Path(base_manifest["run_manifest"]))
    base_identity = source_run_manifest.get("models", {}).get("base", {})
    expert_identity = source_run_manifest.get("models", {}).get("expert", {})
    data_version = {
        "source_data_dir": str(data_dir),
        "source_q_dataset_sha256": audit["q_sha256"],
        "source_trajectory_sha256": audit["trajectory_sha256"],
        "source_merged_manifest_sha256": sha256_file(data_dir / "merged_manifest.json"),
        "source_config_sha256": audit["config_sha256"],
        "official_case_split_sha256": audit["case_split_sha256"],
        "source_status": audit["manifest"].get("status"),
        "case_count": len(audit["cases"]), "state_count": len(keys),
        "trajectory_count": len(trajectories), "action_order": list(ACTION_ORDER),
        "valid_actions_by_action": dict(Counter(row["action"] for row in trajectories)),
        "continuation_policy_id": audit["continuation_policy_id"],
        "original_costs": audit["original_costs"],
        "Base": {"model": base_identity.get("model"), "revision": base_identity.get("model_revision")},
        "Expert": {"model": expert_identity.get("model"), "revision": expert_identity.get("model_revision")},
        "frozen_holdout_opened": False,
        "validation_or_internal_test_used_for_fitting": False,
        "manifest_source_batch_count": audit["n_source_batches"],
        "manifest_frozen_train_public_state_count": audit["n_total_train_states_from_manifest"],
        "source_project_commits": audit["source_project_commits"],
        "source_project_branches": audit["source_project_branches"],
        "runner_source_hash_pinned": audit["runner_source_hash_pinned"],
        "inference_config": {key: audit["config"]["inference"].get(key) for key in ("temperature", "patient_temperature", "top_p", "api_retries")},
    }
    (output_dir / "data_version.json").write_text(json.dumps(data_version, indent=2) + "\n", encoding="utf-8")

    answer_idx = ANSWER_INDEX
    state_counts = {action: int(np.sum(accuracy[:, index] == 1)) for index, action in enumerate(ACTION_ORDER)}
    case_any = {}
    for action, index in (("ASK", 0), ("ANSWER", 1), ("DEFER", 2)):
        case_any[action] = sum(any(accuracy[j, index] == 1 for j, case in enumerate(case_ids) if case == c) for c in set(case_ids))
    main_rows = {row["policy"]: row for row in cost_rows if row["ask_cost"] == MAIN_ASK_COST and row["defer_cost"] == MAIN_DEFER_COST}
    analysis_summary = {
        "data_version": data_version,
        "state_outcome_correct_counts": state_counts,
        "cases_with_at_least_one_correct_outcome_by_action": case_any,
        "expert_vs_base": {row["category"]: {k: row[k] for k in ("n_states", "state_share", "n_cases_with_at_least_one", "case_macro_share", "case_macro_share_ci95_low", "case_macro_share_ci95_high")} for row in complementarity if row["comparison"] == "Expert vs Base"},
        "ask_vs_answer": {row["category"]: {k: row[k] for k in ("n_states", "state_share", "n_cases_with_at_least_one", "case_macro_share", "case_macro_share_ci95_low", "case_macro_share_ci95_high")} for row in complementarity if row["comparison"] == "ASK vs ANSWER"},
        "ask_information": ask_info,
        "primary_cost_policy_metrics": main_rows,
        "oracle_upper_bound": oracle_summary,
        "oracle_cost_grid_best_action_distribution": oracle_distribution,
        "oracle_action_switch_summary": switch_summary,
        "q_advantage_distribution": advantage_rows,
        "advantage_cost_summary": advantage_summary,
        "feature_advantage_diagnostic": feature_diagnostic,
        "label_noise_diagnostics": noise,
        "repeat_sampling_estimate": call_estimates,
        "statistics": {
            "bootstrap_unit": "case_id; all states for sampled case remain together",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "seed": SEED,
            "point_estimate_for_policy_metrics": "macro average: state metric within case, then equal weight per case",
            "oracle_tie_handling": "fractional mass equally across tied maximizing actions",
        },
        "limitations": [
            "K=1, so within-state/action return variance and stable expected Q are not estimable",
            "Retrospective Oracle uses observed labels and is an upper bound only",
            "ask_response_novel_text is lexical novelty, not clinical information validity",
            "case-grouped Ridge CV is a low-capacity feature diagnostic, not a trained Router or generalization claim",
        ],
    }
    (output_dir / "analysis_summary.json").write_text(json.dumps(analysis_summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    run_manifest = {
        "script": str(Path(__file__).resolve()),
        "command": ["python", "-m", "experiments.ea_atpo_v2.window_b_diagnostics.analyze_window_b",
                    "--data-dir", str(data_dir), "--case-split", str(split_path), "--output-dir", str(output_dir)],
        "python": __import__("sys").version,
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "created_from_source_commit": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "model_services_started": False,
        "window_a_artifacts_modified": False,
    }
    (output_dir / "analysis_run_manifest.json").write_text(json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "completed",
        "cases": len(audit["cases"]), "states": len(keys), "trajectories": len(trajectories),
        "output_dir": str(output_dir),
        "main_costs": {"ASK": MAIN_ASK_COST, "DEFER": MAIN_DEFER_COST},
        "oracle_upper_bound": oracle_summary["retrospective_oracle"],
        "feature_cv": feature_diagnostic["results"],
        "repeat_sampling_estimate": call_estimates["full_70_case_train_pool_K3"],
    }, indent=2))


if __name__ == "__main__":
    main()
