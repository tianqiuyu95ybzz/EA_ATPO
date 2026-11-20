#!/usr/bin/env python3
"""Offline answer-key Oracle evaluation on the already-used V2 Dev cases.

This runner does not train models, call a model service, or use Gold as a
router feature. It verifies the frozen Q-guided checkpoints against stored
Dev scores, uses their stored public-state decisions, and joins outcomes only
after decisions have been fixed. Per-case output is private by default.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import stat
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

# Allow direct script execution from any working directory in this repository.
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.ea_atpo_v2.training.data import (
    load_public_embedding_cache,
    load_q_dataset,
)
from experiments.ea_atpo_v2.training.features import Standardizer


ACTIONS = ("ASK", "ANSWER", "DEFER")
SEEDS = (17, 42, 2026)
DEFAULT_ROUTER_ROOT = Path(
    "/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO_V1_router_training"
)
DEFAULT_SOURCE = Path(
    "/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO_V1/"
    "experiments/expert_sft/data/source/dataset/merged_train_dataset.jsonl"
)
EXPECTED_SOURCE_SHA256 = "434a6734dce931262f2ed9e3201a3751e57db3609b1612ab9666835ac8832fdc"
EXPECTED_V3_SPLIT_SHA256 = "b55d889a713b2b1d32afc25d89aaf9205891a1ee5fbaae470702ff31a9f81267"
DEFAULT_COSTS = (0.05, 0.10, 0.20, 0.60)
ASK_COST = 0.01


class TinyHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_option(value: Any) -> str | None:
    match = re.fullmatch(r"\s*(?:Final Answer:\s*)?([A-D])\s*", str(value), re.I)
    return match.group(1).upper() if match else None


def source_answer(source_row: Mapping[str, Any]) -> str:
    value = source_row.get("ground_truth")
    value = value.get("answer") if isinstance(value, Mapping) else value
    parsed = parse_option(value)
    if parsed is None:
        raise ValueError("unparseable source answer key")
    return parsed


def answer_prompt(source_row: Mapping[str, Any]) -> str:
    prompt = source_row.get("prompt")
    if not isinstance(prompt, list):
        raise ValueError("source row prompt is not a message list")
    for message in prompt:
        if isinstance(message, Mapping) and message.get("role") == "user":
            return str(message.get("content", ""))
    raise ValueError("source row has no user prompt")


def extract_user(state: Sequence[Mapping[str, Any]]) -> str | None:
    for message in state:
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return None


def _load_checkpoint_scores(
    *,
    checkpoint: Path,
    expected_sha256: str,
    expected_seed: int,
    dev_examples: Sequence[Any],
    embedding_vectors: Mapping[tuple[str, str], np.ndarray],
    saved_predictions: Mapping[tuple[str, str, int], Mapping[str, Any]],
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    actual_sha = sha256_file(checkpoint)
    if actual_sha != expected_sha256:
        raise ValueError(f"checkpoint hash mismatch for seed {expected_seed}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("feature_kind") != "embedding" or payload.get("model") != "q_guided_distillation":
        raise ValueError("checkpoint is not the frozen embedding Q-guided distillation model")
    if int(payload.get("seed", -1)) != expected_seed:
        raise ValueError("checkpoint seed differs from the frozen candidate manifest")
    vectors = np.stack([
        np.asarray(embedding_vectors[(row.case_id, row.state_id)], dtype=np.float32)
        for row in dev_examples
    ])
    standardizer = Standardizer.from_json(payload["standardizer"])
    x = standardizer.transform(vectors)
    state = payload["state"]
    input_dim = int(state["net.0.weight"].shape[1])
    hidden_dim = int(state["net.0.weight"].shape[0])
    output_dim = int(state["net.2.weight"].shape[0])
    head = TinyHead(input_dim, hidden_dim, output_dim)
    head.load_state_dict(state)
    head.eval()
    with torch.no_grad():
        scores = head(torch.as_tensor(x, dtype=torch.float32)).numpy()
    scores = scores * float(payload["temperature"])
    result: dict[tuple[str, str], dict[str, Any]] = {}
    max_abs_error = 0.0
    for row, score in zip(dev_examples, scores):
        key = (row.case_id, row.state_id)
        saved = saved_predictions.get((key[0], key[1], expected_seed))
        if saved is None:
            raise ValueError(f"stored prediction is missing for seed {expected_seed}")
        saved_scores = np.asarray(saved["predicted_scores"], dtype=np.float32)
        if saved_scores.shape != score.shape:
            raise ValueError("stored prediction has a different score shape")
        max_abs_error = max(max_abs_error, float(np.max(np.abs(saved_scores - score))))
        action_by_argmax = _prediction_argmax(score)
        if str(saved.get("predicted_action")) != action_by_argmax:
            raise ValueError("stored action does not match the frozen checkpoint argmax")
        result[key] = {
            "action": action_by_argmax,
            "scores": [float(value) for value in score],
        }
    if max_abs_error > 1e-5:
        raise ValueError(f"checkpoint replay differs from stored scores: {max_abs_error}")
    audit = {
        "seed": expected_seed,
        "checkpoint_sha256": actual_sha,
        "states_replayed": len(result),
        "max_abs_score_difference": max_abs_error,
        "matched_saved_action_and_scores": True,
    }
    return result, audit


def oracle_conditional_result(base_correct: int) -> tuple[int, int]:
    """Return (final correctness, Oracle calls) for the ideal selector."""
    return (1, 0) if base_correct else (1, 1)


def selected_oracle_result(
    proposed_action: str, base_correct: int, ask_correct: int
) -> tuple[int, int, str]:
    """Oracle safety gate: preserve a correct Base answer; correct Base errors."""
    if proposed_action == "DEFER":
        if base_correct:
            return base_correct, 0, "ANSWER"
        return 1, 1, "DEFER"
    if proposed_action == "ASK":
        return ask_correct, 0, "ASK"
    return base_correct, 0, "ANSWER"


def row_result(row: Mapping[str, Any], method: str, seed: int | None, defer_cost: float) -> dict[str, Any]:
    base = int(row["correctness"]["ANSWER"])
    ask = int(row["correctness"]["ASK"])
    expert = int(row["correctness"]["DEFER"])
    counts = row["action_counts"]
    if method == "always_answer":
        action, final, asks, defers, executed = "ANSWER", base, 0, 0, "ANSWER"
    elif method == "always_ask":
        action, final, asks, defers, executed = "ASK", ask, int(counts["ASK"]), 0, "ASK"
    elif method == "always_defer_oracle":
        action, final, asks, defers, executed = "DEFER", 1, 0, 1, "DEFER"
    elif method == "always_defer_actual":
        action, final, asks, defers, executed = "DEFER", expert, 0, 1, "DEFER"
    elif method == "oracle_conditional":
        final, defers = oracle_conditional_result(base)
        action, asks, executed = ("DEFER" if defers else "ANSWER"), 0, ("DEFER" if defers else "ANSWER")
    elif method in {"learned_actual", "learned_oracle"}:
        if seed is None:
            raise ValueError("learned policy requires a fixed seed")
        action = str(row["router"][str(seed)]["action"])
        if method == "learned_actual":
            final = {"ANSWER": base, "ASK": ask, "DEFER": expert}[action]
            asks = int(counts["ASK"]) if action == "ASK" else 0
            defers = 1 if action == "DEFER" else 0
            executed = action
        else:
            final, defers, executed = selected_oracle_result(action, base, ask)
            asks = int(counts["ASK"]) if executed == "ASK" else 0
    else:
        raise ValueError(f"unknown method: {method}")
    return {
        "base_correct": base,
        "action": action,
        "executed_action": executed,
        "final_correct": int(final),
        "ask_count": int(asks),
        "defer_count": int(defers),
        "utility": float(final) - ASK_COST * int(asks) - defer_cost * int(defers),
    }


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else float(numerator) / float(denominator)


PRIMARY_METRICS = (
    "accuracy", "utility", "ask_rate", "answer_rate", "defer_rate",
    "oracle_call_rate", "average_cost",
)


def case_metric_vectors(
    rows: Sequence[Mapping[str, Any]], method: str, defer_cost: float,
    seeds: Sequence[int] = (),
) -> dict[str, np.ndarray]:
    """One equally weighted metric value per case, averaging fixed seeds if given."""
    seed_list: Sequence[int | None] = list(seeds) if seeds else (None,)
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["case_id"])].append(row)
    output = {metric: [] for metric in PRIMARY_METRICS}
    for case_rows in grouped.values():
        by_seed = {metric: [] for metric in PRIMARY_METRICS}
        for seed in seed_list:
            results = [row_result(row, method, seed, defer_cost) for row in case_rows]
            by_seed["accuracy"].append(float(np.mean([x["final_correct"] for x in results])))
            by_seed["utility"].append(float(np.mean([x["utility"] for x in results])))
            by_seed["ask_rate"].append(float(np.mean([x["action"] == "ASK" for x in results])))
            by_seed["answer_rate"].append(float(np.mean([x["action"] == "ANSWER" for x in results])))
            by_seed["defer_rate"].append(float(np.mean([x["action"] == "DEFER" for x in results])))
            by_seed["oracle_call_rate"].append(float(np.mean([x["defer_count"] for x in results])))
            by_seed["average_cost"].append(float(np.mean([
                ASK_COST * x["ask_count"] + defer_cost * x["defer_count"] for x in results
            ])))
        for metric in PRIMARY_METRICS:
            output[metric].append(float(np.mean(by_seed[metric])))
    return {key: np.asarray(values, dtype=np.float64) for key, values in output.items()}


def conditional_metrics(
    rows: Sequence[Mapping[str, Any]], method: str, defer_cost: float,
    seeds: Sequence[int] = (),
) -> dict[str, float | None]:
    seed_list: Sequence[int | None] = list(seeds) if seeds else (None,)
    accum = Counter()
    for row in rows:
        for seed in seed_list:
            result = row_result(row, method, seed, defer_cost)
            if result["base_correct"]:
                accum["base_correct_den"] += 1
                accum["base_correct_kept"] += result["final_correct"]
            else:
                accum["base_error_den"] += 1
                accum["base_error_fixed"] += result["final_correct"]
            if result["defer_count"]:
                accum["defer_calls"] += result["defer_count"]
                if not result["base_correct"] and result["final_correct"]:
                    accum["useful_defer"] += 1
                if not result["base_correct"]:
                    accum["deferred_base_errors"] += result["defer_count"]
                    if not result["final_correct"]:
                        accum["failed_deferred_base_errors"] += result["defer_count"]
    return {
        "base_correct_retention": _safe_ratio(accum["base_correct_kept"], accum["base_correct_den"]),
        "base_error_correction": _safe_ratio(accum["base_error_fixed"], accum["base_error_den"]),
        "defer_precision_useful_corrections_per_call": _safe_ratio(accum["useful_defer"], accum["defer_calls"]),
        "defer_error_correction_failure_rate": _safe_ratio(accum["failed_deferred_base_errors"], accum["deferred_base_errors"]),
    }


def bootstrap_interval(
    case_values: np.ndarray, *, replicates: int, seed: int,
) -> list[float]:
    if case_values.ndim != 1 or len(case_values) < 2 or not np.isfinite(case_values).all():
        raise ValueError("case-cluster bootstrap requires at least two finite case values")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(case_values), size=(replicates, len(case_values)))
    draws = case_values[indices].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return [float(low), float(high)]


def conditional_bootstrap_intervals(
    rows: Sequence[Mapping[str, Any]], method: str, defer_cost: float,
    *, seeds: Sequence[int] = (), replicates: int = 10000, seed: int = 20261131,
) -> dict[str, list[float] | None]:
    seed_list: Sequence[int | None] = list(seeds) if seeds else (None,)
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["case_id"])].append(row)
    cases = sorted(grouped)
    names = ("retained", "base_correct", "corrected", "base_error", "useful_defer", "defer_calls",
             "failed_defer", "deferred_base_errors")
    cluster = {name: [] for name in names}
    for case_id in cases:
        tally = Counter()
        for row in grouped[case_id]:
            for model_seed in seed_list:
                result = row_result(row, method, model_seed, defer_cost)
                if result["base_correct"]:
                    tally["base_correct"] += 1
                    tally["retained"] += result["final_correct"]
                else:
                    tally["base_error"] += 1
                    tally["corrected"] += result["final_correct"]
                if result["defer_count"]:
                    tally["defer_calls"] += result["defer_count"]
                    if not result["base_correct"]:
                        tally["deferred_base_errors"] += result["defer_count"]
                        tally["useful_defer"] += int(result["final_correct"] == 1)
                        tally["failed_defer"] += int(result["final_correct"] == 0)
        for name in names:
            cluster[name].append(float(tally[name]))
    matrix = {name: np.asarray(values, dtype=np.float64) for name, values in cluster.items()}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(cases), size=(replicates, len(cases)))
    def ratio_ci(numerator: str, denominator: str) -> list[float] | None:
        numerator_sum = matrix[numerator][indices].sum(axis=1)
        denominator_sum = matrix[denominator][indices].sum(axis=1)
        valid = denominator_sum > 0
        if not valid.any():
            return None
        draws = numerator_sum[valid] / denominator_sum[valid]
        return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]
    return {
        "base_correct_retention": ratio_ci("retained", "base_correct"),
        "base_error_correction": ratio_ci("corrected", "base_error"),
        "defer_precision_useful_corrections_per_call": ratio_ci("useful_defer", "defer_calls"),
        "defer_error_correction_failure_rate": ratio_ci("failed_defer", "deferred_base_errors"),
    }


def metrics_for_rows(
    rows: Sequence[Mapping[str, Any]], method: str, defer_cost: float,
    seeds: Sequence[int] = (),
) -> dict[str, Any]:
    vectors = case_metric_vectors(rows, method, defer_cost, seeds=seeds)
    metrics: dict[str, Any] = {key: float(values.mean()) for key, values in vectors.items()}
    for key, values in vectors.items():
        metrics[f"{key}_seed_sd"] = 0.0
        if seeds:
            per_seed_values = [case_metric_vectors(rows, method, defer_cost, seeds=(seed,))[key].mean() for seed in seeds]
            metrics[f"{key}_seed_sd"] = float(np.std(per_seed_values, ddof=1))
    metrics.update(conditional_metrics(rows, method, defer_cost, seeds=seeds))
    metrics["case_cluster_bootstrap_ci95_conditional"] = conditional_bootstrap_intervals(
        rows, method, defer_cost, seeds=seeds,
    )
    seed_list: Sequence[int | None] = list(seeds) if seeds else (None,)
    flat_results = [row_result(row, method, seed, defer_cost) for seed in seed_list for row in rows]
    metrics["state_weighted"] = {
        "accuracy": float(np.mean([x["final_correct"] for x in flat_results])),
        "utility": float(np.mean([x["utility"] for x in flat_results])),
        "ask_rate": float(np.mean([x["action"] == "ASK" for x in flat_results])),
        "answer_rate": float(np.mean([x["action"] == "ANSWER" for x in flat_results])),
        "defer_proposal_rate": float(np.mean([x["action"] == "DEFER" for x in flat_results])),
        "oracle_call_rate": float(np.mean([x["defer_count"] for x in flat_results])),
        "average_cost": float(np.mean([ASK_COST * x["ask_count"] + defer_cost * x["defer_count"] for x in flat_results])),
    }
    metrics["per_seed_case_macro"] = {
        str(seed): {key: float(case_metric_vectors(rows, method, defer_cost, seeds=(seed,))[key].mean())
                    for key in PRIMARY_METRICS}
        for seed in seeds
    }
    return metrics


def _prediction_argmax(scores: Sequence[float]) -> str:
    if len(scores) != len(ACTIONS):
        raise ValueError("router scores must follow ASK, ANSWER, DEFER order")
    return ACTIONS[int(np.argmax(np.asarray(scores, dtype=np.float64)))]


def build_state_rows(
    *, router_root: Path, source_path: Path, q_path: Path, trajectory_path: Path,
    split_path: Path, prediction_path: Path, candidate_path: Path,
    embedding_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    q_examples = load_q_dataset(q_path)
    q_by_key = {(row.case_id, row.state_id): row for row in q_examples}
    if len(q_by_key) != 146:
        raise ValueError("expected the frozen 146-state Q dataset")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if split.get("unit") != "case_id":
        raise ValueError("frozen split is not case_id based")
    train_ids, dev_ids = set(map(str, split["train"])), set(map(str, split["dev"]))
    if len(train_ids) != 35 or len(dev_ids) != 15 or train_ids & dev_ids:
        raise ValueError("frozen split must have disjoint 35 Train / 15 Dev cases")
    dev_examples = [row for row in q_examples if row.case_id in dev_ids]
    if len(dev_examples) != 43 or {row.case_id for row in dev_examples} != dev_ids:
        raise ValueError("frozen Dev set must contain 43 states from 15 cases")
    if any(not row.valid_mask.all() for row in dev_examples):
        raise ValueError("a frozen Dev state has an invalid action label")
    if any(row.raw_metadata.get("gold_or_future_fields_in_features") is not False
           or row.raw_metadata.get("feature_boundary") != "public_state_before_action_only"
           for row in q_examples):
        raise ValueError("Q dataset does not document a public-state-only feature boundary")

    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    if candidate.get("status") != "frozen_candidates_confirmation_blocked_no_unseen_cases":
        raise ValueError("unexpected frozen candidate status")
    if candidate["router_common"]["action_order"] != list(ACTIONS):
        raise ValueError("frozen Router action order differs")
    if candidate["router_common"]["costs"] != {"ANSWER": 0.0, "ASK": ASK_COST, "DEFER": 0.05}:
        raise ValueError("frozen Router cost definition differs")
    if candidate["feature_extractor"]["feature_boundary"] != "public_state_before_action_only":
        raise ValueError("frozen Router feature boundary is not public-state-only")
    if candidate["feature_extractor"]["revision"] != "b968826d9c46dd6066d109eabc6255188de91218":
        raise ValueError("unexpected frozen embedding model revision")

    embeddings = load_public_embedding_cache(
        embedding_path,
        expected_examples=q_examples,
        expected_config_hash=candidate["feature_extractor"]["embedding_config_sha256"],
    )
    saved = read_jsonl(prediction_path)
    saved_predictions: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for prediction in saved:
        if prediction.get("feature") != "embedding" or prediction.get("model") != "q_guided_distillation":
            continue
        if str(prediction.get("case_id")) not in dev_ids:
            continue
        key = (str(prediction["case_id"]), str(prediction["state_id"]), int(prediction["seed"]))
        saved_predictions[key] = prediction
    checkpoints = {
        item["seed"]: item
        for item in next(f for f in candidate["candidate_families"] if f["name"] == "q_guided_distillation")["checkpoints"]
    }
    router_by_seed: dict[int, dict[tuple[str, str], dict[str, Any]]] = {}
    checkpoint_audits = []
    for seed in SEEDS:
        entry = checkpoints[seed]
        checkpoint = router_root / entry["path"]
        scores, audit = _load_checkpoint_scores(
            checkpoint=checkpoint,
            expected_sha256=entry["sha256"],
            expected_seed=seed,
            dev_examples=dev_examples,
            embedding_vectors=embeddings.vectors,
            saved_predictions=saved_predictions,
        )
        router_by_seed[seed] = scores
        checkpoint_audits.append(audit)
    if len(saved_predictions) != len(SEEDS) * len(dev_examples):
        raise ValueError("stored frozen Router predictions are incomplete or duplicated")

    source_sha = sha256_file(source_path)
    if source_sha != EXPECTED_SOURCE_SHA256:
        raise ValueError("source answer-key dataset hash differs from audited ATPO source")
    source_rows = [json.loads(line) for line in source_path.open(encoding="utf-8") if line.strip()]
    if len(source_rows) != 14256:
        raise ValueError("source answer-key dataset must contain the audited 14,256 rows")
    trajectories = read_jsonl(trajectory_path)
    dev_source_categories: Counter[str] = Counter()
    q_raw_by_key = {}
    for item in read_jsonl(q_path):
        metadata = item.get("metadata", {})
        q_raw_by_key[(str(metadata.get("case_id")), str(metadata.get("state_id")))] = item
    by_key_action: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    source_index_by_case: dict[str, set[int]] = defaultdict(set)
    format_valid_count = 0
    prompt_match_count = 0
    source_answer_valid_count = 0
    correctness_reconciled_count = 0
    raw_gold_fields = 0
    for trajectory in trajectories:
        action = str(trajectory.get("action"))
        case_id, state_id = str(trajectory.get("case_id")), str(trajectory.get("state_id"))
        if action not in ACTIONS or not trajectory.get("valid"):
            continue
        forbidden = {"gold", "ground_truth", "atomic_facts", "answer_idx"}
        raw_gold_fields += int(bool(forbidden.intersection(trajectory)))
        if case_id not in dev_ids:
            continue
        key = (case_id, state_id)
        if key not in q_by_key or action in by_key_action[key]:
            raise ValueError("trajectory has unknown state or duplicate state/action")
        index = int(trajectory["source_row_index"])
        if not 0 <= index < len(source_rows):
            raise ValueError("source row index is out of range")
        source_row = source_rows[index]
        dev_source_categories[str(source_row.get("data_source", "UNSPECIFIED"))] += 1
        source_index_by_case[case_id].add(index)
        gold = source_answer(source_row)
        source_answer_valid_count += 1
        predicted = parse_option(trajectory.get("final_answer"))
        if predicted is None:
            raise ValueError("a recorded V2 final answer does not parse as an A-D option")
        observed_correct = int(gold == predicted)
        recorded_correct = int(round(float(trajectory["final_accuracy"])))
        if observed_correct != recorded_correct:
            raise ValueError("source answer key does not reconcile with a rollout correctness label")
        correctness_reconciled_count += 1
        if extract_user(trajectory.get("public_state", [])) == answer_prompt(source_row):
            prompt_match_count += 1
        if not trajectory.get("format_valid", False):
            raise ValueError("V2 rollout has a format-invalid action output")
        format_valid_count += 1
        by_key_action[key][action] = trajectory
    if raw_gold_fields:
        raise ValueError("Gold/hidden fields were found in serialized rollout records")
    if source_answer_valid_count != 3 * len(dev_examples):
        raise ValueError("Dev rollout/source answer-key coverage is incomplete")
    if prompt_match_count != 3 * len(dev_examples):
        raise ValueError("Dev initial public prompt does not match its source record")
    if format_valid_count != 3 * len(dev_examples) or correctness_reconciled_count != 3 * len(dev_examples):
        raise ValueError("Dev action output format or Gold correctness reconciliation failed")
    if any(len(indices) != 1 for indices in source_index_by_case.values()):
        raise ValueError("a case_id maps to multiple source rows")
    if set(by_key_action) != {(row.case_id, row.state_id) for row in dev_examples}:
        raise ValueError("counterfactual trajectories do not cover the exact Dev Q states")

    rows = []
    for example in sorted(dev_examples, key=lambda row: (row.case_id, row.state_id)):
        key = (example.case_id, example.state_id)
        action_rows = by_key_action[key]
        if set(action_rows) != set(ACTIONS):
            raise ValueError("counterfactual state is missing one of the three actions")
        correctness, action_counts = {}, {}
        for action, trajectory in action_rows.items():
            correctness[action] = int(round(float(trajectory["final_accuracy"])))
            action_counts[action] = {
                "ask_count": int(trajectory.get("ask_count", 0)),
                "defer_count": int(trajectory.get("defer_count", 0)),
            }
        # The realized branch costs are taken from raw outcomes; never add action costs to an already-net Q target.
        q_raw = q_raw_by_key[key]
        if q_raw.get("metadata", {}).get("target_definition") != "terminal_correctness - 0.01*cumulative_ask_count - 0.05*cumulative_defer_count":
            raise ValueError("Q dataset has a different frozen target definition")
        for action in ACTIONS:
            expected = correctness[action] - ASK_COST * action_counts[action]["ask_count"] - 0.05 * action_counts[action]["defer_count"]
            if abs(float(q_raw[f"Q_{action}"]) - expected) > 1e-8:
                raise ValueError("frozen Q target does not reconcile to source correctness and single-count costs")
        source_index = next(iter(source_index_by_case[example.case_id]))
        router = {str(seed): router_by_seed[seed][key] for seed in SEEDS}
        rows.append({
            "case_id": example.case_id,
            "state_id": example.state_id,
            "source_row_index": source_index,
            "correctness": correctness,
            "action_counts": {action: action_counts[action]["ask_count"] if action == "ASK" else action_counts[action]["defer_count"] for action in ACTIONS},
            "router": router,
        })
    audit = {
        "source_rows": len(source_rows),
        "source_dataset_sha256": source_sha,
        "dev_source_row_categories_action_rollouts": dict(sorted(dev_source_categories.items())),
        "dev_cases": len(dev_ids),
        "train_cases": len(train_ids),
        "dev_states": len(dev_examples),
        "dev_action_rollouts": 3 * len(dev_examples),
        "source_answer_keys_parsed": source_answer_valid_count,
        "rollout_correctness_reconciled_to_source_answer_key": correctness_reconciled_count,
        "initial_public_prompts_exactly_match_source": prompt_match_count,
        "format_valid_rollouts": format_valid_count,
        "serialized_rollout_gold_or_hidden_fields": raw_gold_fields,
        "checkpoint_replay": checkpoint_audits,
        "embedding_cache_sha256": embeddings.sha256,
        "embedding_manifest_sha256": embeddings.metadata["manifest_sha256"],
    }
    return rows, audit


def make_policy_records(rows: Sequence[Mapping[str, Any]], *, method: str, defer_cost: float) -> dict[str, Any]:
    seed_spec = SEEDS if method in {"learned_actual", "learned_oracle"} else ()
    metrics = metrics_for_rows(rows, method, defer_cost, seeds=seed_spec)
    baseline_vectors = case_metric_vectors(rows, "always_answer", defer_cost)
    vectors = case_metric_vectors(rows, method, defer_cost, seeds=seed_spec)
    metrics["net_gain_vs_always_answer_utility"] = float(vectors["utility"].mean() - baseline_vectors["utility"].mean())
    metrics["paired_ci95_accuracy_vs_always_answer"] = bootstrap_interval(
        vectors["accuracy"] - baseline_vectors["accuracy"], replicates=10000, seed=20261121,
    )
    metrics["paired_ci95_utility_vs_always_answer"] = bootstrap_interval(
        vectors["utility"] - baseline_vectors["utility"], replicates=10000, seed=20261122,
    )
    for i, metric in enumerate(PRIMARY_METRICS):
        metrics[f"bootstrap_ci95_{metric}"] = bootstrap_interval(vectors[metric], replicates=10000, seed=20261123 + i)
    return metrics


def evaluate(rows: Sequence[Mapping[str, Any]], *, costs: Sequence[float], bootstrap_replicates: int = 10000,
             bootstrap_seed: int = 20261121) -> dict[str, Any]:
    if bootstrap_replicates != 10000 or bootstrap_seed != 20261121:
        # Keep one predeclared execution for this frozen Dev audit.
        if bootstrap_replicates < 1000:
            raise ValueError("use at least 1,000 case-cluster bootstrap replicates")
    methods = (
        ("always_answer", "Always ANSWER"),
        ("always_ask", "Always ASK (actual ASK trajectory)"),
        ("always_defer_actual", "Always DEFER (Qwen3-32B-AWQ)"),
        ("always_defer_oracle", "Always DEFER (direct answer-key Oracle)"),
        ("oracle_conditional", "Oracle Conditional"),
        ("learned_actual", "Frozen Q-guided Router + actual Expert"),
        ("learned_oracle", "Frozen Q-guided Router + Oracle safety gate"),
    )
    ordered_rows = list(rows)
    four_cell_count = Counter()
    four_cell_case_sum = Counter()
    case_state_counts = Counter(row["case_id"] for row in ordered_rows)
    for row in ordered_rows:
        base, expert = row["correctness"]["ANSWER"], row["correctness"]["DEFER"]
        four_cell_count[(base, expert)] += 1
        four_cell_case_sum[(base, expert)] += 1.0 / case_state_counts[row["case_id"]]
    n_cases = len(case_state_counts)
    four_cells = {
        "counts_states": {
            "base_correct_expert_correct": four_cell_count[(1, 1)],
            "base_correct_expert_wrong": four_cell_count[(1, 0)],
            "base_wrong_expert_correct": four_cell_count[(0, 1)],
            "base_wrong_expert_wrong": four_cell_count[(0, 0)],
        },
        "case_macro_proportions": {
            "base_correct_expert_correct": four_cell_case_sum[(1, 1)] / n_cases,
            "base_correct_expert_wrong": four_cell_case_sum[(1, 0)] / n_cases,
            "base_wrong_expert_correct": four_cell_case_sum[(0, 1)] / n_cases,
            "base_wrong_expert_wrong": four_cell_case_sum[(0, 0)] / n_cases,
        },
        "expert_correction_rate_given_base_wrong_state_weighted": _safe_ratio(four_cell_count[(0, 1)], four_cell_count[(0, 1)] + four_cell_count[(0, 0)]),
        "base_correct_spoilage_rate_state_weighted": _safe_ratio(four_cell_count[(1, 0)], four_cell_count[(1, 1)] + four_cell_count[(1, 0)]),
        "expert_net_accuracy_difference_state_weighted": (four_cell_count[(0, 1)] - four_cell_count[(1, 0)]) / len(ordered_rows),
    }
    # Paired, case-cluster bootstrap for the actual Expert-vs-Base cells and
    # conditional correction/spoilage rates.
    case_ids = sorted(case_state_counts)
    case_data = defaultdict(list)
    for row in ordered_rows:
        base = row["correctness"]["ANSWER"]
        expert = row["correctness"]["DEFER"]
        case_data[row["case_id"]].append((base, expert))
    cell_keys = ((1, 1), (1, 0), (0, 1), (0, 0))
    cell_boot = {key: [] for key in cell_keys}
    correction_boot, spoilage_boot, net_boot = [], [], []
    rng = np.random.default_rng(20261121)
    for sample in rng.integers(0, len(case_ids), size=(10000, len(case_ids))):
        chosen = [case_ids[int(i)] for i in sample]
        values = [pair for case_id in chosen for pair in case_data[case_id]]
        n = len(values)
        for key in cell_keys:
            cell_boot[key].append(sum(pair == key for pair in values) / n)
        n_base_wrong = sum(not base for base, _ in values)
        n_base_right = sum(base for base, _ in values)
        correction_boot.append(_safe_ratio(sum((not base) and expert for base, expert in values), n_base_wrong))
        spoilage_boot.append(_safe_ratio(sum(base and (not expert) for base, expert in values), n_base_right))
        net_boot.append(sum(int(expert) - int(base) for base, expert in values) / n)
    def interval(values):
        finite = np.asarray([v for v in values if v is not None], dtype=np.float64)
        return [float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))]
    four_cells["bootstrap_ci95_case_cluster"] = {
        "base_correct_expert_correct_proportion": interval(cell_boot[(1, 1)]),
        "base_correct_expert_wrong_proportion": interval(cell_boot[(1, 0)]),
        "base_wrong_expert_correct_proportion": interval(cell_boot[(0, 1)]),
        "base_wrong_expert_wrong_proportion": interval(cell_boot[(0, 0)]),
        "expert_correction_rate_given_base_wrong": interval(correction_boot),
        "base_correct_spoilage_rate": interval(spoilage_boot),
        "paired_expert_minus_base_accuracy": interval(net_boot),
    }
    action_counts_by_seed = {}
    for seed in SEEDS:
        counts = Counter(row["router"][str(seed)]["action"] for row in ordered_rows)
        calls = sum(row["router"][str(seed)]["action"] == "DEFER" and not row["correctness"]["ANSWER"] for row in ordered_rows)
        action_counts_by_seed[str(seed)] = {
            "router_proposed_actions_state_counts": {action: counts[action] for action in ACTIONS},
            "router_proposed_defer_count": counts["DEFER"],
            "oracle_calls_after_preserve_correct_base_gate": calls,
        }
    cost_results = {}
    for cost in costs:
        if cost < 0 or not math.isfinite(cost):
            raise ValueError("DEFER costs must be finite and nonnegative")
        entries = {}
        for method, label in methods:
            metrics = make_policy_records(ordered_rows, method=method, defer_cost=float(cost))
            entries[method] = {"label": label, **metrics}
        cost_results[f"{cost:.2f}"] = entries
    return {
        "schema_version": 1,
        "status": "completed_development_only_answer_key_oracle_reanalysis",
        "estimand": "case-macro mean over 43 previously-used public states clustered in 15 case_id groups; state-weighted denominators are additionally reported for paired expert cells",
        "gold_interpretation": "source benchmark answer key; provenance-verified, not clinician-adjudicated clinical truth",
        "feature_boundary": "the learned Router consumes only frozen public-state embeddings/checkpoint scores; answer keys and rollout outcomes are joined after predicted actions are fixed",
        "cost_definition": {
            "ANSWER": 0.0,
            "ASK_per_count": ASK_COST,
            "DEFER_grid": [float(x) for x in costs],
            "utility": "terminal_correctness - 0.01*cumulative_ASK - defer_cost*cumulative_DEFER",
            "defer_decisions_frozen_across_cost_grid": True,
            "resource_token_latency_costs_in_utility": False,
        },
        "bootstrap": {
            "unit": "case_id cluster; resample all states in a selected case together",
            "replicates": 10000,
            "base_seed": 20261121,
            "seed_policy": "deterministic distinct seed per reported metric, starting at the base seed",
            "interval": "percentile 95%",
            "interpretation": "exploratory and unstable with only 15 clusters",
        },
        "four_cell_table": four_cells,
        "frozen_router_action_counts": action_counts_by_seed,
        "cost_sensitivity": cost_results,
    }


def parse_costs(value: str) -> list[float]:
    result = [float(item) for item in value.split(",") if item.strip()]
    if result != list(DEFAULT_COSTS):
        raise ValueError("cost grid is frozen as 0.05,0.10,0.20,0.60 for this run")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router-root", type=Path, default=DEFAULT_ROUTER_ROOT)
    parser.add_argument("--source-dataset", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cost-grid", default="0.05,0.10,0.20,0.60")
    parser.add_argument(
        "--config", type=Path,
        default=REPO_ROOT / "experiments/ea_atpo_v3/oracle_expert/configs/dev_cost_sensitivity_20261121.json",
    )
    parser.add_argument(
        "--v3-case-split", type=Path,
        default=Path("/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO_V1/experiments/ea_atpo_value_router_20261105/artifacts/case_split.json"),
    )
    args = parser.parse_args()
    router_root = args.router_root.resolve()
    out_dir = args.output_dir.resolve()
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {out_dir}")
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    costs = parse_costs(args.cost_grid)
    if [float(x) for x in config["utility"]["defer_costs"]] != costs:
        raise ValueError("CLI DEFER cost grid differs from the frozen experiment configuration")
    if float(config["utility"]["ask_cost"]) != ASK_COST:
        raise ValueError("ASK cost differs from the frozen experiment configuration")
    if config["source_dataset_sha256"] != EXPECTED_SOURCE_SHA256:
        raise ValueError("configured source answer-key hash is not the audited dataset")
    if Path(config["source_dataset"]).resolve() != args.source_dataset.resolve():
        raise ValueError("source dataset path differs from the frozen experiment configuration")
    v3_split_path = args.v3_case_split.resolve()
    if (sha256_file(v3_split_path) != EXPECTED_V3_SPLIT_SHA256
            or config["v3_case_split"]["sha256"] != EXPECTED_V3_SPLIT_SHA256
            or Path(config["v3_case_split"]["path"]).resolve() != v3_split_path):
        raise ValueError("V3 case split hash differs from the audited split")
    if config["utility"]["costs_applied_once"] is not True or config["router"]["router_predictions_frozen_across_cost_grid"] is not True:
        raise ValueError("cost or policy-freeze rules differ from the predeclared configuration")
    base_dir = router_root / "experiments/ea_atpo_v2/router_training/outputs/real_q_router_20261120"
    candidate = router_root / "experiments/ea_atpo_v2/router_training/confirmation_20261120/configs/frozen_candidate_20261120.json"
    paths = {
        "source_dataset": args.source_dataset.resolve(),
        "q_dataset": base_dir / "q_dataset_cost_ask001_defer005.jsonl",
        "trajectories": router_root / "experiments/ea_atpo_v2/recovery_bundle_20261120/pilot_trajectories.jsonl",
        "split": base_dir / "case_split_35_train_15_dev.json",
        "predictions": base_dir / "training_final/dev_predictions.jsonl",
        "candidate_manifest": candidate,
        "embedding_cache": base_dir / "qwen3_8b_frozen_public_embeddings.jsonl",
        "recovery_manifest": router_root / "experiments/ea_atpo_v2/recovery_bundle_20261120/recovery_manifest.json",
        "experiment_config": config_path,
        "v3_case_split": v3_split_path,
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("required frozen input is missing: " + ", ".join(missing))
    v2_split = json.loads(paths["split"].read_text(encoding="utf-8"))
    v3_split = json.loads(paths["v3_case_split"].read_text(encoding="utf-8"))
    v2_cases = set(map(str, v2_split["train"])) | set(map(str, v2_split["dev"]))
    v3_train = set(map(str, v3_split["train"]))
    v3_nontrain = set(map(str, v3_split["validation"])) | set(map(str, v3_split["internal_test"]))
    if len(v2_cases) != 50 or not v2_cases.issubset(v3_train) or v2_cases & v3_nontrain:
        raise ValueError("V2 cases are not strictly contained in the V3 source Train split")
    recovery_manifest = json.loads(paths["recovery_manifest"].read_text(encoding="utf-8"))
    if recovery_manifest["models"]["base"]["revision"] != config["router"]["base_revision"]:
        raise ValueError("rollout Base revision differs from the frozen Router embedding revision")
    if recovery_manifest["models"]["expert"]["revision"] != "0499c3ac83fdef8810b907a23894ba91e95eddd8":
        raise ValueError("unexpected actual Expert revision in the V2 rollout manifest")
    if recovery_manifest.get("continuation_policy_id") != "ASK_ONCE_THEN_FORCE_BASE_ANSWER_PATIENT_REVIEW_V2":
        raise ValueError("V2 rollout continuation policy differs from the audited policy")
    if recovery_manifest["costs"]["ask"] != 0.05 or recovery_manifest["costs"]["defer"] != 0.6:
        raise ValueError("raw V2 rollout cost protocol differs from the recorded run")
    rows, audit = build_state_rows(
        router_root=router_root,
        source_path=paths["source_dataset"],
        q_path=paths["q_dataset"],
        trajectory_path=paths["trajectories"],
        split_path=paths["split"],
        prediction_path=paths["predictions"],
        candidate_path=paths["candidate_manifest"],
        embedding_path=paths["embedding_cache"],
    )
    summary = evaluate(rows, costs=costs)
    summary["data_audit"] = audit
    summary["data_audit"]["v2_rollout_models"] = recovery_manifest["models"]
    summary["data_audit"]["v2_rollout_raw_costs"] = recovery_manifest["costs"]
    summary["data_audit"]["continuation_policy_id"] = recovery_manifest["continuation_policy_id"]
    summary["data_audit"]["v2_case_split_source"] = "case_id split; all 50 Router Train/Dev cases are in V3 source Train"
    summary["data_audit"]["v2_case_split_sha256"] = sha256_file(paths["split"])
    summary["data_audit"]["v3_case_split_sha256"] = sha256_file(paths["v3_case_split"])
    out_dir.mkdir(parents=True, mode=0o700)
    out_dir.chmod(0o700)
    private_path = out_dir / "private_state_results.jsonl"
    with private_path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    private_path.chmod(0o600)
    summary_path = out_dir / "aggregate_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.chmod(0o600)
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    git_branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO_ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True).strip())
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_branch": git_branch,
        "git_head": git_head,
        "git_worktree_dirty_at_run": dirty,
        "evaluation_script_sha256": sha256_file(Path(__file__).resolve()),
        "device": "CPU only",
        "model_generation_or_training": False,
        "frozen_holdout_opened": False,
        "cost_grid": costs,
        "input_sha256": {name: sha256_file(path) for name, path in paths.items()},
        "summary_sha256": sha256_file(summary_path),
        "private_case_results_sha256": sha256_file(private_path),
        "file_modes": {"directory": oct(stat.S_IMODE(out_dir.stat().st_mode)),
                       "aggregate_summary.json": oct(stat.S_IMODE(summary_path.stat().st_mode)),
                       "private_state_results.jsonl": oct(stat.S_IMODE(private_path.stat().st_mode))},
    }
    manifest_path = out_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_path.chmod(0o600)
    print(json.dumps({
        "status": summary["status"],
        "cases": 15,
        "states": len(rows),
        "checkpoint_replay_max_abs_difference": max(x["max_abs_score_difference"] for x in audit["checkpoint_replay"]),
        "output_dir": str(out_dir),
        "summary_sha256": manifest["summary_sha256"],
        "frozen_holdout_opened": False,
        "gpu_used": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
