#!/usr/bin/env python3
"""Offline, Train/Dev-only selector diagnostic for the existing Router heads.

Reads the already-created costed Q dataset, public embedding cache, selected
small-head checkpoints, and saved Dev predictions. It never reads the Window A
new-case confirmation set and writes only aggregate summaries.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .selection import ACTION_ORDER, advance_after_ask, select_distilled_action, select_relative_action

SEEDS = (17, 42, 2026)
REGISTERED = json.loads(Path(__file__).with_name("margin_grid.json").read_text(encoding="utf-8"))
MARGINS = tuple(float(value) for value in REGISTERED["margins"])
COSTS = np.asarray([REGISTERED["registered_costs"][action] for action in ACTION_ORDER], dtype=np.float64)
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 20261120


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: {exc}") from exc
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def public_state_digest(messages: list[dict[str, Any]]) -> str:
    visible = []
    for message in messages:
        role = str(message.get("role", ""))
        if role not in {"user", "assistant"}:
            raise ValueError("embedding audit found a non-public message role")
        visible.append({"role": role, "content": str(message.get("content", ""))})
    canonical = json.dumps(visible, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def action_values(row: dict[str, Any], key: str, *, default: float | None = None) -> np.ndarray:
    value = row.get(key)
    if isinstance(value, dict):
        return np.asarray([value.get(action, value.get(f"Q_{action}", default)) for action in ACTION_ORDER], dtype=object)
    if isinstance(value, list) and len(value) == 3:
        return np.asarray(value, dtype=object)
    if value is None:
        return np.asarray([default] * 3, dtype=object)
    raise ValueError(f"unsupported {key} format")


def load_q_rows(path: Path) -> list[dict[str, Any]]:
    result = []
    for raw in read_jsonl(path):
        md = raw.get("metadata") or {}
        case_id = str(raw.get("case_id", md.get("case_id", "")))
        state_id = str(raw.get("state_id", md.get("state_id", "")))
        if not case_id or not state_id:
            raise ValueError("costed Q row lacks case_id/state_id")
        mask_map = raw.get("valid_mask") or {}
        if not isinstance(mask_map, dict) or list(mask_map) != list(ACTION_ORDER):
            raise ValueError(f"{case_id}/{state_id}: valid_mask order must be ASK/ANSWER/DEFER")
        mask = np.asarray([bool(mask_map[action]) for action in ACTION_ORDER], dtype=bool)
        q = np.asarray([float(raw[f"Q_{action}"]) for action in ACTION_ORDER], dtype=np.float64)
        cost_map = raw.get("action_cost") or {}
        costs = np.asarray([float(cost_map[action]) for action in ACTION_ORDER], dtype=np.float64)
        if not np.allclose(costs, COSTS, atol=1e-9, rtol=0.0):
            raise ValueError(f"{case_id}/{state_id}: action costs differ from registered primary costs")
        if raw.get("costs_included_in_targets") is not True or md.get("costs_included_in_targets") is not True:
            raise ValueError(f"{case_id}/{state_id}: costed target marker is missing")
        if md.get("feature_boundary") != "public_state_before_action_only" or md.get("gold_or_future_fields_in_features") is not False:
            raise ValueError(f"{case_id}/{state_id}: feature boundary is not public-state-only")
        counts = action_values(raw, "valid_sample_count", default=1)
        stds = action_values(raw, "return_std", default=None)
        for index in range(3):
            if mask[index] and int(counts[index]) != 1:
                raise ValueError(f"{case_id}/{state_id}: expected K=1 labels")
            if mask[index] and stds[index] is not None:
                raise ValueError(f"{case_id}/{state_id}: K=1 return_std must remain missing")
        correct = q + costs
        observed = correct[mask]
        if not np.isfinite(observed).all() or not np.all(np.minimum(np.abs(observed), np.abs(observed - 1.0)) < 1e-8):
            raise ValueError(f"{case_id}/{state_id}: Q + one-time cost does not recover binary correctness")
        public_state = md.get("public_state")
        if not isinstance(public_state, list) or not public_state:
            raise ValueError(f"{case_id}/{state_id}: public_state is absent")
        result.append({"case_id": case_id, "state_id": state_id, "q": q, "mask": mask,
                       "correct": correct, "public_state_hash": public_state_digest(public_state)})
    keys = [(row["case_id"], row["state_id"]) for row in result]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate state in costed Q data")
    if not result:
        raise ValueError("costed Q dataset is empty")
    return result


def load_embeddings(path: Path, q_rows: list[dict[str, Any]]) -> tuple[dict[tuple[str, str], np.ndarray], dict[str, Any]]:
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    manifest = read_json(manifest_path)
    cache_hash = sha256(path)
    if manifest.get("output_sha256") != cache_hash:
        raise ValueError("embedding cache checksum differs from manifest")
    if manifest.get("feature_boundary") != "public_state_before_action_only" or manifest.get("privileged_fields_used") is not False:
        raise ValueError("embedding manifest does not certify public-state-only features")
    audit = manifest.get("case_split_audit") or {}
    if audit.get("no_gold_or_outcome_fields_used") is not True or audit.get("official_split") != "train":
        raise ValueError("embedding manifest does not certify the Train-only public feature boundary")
    allowed = {"case_id", "state_id", "feature_boundary", "embedding", "state_hash", "rendered_prompt_hash",
               "model_id", "model_revision", "pooling", "max_length", "prompt_template_hash", "embedding_config_hash"}
    vectors: dict[tuple[str, str], np.ndarray] = {}
    for line_no, row in enumerate(read_jsonl(path), 1):
        extras = set(row) - allowed
        if extras:
            raise ValueError(f"embedding row {line_no} has unsupported feature fields: {sorted(extras)}")
        key = (str(row.get("case_id", "")), str(row.get("state_id", "")))
        if key in vectors or row.get("feature_boundary") != "public_state_before_action_only":
            raise ValueError(f"duplicate or non-public embedding row at line {line_no}")
        vec = np.asarray(row.get("embedding"), dtype=np.float32)
        if vec.ndim != 1 or vec.size != 4096 or not np.isfinite(vec).all():
            raise ValueError(f"invalid 4096-dimensional embedding at line {line_no}")
        expected_hash = next((r["public_state_hash"] for r in q_rows if (r["case_id"], r["state_id"]) == key), None)
        if expected_hash is None or row.get("state_hash") != expected_hash:
            raise ValueError(f"public-state embedding hash mismatch at line {line_no}")
        vectors[key] = vec
    expected_keys = {(row["case_id"], row["state_id"]) for row in q_rows}
    if set(vectors) != expected_keys:
        raise ValueError("embedding cache keys do not match the 50-case Q dataset")
    if int(manifest.get("state_count", -1)) != len(vectors) or int(manifest.get("embedding_dimension", -1)) != 4096:
        raise ValueError("embedding manifest count/dimension mismatch")
    return vectors, {**manifest, "manifest_sha256": sha256(manifest_path), "cache_sha256": cache_hash}


class TinyHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_split(path: Path, q_rows: list[dict[str, Any]]) -> tuple[set[str], set[str], dict[str, Any]]:
    split = read_json(path)
    if split.get("unit") != "case_id":
        raise ValueError("Router split must be case_id based")
    train_ids = set(map(str, split.get("train", [])))
    dev_ids = set(map(str, split.get("dev", [])))
    observed = {row["case_id"] for row in q_rows}
    if len(train_ids) != 35 or len(dev_ids) != 15 or train_ids & dev_ids or observed != train_ids | dev_ids:
        raise ValueError("expected the existing 35/15 case-disjoint Train/Dev split")
    return train_ids, dev_ids, split


def rows_for(rows: list[dict[str, Any]], cases: set[str]) -> list[dict[str, Any]]:
    return [row for row in rows if row["case_id"] in cases]


def model_scores(checkpoint_path: Path, payload: dict[str, Any], x: np.ndarray) -> np.ndarray:
    state = payload["state"]
    input_dim = int(state["net.0.weight"].shape[1])
    hidden_dim = int(state["net.0.weight"].shape[0])
    output_dim = int(state["net.2.weight"].shape[0])
    if x.shape[1] != input_dim:
        raise ValueError(f"{checkpoint_path}: embedding dimension mismatches checkpoint")
    model = TinyHead(input_dim, hidden_dim, output_dim)
    model.load_state_dict(state)
    model.eval()
    with torch.inference_mode():
        out = model(torch.as_tensor(x, dtype=torch.float32)).cpu().numpy().astype(np.float64)
    if payload["model"] == "mlp_relative":
        if output_dim != 2:
            raise ValueError("relative checkpoint must predict ASK-ANSWER and DEFER-ANSWER")
        return out
    if payload["model"] == "q_guided_distillation":
        if output_dim != 3 or not np.isclose(float(payload["temperature"]), 0.1):
            raise ValueError("distillation checkpoint config differs from the registered three-action head")
        return out * float(payload["temperature"])
    raise ValueError(f"unsupported comparison checkpoint model: {payload['model']}")


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if not positives or not negatives:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and scores[order[j]] == scores[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def choose_actions(rows: list[dict[str, Any]], scores: np.ndarray, model: str, margin: float = 0.0) -> np.ndarray:
    actions = []
    for row, value in zip(rows, scores):
        if model == "q_guided_distillation":
            action = select_distilled_action(value, row["mask"])
        else:
            action, _ = select_relative_action(value[[0, 2]], row["mask"], margin=margin,
                                                ask_budget_remaining=1,
                                                after_ask_without_new_information=False,
                                                costs_included_in_values=True)
        actions.append(action)
    return np.asarray(actions, dtype=np.int64)


def action_scores(scores: np.ndarray, model: str) -> np.ndarray:
    if model == "mlp_relative":
        return np.column_stack([scores[:, 0], np.zeros(len(scores)), scores[:, 1]])
    return scores


def case_metrics(rows: list[dict[str, Any]], actions: np.ndarray) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        grouped[row["case_id"]].append(i)
    result = {}
    for case_id, indices in grouped.items():
        selected = actions[indices]
        q = np.stack([rows[i]["q"] for i in indices])
        correct = np.stack([rows[i]["correct"] for i in indices])
        state_indices = np.arange(len(indices))
        result[case_id] = {
            "accuracy": float(np.mean(correct[state_indices, selected])),
            "utility": float(np.mean(q[state_indices, selected])),
            "ask_rate": float(np.mean(selected == 0)),
            "defer_rate": float(np.mean(selected == 2)),
        }
    return result


def ci_from_case_values(values: np.ndarray, seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    # Input is [seed, case], keeping each case's states paired across all seeds.
    if values.ndim != 2 or values.shape[1] == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, values.shape[1], size=(BOOTSTRAP_REPLICATES, values.shape[1]))
    sample = values[:, draws].mean(axis=2).mean(axis=0)
    return float(np.quantile(sample, .025)), float(np.quantile(sample, .975))


def summarize_policy(rows: list[dict[str, Any]], actions_by_seed: dict[int, np.ndarray], *, label: str,
                     baseline_case_values: dict[str, dict[str, float]] | None = None) -> dict[str, Any]:
    case_ids = sorted({row["case_id"] for row in rows})
    per_seed = {seed: case_metrics(rows, actions) for seed, actions in actions_by_seed.items()}
    pooled_rows = [r for r in rows]
    metric_rows = []
    for metric_index, metric in enumerate(("accuracy", "utility", "ask_rate", "defer_rate")):
        values = np.asarray([[per_seed[seed][case][metric] for case in case_ids]
                             for seed in sorted(per_seed)], dtype=np.float64)
        point = float(values.mean())
        low, high = ci_from_case_values(values, BOOTSTRAP_SEED + metric_index)
        row = {"split": label, "metric": metric, "case_macro": point, "case_bootstrap_ci95_low": low,
               "case_bootstrap_ci95_high": high, "seed_count": len(per_seed), "case_count": len(case_ids),
               "state_count": len(pooled_rows)}
        if baseline_case_values is not None:
            deltas = np.asarray([[per_seed[seed][case][metric] - baseline_case_values[case][metric]
                                  for case in case_ids] for seed in sorted(per_seed)], dtype=np.float64)
            dlow, dhigh = ci_from_case_values(deltas, BOOTSTRAP_SEED + 100 + metric_index)
            row["delta_vs_always_answer"] = float(deltas.mean())
            row["delta_vs_always_answer_ci95_low"] = dlow
            row["delta_vs_always_answer_ci95_high"] = dhigh
        metric_rows.append(row)

    state_pooled = {}
    for metric in ("accuracy", "utility", "ask_rate", "defer_rate"):
        vals = []
        for seed in sorted(actions_by_seed):
            actions = actions_by_seed[seed]
            correct = np.stack([r["correct"] for r in rows])
            q = np.stack([r["q"] for r in rows])
            idx = np.arange(len(rows))
            if metric == "accuracy": vals.append(float(correct[idx, actions].mean()))
            elif metric == "utility": vals.append(float(q[idx, actions].mean()))
            elif metric == "ask_rate": vals.append(float((actions == 0).mean()))
            else: vals.append(float((actions == 2).mean()))
        state_pooled[metric] = float(np.mean(vals))
    return {"label": label, "metrics": metric_rows, "state_pooled": state_pooled,
            "per_seed_case_values": per_seed, "case_ids": case_ids}


def bootstrap_mean(values_by_seed_case: np.ndarray, seed_offset: int = 0) -> tuple[float, float]:
    return ci_from_case_values(values_by_seed_case, BOOTSTRAP_SEED + seed_offset)


def advantage_diagnostics(rows: list[dict[str, Any]], score_by_seed: dict[int, np.ndarray], *, split_name: str,
                          model: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    seeds = sorted(score_by_seed)
    true_delta = np.stack([r["q"][[0, 2]] - r["q"][1] for r in rows])
    pred = np.stack([score_by_seed[seed][:, [0, 2]] for seed in seeds])
    positive_rows, near_zero_rows, calibration_rows = [], [], []
    for action_col, action_name in enumerate(("ASK", "DEFER")):
        actual = true_delta[:, action_col]
        actual_positive = actual > 1e-12
        per_seed_mae = np.mean(np.abs(pred[:, :, action_col] - actual[None, :]), axis=1)
        case_ids = sorted({r["case_id"] for r in rows})
        by_case = np.asarray([[np.mean([abs(pred[s, i, action_col] - actual[i]) for i, r in enumerate(rows) if r["case_id"] == c])
                               for c in case_ids] for s in range(len(seeds))], dtype=np.float64)
        mae_low, mae_high = bootstrap_mean(by_case, 300 + action_col)
        auc_values = [safe_auc(actual_positive, pred[s, :, action_col]) for s in range(len(seeds))]
        positive_rows.append({
            "split": split_name, "model": model, "action": action_name, "states": len(rows),
            "actual_positive_states": int(actual_positive.sum()), "actual_positive_share": float(actual_positive.mean()),
            "predicted_positive_share_mean_seeds": float(np.mean((pred[:, :, action_col] > 0).mean(axis=1))),
            "mean_predicted_advantage": float(pred[:, :, action_col].mean()),
            "mean_actual_advantage": float(actual.mean()),
            "advantage_mae_state_pooled_mean_seeds": float(per_seed_mae.mean()),
            "advantage_mae_case_macro": float(by_case.mean()), "advantage_mae_ci95_low": mae_low,
            "advantage_mae_ci95_high": mae_high,
            "positive_advantage_auroc_mean_seeds": float(np.mean([x for x in auc_values if x is not None])) if any(x is not None for x in auc_values) else None,
            "case_count": len(case_ids), "positive_class_base_rate": float(actual_positive.mean()),
        })

        mean_prediction = pred[:, :, action_col].mean(axis=0)
        near = np.abs(mean_prediction) <= 0.01 + 1e-12
        sign_error = (mean_prediction > 0) != actual_positive
        near_cases = sorted({rows[i]["case_id"] for i in np.where(near)[0]})
        case_error = np.asarray([[np.mean([sign_error[i] for i, row in enumerate(rows)
                                           if row["case_id"] == case and near[i]])
                                  for case in near_cases if any(near[i] and rows[i]["case_id"] == case for i in range(len(rows)))]], dtype=np.float64)
        if case_error.size:
            near_low, near_high = ci_from_case_values(case_error, BOOTSTRAP_SEED + 400 + action_col)
            near_error_rate = float(case_error.mean())
        else:
            near_low, near_high, near_error_rate = None, None, None
        near_zero_rows.append({"split": split_name, "model": model, "action": action_name,
                               "near_zero_threshold": 0.01, "near_zero_states": int(near.sum()),
                               "near_zero_state_share": float(near.mean()), "near_zero_cases": len(near_cases),
                               "sign_error_rate_case_macro": near_error_rate,
                               "sign_error_ci95_low": near_low, "sign_error_ci95_high": near_high,
                               "sign_errors_state_count": int((near & sign_error).sum()),
                               "actual_positive_near_zero_states": int((near & actual_positive).sum())})

        order = np.argsort(mean_prediction, kind="mergesort")
        chunks = np.array_split(order, min(4, len(order)))
        for bin_index, indices in enumerate(chunks, 1):
            if not len(indices):
                continue
            calibration_rows.append({"split": split_name, "model": model, "action": action_name,
                                     "predicted_score_quantile_bin": bin_index, "states": len(indices),
                                     "cases": len({rows[i]["case_id"] for i in indices}),
                                     "mean_predicted_net_advantage": float(mean_prediction[indices].mean()),
                                     "mean_observed_net_advantage": float(actual[indices].mean()),
                                     "positive_observed_share": float(actual_positive[indices].mean()),
                                     "abs_mean_calibration_gap": float(abs(mean_prediction[indices].mean() - actual[indices].mean()))})
    return positive_rows, near_zero_rows, calibration_rows


def selected_action_errors(rows: list[dict[str, Any]], actions_by_seed: dict[int, np.ndarray], *,
                           split_name: str, model_name: str) -> list[dict[str, Any]]:
    records = []
    for action_index, action in ((0, "ASK"), (2, "DEFER")):
        selected_total = false_total = accuracy_regressions = no_accuracy_gain = 0
        actual_positive_missed = 0
        for seed, selected in actions_by_seed.items():
            for i, row in enumerate(rows):
                delta = float(row["q"][action_index] - row["q"][1])
                if selected[i] == action_index:
                    selected_total += 1
                    if delta <= 1e-12:
                        false_total += 1
                        raw_correctness_delta = float(row["correct"][action_index] - row["correct"][1])
                        accuracy_regressions += int(raw_correctness_delta < -1e-12)
                        no_accuracy_gain += int(abs(raw_correctness_delta) <= 1e-12)
                elif delta > 1e-12:
                    actual_positive_missed += 1
        records.append({"split": split_name, "model": model_name, "action": action,
                        "seed_state_decisions": len(rows) * len(actions_by_seed),
                        "selected_state_seed_count": selected_total,
                        "selected_without_positive_net_advantage": false_total,
                        "false_selected_share": false_total / selected_total if selected_total else None,
                        "false_selected_with_accuracy_regression": accuracy_regressions,
                        "false_selected_with_no_accuracy_gain_cost_only": no_accuracy_gain,
                        "missed_positive_net_advantage_state_seed_count": actual_positive_missed})
    return records


def four_cell_rows(rows: list[dict[str, Any]], split_name: str) -> list[dict[str, Any]]:
    keys = (("base_correct_expert_correct", 1, 1), ("base_correct_expert_wrong", 1, 0),
            ("base_wrong_expert_correct", 0, 1), ("base_wrong_expert_wrong", 0, 0))
    case_count = len({r["case_id"] for r in rows})
    total_states = len(rows)
    out = []
    for name, base_value, expert_value in keys:
        selected = [r for r in rows if int(round(r["correct"][1])) == base_value and int(round(r["correct"][2])) == expert_value]
        out.append({"split": split_name, "cell": name, "states": len(selected), "state_denominator": total_states,
                    "state_share": len(selected) / total_states if total_states else None,
                    "cases_with_at_least_one_state": len({r["case_id"] for r in selected}), "case_denominator": case_count})
    return out


def self_check_selector() -> None:
    # Small deterministic invariants for masks/state handling; no training data is used.
    action, _ = select_relative_action([0.03, -0.2], [True, True, True], margin=0.01)
    assert action == 0
    action, _ = select_relative_action([0.2, 0.04], [True, True, True], margin=0.01, ask_budget_remaining=0)
    assert action == 2
    action, _ = select_relative_action([0.2, -0.2], [True, True, True], margin=0.0,
                                       after_ask_without_new_information=True)
    assert action == 1
    action, _ = select_relative_action([0.2, 0.1], [True, False, True], margin=0.0)
    assert action == 0
    remaining, no_info = advance_after_ask(1, information_received=False)
    assert remaining == 0 and no_info
    action, _ = select_relative_action([0.9, 0.2], [True, True, True], ask_budget_remaining=remaining,
                                       after_ask_without_new_information=no_info)
    assert action == 2
    action = select_distilled_action([9.0, 1.0, 2.0], [False, True, True])
    assert action == 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True,
                        help="existing 50-case Train-only router run root; never point at confirmation outputs")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    training_dir = run_root / "training_final"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    self_check_selector()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    q_path = run_root / "q_dataset_cost_ask001_defer005.jsonl"
    split_path = run_root / "case_split_35_train_15_dev.json"
    embedding_path = run_root / "qwen3_8b_frozen_public_embeddings.jsonl"
    training_results_path = training_dir / "training_results.json"
    dev_predictions_path = training_dir / "dev_predictions.jsonl"
    reload_audit_path = training_dir / "checkpoint_reload_audit.json"
    for path in (q_path, split_path, embedding_path, training_results_path, dev_predictions_path, reload_audit_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    q_rows = load_q_rows(q_path)
    train_ids, dev_ids, split = load_split(split_path, q_rows)
    train_rows, dev_rows = rows_for(q_rows, train_ids), rows_for(q_rows, dev_ids)
    if len(q_rows) != 146 or len(train_rows) != 103 or len(dev_rows) != 43:
        raise ValueError("expected the audited 146-state, 103/43 Train/Dev state counts")
    vectors, embedding_manifest = load_embeddings(embedding_path, q_rows)

    source_summary = read_json(training_results_path)
    if source_summary.get("official_validation_internal_test_and_frozen_holdout_opened") is not False:
        raise ValueError("source Router report indicates a protected split was opened")
    if source_summary.get("source_q_sha256") != sha256(q_path) or source_summary.get("split_sha256") != sha256(split_path):
        raise ValueError("training result hashes do not match the selected 50-case run")
    if source_summary.get("costs_included_in_targets") != {"ASK": 0.01, "ANSWER": 0.0, "DEFER": 0.05}:
        raise ValueError("training result cost contract differs from selector")
    reload_audit = read_json(reload_audit_path)
    if reload_audit.get("status") != "passed" or int(reload_audit.get("checkpoint_count", 0)) != 30:
        raise ValueError("source checkpoint reload audit did not pass")

    dev_saved = {}
    for row in read_jsonl(dev_predictions_path):
        if row.get("feature") != "embedding" or row.get("model") not in {"mlp_relative", "q_guided_distillation"}:
            continue
        key = (str(row["model"]), int(row["seed"]), str(row["case_id"]), str(row["state_id"]))
        dev_saved[key] = np.asarray(row["predicted_scores"], dtype=np.float64)

    keys = [(row["case_id"], row["state_id"]) for row in q_rows]
    x_raw = np.stack([vectors[key] for key in keys]).astype(np.float32)
    all_scores: dict[str, dict[int, np.ndarray]] = {"mlp_relative": {}, "q_guided_distillation": {}}
    checkpoint_meta = []
    dev_score_mismatch_max = 0.0
    for model_name in ("mlp_relative", "q_guided_distillation"):
        for seed in SEEDS:
            checkpoint_path = training_dir / "checkpoints" / "embedding" / model_name / f"seed_{seed}.pt"
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if payload.get("format") != "ea_atpo_v2_router_training_v1" or payload.get("model") != model_name:
                raise ValueError(f"unexpected checkpoint format/model: {checkpoint_path}")
            if payload.get("feature_kind") != "embedding" or payload.get("action_order") != list(ACTION_ORDER):
                raise ValueError(f"checkpoint feature/action order mismatch: {checkpoint_path}")
            if payload.get("costs_already_in_q_targets") is not True or payload.get("reference_costs") != {"ANSWER": 0.0, "ASK": 0.01, "DEFER": 0.05}:
                raise ValueError(f"checkpoint cost contract mismatch: {checkpoint_path}")
            if set(map(str, payload.get("train_case_ids", []))) != train_ids or set(map(str, payload.get("dev_case_ids", []))) != dev_ids:
                raise ValueError(f"checkpoint split differs from the audited case split: {checkpoint_path}")
            standardizer = payload["standardizer"]
            mean = np.asarray(standardizer["mean"], dtype=np.float32)
            scale = np.asarray(standardizer["scale"], dtype=np.float32)
            if mean.shape != (4096,) or scale.shape != (4096,) or np.any(scale <= 0):
                raise ValueError(f"checkpoint standardizer is invalid: {checkpoint_path}")
            x = ((x_raw - mean) / scale).astype(np.float32)
            raw_scores = model_scores(checkpoint_path, payload, x)
            scores = action_scores(raw_scores, model_name)
            all_scores[model_name][seed] = scores
            dev_indices = [i for i, row in enumerate(q_rows) if row["case_id"] in dev_ids]
            expected = [dev_saved.get((model_name, seed, q_rows[i]["case_id"], q_rows[i]["state_id"])) for i in dev_indices]
            if any(value is None for value in expected):
                raise ValueError(f"saved Dev predictions missing for {model_name}, seed={seed}")
            difference = np.max(np.abs(scores[dev_indices] - np.stack(expected)))
            dev_score_mismatch_max = max(dev_score_mismatch_max, float(difference))
            training = payload.get("training_metadata") or {}
            checkpoint_meta.append({"model": model_name, "seed": seed,
                                    "checkpoint_sha256": sha256(checkpoint_path),
                                    "best_epoch": int(training.get("best_epoch", -1)),
                                    "best_validation_loss": float(training.get("best_validation_loss", math.nan)),
                                    "best_train_loss": float(training.get("best_train_loss", math.nan)),
                                    "input_dimension": int(payload["state"]["net.0.weight"].shape[1]),
                                    "hidden_dimension": int(payload["state"]["net.0.weight"].shape[0]),
                                    "output_dimension": int(payload["state"]["net.2.weight"].shape[0]),
                                    "reference_costs": payload["reference_costs"],
                                    "costs_already_in_q_targets": payload["costs_already_in_q_targets"]})
    if dev_score_mismatch_max > 1e-5:
        raise ValueError(f"reloaded checkpoints do not reproduce saved Dev scores: {dev_score_mismatch_max}")

    # Policies: the two existing heads plus the same relative head with fixed margins.
    policy_actions: dict[str, dict[str, dict[int, np.ndarray]]] = {"train": {}, "dev": {}}
    policy_names = ["embedding_q_guided_distillation", "embedding_relative_advantage_margin_0"] + [
        f"conservative_relative_advantage_margin_{margin:g}" for margin in MARGINS[1:]]
    for split_name, subset, case_set in (("train", train_rows, train_ids), ("dev", dev_rows, dev_ids)):
        indices = [i for i, row in enumerate(q_rows) if row["case_id"] in case_set]
        policy_actions[split_name][policy_names[0]] = {}
        policy_actions[split_name][policy_names[1]] = {}
        for margin in MARGINS[1:]:
            policy_actions[split_name][f"conservative_relative_advantage_margin_{margin:g}"] = {}
        for seed in SEEDS:
            qg_scores = all_scores["q_guided_distillation"][seed][indices]
            rel_scores = all_scores["mlp_relative"][seed][indices]
            policy_actions[split_name][policy_names[0]][seed] = np.asarray([
                select_distilled_action(score, row["mask"]) for score, row in zip(qg_scores, subset)], dtype=np.int64)
            policy_actions[split_name][policy_names[1]][seed] = np.asarray([
                select_relative_action(score[[0, 2]], row["mask"], margin=0.0, ask_budget_remaining=1,
                                       costs_included_in_values=True)[0]
                for score, row in zip(rel_scores, subset)], dtype=np.int64)
            for margin in MARGINS[1:]:
                name = f"conservative_relative_advantage_margin_{margin:g}"
                policy_actions[split_name][name][seed] = np.asarray([
                    select_relative_action(score[[0, 2]], row["mask"], margin=margin,
                                           ask_budget_remaining=1, costs_included_in_values=True)[0]
                    for score, row in zip(rel_scores, subset)], dtype=np.int64)

    # The margin-zero selector should reproduce the existing relative head choices.
    saved_action_map = {}
    for row in read_jsonl(dev_predictions_path):
        if row.get("feature") == "embedding" and row.get("model") == "mlp_relative":
            saved_action_map[(int(row["seed"]), str(row["case_id"]), str(row["state_id"]))] = ACTION_ORDER.index(row["predicted_action"])
    relative_zero_mismatches = 0
    for seed in SEEDS:
        for row, action in zip(dev_rows, policy_actions["dev"]["embedding_relative_advantage_margin_0"][seed]):
            relative_zero_mismatches += int(saved_action_map.get((seed, row["case_id"], row["state_id"])) != int(action))

    comparison_rows = []
    policy_case_summaries: dict[str, dict[str, Any]] = {"train": {}, "dev": {}}
    baseline_case_values: dict[str, dict[str, float]] = {}
    for split_name, subset in (("train", train_rows), ("dev", dev_rows)):
        answer_actions = {seed: np.full(len(subset), 1, dtype=np.int64) for seed in SEEDS}
        base_summary = summarize_policy(subset, answer_actions, label=split_name)
        baseline_case_values[split_name] = base_summary["per_seed_case_values"][SEEDS[0]]
        for model_name in policy_names:
            summary = summarize_policy(subset, policy_actions[split_name][model_name], label=split_name,
                                       baseline_case_values=baseline_case_values[split_name])
            policy_case_summaries[split_name][model_name] = summary
            for metric_row in summary["metrics"]:
                comparison_rows.append({"model": model_name, "split": split_name, **metric_row,
                                        "state_pooled": summary["state_pooled"][metric_row["metric"]]})

    # Paired comparisons reuse each case's outcomes under the same seed and
    # bootstrap case clusters. This is the primary uncertainty estimate for
    # whether a margin improves over either existing policy.
    pairwise_rows = []
    pairwise_pairs = [(policy_names[1], policy_names[0])] + [
        (f"conservative_relative_advantage_margin_{margin:g}", policy_names[1]) for margin in MARGINS[1:]
    ] + [
        (f"conservative_relative_advantage_margin_{margin:g}", policy_names[0]) for margin in MARGINS[1:]
    ]
    pairwise_metric_index = {name: i for i, name in enumerate(("accuracy", "utility", "ask_rate", "defer_rate"))}
    for split_name in ("train", "dev"):
        for candidate, reference in pairwise_pairs:
            candidate_values = policy_case_summaries[split_name][candidate]["per_seed_case_values"]
            reference_values = policy_case_summaries[split_name][reference]["per_seed_case_values"]
            case_ids = policy_case_summaries[split_name][candidate]["case_ids"]
            if case_ids != policy_case_summaries[split_name][reference]["case_ids"]:
                raise ValueError("paired policy comparison case order mismatch")
            for metric in ("accuracy", "utility", "ask_rate", "defer_rate"):
                deltas = np.asarray([
                    [candidate_values[seed][case][metric] - reference_values[seed][case][metric]
                     for case in case_ids]
                    for seed in SEEDS
                ], dtype=np.float64)
                low, high = ci_from_case_values(deltas, BOOTSTRAP_SEED + 700 +
                                                100 * len(pairwise_rows) + pairwise_metric_index[metric])
                pairwise_rows.append({"split": split_name, "candidate": candidate, "reference": reference,
                                      "metric": metric, "paired_delta": float(deltas.mean()),
                                      "case_bootstrap_ci95_low": low, "case_bootstrap_ci95_high": high,
                                      "case_count": len(case_ids), "seed_count": len(SEEDS),
                                      "bootstrap_unit": "case_id"})

    # Advantage learning diagnostics for relative head, per action, on the existing split only.
    advantage_rows, near_zero_rows, calibration_rows = [], [], []
    for split_name, rows, case_set in (("train", train_rows, train_ids), ("dev", dev_rows, dev_ids)):
        indices = [i for i, row in enumerate(q_rows) if row["case_id"] in case_set]
        a, nz, cal = advantage_diagnostics(rows, {seed: all_scores["mlp_relative"][seed][indices] for seed in SEEDS},
                                           split_name=split_name, model="embedding_mlp_relative")
        advantage_rows.extend(a); near_zero_rows.extend(nz); calibration_rows.extend(cal)

    error_rows = []
    for split_name, subset in (("train", train_rows), ("dev", dev_rows)):
        for model_name in policy_names:
            error_rows.extend(selected_action_errors(subset, policy_actions[split_name][model_name],
                                                     split_name=split_name, model_name=model_name))

    four_cells = four_cell_rows(train_rows, "train") + four_cell_rows(dev_rows, "dev")

    # Per-action value-error audit for the existing absolute-Q head (diagnostic only).
    abs_errors = []
    for split_name, rows, case_set in (("train", train_rows, train_ids), ("dev", dev_rows, dev_ids)):
        indices = [i for i, row in enumerate(q_rows) if row["case_id"] in case_set]
        for seed in SEEDS:
            checkpoint_path = training_dir / "checkpoints" / "embedding" / "mlp_absolute" / f"seed_{seed}.pt"
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            standardizer = payload["standardizer"]
            x = ((x_raw - np.asarray(standardizer["mean"], dtype=np.float32)) /
                 np.asarray(standardizer["scale"], dtype=np.float32)).astype(np.float32)
            state = payload["state"]
            head = TinyHead(int(state["net.0.weight"].shape[1]), int(state["net.0.weight"].shape[0]), 3)
            head.load_state_dict(state); head.eval()
            with torch.inference_mode():
                pred = head(torch.as_tensor(x[indices], dtype=torch.float32)).numpy().astype(np.float64)
            target = np.stack([r["q"] for r in rows])
            for action_index, action in enumerate(ACTION_ORDER):
                abs_errors.append({"split": split_name, "seed": seed, "action": action,
                                   "q_mae": float(np.mean(np.abs(pred[:, action_index] - target[:, action_index]))),
                                   "states": len(rows), "cases": len(case_set)})

    # Aggregate over the three seeds without retaining per-state predictions.
    absolute_error_summary = []
    for split_name in ("train", "dev"):
        for action in ACTION_ORDER:
            vals = [r["q_mae"] for r in abs_errors if r["split"] == split_name and r["action"] == action]
            absolute_error_summary.append({"split": split_name, "head": "embedding_mlp_absolute_q_diagnostic",
                                           "action": action, "q_mae_mean_seeds": float(np.mean(vals)),
                                           "q_mae_sd_across_seeds": float(np.std(vals, ddof=1)),
                                           "seed_count": len(vals), "states": len(train_rows) if split_name == "train" else len(dev_rows),
                                           "cases": len(train_ids) if split_name == "train" else len(dev_ids)})

    # Aggregate only; never write case_id, state_id, text, or per-state predictions.
    write_csv(output_dir / "policy_comparison.csv", comparison_rows)
    write_csv(output_dir / "pairwise_policy_deltas.csv", pairwise_rows)
    write_csv(output_dir / "relative_advantage_diagnostics.csv", advantage_rows)
    write_csv(output_dir / "near_zero_diagnostics.csv", near_zero_rows)
    write_csv(output_dir / "advantage_calibration_bins.csv", calibration_rows)
    write_csv(output_dir / "selected_action_error_breakdown.csv", error_rows)
    write_csv(output_dir / "base_expert_four_cell_counts.csv", four_cells)
    write_csv(output_dir / "absolute_q_action_mae.csv", absolute_error_summary)

    checkpoint_config = {
        "source_model_family": "existing embedding heads; no retraining",
        "embedding_model": embedding_manifest.get("model_id"),
        "embedding_revision": embedding_manifest.get("model_revision"),
        "embedding_dimension": embedding_manifest.get("embedding_dimension"),
        "feature_boundary": embedding_manifest.get("feature_boundary"),
        "relative_head": "4096 -> 16 -> 2; outputs [Q_ASK-Q_ANSWER, Q_DEFER-Q_ANSWER]",
        "distillation_head": "4096 -> 16 -> 3; action logits multiplied by registered T=0.1 for saved policy scores",
        "costs": {"ASK": 0.01, "ANSWER": 0.0, "DEFER": 0.05},
        "costs_already_in_q_targets": True,
        "checkpoints": checkpoint_meta,
        "checkpoint_reload_audit": reload_audit["status"],
        "max_abs_difference_vs_saved_dev_scores": dev_score_mismatch_max,
        "relative_margin_zero_saved_action_mismatches": relative_zero_mismatches,
        "torch_device": "cpu",
        "formal_training_or_large_model_inference_run": False,
    }
    (output_dir / "checkpoint_and_data_audit.json").write_text(json.dumps({
        "source_run_name": run_root.name,
        "q_dataset_sha256": sha256(q_path),
        "split_sha256": sha256(split_path),
        "embedding_cache_sha256": embedding_manifest["cache_sha256"],
        "embedding_manifest_sha256": embedding_manifest["manifest_sha256"],
        "training_results_sha256": sha256(training_results_path),
        "saved_dev_predictions_sha256": sha256(dev_predictions_path),
        "case_count": len(train_ids | dev_ids), "state_count": len(q_rows),
        "train_cases": len(train_ids), "train_states": len(train_rows),
        "dev_cases": len(dev_ids), "dev_states": len(dev_rows),
        "action_order": list(ACTION_ORDER), "valid_mask_all_actions_complete": all(bool(r["mask"].all()) for r in q_rows),
        "sample_count_distribution": {"K=1": len(q_rows) * 3},
        "return_std_missing_for_k1": True,
        "all_costs_in_q_targets_once": True,
        "frozen_holdout_or_official_validation_opened": False,
        "window_a_confirmation_data_read": False,
        "gold_or_hidden_facts_used_as_features": False,
        "checkpoint_config": checkpoint_config,
        "bootstrap": {"unit": "case_id cluster", "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
                      "metric_point": "mean across three seeds, then equal-case macro average"},
        "registered_margin_grid": list(MARGINS),
        "margin_selection_note": "all margins are reported; any selected candidate is development-only because Dev also selected checkpoints",
        "command": ["python", "-m", "experiments.ea_atpo_v2.window_b_router_diagnostics.run_diagnostic",
                    "--run-root", "<EXISTING_50_CASE_TRAINING_RUN_ROOT>",
                    "--output-dir", "<NEW_OUTPUT_DIRECTORY>"],
    }, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "case_counts": {"train": len(train_ids), "dev": len(dev_ids)},
                      "state_counts": {"train": len(train_rows), "dev": len(dev_rows)},
                      "max_abs_difference_vs_saved_dev_scores": dev_score_mismatch_max,
                      "relative_zero_policy_action_mismatches": relative_zero_mismatches,
                      "output_dir": str(output_dir), "models_called": False}, indent=2))


if __name__ == "__main__":
    main()
