#!/usr/bin/env python3
"""Paired relative-advantage Router with public-state-only features.

The model predicts each collaboration action's outcome and resource deltas
relative to ANSWER from the same pre-action state.  The stored gold/outcomes
are used only to build training targets and evaluate completed rollouts.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from router_core import PublicStateVectorizer, fit_kernel_ridge, predict_kernel_ridge


ACTIONS = ("ANSWER", "ASK", "CONSULT", "DEFER")
COLLAB_ACTIONS = ("ASK", "CONSULT", "DEFER")
METRICS = ("accuracy", "patient_turns", "expert_tokens_k", "total_tokens_k", "latency_s")
FEATURES = 128


def _trace_sum(row: dict[str, Any], actor: str, key: str) -> float:
    return float(sum(float(e.get(key, 0) or 0) for e in (row.get("trace") or []) if e.get("actor") == actor))


def rollout_metrics(row: dict[str, Any]) -> dict[str, float]:
    """Extract post-rollout outcomes/costs; never used as decision features."""
    trace = row.get("trace") or []
    correct = row.get("correctness", row.get("outcome", 0.0))
    if correct is None:
        correct = 0.0
    patient_turns = row.get("patient_turns")
    if patient_turns is None:
        patient_turns = sum(e.get("actor") == "patient" for e in trace)
    if "total_prompt_tokens" in row and "total_completion_tokens" in row:
        total_tokens = float(row.get("total_prompt_tokens", 0) or 0) + float(row.get("total_completion_tokens", 0) or 0)
    else:
        total_tokens = sum(float(e.get("prompt_tokens", 0) or 0) + float(e.get("completion_tokens", 0) or 0) for e in trace)
    if "total_latency_s" in row:
        latency = float(row.get("total_latency_s", 0) or 0)
    elif "total_latency_s" not in row and "total_latency_s" in (row.get("trace_metrics") or {}):
        latency = float(row["trace_metrics"]["total_latency_s"])
    else:
        latency = sum(float(e.get("latency_s", 0) or 0) for e in trace)
    expert_tokens = sum(
        float(e.get("prompt_tokens", 0) or 0) + float(e.get("completion_tokens", 0) or 0)
        for e in trace if e.get("actor") == "expert"
    )
    # Older Pilot rows have actor totals but the same trace-level accounting.
    if not trace and any(k in row for k in ("expert_prompt_tokens", "expert_completion_tokens")):
        expert_tokens = float(row.get("expert_prompt_tokens", 0) or 0) + float(row.get("expert_completion_tokens", 0) or 0)
    return {
        "accuracy": float(correct) if bool(row.get("valid", True)) else 0.0,
        "patient_turns": float(patient_turns),
        "expert_tokens_k": expert_tokens / 1000.0,
        "total_tokens_k": total_tokens / 1000.0,
        "latency_s": latency,
    }


def paired_relative_targets(records: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Aggregate replicate-paired action-minus-ANSWER outcomes by state.

    Invalid episodes remain in the intention-to-treat outcome as accuracy 0.
    Only matched state/action/replicate pairs contribute to an advantage.
    """
    states: dict[str, dict[str, Any]] = {}
    groups: dict[tuple[str, str], dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in records:
        if row.get("record_type") != "trajectory":
            continue
        sid, action = str(row["state_id"]), str(row["action"])
        if action not in ACTIONS:
            continue
        state = {"state_id": sid, "case_id": str(row["case_id"]), "public_state": row["public_state"]}
        if sid in states and (states[sid]["case_id"] != state["case_id"] or states[sid]["public_state"] != state["public_state"]):
            raise ValueError(f"state collision or state changed across paired rows: {sid}")
        states[sid] = state
        replicate = int(row.get("rollout_index", row.get("replicate_index", 0)))
        if replicate in groups[(sid, action)]:
            raise ValueError(f"duplicate rollout for {sid}/{action}/{replicate}")
        groups[(sid, action)][replicate] = rollout_metrics(row)

    target_rows: list[dict[str, Any]] = []
    noise_rows: list[dict[str, Any]] = []
    for sid, state in sorted(states.items()):
        answer_reps = groups.get((sid, "ANSWER"), {})
        if not answer_reps:
            continue
        target: dict[str, Any] = {"state_id": sid, "case_id": state["case_id"]}
        for action in COLLAB_ACTIONS:
            action_reps = groups.get((sid, action), {})
            paired_ids = sorted(set(answer_reps) & set(action_reps))
            if not paired_ids:
                for metric in METRICS:
                    target[f"{action}_{metric}_delta"] = math.nan
                continue
            deltas = {
                metric: np.asarray([action_reps[r][metric] - answer_reps[r][metric] for r in paired_ids], dtype=float)
                for metric in METRICS
            }
            for metric in METRICS:
                target[f"{action}_{metric}_delta"] = float(np.mean(deltas[metric]))
            adv = deltas["accuracy"]
            signs = np.sign(adv)
            counts = [int(np.sum(signs == s)) for s in (-1, 0, 1)]
            majority_sign_share = max(counts) / len(signs)
            noise_rows.append({
                "case_id": state["case_id"], "state_id": sid, "action": action,
                "paired_replicates": len(paired_ids),
                "mean_accuracy_advantage": float(np.mean(adv)),
                "sd_accuracy_advantage": float(np.std(adv, ddof=1)) if len(adv) > 1 else 0.0,
                "accuracy_advantage_sign_agreement": majority_sign_share,
                "replicate_advantages": json.dumps(adv.tolist()),
                "mean_patient_turn_delta": float(np.mean(deltas["patient_turns"])),
                "sd_patient_turn_delta": float(np.std(deltas["patient_turns"], ddof=1)) if len(paired_ids) > 1 else 0.0,
                "mean_expert_tokens_delta_k": float(np.mean(deltas["expert_tokens_k"])),
                "sd_expert_tokens_delta_k": float(np.std(deltas["expert_tokens_k"], ddof=1)) if len(paired_ids) > 1 else 0.0,
            })
        target_rows.append(target)

    return [states[row["state_id"]] for row in target_rows], target_rows, noise_rows


def _target_matrix(target_rows: list[dict[str, Any]]) -> np.ndarray:
    names = [f"{a}_{m}_delta" for a in COLLAB_ACTIONS for m in METRICS]
    y = np.asarray([[float(row[name]) for name in names] for row in target_rows], dtype=np.float64)
    if not np.isfinite(y).all():
        raise ValueError("relative target matrix contains missing/non-finite paired labels")
    return y


def fit_relative_model(states: list[dict[str, Any]], target_rows: list[dict[str, Any]], alpha: float, max_features: int = FEATURES) -> dict[str, Any]:
    if len(states) != len(target_rows) or not states:
        raise ValueError("non-empty, aligned state/target rows required")
    vectorizer = PublicStateVectorizer(max_features=max_features).fit(states)
    x = vectorizer.transform(states)
    y = _target_matrix(target_rows)
    dual = fit_kernel_ridge(x, y, alpha)
    return {
        "vectorizer": vectorizer, "x_train": x, "dual": dual,
        "target_columns": [f"{a}_{m}_delta" for a in COLLAB_ACTIONS for m in METRICS],
        "alpha": float(alpha), "max_features": int(max_features),
        "train_state_ids": [str(s["state_id"]) for s in states],
        "train_case_ids": sorted({str(s["case_id"]) for s in states}),
    }


def predict_relative(model: dict[str, Any], states: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    if not states:
        return {}
    x = model["vectorizer"].transform(states)
    values = predict_kernel_ridge(x, model["x_train"], model["dual"])
    result: dict[str, dict[str, dict[str, float]]] = {}
    for i, state in enumerate(states):
        sid = str(state["state_id"])
        result[sid] = {"ANSWER": {m: 0.0 for m in METRICS}}
        offset = 0
        for action in COLLAB_ACTIONS:
            result[sid][action] = {}
            for metric in METRICS:
                result[sid][action][metric] = float(values[i, offset])
                offset += 1
    return result


def save_relative_model(model: dict[str, Any], model_dir: Path, *, training_metadata: dict[str, Any]) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(model_dir / "relative_value_checkpoint.npz", x_train=model["x_train"], dual=model["dual"])
    payload = {
        "method": "linear-kernel ridge regression on paired action-minus-ANSWER outcomes and costs",
        "feature_contract": "only PublicStateVectorizer(public_state): visible user/assistant text and structural counts",
        "vectorizer": model["vectorizer"].to_json(),
        "actions": list(ACTIONS), "collaboration_actions": list(COLLAB_ACTIONS), "metrics": list(METRICS),
        "alpha": model["alpha"], "max_features": model["max_features"],
        "target_columns": model["target_columns"], "train_state_ids": model["train_state_ids"],
        "train_case_ids": model["train_case_ids"], "training": training_metadata,
        "cost_rule": "predict patient-turn and expert-token deltas separately; policy utility subtracts each exactly once",
    }
    (model_dir / "relative_value_checkpoint.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_relative_model(model_dir: Path) -> dict[str, Any]:
    meta = json.loads((model_dir / "relative_value_checkpoint.json").read_text(encoding="utf-8"))
    arrays = np.load(model_dir / "relative_value_checkpoint.npz")
    return {
        "vectorizer": PublicStateVectorizer.from_json(meta["vectorizer"]),
        "x_train": arrays["x_train"], "dual": arrays["dual"],
        "target_columns": meta["target_columns"], "alpha": float(meta["alpha"]),
        "max_features": int(meta["max_features"]), "train_state_ids": meta["train_state_ids"],
        "train_case_ids": meta["train_case_ids"],
    }
