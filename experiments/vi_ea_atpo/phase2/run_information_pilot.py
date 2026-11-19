#!/usr/bin/env python3
"""Run the frozen one-step ASK/CONSULT pilot; default behavior is preflight only."""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PHASE2 = ROOT / "experiments/vi_ea_atpo/phase2"
PROJECT_DIR = ROOT / "ea_atpo_stage"
AUDIT_DIR = ROOT / "experiments/real_case_pilot_20261103/audit"
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(AUDIT_DIR))

import run_direct_reliability_eval as direct_audit  # noqa: E402

from recipe.ea_atpo.collect_counterfactuals import (  # noqa: E402
    ACTION_REGEX,
    BASE_FORCE,
    ChatAPI,
    CONTINUATION_SYSTEM,
    PATIENT_SYSTEM,
    _answer_letter,
    _base_action,
    _call_metrics,
    _consult_expert,
    _defer_to_expert,
    _extend_state,
    _gold_letter,
    _jsonable_messages,
    _patient_messages,
    rollout_action,
)
from recipe.ea_atpo.protocol import Action, PublicState, parse_action, visible_response  # noqa: E402

COSTS = (0.00, 0.05, 0.10, 0.20, 0.40, 0.60)
TERMINAL_DECISION_REGEX = r"^(?:Final Answer: [A-D]|Defer to Expert)$"
DECISION_SYSTEM = (
    "This is the terminal decision after one information acquisition. Choose exactly one: "
    "answer now as 'Final Answer: <one option letter A-D>' or transfer control as "
    "'Defer to Expert'. Do not ask or consult again. Do not include private reasoning "
    "in the visible action."
)
PROMPT_VERSION = "vi-ea-atpo-one-step-pilot-v2"
CONTINUATION_VERSION = "forced-answer-v1; terminal-answer-or-defer-v1"
PAIRED_METRIC_FIELDS = (
    "split", "strategy", "cost_defer", "n_cases", "accuracy", "utility", "ask_rate", "consult_rate",
    "defer_rate", "negative_utility_rate", "average_action_cost", "expert_calls",
    "accuracy_delta_vs_direct_answer", "accuracy_delta_vs_direct_answer_ci95_low",
    "accuracy_delta_vs_direct_answer_ci95_high", "accuracy_delta_vs_direct_defer",
    "accuracy_delta_vs_direct_defer_ci95_low", "accuracy_delta_vs_direct_defer_ci95_high",
    "utility_lift_vs_direct_answer", "utility_lift_vs_direct_answer_ci95_low",
    "utility_lift_vs_direct_answer_ci95_high", "utility_lift_vs_direct_defer",
    "utility_lift_vs_direct_defer_ci95_low", "utility_lift_vs_direct_defer_ci95_high",
    "frozen_static_action_selected_on_train", "utility_lift_vs_frozen_static",
    "utility_lift_vs_frozen_static_ci95_low", "utility_lift_vs_frozen_static_ci95_high",
)
VOI_ANALYSIS_FIELDS = (
    "split", "group", "strategy", "analysis", "cost_defer", "n_cases", "value", "ci95_low", "ci95_high",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def answer_correct(answer: str | None, gold: Any) -> bool:
    return answer is not None and answer.strip().upper() == _gold_letter(gold)


def utility(correct: bool, n_ask: int, n_consult: int, n_defer: int,
            c_ask: float, c_consult: float, c_defer: float) -> float:
    return float(bool(correct)) - c_ask * n_ask - c_consult * n_consult - c_defer * n_defer


def bootstrap_mean_ci(values: Sequence[float], seed: int, repeats: int = 5000) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, arr.size, size=(repeats, arr.size))
    means = arr[draws].mean(axis=1)
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))


def public_features(state: PublicState) -> dict[str, Any]:
    text = "\n".join(str(message.get("content", "")) for message in state.messages
                     if message.get("role") != "system")
    words = text.split()
    return {
        "public_chars": len(text),
        "public_words": len(words),
        "question_marks": text.count("?"),
        "base_confidence": None,
        "base_entropy": None,
        "base_top1_top2_margin": None,
        "uncertainty_status": "unavailable_from_current_completion_api",
        "option_features": "unavailable_not_applicable_to_all_source_cases",
    }


def endpoint_status(settings: Mapping[str, Any]) -> dict[str, Any]:
    url = str(settings["base_url"]).rstrip("/") + "/models"
    try:
        request = urllib.request.Request(url)
        with urllib.request.urlopen(request, timeout=3) as response:
            body = json.loads(response.read().decode("utf-8"))
            ids = [str(item.get("id", "")) for item in body.get("data", [])]
            expected = str(settings["model"])
            result = {
                "url": url,
                "reachable": True,
                "http_status": response.status,
                "model_ids": ids,
                "configured_model": expected,
                "configured_model_listed": expected in ids if ids else None,
                "model_metadata": body.get("data", []),
            }
            expected_revision = settings.get("revision")
            manifest_ref = settings.get("service_manifest")
            manifest: dict[str, Any] | None = None
            manifest_path: Path | None = None
            if manifest_ref:
                manifest_path = Path(str(manifest_ref))
                if not manifest_path.is_absolute():
                    manifest_path = ROOT / manifest_path
                if manifest_path.is_file():
                    try:
                        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        manifest = None
            identity_ok = (
                manifest is not None
                and manifest.get("served_model_name") == expected
                and manifest.get("model_revision") == expected_revision
                and (not settings.get("vllm_version") or manifest.get("framework_version") == settings.get("vllm_version"))
            ) if expected_revision else None
            result.update({
                "expected_model_revision": expected_revision,
                "service_manifest_path": str(manifest_path) if manifest_path else None,
                "service_manifest_found": manifest is not None,
                "service_manifest_revision": manifest.get("model_revision") if manifest else None,
                "service_manifest_matches": identity_ok,
            })
            return result
    except urllib.error.HTTPError as exc:
        return {"url": url, "reachable": True, "http_status": exc.code,
                "configured_model": str(settings["model"]), "error": f"HTTP {exc.code}"}
    except Exception as exc:
        return {"url": url, "reachable": False, "configured_model": str(settings["model"]),
                "error": f"{type(exc).__name__}: {exc}"}


def gpu_inventory() -> list[dict[str, Any]]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in result.stdout.splitlines():
        columns = [x.strip() for x in line.split(",")]
        if len(columns) == 5:
            total, used = int(columns[2]), int(columns[3])
            rows.append({"index": columns[0], "name": columns[1], "memory_total_mib": total,
                         "memory_used_mib": used, "memory_free_mib": total - used,
                         "utilization_pct": int(columns[4])})
    return rows


def safe_gpu_available(gpus: Sequence[Mapping[str, Any]]) -> bool:
    return any(int(gpu["memory_free_mib"]) >= 24576 and int(gpu["utilization_pct"]) <= 50
               for gpu in gpus)


def load_pilot_rows(selection_csv: Path, data_path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    cohort = pd.read_csv(selection_csv)
    if cohort.case_id.duplicated().any() or cohort.prompt_hash.duplicated().any():
        raise ValueError("frozen pilot cohort has duplicate case_id or prompt_hash")
    if set(cohort["split"]) - {"train", "validation"}:
        raise ValueError("pilot selection contains an unexpected split")
    if cohort.groupby("split").size().to_dict() != {"train": 100, "validation": 50}:
        raise ValueError("pilot cohort must contain exactly 100 train and 50 validation cases")
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    frame = pd.read_parquet(data_path)
    rows = []
    for record in cohort.to_dict(orient="records"):
        source_ix = int(record["source_row_index"])
        row = frame.iloc[source_ix].to_dict()
        extra = row["extra_info"]
        initial_messages = _jsonable_messages(extra["prompt"])
        state = PublicState(str(record["case_id"]), initial_messages)
        identity_messages = direct_audit.build_messages(row)
        actual_hash = hashlib.sha256(
            json.dumps(identity_messages, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if actual_hash != str(record["prompt_hash"]):
            raise ValueError(f"prompt hash mismatch for frozen case {record['case_id']}")
        rows.append({
            "case_id": str(record["case_id"]),
            "split": str(record["split"]),
            "source_row_index": source_ix,
            "state": state,
            "atomic_facts": [str(value) for value in extra.get("atomic_facts", [])],
            "ground_truth": row["reward_model"]["ground_truth"],
        })
    if limit is not None:
        # Smoke testing takes a deterministic prefix separately within each split.
        counts = {"train": max(1, limit * 2 // 3), "validation": max(1, limit - limit * 2 // 3)}
        limited = []
        for split in ("train", "validation"):
            limited.extend([row for row in rows if row["split"] == split][:counts[split]])
        rows = limited[:limit]
    return rows


async def terminal_actor_decision(base: ChatAPI, state: PublicState, temperature: float = 0.0) -> tuple[Action, str, dict[str, int | float]]:
    messages = state.messages
    system_index = next((i for i, message in enumerate(messages) if message["role"] == "system"), None)
    if system_index is None:
        messages.insert(0, {"role": "system", "content": DECISION_SYSTEM})
    else:
        messages[system_index] = {
            **messages[system_index],
            "content": DECISION_SYSTEM + "\n\n" + messages[system_index]["content"],
        }
    call = await base.generate(
        messages, temperature=temperature, max_tokens=16,
        structured_outputs={"regex": TERMINAL_DECISION_REGEX},
    )
    action, content = parse_action(visible_response(call.content))
    if action not in (Action.ANSWER, Action.DEFER):
        raise ValueError(f"terminal decision emitted disallowed action {action.value}")
    return action, content, _call_metrics(call)


async def evaluate_information_branch(
    action: Action,
    row: Mapping[str, Any],
    base: ChatAPI,
    patient: ChatAPI,
    expert: ChatAPI,
    temperature: float = 0.0,
    model_versions: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    state: PublicState = row["state"]
    facts: Sequence[str] = row["atomic_facts"]
    action_content, action_call = await _base_action(base, state, action, temperature=temperature)
    visible_action = {"role": "assistant", "content": (
        f"Question: {action_content}" if action is Action.ASK else f"Consult Expert: {action_content}"
    )}
    if action is Action.ASK:
        patient_call = await patient.generate(
            _patient_messages(facts, action_content), temperature=temperature, max_tokens=256
        )
        acquired = visible_response(patient_call.content)
        next_state = _extend_state(state, visible_action, {"role": "user", "content": acquired})
        acquisition_trace = [
            {"actor": "base", "action": action.value, "response": visible_action["content"], **_call_metrics(action_call)},
            {"actor": "patient", "response": acquired, **_call_metrics(patient_call)},
        ]
    else:
        acquired, expert_call = await _consult_expert(
            expert, state, action_content, facts, temperature=temperature
        )
        next_state = _extend_state(state, visible_action, {"role": "user", "content": acquired})
        acquisition_trace = [
            {"actor": "base", "action": action.value, "response": visible_action["content"],
             "consult_question": action_content, **_call_metrics(action_call)},
            {"actor": "expert", "action": action.value, "response": acquired, **_call_metrics(expert_call)},
        ]

    forced_answer, task_call = await _base_action(base, next_state, Action.ANSWER, temperature=temperature)
    task_action = Action.ANSWER
    decision_answer: str | None = None
    decision_action: Action
    decision_usage: dict[str, int | float]
    decision_expert_usage: dict[str, int | float] | None = None
    try:
        decision_action, decision_content, decision_usage = await terminal_actor_decision(base, next_state, temperature)
        if decision_action is Action.ANSWER:
            decision_answer = decision_content.strip().upper()
        else:
            decision_answer, decision_call = await _defer_to_expert(expert, next_state, facts, temperature=temperature)
            decision_expert_usage = _call_metrics(decision_call)
    except Exception as exc:
        decision_action = Action.ANSWER
        decision_answer = None
        decision_usage = {"prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0.0}
        decision_expert_usage = {"error": f"{type(exc).__name__}: {exc}"}

    gold = row["ground_truth"]
    task_correct = answer_correct(forced_answer, gold)
    decision_correct = answer_correct(decision_answer, gold)
    task_calls = [_call_metrics(action_call), _call_metrics(task_call)]
    decision_calls: list[Mapping[str, Any]] = [_call_metrics(action_call), decision_usage]
    if action is Action.ASK:
        decision_calls.append(_call_metrics(patient_call))
    else:
        decision_calls.append(_call_metrics(expert_call))
    if decision_expert_usage and "prompt_tokens" in decision_expert_usage:
        decision_calls.append(decision_expert_usage)
    common = {
        "initial_state_id": state.state_id,
        "pre_action_features": public_features(state),
        "acquired_information": acquired,
        "next_state": next_state.messages,
        "sample_seed": int(row.get("sample_seed", 20261118)),
        "rollout_seed": 20261118,
        "model_version": dict(model_versions or {}),
        "prompt_version": PROMPT_VERSION,
        "action_cost": 0.05 if action is Action.ASK else 0.20,
        "intervention_action": action.value,
    }
    def usage_total(calls: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
        return {
            "prompt_tokens": sum(int(x.get("prompt_tokens", 0)) for x in calls),
            "completion_tokens": sum(int(x.get("completion_tokens", 0)) for x in calls),
            "latency_s": sum(float(x.get("latency_s", 0.0)) for x in calls),
        }
    task = {
        **common,
        "intervention": f"{action.value}_TASK_ANSWER",
        "continuation_policy_version": "forced-answer-v1",
        "terminal_action": task_action.value,
        "terminal_correct": task_correct,
        "terminal_answer": forced_answer,
        "action_counts": {"ask": int(action is Action.ASK), "consult": int(action is Action.CONSULT), "defer": 0},
        "usage": usage_total(task_calls + ([_call_metrics(patient_call)] if action is Action.ASK
                                             else [_call_metrics(expert_call)])),
        "trace": acquisition_trace + [{"actor": "base", "action": "ANSWER",
                                         "response": f"Final Answer: {forced_answer}", **_call_metrics(task_call)}],
    }
    decision = {
        **common,
        "intervention": f"{action.value}_DECISION",
        "continuation_policy_version": "terminal-answer-or-defer-v1",
        "terminal_action": decision_action.value,
        "terminal_correct": decision_correct,
        "terminal_answer": decision_answer,
        "action_counts": {"ask": int(action is Action.ASK), "consult": int(action is Action.CONSULT),
                          "defer": int(decision_action is Action.DEFER)},
        "usage": usage_total(decision_calls),
        "trace": acquisition_trace + [{"actor": "base", "action": decision_action.value,
                                         "response": f"Final Answer: {decision_answer}" if decision_action is Action.ANSWER else "Defer to Expert",
                                         **decision_usage}],
    }
    if decision_expert_usage and "error" in decision_expert_usage:
        decision["error"] = decision_expert_usage["error"]
    if decision_expert_usage and "prompt_tokens" in decision_expert_usage:
        decision["trace"].append({"actor": "expert", "action": "DEFER",
                                  "response": f"Expert Final Answer: {decision_answer}", **decision_expert_usage})
    return task, decision


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: Sequence[Mapping[str, Any]], costs: Mapping[str, float]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_key = {(str(row["case_id"]), str(row["intervention"])): row for row in rows}
    outcome_rows: list[dict[str, Any]] = []
    analysis_rows: list[dict[str, Any]] = []
    branches = ("ANSWER", "DEFER", "ASK_TASK_ANSWER", "ASK_DECISION", "CONSULT_TASK_ANSWER", "CONSULT_DECISION")

    def row_utility(row: Mapping[str, Any], c_defer: float) -> float:
        counts = row["action_counts"]
        return utility(bool(row["terminal_correct"]), int(counts["ask"]), int(counts["consult"]),
                       int(counts["defer"]), float(costs["ask"]), float(costs["consult"]), c_defer)

    def paired_summary(case_ids: Sequence[str], intervention: str, baseline: str, c_defer: float,
                       seed: int) -> tuple[float, float, float, float, float, float]:
        current = [by_key[(case, intervention)] for case in case_ids
                   if (case, intervention) in by_key and (case, baseline) in by_key]
        accuracy_delta = [float(row["terminal_correct"])
                          - float(by_key[(str(row["case_id"]), baseline)]["terminal_correct"])
                          for row in current]
        utility_delta = [row_utility(row, c_defer)
                         - row_utility(by_key[(str(row["case_id"]), baseline)], c_defer)
                         for row in current]
        acc_low, acc_high = bootstrap_mean_ci(accuracy_delta, seed)
        util_low, util_high = bootstrap_mean_ci(utility_delta, seed + 1)
        return (
            float(np.mean(accuracy_delta)) if accuracy_delta else float("nan"), acc_low, acc_high,
            float(np.mean(utility_delta)) if utility_delta else float("nan"), util_low, util_high,
        )

    for split in ("train", "validation"):
        cases = sorted({str(row["case_id"]) for row in rows if row["split"] == split})
        static_action_by_cost = {}
        train_cases = sorted({str(row["case_id"]) for row in rows if row["split"] == "train"})
        for c_d in costs["defer_grid"]:
            ans = [float(by_key[(case, "DIRECT_ANSWER")]["terminal_correct"]) for case in train_cases]
            deff = [float(by_key[(case, "DIRECT_DEFER")]["terminal_correct"]) - c_d for case in train_cases]
            static_action_by_cost[c_d] = "ANSWER" if np.mean(ans) >= np.mean(deff) else "DEFER"
        for c_d in costs["defer_grid"]:
            for name in branches:
                intervention = {"ANSWER": "DIRECT_ANSWER", "DEFER": "DIRECT_DEFER"}.get(name, name)
                matched_cases = [case for case in cases if (case, intervention) in by_key]
                selected = [by_key[(case, intervention)] for case in matched_cases]
                if not selected:
                    continue
                accuracies = [float(x["terminal_correct"]) for x in selected]
                utilities = [row_utility(x, float(c_d)) for x in selected]
                average_costs = [float(x["action_counts"]["ask"]) * float(costs["ask"])
                                 + float(x["action_counts"]["consult"]) * float(costs["consult"])
                                 + float(x["action_counts"]["defer"]) * float(c_d) for x in selected]
                static_name = "DIRECT_ANSWER" if static_action_by_cost[c_d] == "ANSWER" else "DIRECT_DEFER"
                seed = 20261118 + int(c_d * 1000) + len(selected) + sum(map(ord, name))
                acc_answer, acc_answer_low, acc_answer_high, util_answer, util_answer_low, util_answer_high = paired_summary(
                    matched_cases, intervention, "DIRECT_ANSWER", float(c_d), seed
                )
                acc_defer, acc_defer_low, acc_defer_high, util_defer, util_defer_low, util_defer_high = paired_summary(
                    matched_cases, intervention, "DIRECT_DEFER", float(c_d), seed + 10
                )
                acc_static, acc_static_low, acc_static_high, util_static, util_static_low, util_static_high = paired_summary(
                    matched_cases, intervention, static_name, float(c_d), seed + 20
                )
                row = {
                    "split": split, "strategy": name, "cost_defer": c_d, "n_cases": len(selected),
                    "accuracy": float(np.mean(accuracies)), "utility": float(np.mean(utilities)),
                    "ask_rate": float(np.mean([int(x["action_counts"]["ask"]) for x in selected])),
                    "consult_rate": float(np.mean([int(x["action_counts"]["consult"]) for x in selected])),
                    "defer_rate": float(np.mean([int(x["action_counts"]["defer"]) for x in selected])),
                    "negative_utility_rate": float(np.mean(np.asarray(utilities) < 0.0)),
                    "average_action_cost": float(np.mean(average_costs)),
                    "expert_calls": sum(int(x["action_counts"]["consult"]) + int(x["action_counts"]["defer"])
                                        for x in selected),
                    "accuracy_delta_vs_direct_answer": acc_answer,
                    "accuracy_delta_vs_direct_answer_ci95_low": acc_answer_low,
                    "accuracy_delta_vs_direct_answer_ci95_high": acc_answer_high,
                    "accuracy_delta_vs_direct_defer": acc_defer,
                    "accuracy_delta_vs_direct_defer_ci95_low": acc_defer_low,
                    "accuracy_delta_vs_direct_defer_ci95_high": acc_defer_high,
                    "utility_lift_vs_direct_answer": util_answer,
                    "utility_lift_vs_direct_answer_ci95_low": util_answer_low,
                    "utility_lift_vs_direct_answer_ci95_high": util_answer_high,
                    "utility_lift_vs_direct_defer": util_defer,
                    "utility_lift_vs_direct_defer_ci95_low": util_defer_low,
                    "utility_lift_vs_direct_defer_ci95_high": util_defer_high,
                    "frozen_static_action_selected_on_train": static_action_by_cost[c_d],
                    "utility_lift_vs_frozen_static": util_static,
                    "utility_lift_vs_frozen_static_ci95_low": util_static_low,
                    "utility_lift_vs_frozen_static_ci95_high": util_static_high,
                }
                outcome_rows.append(row)
                for metric, value, low, high in (
                    ("accuracy_delta_vs_direct_answer", acc_answer, acc_answer_low, acc_answer_high),
                    ("accuracy_delta_vs_direct_defer", acc_defer, acc_defer_low, acc_defer_high),
                    ("accuracy_delta_vs_frozen_static_action", acc_static, acc_static_low, acc_static_high),
                    ("utility_delta_vs_direct_answer", util_answer, util_answer_low, util_answer_high),
                    ("utility_delta_vs_direct_defer", util_defer, util_defer_low, util_defer_high),
                    ("utility_delta_vs_frozen_static_action", util_static, util_static_low, util_static_high),
                ):
                    analysis_rows.append({"split": split, "strategy": name, "analysis": metric,
                                          "cost_defer": c_d, "n_cases": len(selected), "value": value,
                                          "ci95_low": low, "ci95_high": high})

    # Outcome-defined helpful/harmful groups are offline strata only; never model inputs.
    for split in ("train", "validation"):
        cases = sorted({str(row["case_id"]) for row in rows if row["split"] == split})
        groups: dict[str, list[str]] = {"helpful_defer": [], "harmful_defer": [], "both_correct": [], "both_wrong": []}
        for case in cases:
            a = bool(by_key[(case, "DIRECT_ANSWER")]["terminal_correct"])
            d = bool(by_key[(case, "DIRECT_DEFER")]["terminal_correct"])
            group = "helpful_defer" if d and not a else "harmful_defer" if a and not d else "both_correct" if a else "both_wrong"
            groups[group].append(case)
        train_cases = sorted({str(row["case_id"]) for row in rows if row["split"] == "train"})
        for group, case_ids in groups.items():
            for strategy in ("ASK_TASK_ANSWER", "ASK_DECISION", "CONSULT_TASK_ANSWER", "CONSULT_DECISION"):
                if not case_ids:
                    continue
                for c_d in costs["defer_grid"]:
                    selected_cases = [case for case in case_ids if (case, strategy) in by_key]
                    if not selected_cases:
                        continue
                    static_action = "ANSWER" if np.mean([
                        float(by_key[(case, "DIRECT_ANSWER")]["terminal_correct"]) for case in train_cases
                    ]) >= np.mean([
                        float(by_key[(case, "DIRECT_DEFER")]["terminal_correct"]) - c_d for case in train_cases
                    ]) else "DEFER"
                    static_name = "DIRECT_ANSWER" if static_action == "ANSWER" else "DIRECT_DEFER"
                    seed = 20261121 + len(selected_cases) + sum(map(ord, group + strategy)) + int(c_d * 1000)
                    for baseline, metric_name in (
                        ("DIRECT_ANSWER", "direct_answer"),
                        ("DIRECT_DEFER", "direct_defer"),
                        (static_name, "frozen_static_action"),
                    ):
                        acc, acc_low, acc_high, util_delta, util_low, util_high = paired_summary(
                            selected_cases, strategy, baseline, float(c_d), seed
                        )
                        for metric, value, low, high in (
                            (f"accuracy_delta_vs_{metric_name}", acc, acc_low, acc_high),
                            (f"utility_delta_vs_{metric_name}", util_delta, util_low, util_high),
                        ):
                            analysis_rows.append({
                                "split": split, "group": group, "strategy": strategy, "analysis": metric,
                                "cost_defer": c_d, "n_cases": len(selected_cases), "value": value,
                                "ci95_low": low, "ci95_high": high,
                            })
    return outcome_rows, analysis_rows


async def run_rollouts(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    endpoints = config["endpoints"]
    sampling = config.get("sampling", {})
    temperature = float(sampling.get("temperature", config.get("temperature", 0.0)))
    top_p = float(sampling.get("top_p", 1.0))
    model_versions = {
        "base_revision": endpoints["base"].get("revision"),
        "patient_revision": endpoints["patient"].get("revision"),
        "expert_revision": endpoints["expert"].get("revision"),
        "vllm_version": endpoints["base"].get("vllm_version"),
    }
    base = ChatAPI(endpoints["base"]["base_url"], endpoints["base"]["model"], enable_thinking=endpoints["base"].get("enable_thinking"), top_p=top_p)
    patient = ChatAPI(endpoints["patient"]["base_url"], endpoints["patient"]["model"], enable_thinking=endpoints["patient"].get("enable_thinking"), top_p=top_p)
    expert = ChatAPI(endpoints["expert"]["base_url"], endpoints["expert"]["model"], enable_thinking=endpoints["expert"].get("enable_thinking"), top_p=top_p)
    rollout_path = out / "intervention_rollouts.jsonl"
    all_rows: list[dict[str, Any]] = []
    costs = config["costs"]
    try:
        for case in rows:
            direct_answer = await rollout_action(Action.ANSWER, case["state"], case["atomic_facts"], case["ground_truth"],
                                                 base, patient, expert, temperature=temperature, max_turns=2)
            direct_defer = await rollout_action(Action.DEFER, case["state"], case["atomic_facts"], case["ground_truth"],
                                                base, patient, expert, temperature=temperature, max_turns=2)
            branch_rows = []
            for intervention in (Action.ASK, Action.CONSULT):
                task, decision = await evaluate_information_branch(
                    intervention, case, base, patient, expert, temperature, model_versions
                )
                branch_rows.extend((task, decision))
            baseline_rows = []
            for name, result, action in (("DIRECT_ANSWER", direct_answer, Action.ANSWER),
                                         ("DIRECT_DEFER", direct_defer, Action.DEFER)):
                correct = bool(result["valid"] and result.get("outcome", 0.0) == 1.0)
                baseline_rows.append({
                    "initial_state_id": case["state"].state_id,
                    "intervention": name,
                    "pre_action_features": public_features(case["state"]),
                    "acquired_information": "",
                    "next_state": result.get("public_messages", case["state"].messages),
                    "continuation_policy_version": "direct-terminal-v1",
                    "terminal_action": action.value,
                    "terminal_correct": correct,
                    "terminal_answer": result.get("answer"),
                    "action_counts": {"ask": 0, "consult": 0, "defer": int(action is Action.DEFER)},
                    "action_cost": 0.0 if action is Action.ANSWER else float(costs["primary_defer"]),
                    "usage": {"prompt_tokens": int(result.get("total_prompt_tokens", 0)),
                              "completion_tokens": int(result.get("total_completion_tokens", 0)),
                              "latency_s": float(result.get("total_latency_s", 0.0))},
                    "trace": result.get("trace", []),
                    "error": result.get("error"),
                    "prompt_version": PROMPT_VERSION,
                    "model_version": dict(model_versions),
                    "sample_seed": config["seed"], "rollout_seed": config["seed"],
                })
            for item in baseline_rows + branch_rows:
                item.update({"case_id": case["case_id"], "split": case["split"]})
                item["action_cost"] = float(item["action_counts"]["ask"]) * float(costs["ask"]) + \
                    float(item["action_counts"]["consult"]) * float(costs["consult"]) + \
                    float(item["action_counts"]["defer"]) * float(costs["primary_defer"])
                item["total_utility"] = utility(
                    bool(item["terminal_correct"]), int(item["action_counts"]["ask"]),
                    int(item["action_counts"]["consult"]), int(item["action_counts"]["defer"]),
                    float(costs["ask"]), float(costs["consult"]), float(costs["primary_defer"]),
                )
                item["baseline_utility_direct_answer"] = float(baseline_rows[0]["terminal_correct"])
                item["baseline_utility_direct_defer"] = float(baseline_rows[1]["terminal_correct"]) - float(costs["primary_defer"])
                item["observed_utility_difference_vs_direct_answer"] = item["total_utility"] - item["baseline_utility_direct_answer"]
                item["observed_utility_difference_vs_direct_defer"] = item["total_utility"] - item["baseline_utility_direct_defer"]
                all_rows.append(item)
                with rollout_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(item, ensure_ascii=False) + "\n")
                    stream.flush()
        return all_rows, {"status": "completed", "cases": len(rows), "rows": len(all_rows)}
    finally:
        await asyncio.gather(base.close(), patient.close(), expert.close())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PHASE2 / "pilot_config.json")
    parser.add_argument("--selection", type=Path, default=PHASE2 / "pilot_cases.csv")
    parser.add_argument("--data", type=Path, default=None,
                        help="Override source parquet; defaults to source_train_path in selection_manifest.json")
    parser.add_argument("--output-dir", type=Path, default=PHASE2)
    parser.add_argument("--run-pilot", action="store_true", help="Run all selected 150 cases; not used by default.")
    parser.add_argument("--smoke-cases", type=int, default=0, help="Run up to 2 cases only if services and GPU gate pass.")
    args = parser.parse_args()
    if args.smoke_cases < 0 or args.smoke_cases > 2:
        raise SystemExit("--smoke-cases must be between 0 and 2")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    selection_manifest_path = args.selection.with_name("selection_manifest.json")
    if args.data is not None:
        data_path = args.data.resolve()
    elif selection_manifest_path.is_file():
        data_path = Path(json.loads(selection_manifest_path.read_text(encoding="utf-8"))["source_train_path"])
    else:
        data_path = Path("")
    endpoints = {name: endpoint_status(spec) for name, spec in config["endpoints"].items()}
    gpus = gpu_inventory()
    services_ready = all(
        x.get("reachable") and x.get("configured_model_listed")
        and (x.get("expected_model_revision") is None or x.get("service_manifest_matches") is True)
        for x in endpoints.values()
    )
    gpu_safe = safe_gpu_available(gpus)
    should_run = args.run_pilot or args.smoke_cases > 0
    if args.run_pilot and args.smoke_cases:
        raise SystemExit("choose either --run-pilot or --smoke-cases")
    status = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experiment": config["experiment"],
        "repo_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).strip().splitlines(),
        "python": platform.python_version(),
        "config_path": str(args.config.resolve()), "config_sha256": sha256_file(args.config),
        "selection_path": str(args.selection.resolve()),
        "selection_sha256": sha256_file(args.selection) if args.selection.is_file() else None,
        "selection_manifest_sha256": sha256_file(selection_manifest_path) if selection_manifest_path.is_file() else None,
        "selection_exists": args.selection.is_file(),
        "data_path": str(data_path.resolve()), "data_exists": data_path.is_file(),
        "data_sha256": sha256_file(data_path) if data_path.is_file() else None,
        "pilot_script_sha256": sha256_file(Path(__file__).resolve()),
        "collector_sha256": sha256_file(PROJECT_DIR / "recipe/ea_atpo/collect_counterfactuals.py"),
        "protocol_sha256": sha256_file(PROJECT_DIR / "recipe/ea_atpo/protocol.py"),
        "prompt_protocol_sha256": hashlib.sha256(json.dumps({
            "version": PROMPT_VERSION,
            "continuation_version": CONTINUATION_VERSION,
            "base_force": {action.value: prompt for action, prompt in BASE_FORCE.items()},
            "continue": CONTINUATION_SYSTEM,
            "patient": PATIENT_SYSTEM,
            "decision": DECISION_SYSTEM,
        }, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest(),
        "endpoints": endpoints, "gpu_inventory": gpus,
        "sampling": config.get("sampling", {"temperature": config.get("temperature", 0.0), "top_p": 1.0}),
        "gpu_gate_passed": gpu_safe,
        "services_ready": services_ready,
        "nominal_full_pilot_api_calls_excluding_retries": config["request_budget"]["nominal_total_api_calls_for_150_cases"],
        "model_inference_started": False,
    }
    if args.selection.is_file() and data_path.is_file():
        try:
            frozen_rows = load_pilot_rows(args.selection, data_path)
            status["frozen_cohort_counts"] = {
                split: sum(row["split"] == split for row in frozen_rows)
                for split in ("train", "validation")
            }
            status["frozen_case_ids_unique"] = len({row["case_id"] for row in frozen_rows}) == len(frozen_rows)
        except Exception as exc:
            status.update({"status": "blocked_frozen_cohort_invalid",
                           "reason": f"{type(exc).__name__}: {exc}"})
    if status.get("status") == "blocked_frozen_cohort_invalid":
        pass
    elif not args.selection.is_file():
        status.update({"status": "blocked_selection_missing", "reason": "pilot_cases.csv has not been frozen"})
    elif not data_path.is_file():
        status.update({"status": "blocked_data_missing", "reason": "source ATPO train parquet is unavailable"})
    elif should_run and not services_ready:
        status.update({"status": "not_run_service_unavailable", "reason": "one or more configured local model aliases are not reachable"})
    elif should_run and not gpu_safe:
        status.update({"status": "not_run_gpu_busy", "reason": "no local GPU meets >=24 GiB free and <=50% utilization"})
    elif not should_run and status.get("status") != "blocked_frozen_cohort_invalid":
        status.update({
            "status": "preflight_only" if services_ready and gpu_safe else "preflight_only_service_or_gpu_blocked",
            "reason": "no inference requested",
        })
    elif should_run and status.get("status") != "blocked_frozen_cohort_invalid":
        n = args.smoke_cases if args.smoke_cases else None
        rollout_file = out / "intervention_rollouts.jsonl"
        if rollout_file.exists() and rollout_file.stat().st_size > 0:
            raise FileExistsError(f"refusing to overwrite existing rollouts: {rollout_file}")
        rows = load_pilot_rows(args.selection, data_path, n)
        status["selected_run_cases"] = len(rows)
        status["nominal_max_api_calls_excluding_retries"] = 12 * len(rows)
        status["model_inference_started"] = True
        status["status"] = "running"
        (out / "execution_manifest.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        try:
            records, result = asyncio.run(run_rollouts(rows, config, out))
            cost_rows, analysis = summarize(records, config["costs"])
            write_csv(out / "paired_intervention_metrics.csv", cost_rows, PAIRED_METRIC_FIELDS)
            write_csv(out / "voi_pilot_analysis.csv", analysis, VOI_ANALYSIS_FIELDS)
            status.update(result)
        except Exception as exc:
            status.update({"status": "failed_with_partial_rollouts", "error": f"{type(exc).__name__}: {exc}"})
    if not (out / "intervention_rollouts.jsonl").exists():
        (out / "intervention_rollouts.jsonl").write_text("", encoding="utf-8")
    if status["status"] not in ("completed", "running"):
        write_csv(out / "paired_intervention_metrics.csv", [], PAIRED_METRIC_FIELDS)
        write_csv(out / "voi_pilot_analysis.csv", [], VOI_ANALYSIS_FIELDS)
        with (out / "failure.log").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"status": status["status"], "reason": status.get("reason"),
                                     "endpoints": endpoints, "timestamp": status["created_at_utc"]}, ensure_ascii=False) + "\n")
    (out / "execution_manifest.json").write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: status.get(key) for key in (
        "status", "reason", "selection_exists", "data_exists", "services_ready", "gpu_gate_passed",
        "nominal_full_pilot_api_calls_excluding_retries", "selected_run_cases", "model_inference_started",
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
