#!/usr/bin/env python3
"""Train, validate, and independently evaluate the paired Relative Advantage Router.

The policy chooses one initial action from a public state. For ANSWER, ASK, and
CONSULT branches, the same pinned Base Agent continuation then runs to a final
answer under the fixed ten-turn / one-consult budget. DEFER terminates at the
Expert. This is the same continuation policy used to construct the Pilot labels.
"""

from __future__ import annotations

import argparse
import asyncio
import contextvars
import csv
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PILOT_DIR = REPO / "experiments/real_case_pilot_20261103"
OUTPUT_ROOT = HERE / "outputs/relative_value_20261106"
PILOT_ROWS = PILOT_DIR / "outputs/full/pilot_trajectories.jsonl"
PILOT_SPLIT = HERE / "artifacts/case_split.json"
PILOT_ABSOLUTE_CHECKPOINT = HERE / "artifacts/checkpoint"
TEST_DATA = PILOT_DIR / "assets/test.pilot.parquet"
MODEL_ALIAS = "qwen3-8b-relative-20261106"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
MODEL_PATH = PILOT_DIR / "assets/models/Qwen3-8B"
BASE_URL = "http://127.0.0.1:30143/v1"
PORT = 30143
GENERATION = {"temperature": 0.2, "patient_temperature": 0.0, "max_turns": 10, "max_consults": 1, "max_concurrency": 2}
REFERENCE_COST = {"patient_per_turn": 0.01, "expert_per_1k_tokens": 0.02}
COST_GRID = [
    {"name": "no_cost", "patient_per_turn": 0.0, "expert_per_1k_tokens": 0.0},
    {"name": "patient_0005_expert_001", "patient_per_turn": 0.005, "expert_per_1k_tokens": 0.01},
    {"name": "reference", "patient_per_turn": 0.01, "expert_per_1k_tokens": 0.02},
    {"name": "patient_001_expert_005", "patient_per_turn": 0.01, "expert_per_1k_tokens": 0.05},
    {"name": "patient_003_expert_010", "patient_per_turn": 0.03, "expert_per_1k_tokens": 0.10},
]
VALIDATION_N, TEST_N = 50, 200
VALIDATION_SEED, TEST_SEED = 20261107, 20261108
BOOTSTRAP_REPS = 4000
ROLLOUT_CONTEXT: contextvars.ContextVar[str] = contextvars.ContextVar("ea_atpo_rollout_context", default="")
API_CALL_LEDGER: contextvars.ContextVar[list[dict[str, Any]] | None] = contextvars.ContextVar("ea_atpo_api_call_ledger", default=None)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from ea_atpo_stage.recipe.ea_atpo.collect_counterfactuals import (  # noqa: E402
    APICall, ChatAPI, _gold_letter, rollout_action,
)
from ea_atpo_stage.recipe.ea_atpo.protocol import Action, PublicState  # noqa: E402
from relative_value_router import (  # noqa: E402
    ACTIONS, COLLAB_ACTIONS, FEATURES, METRICS, fit_relative_model,
    load_relative_model, paired_relative_targets, predict_relative, rollout_metrics,
    save_relative_model,
)
from router_core import file_sha256, load_jsonl, write_json  # noqa: E402
from train_router import _load_model as load_absolute_model, predict as predict_absolute  # noqa: E402


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def _macro_mse(y_true: np.ndarray, y_pred: np.ndarray, case_ids: list[str]) -> float:
    by_case: dict[str, list[float]] = defaultdict(list)
    for i, case in enumerate(case_ids):
        by_case[case].extend(np.square(y_true[i] - y_pred[i]).tolist())
    return float(np.mean([np.mean(values) for values in by_case.values()]))


def _relative_y(targets: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([
        [float(t[f"{a}_accuracy_delta"]) for a in COLLAB_ACTIONS]
        for t in targets
    ], dtype=float)


def train_router(output_dir: Path, alphas: list[float]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_records = load_jsonl(PILOT_ROWS)
    # The deployed Router makes one initial-state decision. Train on the
    # matching state distribution instead of adding enriched later states.
    records = [r for r in raw_records if r.get("record_type") != "trajectory" or int(r.get("state_index", 0)) == 0]
    states, targets, noise = paired_relative_targets(records)
    if len({str(s["case_id"]) for s in states}) != len(states):
        raise ValueError("initial-state training filter must yield exactly one paired state per case_id")
    state_by_id = {s["state_id"]: s for s in states}
    target_by_id = {t["state_id"]: t for t in targets}
    split = json.loads(PILOT_SPLIT.read_text(encoding="utf-8"))
    case_split = {name: set(split[name]) for name in ("train", "validation", "internal_test")}
    split_ids = {
        name: sorted(sid for sid, state in state_by_id.items() if state["case_id"] in cases)
        for name, cases in case_split.items()
    }
    if set(split_ids["train"]) & set(split_ids["validation"]) or set(split_ids["train"]) & set(split_ids["internal_test"]) or set(split_ids["validation"]) & set(split_ids["internal_test"]):
        raise ValueError("case split is not disjoint")

    train_states = [state_by_id[sid] for sid in split_ids["train"]]
    train_targets = [target_by_id[sid] for sid in split_ids["train"]]
    val_states = [state_by_id[sid] for sid in split_ids["validation"]]
    val_targets = [target_by_id[sid] for sid in split_ids["validation"]]
    val_y = _relative_y(val_targets)
    alpha_rows = []
    candidate_models = []
    for alpha in alphas:
        model = fit_relative_model(train_states, train_targets, alpha, max_features=FEATURES)
        prediction = predict_relative(model, val_states)
        pred_y = np.asarray([[prediction[s["state_id"]][a]["accuracy"] for a in COLLAB_ACTIONS] for s in val_states])
        score = _macro_mse(val_y, pred_y, [s["case_id"] for s in val_states])
        alpha_rows.append({"alpha": alpha, "pilot_validation_case_macro_advantage_mse": score,
                           "training_cases": len({s["case_id"] for s in train_states}),
                           "validation_cases": len({s["case_id"] for s in val_states}),
                           "max_features": FEATURES})
        candidate_models.append((score, -alpha, float(alpha)))
    best_alpha = min(candidate_models)[2]
    train_only_selected = fit_relative_model(train_states, train_targets, best_alpha, max_features=FEATURES)

    # Freeze alpha on the Pilot's case-held-out validation split; fit final
    # parameters on train+validation only, leaving internal_test untouched.
    fit_ids = sorted(split_ids["train"] + split_ids["validation"])
    final_states = [state_by_id[sid] for sid in fit_ids]
    final_targets = [target_by_id[sid] for sid in fit_ids]
    model = fit_relative_model(final_states, final_targets, best_alpha, max_features=FEATURES)
    relative_checkpoint_dir = output_dir / "checkpoint_relative_initial_only"
    save_relative_model(model, relative_checkpoint_dir, training_metadata={
        "pilot_source": str(PILOT_ROWS.resolve()), "pilot_sha256": file_sha256(PILOT_ROWS),
        "case_split_path": str(PILOT_SPLIT.resolve()), "case_split_sha256": file_sha256(PILOT_SPLIT),
        "case_split_seed": split.get("seed"), "split_counts": {k: len(v) for k, v in case_split.items()},
        "fit_cases": len({s["case_id"] for s in final_states}), "fit_states": len(final_states),
        "alpha_selection": "minimum case-macro MSE of paired accuracy advantages on the 15-case Pilot validation split; ties favor larger alpha",
        "candidate_alphas": alphas, "selected_alpha": best_alpha,
        "pilot_state_filter": "state_index == 0 (initial public state); matches one-shot deployed Router input distribution",
        "excluded_noninitial_states": len({str(r["state_id"]) for r in raw_records if r.get("record_type") == "trajectory" and int(r.get("state_index", 0)) != 0}),
        "costs_separate_from_accuracy": True,
    })
    _write_csv(output_dir / "alpha_validation.csv", alpha_rows)

    # Honest case-held-out pilot validation uses the train-only fit. The
    # internal test uses the final train+validation refit and is never used for
    # alpha selection or fitting.
    from relative_value_router import _target_matrix
    split_metrics = []
    for name in ("train", "validation", "internal_test"):
        ids = split_ids[name]
        subset_states = [state_by_id[sid] for sid in ids]
        subset_targets = [target_by_id[sid] for sid in ids]
        evaluation_model = train_only_selected if name in ("train", "validation") else model
        p = predict_relative(evaluation_model, subset_states)
        yt = _relative_y(subset_targets)
        yp = np.asarray([[p[s["state_id"]][a]["accuracy"] for a in COLLAB_ACTIONS] for s in subset_states])
        split_metrics.append({
            "split": name, "cases": len({s["case_id"] for s in subset_states}), "states": len(subset_states),
            "paired_advantage_mae": float(np.mean(np.abs(yp - yt))) if len(yt) else math.nan,
            "paired_advantage_rmse": float(np.sqrt(np.mean(np.square(yp - yt)))) if len(yt) else math.nan,
            "per_action_mae": json.dumps({a: float(np.mean(np.abs(yp[:, i] - yt[:, i]))) for i, a in enumerate(COLLAB_ACTIONS)}, sort_keys=True),
        })
    _write_csv(output_dir / "pilot_split_metrics.csv", split_metrics)
    _write_csv(output_dir / "paired_label_noise.csv", noise)

    # State-level target table is auditable and intentionally has no raw gold,
    # hidden facts, expert advice, or future messages.
    target_cols = [k for k in targets[0] if k not in ("case_id", "state_id")]
    _write_csv(output_dir / "paired_state_targets.csv", [
        {"case_id": t["case_id"], "state_id": t["state_id"], **{k: t[k] for k in target_cols}}
        for t in targets
    ])
    manifest = {
        "status": "trained", "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method": "paired relative action value regression; fixed one-shot Router followed by pinned Base continuation",
        "pilot_sha256": file_sha256(PILOT_ROWS), "pilot_split_sha256": file_sha256(PILOT_SPLIT),
        "case_split_counts": {k: len(v) for k, v in case_split.items()},
        "all_paired_states": len(states), "paired_target_rows": len(targets),
        "paired_rollout_count": sum(x.get("record_type") == "trajectory" for x in records),
        "pilot_state_filter": "state_index == 0; one initial public state per case",
        "excluded_noninitial_states": len({str(r["state_id"]) for r in raw_records if r.get("record_type") == "trajectory" and int(r.get("state_index", 0)) != 0}),
        "invalid_rollouts_retained_as_zero_outcome": sum(not bool(x.get("valid")) for x in records if x.get("record_type") == "trajectory"),
        "selected_alpha": best_alpha, "max_features": FEATURES,
        "relative_checkpoint_dir": str(relative_checkpoint_dir.resolve()),
        "continuation": {"policy": "Base Agent selects subsequent actions after forced initial action", "max_assistant_turns": 10, "max_consults": 1, "terminal_answer_on_final_turn": True},
        "feature_boundary": "public_state only; only visible user/assistant text and derived public-state counts",
        "cost_contract": "accuracy, patient turns, expert tokens, total tokens, and latency are separate targets; only patient turns and expert tokens enter reference utility",
    }
    write_json(output_dir / "training_manifest.json", manifest)
    return manifest


def _source_indices_seen() -> tuple[set[int], list[dict[str, Any]]]:
    output_dir = HERE / "outputs"
    target_sha = file_sha256(TEST_DATA)
    seen: set[int] = set()
    sources: list[dict[str, Any]] = []
    for path in sorted(output_dir.rglob("run_manifest.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if doc.get("data_sha256") != target_sha:
            continue
        local: set[int] = set()
        for key, value in doc.items():
            if isinstance(value, list) and ("row" in key.lower() or "sampled" in key.lower() or "selected" in key.lower() or "excluded" in key.lower()):
                for item in value:
                    if isinstance(item, (int, np.integer)):
                        local.add(int(item))
        seen |= local
        sources.append({"manifest": str(path.relative_to(REPO)), "rows_recovered": len(local)})
    for path in sorted(output_dir.rglob("*source_rows*.json")):
        try:
            values = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(values, list) and all(isinstance(x, int) for x in values):
            seen.update(map(int, values))
            sources.append({"source_rows_file": str(path.relative_to(REPO)), "rows_recovered": len(values)})
    return seen, sources


def prepare_samples(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_rows_path = output_dir / "validation_source_rows.json"
    test_rows_path = output_dir / "test_source_rows.json"
    if validation_rows_path.exists() and test_rows_path.exists():
        val_rows = json.loads(validation_rows_path.read_text(encoding="utf-8"))
        test_rows = json.loads(test_rows_path.read_text(encoding="utf-8"))
        manifest_path = output_dir / "sample_selection_manifest.json"
        if manifest_path.exists():
            return json.loads(manifest_path.read_text(encoding="utf-8"))
    data = pd.read_parquet(TEST_DATA)
    seen, prior_sources = _source_indices_seen()
    eligible = sorted(set(range(len(data))) - seen)
    if len(eligible) < VALIDATION_N + TEST_N:
        raise ValueError(f"only {len(eligible)} untouched official test rows remain")
    val_rows = random.Random(VALIDATION_SEED).sample(eligible, VALIDATION_N)
    after_val = sorted(set(eligible) - set(val_rows))
    test_rows = random.Random(TEST_SEED).sample(after_val, TEST_N)
    write_json(validation_rows_path, val_rows)
    write_json(test_rows_path, test_rows)
    manifest = {
        "source_data": str(TEST_DATA.resolve()), "data_sha256": file_sha256(TEST_DATA), "source_rows": len(data),
        "prior_evaluations": prior_sources, "previously_used_unique_rows": len(seen),
        "eligible_untouched_rows_before_sampling": len(eligible),
        "validation_seed": VALIDATION_SEED, "validation_count": len(val_rows), "validation_rows": val_rows,
        "test_seed": TEST_SEED, "test_count": len(test_rows), "test_rows": test_rows,
        "validation_test_overlap": sorted(set(val_rows) & set(test_rows)),
        "note": "Rows from every prior run manifest using this exact Parquet SHA and explicit source_rows files are excluded. Test outcomes are not read while selecting validation thresholds.",
    }
    write_json(output_dir / "sample_selection_manifest.json", manifest)
    return manifest


def _case_from_row(row: dict[str, Any], source_index: int) -> dict[str, Any]:
    extra = row.get("extra_info") or {}
    raw = extra.get("prompt", row.get("prompt"))
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if not isinstance(raw, list):
        raise ValueError(f"official test row {source_index} has no prompt list")
    messages = [{"role": str(m.get("role", "")), "content": str(m.get("content", ""))}
                for m in raw if m.get("role") in ("system", "user", "assistant")]
    facts = extra.get("atomic_facts", [])
    if hasattr(facts, "tolist"):
        facts = facts.tolist()
    gold = row.get("ground_truth") or (row.get("reward_model") or {}).get("ground_truth") or extra.get("ground_truth")
    item_id = extra.get("index", row.get("index", source_index))
    case_id = f"relative-test-row-{source_index}:item-{item_id}"
    prompt_sha = hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    return {"case_id": case_id, "source_row_index": int(source_index), "initial_messages": messages,
            "atomic_facts": [str(x) for x in (facts or [])], "gold": gold, "prompt_sha256": prompt_sha}


class StableSeedChatAPI(ChatAPI):
    def __init__(self, base_url: str, model: str, role: str, timeout_s: float = 180.0, enable_thinking: bool | None = False):
        super().__init__(base_url, model, timeout_s=timeout_s, enable_thinking=enable_thinking)
        self.role = role

    async def generate(self, messages, *, temperature, max_tokens=768, retries=3, structured_outputs=None):
        request_messages = list(messages)
        request_body = {"model": self.model, "role": self.role, "context": ROLLOUT_CONTEXT.get(),
                        "messages": request_messages, "temperature": float(temperature), "max_tokens": int(max_tokens)}
        seed = int(hashlib.sha256(json.dumps(request_body, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:8], 16) & 0x7FFFFFFF
        payload = {"model": self.model, "messages": request_messages, "temperature": temperature,
                   "max_tokens": max_tokens, "seed": seed, "top_p": 1.0}
        if self.enable_thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}
        if structured_outputs is not None:
            payload["structured_outputs"] = dict(structured_outputs)
        last_error = None
        for attempt in range(retries):
            started = time.perf_counter()
            try:
                async with self.client.post(self.url, json=payload) as response:
                    response.raise_for_status()
                    body = await response.json()
                usage = body.get("usage") or {}
                content = body["choices"][0]["message"]["content"]
                latency_s = time.perf_counter() - started
                call = APICall(content, int(usage.get("prompt_tokens", 0)),
                               int(usage.get("completion_tokens", 0)), latency_s)
                ledger = API_CALL_LEDGER.get()
                if ledger is not None:
                    ledger.append({
                        "role": self.role, "seed": seed,
                        "request_sha256": hashlib.sha256(json.dumps(request_body, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
                        "prompt_tokens": call.prompt_tokens, "completion_tokens": call.completion_tokens,
                        "latency_s": call.latency_s, "response": content,
                    })
                return call
            except Exception as exc:
                last_error = exc
                await asyncio.sleep(2 ** attempt)
        raise RuntimeError(f"generation failed after {retries} stable-seed attempts: {last_error}")


def _attach_api_ledger(result: dict[str, Any], ledger: list[dict[str, Any]]) -> dict[str, Any]:
    """Ensure successful API calls rejected before trace append still count."""
    trace = list(result.get("trace") or [])
    traced_counts = Counter(
        (str(event.get("actor")), int(event.get("prompt_tokens", 0) or 0),
         int(event.get("completion_tokens", 0) or 0), round(float(event.get("latency_s", 0.0) or 0.0), 6))
        for event in trace if event.get("actor") in ("base", "patient", "expert")
    )
    for call in ledger:
        key = (str(call["role"]), int(call["prompt_tokens"]), int(call["completion_tokens"]), round(float(call["latency_s"]), 6))
        if traced_counts[key]:
            traced_counts[key] -= 1
            continue
        # Preserve successful model calls omitted when a nested continuation
        # raised before its local trace was returned to rollout_action.
        protocol_rejected = (call["role"] == "expert" and "CONSULT leaked or asserted a final answer" in str(result.get("error", "")))
        trace.append({
            "actor": call["role"],
            "action": "expert_response_rejected_by_protocol" if protocol_rejected else "continued_action_lost_on_exception",
            "protocol_rejected": protocol_rejected, "trace_recovery_reason": "collector_exception_dropped_nested_continuation_trace",
            "protocol_error": result.get("error"),
            "response": call["response"], "request_sha256": call["request_sha256"], "seed": call["seed"],
            "prompt_tokens": call["prompt_tokens"], "completion_tokens": call["completion_tokens"],
            "latency_s": call["latency_s"],
        })
    result["trace"] = trace
    result["api_call_ledger"] = ledger
    usage = {}
    for role in ("base", "patient", "expert"):
        calls = [event for event in ledger if event["role"] == role]
        usage[role] = {
            "prompt_tokens": sum(int(event["prompt_tokens"]) for event in calls),
            "completion_tokens": sum(int(event["completion_tokens"]) for event in calls),
            "latency_s": sum(float(event["latency_s"]) for event in calls),
        }
    for role in usage:
        for measure, value in usage[role].items():
            result[f"{role}_{measure}"] = value
    result["patient_turns"] = sum(event["role"] == "patient" for event in ledger)
    result["total_prompt_tokens"] = sum(int(event["prompt_tokens"]) for event in ledger)
    result["total_completion_tokens"] = sum(int(event["completion_tokens"]) for event in ledger)
    result["total_latency_s"] = sum(float(event["latency_s"]) for event in ledger)
    return result


async def repair_test_protocol_failures(output_dir: Path) -> None:
    """Re-run only protocol-invalid test branches to restore omitted API usage.

    The action, answer-validity, and error must reproduce exactly. These
    repairs affect only missing token/latency records and never policy tuning.
    """
    raw_path = output_dir / "relative_rollouts.jsonl"
    repaired_path = output_dir / "protocol_repaired_rollouts.jsonl"
    manifest_path = output_dir / "protocol_repair_manifest.json"
    raw_rows = _read_jsonl(raw_path)
    raw_sha = file_sha256(raw_path)
    if repaired_path.exists() and manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("original_rollout_sha256") == raw_sha and old.get("status") == "completed":
            return
        raise ValueError("existing protocol repair does not match the current raw rollout file")
    invalid_rows = [row for row in raw_rows if not bool(row.get("valid", False))]
    if not invalid_rows:
        repaired_path.write_text(raw_path.read_text(encoding="utf-8"), encoding="utf-8")
        write_json(manifest_path, {"status": "completed", "original_rollout_sha256": raw_sha, "repaired_rollout_sha256": file_sha256(repaired_path), "repaired_branches": 0})
        return

    selected = [
        (int(row["source_row_index"]), str(row["case_id"]), str(row["action"]), int(row["replicate_index"]))
        for row in invalid_rows
    ]
    data = pd.read_parquet(TEST_DATA)
    cases = {idx: _case_from_row(data.iloc[idx].to_dict(), idx) for idx, *_ in selected}
    base = StableSeedChatAPI(BASE_URL, MODEL_ALIAS, "base")
    patient = StableSeedChatAPI(BASE_URL, MODEL_ALIAS, "patient")
    expert = StableSeedChatAPI(BASE_URL, MODEL_ALIAS, "expert")
    repaired_by_key: dict[tuple[int, str, int], dict[str, Any]] = {}
    try:
        for source_index, original_case_id, action_name, replicate in selected:
            case = cases[source_index]
            if case["case_id"] != original_case_id:
                raise ValueError(f"source row identity changed for {source_index}")
            token = ROLLOUT_CONTEXT.set(f"test:{case['case_id']}:{action_name}:replicate-{replicate}")
            ledger: list[dict[str, Any]] = []
            ledger_token = API_CALL_LEDGER.set(ledger)
            try:
                state = PublicState(case["case_id"], tuple(case["initial_messages"]), ())
                result = await rollout_action(Action(action_name), state, case["atomic_facts"], case["gold"],
                                              base, patient, expert, temperature=float(GENERATION["temperature"]),
                                              max_turns=int(GENERATION["max_turns"]), max_consults=int(GENERATION["max_consults"]))
                original = next(row for row in invalid_rows if int(row["source_row_index"]) == source_index and row["action"] == action_name and int(row["replicate_index"]) == replicate)
                if (result.get("answer"), bool(result.get("valid")), result.get("error")) != (original.get("answer"), bool(original.get("valid")), original.get("error")):
                    raise RuntimeError(f"replay outcome differs for {source_index}/{action_name}/{replicate}; refusing to replace test outcome")
                result = _attach_api_ledger(result, ledger)
                key = (source_index, action_name, replicate)
                repaired_by_key[key] = {
                    "record_type": "relative_rollout", "split": "test", "case_id": case["case_id"],
                    "source_row_index": source_index, "prompt_sha256": case["prompt_sha256"],
                    "replicate_index": replicate, "action": action_name,
                    "initial_public_state": case["initial_messages"], "gold": _gold_letter(case["gold"]),
                    **result,
                }
            finally:
                API_CALL_LEDGER.reset(ledger_token)
                ROLLOUT_CONTEXT.reset(token)
    finally:
        await asyncio.gather(base.close(), patient.close(), expert.close(), return_exceptions=True)

    merged = []
    for row in raw_rows:
        key = (int(row["source_row_index"]), str(row["action"]), int(row["replicate_index"]))
        merged.append(repaired_by_key.get(key, row))
    _write_jsonl(repaired_path, merged)
    _write_csv(output_dir / "protocol_failure_audit.csv", [
        {"case_id": row["case_id"], "source_row_index": row["source_row_index"],
         "initial_action": row["action"], "replicate_index": row["replicate_index"],
         "valid": row.get("valid"), "failure_reason": row.get("error"),
         "total_tokens": int(row.get("total_prompt_tokens", 0)) + int(row.get("total_completion_tokens", 0)),
         "expert_tokens": int(row.get("expert_prompt_tokens", 0)) + int(row.get("expert_completion_tokens", 0)),
         "model_latency_s": row.get("total_latency_s"),
         "recovered_calls": sum(bool(event.get("trace_recovery_reason")) for event in row.get("trace", []))}
        for row in merged if not bool(row.get("valid", False))
    ])
    write_json(manifest_path, {
        "status": "completed", "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "original_rollout_sha256": raw_sha, "repaired_rollout_sha256": file_sha256(repaired_path),
        "repaired_branches": len(repaired_by_key), "repaired_keys": [list(key) for key in sorted(repaired_by_key)],
        "answer_and_validity_unchanged": True,
        "purpose": "restore API token/latency and calls omitted when a protocol exception dropped the nested continuation trace; no route, threshold, or outcome change",
    })


async def collect_split(split: str, output_dir: Path, *, resume: bool = True) -> None:
    selection = prepare_samples(OUTPUT_ROOT)
    row_indices = selection["validation_rows"] if split == "validation" else selection["test_rows"]
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "relative_rollouts.jsonl"
    manifest_path = output_dir / "run_manifest.json"
    data = pd.read_parquet(TEST_DATA)
    cases = [_case_from_row(data.iloc[i].to_dict(), i) for i in row_indices]
    branch_manifest = {
        "split": split, "source_data": str(TEST_DATA.resolve()), "data_sha256": file_sha256(TEST_DATA),
        "source_rows": row_indices, "count_cases": len(cases), "replicates_per_action": 3,
        "actions": list(ACTIONS), "model_alias": MODEL_ALIAS, "model_revision": MODEL_REVISION,
        "model_path": str(MODEL_PATH.resolve()), "model_config_sha256": file_sha256(MODEL_PATH / "config.json"),
        "model_service": str((HERE / "outputs/relative_value_20261106/service/model_service_manifest.json").resolve()),
        "base_url": BASE_URL, "generation": GENERATION,
        "seed_strategy": "SHA256(model, role, rollout-context, messages, temperature, max_tokens); request retry repeats seed",
        "continuation_policy": {"initial_action_forced": True, "post_action": "same Base Agent continuation policy as official Pilot collector", "max_assistant_turns": 10, "max_consults": 1, "terminal_answer_last_turn": True},
        "input_boundary": "Base and Router see only public dialogue; Patient alone receives hidden atomic facts; Expert sees public dialogue and explicit consult question; gold is scored after rollout_action completes model calls.",
        "cost_measurements": ["patient_turns", "expert_tokens", "total_tokens", "latency_s"],
    }
    if manifest_path.exists() and resume:
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("split", "data_sha256", "source_rows", "replicates_per_action", "model_revision", "generation", "continuation_policy"):
            if old.get(key) != branch_manifest.get(key):
                raise ValueError(f"resume refused: manifest mismatch in {key}")
    branch_manifest["status"] = "running"
    branch_manifest["started_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    write_json(manifest_path, branch_manifest)

    done = set()
    if rows_path.exists() and resume:
        for row in _read_jsonl(rows_path):
            done.add((int(row["source_row_index"]), str(row["action"]), int(row["replicate_index"])))
    else:
        rows_path.write_text("", encoding="utf-8")

    endpoint = BASE_URL
    base = StableSeedChatAPI(endpoint, MODEL_ALIAS, "base")
    patient = StableSeedChatAPI(endpoint, MODEL_ALIAS, "patient")
    expert = StableSeedChatAPI(endpoint, MODEL_ALIAS, "expert")
    semaphore = asyncio.Semaphore(int(GENERATION["max_concurrency"]))

    async def one(case: dict[str, Any], action_name: str, replicate: int) -> dict[str, Any]:
        async with semaphore:
            token = ROLLOUT_CONTEXT.set(f"{split}:{case['case_id']}:{action_name}:replicate-{replicate}")
            try:
                state = PublicState(case["case_id"], tuple(case["initial_messages"]), ())
                result = await rollout_action(Action(action_name), state, case["atomic_facts"], case["gold"],
                                              base, patient, expert, temperature=float(GENERATION["temperature"]),
                                              max_turns=int(GENERATION["max_turns"]), max_consults=int(GENERATION["max_consults"]))
                return {
                    "record_type": "relative_rollout", "split": split, "case_id": case["case_id"],
                    "source_row_index": case["source_row_index"], "prompt_sha256": case["prompt_sha256"],
                    "replicate_index": replicate, "action": action_name,
                    "initial_public_state": case["initial_messages"], "gold": _gold_letter(case["gold"]),
                    **result,
                }
            finally:
                ROLLOUT_CONTEXT.reset(token)

    tasks = []
    for case in cases:
        for action_name in ACTIONS:
            for replicate in range(3):
                key = (case["source_row_index"], action_name, replicate)
                if key not in done:
                    tasks.append(asyncio.create_task(one(case, action_name, replicate)))
    count = len(done)
    started = time.monotonic()
    try:
        with rows_path.open("a", encoding="utf-8") as stream:
            for task in asyncio.as_completed(tasks):
                row = await task
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                count += 1
                if count % 50 == 0:
                    print(f"[{split}] completed {count}/{len(cases) * 4 * 3} branches; elapsed={time.monotonic()-started:.1f}s", flush=True)
    except Exception:
        branch_manifest["status"] = "interrupted"
        branch_manifest["completed_branches"] = count
        branch_manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        write_json(manifest_path, branch_manifest)
        for task in tasks:
            if not task.done():
                task.cancel()
        raise
    finally:
        await asyncio.gather(base.close(), patient.close(), expert.close(), return_exceptions=True)
    branch_manifest["status"] = "completed"
    branch_manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    branch_manifest["branch_count"] = count
    branch_manifest["rollout_sha256"] = file_sha256(rows_path)
    branch_manifest["elapsed_s"] = time.monotonic() - started
    write_json(manifest_path, branch_manifest)


def _state_docs_from_rows(data: pd.DataFrame, indices: list[int]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    states, cases = [], {}
    for index in indices:
        case = _case_from_row(data.iloc[index].to_dict(), index)
        state_id = case["case_id"] + ":initial"
        state = {"state_id": state_id, "case_id": case["case_id"], "public_state": case["initial_messages"]}
        states.append(state)
        cases[case["case_id"]] = {**case, "state_id": state_id}
    return states, cases


def _cost_utility(metrics: dict[str, float], cost: dict[str, float]) -> float:
    # Clinical outcome and resource costs are separate; patient and expert costs
    # are each subtracted once. Total token count is an observed reporting axis.
    return float(metrics["accuracy"] - cost["patient_per_turn"] * metrics["patient_turns"]
                 - cost["expert_per_1k_tokens"] * metrics["expert_tokens_k"])


def _action_metrics(branches: dict[str, dict[int, dict[str, Any]]], case_id: str) -> dict[str, dict[str, float]]:
    result = {}
    for action in ACTIONS:
        reps = branches[case_id][action]
        values = [rollout_metrics(row) for _, row in sorted(reps.items())]
        mean = {metric: float(np.mean([v[metric] for v in values])) for metric in METRICS}
        mean["expert_calls"] = float(np.mean([sum(e.get("actor") == "expert" for e in (row.get("trace") or [])) for row in reps.values()]))
        mean["expert_call_rate"] = float(np.mean([sum(e.get("actor") == "expert" for e in (row.get("trace") or [])) > 0 for row in reps.values()]))
        mean["valid_rate"] = float(np.mean([bool(row.get("valid")) for row in reps.values()]))
        result[action] = mean
    return result


def _absolute_scores(pred: dict[str, Any], cost: dict[str, float]) -> dict[str, float]:
    scores = {}
    for action in ACTIONS:
        q = float(pred[action]["q"])
        c = pred[action]["costs"]
        patient = float(c.get("patient_turns", 0.0))
        expert_k = float(c.get("consult_tokens", 0.0)) + float(c.get("defer_tokens", 0.0))
        scores[action] = q - cost["patient_per_turn"] * patient - cost["expert_per_1k_tokens"] * expert_k
    return scores


def _select_absolute(pred: dict[str, Any], cost: dict[str, float]) -> str:
    scores = _absolute_scores(pred, cost)
    return max(ACTIONS, key=lambda a: (scores[a], -ACTIONS.index(a)))


def _relative_scores(pred: dict[str, Any], cost: dict[str, float]) -> dict[str, float]:
    scores = {"ANSWER": 0.0}
    for action in COLLAB_ACTIONS:
        item = pred[action]
        scores[action] = (float(item["accuracy"])
                          - cost["patient_per_turn"] * float(item["patient_turns"])
                          - cost["expert_per_1k_tokens"] * float(item["expert_tokens_k"]))
    return scores


def _case_utility(case_id: str, action: str, action_metrics: dict[str, dict[str, float]], cost: dict[str, float]) -> float:
    return _cost_utility(action_metrics[case_id][action], cost)


def _threshold_candidates(values: list[float], *, include_zero: bool = True) -> list[float]:
    candidates: set[float] = set()
    for value in values:
        x = float(value)
        if math.isfinite(x):
            candidates.add(x)
            candidates.add(float(np.nextafter(x, -np.inf)))
            candidates.add(float(np.nextafter(x, np.inf)))
    candidates.update((0.0, 1.0, max(values, default=0.0) + 1e-8))
    if include_zero:
        candidates.add(0.0)
    return sorted(candidates)


def _bootstrap_mean_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    sample_idx = rng.integers(0, n, size=(BOOTSTRAP_REPS, n))
    means = values[sample_idx].mean(axis=1)
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))


def _bootstrap_difference_ci(a: np.ndarray, b: np.ndarray, seed: int) -> tuple[float, float]:
    return _bootstrap_mean_ci(a - b, seed)


def _summarize_policy(policy_name: str, chosen: dict[str, str], metrics_by_case: dict[str, dict[str, dict[str, float]]], case_ids: list[str], cost: dict[str, float], seed: int) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    per_case = {}
    for case_id in case_ids:
        per_case[case_id] = metrics_by_case[case_id][chosen[case_id]]
    acc = np.asarray([per_case[c]["accuracy"] for c in case_ids])
    lo, hi = _bootstrap_mean_ci(acc, seed)
    result: dict[str, Any] = {
        "policy": policy_name, "cost_set": cost["name"], "cases": len(case_ids), "rollouts": len(case_ids) * 3,
        "accuracy": float(np.mean(acc)), "accuracy_ci95_low": lo, "accuracy_ci95_high": hi,
        "mean_patient_turns": float(np.mean([per_case[c]["patient_turns"] for c in case_ids])),
        "mean_expert_tokens": float(1000 * np.mean([per_case[c]["expert_tokens_k"] for c in case_ids])),
        "mean_total_tokens": float(1000 * np.mean([per_case[c]["total_tokens_k"] for c in case_ids])),
        "mean_latency_s": float(np.mean([per_case[c]["latency_s"] for c in case_ids])),
        "mean_expert_calls": float(np.mean([per_case[c]["expert_calls"] for c in case_ids])),
        "expert_call_rate": float(np.mean([per_case[c]["expert_call_rate"] for c in case_ids])),
        "valid_rate": float(np.mean([per_case[c]["valid_rate"] for c in case_ids])),
        "net_utility": float(np.mean([_cost_utility(per_case[c], cost) for c in case_ids])),
        "lambda_patient_per_turn": cost["patient_per_turn"],
        "lambda_expert_per_1k_tokens": cost["expert_per_1k_tokens"],
        **{f"first_{a.lower()}_rate": float(np.mean([chosen[c] == a for c in case_ids])) for a in ACTIONS},
    }
    return result, per_case


def evaluate_split(split: str, output_dir: Path, *, freeze_validation: bool = False) -> None:
    selection = json.loads((OUTPUT_ROOT / "sample_selection_manifest.json").read_text(encoding="utf-8"))
    source_rows = selection["validation_rows"] if split == "validation" else selection["test_rows"]
    rollout_path = output_dir / "relative_rollouts.jsonl"
    repaired_rollout_path = output_dir / "protocol_repaired_rollouts.jsonl"
    if split == "test" and repaired_rollout_path.exists():
        rollout_path = repaired_rollout_path
    if not rollout_path.exists():
        raise FileNotFoundError(rollout_path)
    branches: dict[str, dict[str, dict[int, dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    for row in _read_jsonl(rollout_path):
        branches[str(row["case_id"])][str(row["action"])][int(row["replicate_index"])] = row
    data = pd.read_parquet(TEST_DATA)
    states, cases = _state_docs_from_rows(data, source_rows)
    case_ids = [s["case_id"] for s in states]
    if set(case_ids) != set(branches):
        raise ValueError(f"collected case set differs: expected={len(case_ids)}, got={len(branches)}")
    for case_id in case_ids:
        if any(set(branches[case_id].get(a, {})) != {0, 1, 2} for a in ACTIONS):
            raise ValueError(f"incomplete four-action triplet for {case_id}")

    relative_model = load_relative_model(OUTPUT_ROOT / "checkpoint_relative_initial_only")
    absolute_model = load_absolute_model(PILOT_ABSOLUTE_CHECKPOINT)
    relative_predictions = predict_relative(relative_model, states)
    absolute_predictions = predict_absolute(absolute_model, states)
    metrics_by_case = {case_id: _action_metrics(branches, case_id) for case_id in case_ids}
    relative_by_case = {s["case_id"]: relative_predictions[s["state_id"]] for s in states}
    absolute_by_case = {s["case_id"]: absolute_predictions[s["state_id"]] for s in states}

    if split == "validation":
        frozen_path = OUTPUT_ROOT / "validation_policy_config.json"
        if frozen_path.exists() and freeze_validation:
            raise FileExistsError(f"refusing to overwrite frozen validation config: {frozen_path}")
        fitted: dict[str, Any] = {"cost_sets": {}, "selected_on": "official test-parquet validation rows only", "validation_cases": case_ids}
        for cost in COST_GRID:
            rel_scores = {c: _relative_scores(relative_by_case[c], cost) for c in case_ids}
            margins = [max(v[a] for a in COLLAB_ACTIONS) for v in rel_scores.values()]
            best_tau, best_score = 0.0, -float("inf")
            positive_margins = [x for x in margins if x > 0.0]
            tau_candidates = {0.0, max(positive_margins, default=0.0) + 1e-8}
            tau_candidates.update(float(np.nextafter(x, -np.inf)) for x in positive_margins)
            tau_candidates.update(positive_margins)
            tau_candidates = sorted(tau_candidates)
            for tau in tau_candidates:
                util = []
                for c in case_ids:
                    best_action = max(COLLAB_ACTIONS, key=lambda a: (rel_scores[c][a], -ACTIONS.index(a)))
                    selected = best_action if rel_scores[c][best_action] > tau else "ANSWER"
                    util.append(_case_utility(c, selected, metrics_by_case, cost))
                score = float(np.mean(util))
                if score > best_score + 1e-12 or (abs(score - best_score) <= 1e-12 and tau > best_tau):
                    best_tau, best_score = float(tau), score

            # Alternative is fixed from the Pilot training partition alone.
            pilot_records = load_jsonl(PILOT_ROWS)
            pilot_states, pilot_targets, _ = paired_relative_targets(pilot_records)
            split_manifest = json.loads(PILOT_SPLIT.read_text(encoding="utf-8"))
            train_case_ids = set(split_manifest["train"])
            train_targets = [t for t in pilot_targets if t["case_id"] in train_case_ids]
            train_means = {
                a: float(np.mean([
                    float(t[f"{a}_accuracy_delta"])
                    - cost["patient_per_turn"] * float(t[f"{a}_patient_turns_delta"])
                    - cost["expert_per_1k_tokens"] * float(t[f"{a}_expert_tokens_k_delta"])
                    for t in train_targets
                ])) for a in COLLAB_ACTIONS
            }
            alternative = max(COLLAB_ACTIONS, key=lambda a: (train_means[a], -ACTIONS.index(a)))
            answer_conf = [float(absolute_by_case[c]["ANSWER"]["q"]) for c in case_ids]
            best_conf, best_uncertainty_score = 1.0, -float("inf")
            for conf_threshold in _threshold_candidates(answer_conf):
                util = []
                for c in case_ids:
                    selected = alternative if float(absolute_by_case[c]["ANSWER"]["q"]) < conf_threshold else "ANSWER"
                    util.append(_case_utility(c, selected, metrics_by_case, cost))
                score = float(np.mean(util))
                if score > best_uncertainty_score + 1e-12 or (abs(score - best_uncertainty_score) <= 1e-12 and conf_threshold < best_conf):
                    best_conf, best_uncertainty_score = float(conf_threshold), score
            fitted["cost_sets"][cost["name"]] = {
                "cost": cost, "conservative_advantage_threshold": best_tau,
                "validation_utility_at_threshold": best_score,
                "uncertainty_answer_confidence_threshold": best_conf,
                "uncertainty_alternative_action": alternative,
                "pilot_train_mean_net_advantage": train_means,
                "uncertainty_validation_utility": best_uncertainty_score,
            }
        write_json(frozen_path, fitted)
    else:
        frozen_path = OUTPUT_ROOT / "validation_policy_config.json"
        if not frozen_path.exists():
            raise FileNotFoundError("test evaluation requires a frozen validation_policy_config.json")
        fitted = json.loads(frozen_path.read_text(encoding="utf-8"))

    # Post-hoc value calibration diagnostics only; these rows never feed policy
    # selection or threshold fitting.
    value_diagnostic_rows = []
    for action in COLLAB_ACTIONS:
        predicted_delta = np.asarray([relative_by_case[c][action]["accuracy"] for c in case_ids], dtype=float)
        observed_delta = np.asarray([
            metrics_by_case[c][action]["accuracy"] - metrics_by_case[c]["ANSWER"]["accuracy"]
            for c in case_ids
        ], dtype=float)
        corr = float(np.corrcoef(predicted_delta, observed_delta)[0, 1]) if np.std(predicted_delta) > 0 and np.std(observed_delta) > 0 else math.nan
        value_diagnostic_rows.append({
            "split": split, "diagnostic": "paired_accuracy_advantage", "action": action,
            "cases": len(case_ids), "mean_predicted_delta": float(np.mean(predicted_delta)),
            "mean_observed_delta": float(np.mean(observed_delta)),
            "mae": float(np.mean(np.abs(predicted_delta - observed_delta))), "pearson_r": corr,
            "bias_pred_minus_observed": float(np.mean(predicted_delta - observed_delta)),
            "agreement_rate": math.nan, "empirical_action_reference": "",
        })
    reference_cost = next(cost for cost in COST_GRID if cost["name"] == "reference")
    relative_agreement, absolute_agreement = [], []
    for case_id in case_ids:
        rel_scores = _relative_scores(relative_by_case[case_id], reference_cost)
        rel_best = max(ACTIONS, key=lambda a: (rel_scores[a], -ACTIONS.index(a)))
        abs_best = _select_absolute(absolute_by_case[case_id], reference_cost)
        empirical_scores = {a: _cost_utility(metrics_by_case[case_id][a], reference_cost) for a in ACTIONS}
        empirical_best = max(ACTIONS, key=lambda a: (empirical_scores[a], -ACTIONS.index(a)))
        relative_agreement.append(rel_best == empirical_best)
        absolute_agreement.append(abs_best == empirical_best)
    for label, values in (("relative_advantage_router", relative_agreement), ("absolute_q_router", absolute_agreement)):
        value_diagnostic_rows.append({
            "split": split, "diagnostic": "predicted_vs_empirical_best_action", "action": label,
            "cases": len(case_ids), "mean_predicted_delta": math.nan, "mean_observed_delta": math.nan,
            "mae": math.nan, "pearson_r": math.nan, "bias_pred_minus_observed": math.nan,
            "agreement_rate": float(np.mean(values)),
            "empirical_action_reference": "best mean utility over 3 same-state action rollouts",
        })

    policy_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    selected_actions_by_cost: dict[str, dict[str, dict[str, str]]] = {}
    for cost_idx, cost in enumerate(COST_GRID):
        config = fitted["cost_sets"][cost["name"]]
        tau = float(config["conservative_advantage_threshold"])
        conf_threshold = float(config["uncertainty_answer_confidence_threshold"])
        alternative = str(config["uncertainty_alternative_action"])
        chosen: dict[str, dict[str, str]] = {p: {} for p in (
            "always_answer", "uncertainty_threshold", "absolute_q_router", "relative_advantage_router", "conservative_relative_router"
        )}
        for c in case_ids:
            scores = _relative_scores(relative_by_case[c], cost)
            best_action = max(COLLAB_ACTIONS, key=lambda a: (scores[a], -ACTIONS.index(a)))
            chosen["always_answer"][c] = "ANSWER"
            chosen["uncertainty_threshold"][c] = alternative if float(absolute_by_case[c]["ANSWER"]["q"]) < conf_threshold else "ANSWER"
            chosen["absolute_q_router"][c] = _select_absolute(absolute_by_case[c], cost)
            chosen["relative_advantage_router"][c] = max(ACTIONS, key=lambda a: (scores[a], -ACTIONS.index(a)))
            chosen["conservative_relative_router"][c] = best_action if scores[best_action] > tau else "ANSWER"
        for policy, action_map in chosen.items():
            for c, selected_action in action_map.items():
                rel_scores = _relative_scores(relative_by_case[c], cost)
                abs_scores = _absolute_scores(absolute_by_case[c], cost)
                decision_rows.append({
                    "split": split, "cost_set": cost["name"], "case_id": c,
                    "source_row_index": cases[c]["source_row_index"], "policy": policy,
                    "selected_action": selected_action,
                    "conservative_threshold": tau, "uncertainty_threshold": conf_threshold,
                    "uncertainty_alternative_action": alternative,
                    **{f"relative_net_{a.lower()}": rel_scores[a] for a in ACTIONS},
                    **{f"absolute_net_{a.lower()}": abs_scores[a] for a in ACTIONS},
                    **{f"relative_delta_accuracy_{a.lower()}": relative_by_case[c][a]["accuracy"] for a in ACTIONS},
                    **{f"relative_delta_patient_turns_{a.lower()}": relative_by_case[c][a]["patient_turns"] for a in ACTIONS},
                    **{f"relative_delta_expert_tokens_k_{a.lower()}": relative_by_case[c][a]["expert_tokens_k"] for a in ACTIONS},
                })
        selected_actions_by_cost[cost["name"]] = chosen
        summaries: dict[str, dict[str, float]] = {}
        for policy_idx, (policy, actions) in enumerate(chosen.items()):
            summary, per_case = _summarize_policy(policy, actions, metrics_by_case, case_ids, cost, 20261120 + policy_idx + cost_idx)
            policy_rows.append({"split": split, **summary,
                                "conservative_threshold": tau,
                                "uncertainty_threshold": conf_threshold,
                                "uncertainty_alternative_action": alternative})
            summaries[policy] = per_case
            curve_rows.append({"split": split, "policy": policy, "cost_set": cost["name"],
                               "lambda_patient_per_turn": cost["patient_per_turn"],
                               "lambda_expert_per_1k_tokens": cost["expert_per_1k_tokens"],
                               "accuracy": summary["accuracy"], "mean_total_tokens": summary["mean_total_tokens"],
                               "mean_expert_tokens": summary["mean_expert_tokens"], "mean_latency_s": summary["mean_latency_s"],
                               "mean_patient_turns": summary["mean_patient_turns"], "expert_call_rate": summary["expert_call_rate"],
                               "net_utility": summary["net_utility"], "threshold": tau})
        # Cluster bootstrap paired differences versus ANSWER, with three repeated
        # rollouts averaged within each case before resampling.
        answer_values = np.asarray([summaries["always_answer"][c]["accuracy"] for c in case_ids])
        for policy_idx, policy in enumerate(chosen):
            if policy == "always_answer":
                continue
            values = np.asarray([summaries[policy][c]["accuracy"] for c in case_ids])
            delta = values - answer_values
            low, high = _bootstrap_difference_ci(values, answer_values, 20261200 + cost_idx * 20 + policy_idx)
            paired_rows.append({"split": split, "cost_set": cost["name"], "policy": policy,
                                "comparator": "always_answer", "paired_accuracy_delta": float(np.mean(delta)),
                                "case_bootstrap_ci95_low": low, "case_bootstrap_ci95_high": high,
                                "cases": len(case_ids), "bootstrap_replicates": BOOTSTRAP_REPS})

    _write_csv(output_dir / f"{split}_policy_summary.csv", policy_rows)
    _write_csv(output_dir / f"{split}_paired_comparisons.csv", paired_rows)
    _write_csv(output_dir / f"{split}_accuracy_cost_curve.csv", curve_rows)
    _write_csv(output_dir / f"{split}_policy_decisions.csv", decision_rows)
    _write_csv(output_dir / f"{split}_value_diagnostics.csv", value_diagnostic_rows)
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        policies_to_plot = list(dict.fromkeys(row["policy"] for row in curve_rows))
        for policy in policies_to_plot:
            selected = [row for row in curve_rows if row["policy"] == policy]
            ax.plot([row["mean_total_tokens"] for row in selected], [row["accuracy"] for row in selected],
                    marker="o", linewidth=1.7, label=policy.replace("_", " "))
        ax.set_xlabel("Mean total model tokens per case")
        ax.set_ylabel("Empirical accuracy")
        ax.set_title(f"{split.title()} accuracy–token cost curve")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, frameon=False)
        fig.tight_layout()
        fig.savefig(output_dir / f"{split}_accuracy_cost_curve.png", dpi=180)
        plt.close(fig)
    except ImportError:
        pass
    action_rows = []
    for cost_name, policies in selected_actions_by_cost.items():
        for policy, action_map in policies.items():
            action_rows.append({"split": split, "cost_set": cost_name, "policy": policy,
                                **{f"{a.lower()}_count": sum(v == a for v in action_map.values()) for a in ACTIONS},
                                **{f"{a.lower()}_rate": float(np.mean([v == a for v in action_map.values()])) for a in ACTIONS}})
    _write_csv(output_dir / f"{split}_action_rates.csv", action_rows)
    if split == "validation":
        write_json(output_dir / "frozen_policy_predictions.json", {
            "split": split, "case_ids": case_ids,
            "relative_predictions": relative_by_case, "absolute_predictions": absolute_by_case,
            "cost_sets": fitted["cost_sets"],
        })
    summary_manifest = {
        "status": "completed", "split": split, "evaluated_cases": len(case_ids), "rollouts_per_policy": len(case_ids) * 3,
        "experiment_config_path": str((HERE / "relative_value_config.yaml").resolve()),
        "experiment_config_sha256": file_sha256(HERE / "relative_value_config.yaml"),
        "model_service_manifest_sha256": file_sha256(HERE / "outputs/relative_value_20261106/service/model_service_manifest.json"),
        "source_rows_sha256": file_sha256(OUTPUT_ROOT / ("validation_source_rows.json" if split == "validation" else "test_source_rows.json")),
        "rollout_sha256": file_sha256(rollout_path), "relative_checkpoint_sha256": file_sha256(OUTPUT_ROOT / "checkpoint_relative_initial_only/relative_value_checkpoint.npz"),
        "absolute_checkpoint_sha256": file_sha256(PILOT_ABSOLUTE_CHECKPOINT / "router_checkpoint.npz"),
        "policy_config_sha256": file_sha256(OUTPUT_ROOT / "validation_policy_config.json"),
        "bootstrap": {"unit": "case_id", "replicates": BOOTSTRAP_REPS, "rollouts_averaged_within_case": 3},
        "continuation_policy": "initial forced action followed by Base Agent continuation, 10 total assistant turns, one CONSULT maximum, final turn forced ANSWER",
        "cost_rule": "net utility = accuracy - lambda_patient * patient_turns - lambda_expert * expert_tokens/1000; total tokens is reported separately",
        "test_thresholds_read_from_frozen_validation_config": split == "test",
    }
    if split == "test":
        summary_manifest["raw_rollout_sha256"] = file_sha256(output_dir / "relative_rollouts.jsonl")
        summary_manifest["protocol_repair_manifest_sha256"] = file_sha256(output_dir / "protocol_repair_manifest.json") if (output_dir / "protocol_repair_manifest.json").exists() else None
    write_json(output_dir / f"{split}_evaluation_manifest.json", summary_manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("train", "prepare-samples", "collect-validation", "evaluate-validation", "collect-test", "repair-test-protocol", "evaluate-test"), required=True)
    parser.add_argument("--alphas", default="0.03,0.1,0.3,1,3,10,30")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    out = OUTPUT_ROOT
    if args.phase == "train":
        result = train_router(out, [float(x) for x in args.alphas.split(",")])
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.phase == "prepare-samples":
        print(json.dumps(prepare_samples(out), ensure_ascii=False, indent=2))
    elif args.phase == "collect-validation":
        asyncio.run(collect_split("validation", out / "validation", resume=not args.no_resume))
    elif args.phase == "evaluate-validation":
        evaluate_split("validation", out / "validation", freeze_validation=True)
    elif args.phase == "collect-test":
        if not (out / "validation_policy_config.json").exists():
            raise SystemExit("Refusing test collection before validation thresholds are frozen.")
        asyncio.run(collect_split("test", out / "test", resume=not args.no_resume))
    elif args.phase == "repair-test-protocol":
        if not (out / "validation_policy_config.json").exists():
            raise SystemExit("Refusing test branch repair before validation thresholds are frozen.")
        asyncio.run(repair_test_protocol_failures(out / "test"))
    elif args.phase == "evaluate-test":
        evaluate_split("test", out / "test")


if __name__ == "__main__":
    main()
