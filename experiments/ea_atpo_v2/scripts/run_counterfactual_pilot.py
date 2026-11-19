#!/usr/bin/env python3
"""Collect leakage-checked, paired ASK/ANSWER/DEFER counterfactual rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve()
V2_DIR = HERE.parents[1]
PROJECT_DIR = V2_DIR.parents[1]
sys.path.insert(0, str(V2_DIR / "src"))
from v2_protocol import (  # noqa: E402
    ACTION_ORDER,
    Action,
    PublicState,
    action_net_return,
    assert_no_privileged_payload,
    build_base_messages,
    build_defer_messages,
    build_patient_messages,
    extend_after_ask,
    parse_answer,
    parse_question,
    public_state_from_row,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def direct_loopback_opener() -> urllib.request.OpenerDirector:
    """Model endpoints are loopback services; never route them through HTTP_PROXY."""
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def endpoint_status(role: str, settings: Mapping[str, Any]) -> dict[str, Any]:
    base_url = str(settings["base_url"]).rstrip("/")
    url = base_url + "/models"
    key_env = str(settings.get("api_key_env", ""))
    key = os.environ.get(key_env, "") if key_env else ""
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    request = urllib.request.Request(url, headers=headers)
    try:
        with direct_loopback_opener().open(request, timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
        model_ids = [str(item.get("id", "")) for item in payload.get("data", [])]
        expected = str(settings["model"])
        return {
            "role": role,
            "url": url,
            "http_status": 200,
            "reachable": True,
            "expected_model": expected,
            "model_ids": model_ids,
            "model_match": expected in model_ids,
            "api_key_env": key_env or None,
            "api_key_present": bool(key),
        }
    except urllib.error.HTTPError as exc:
        return {
            "role": role,
            "url": url,
            "http_status": int(exc.code),
            "reachable": False,
            "expected_model": str(settings["model"]),
            "model_ids": [],
            "model_match": False,
            "api_key_env": key_env or None,
            "api_key_present": bool(key),
            "error": "endpoint returned HTTP error; response body intentionally omitted",
        }
    except urllib.error.URLError as exc:
        reason = exc.reason
        if isinstance(reason, OSError):
            error = f"{type(reason).__name__} errno={reason.errno}: {reason.strerror}"
        else:
            error = type(reason).__name__
        return {
            "role": role,
            "url": url,
            "http_status": None,
            "reachable": False,
            "expected_model": str(settings["model"]),
            "model_ids": [],
            "model_match": False,
            "api_key_env": key_env or None,
            "api_key_present": bool(key),
            "error": error,
        }
    except Exception as exc:
        return {
            "role": role,
            "url": url,
            "http_status": None,
            "reachable": False,
            "expected_model": str(settings["model"]),
            "model_ids": [],
            "model_match": False,
            "api_key_env": key_env or None,
            "api_key_present": bool(key),
            "error": type(exc).__name__,
        }


def gpu_inventory() -> list[dict[str, int | str]]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 6:
            rows.append({
                "index": fields[0], "name": fields[1], "memory_total_mib": int(fields[2]),
                "memory_used_mib": int(fields[3]), "memory_free_mib": int(fields[4]),
                "utilization_pct": int(fields[5]),
            })
    return rows


def stable_case_rank(case_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{case_id}".encode("utf-8")).hexdigest()


def read_source_rows(path: Path, indices: set[int]) -> dict[int, dict[str, Any]]:
    output: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            if line_index in indices:
                output[line_index] = json.loads(line)
                if len(output) == len(indices):
                    break
    missing = sorted(indices - output.keys())
    if missing:
        raise ValueError(f"source row indices missing: {len(missing)}")
    return output


def case_id_for_source(index: int, row: Mapping[str, Any]) -> str:
    extra = row.get("extra_info") or {}
    item = extra.get("index", row.get("index", index))
    return f"row-{index}:item-{item}"


def audit_and_select(config: Mapping[str, Any], cases_requested: int) -> tuple[dict[str, Any], list[dict[str, Any]], dict[int, dict[str, Any]]]:
    data = config["data"]
    states_path = PROJECT_DIR / data["public_states"]
    split_path = PROJECT_DIR / data["case_split"]
    source_path = PROJECT_DIR / data["source_jsonl"]
    states = read_jsonl(states_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    train_ids = set(map(str, split["train"]))
    validation_ids = set(map(str, split["validation"]))
    internal_test_ids = set(map(str, split["internal_test"]))
    if train_ids & validation_ids or train_ids & internal_test_ids or validation_ids & internal_test_ids:
        raise ValueError("case_id split overlap detected")

    train_states = [row for row in states if str(row["case_id"]) in train_ids]
    if len({str(row["case_id"]) for row in train_states}) != len(train_ids):
        raise ValueError("frozen train split contains a case with no public state")
    per_case = Counter(str(row["case_id"]) for row in train_states)
    selected_ids = sorted(train_ids, key=lambda case_id: stable_case_rank(case_id, int(config["seed"])))[:cases_requested]
    selected_set = set(selected_ids)
    selected_states = sorted(
        (row for row in train_states if str(row["case_id"]) in selected_set),
        key=lambda row: (str(row["case_id"]), int(row["state_index"])),
    )
    selected_source_indices = {int(row["source_row_index"]) for row in selected_states}
    source_rows = read_source_rows(source_path, selected_source_indices)

    invalid_state_count = 0
    role_boundary_checks = 0
    for state_row in selected_states:
        source_index = int(state_row["source_row_index"])
        source_row = source_rows[source_index]
        if str(state_row["case_id"]) != case_id_for_source(source_index, source_row):
            raise ValueError("public-state case_id does not map to the pinned source row")
        if state_row["initial_messages"] != source_row.get("prompt"):
            raise ValueError("frozen public state does not match the indexed source prompt")
        state = public_state_from_row(state_row, max_asks=int(config["policy"]["max_asks_per_episode"]))
        if not state.messages or not state.initial_messages:
            invalid_state_count += 1
        if state.case_id not in train_ids:
            raise ValueError("non-training case entered the selected counterfactual set")
        if "gold" in state_row or "ground_truth" in state_row or "atomic_facts" in state_row:
            raise ValueError("privileged source fields appear in the frozen public-state row")
        facts = hidden_facts_for_row(source_row)
        unrevealed = safe_public_fact_values(state, facts)
        for action in (Action.ANSWER, Action.DEFER):
            messages = build_base_messages(state, Action.ANSWER) if action is Action.ANSWER else build_defer_messages(state)
            assert_no_privileged_payload(messages, public_state=state, unrevealed_facts=unrevealed)
            role_boundary_checks += 1
        if Action.ASK in state.legal_actions():
            messages = build_base_messages(state, Action.ASK)
            assert_no_privileged_payload(messages, public_state=state, unrevealed_facts=unrevealed)
            role_boundary_checks += 1
        patient_messages = build_patient_messages(facts, "Static role-boundary audit question.")
        patient_payload = json.dumps(patient_messages, ensure_ascii=False)
        gold_object = source_row.get("ground_truth")
        if gold_object is not None and json.dumps(gold_object, ensure_ascii=False) in patient_payload:
            raise ValueError("gold answer object leaked into Patient prompt")
        role_boundary_checks += 1

    # Count source rows without logging any case text or labels.
    with source_path.open(encoding="utf-8") as handle:
        source_row_count = sum(1 for line in handle if line.strip())
    audit = {
        "source_jsonl_rows": source_row_count,
        "source_description": "single-turn source cases; not described as multi-turn trajectories",
        "frozen_public_state_rows": len(states),
        "frozen_public_state_unique_cases": len({str(row["case_id"]) for row in states}),
        "train_cases": len(train_ids),
        "validation_cases_not_used": len(validation_ids),
        "internal_test_cases_not_used": len(internal_test_ids),
        "train_public_state_rows": len(train_states),
        "train_states_per_case_min": min(per_case.values()),
        "train_states_per_case_max": max(per_case.values()),
        "selected_train_cases": cases_requested,
        "selected_public_states": len(selected_states),
        "selected_source_rows": len(selected_source_indices),
        "state_schema_invalid_count": invalid_state_count,
        "static_base_expert_boundary_checks_passed": role_boundary_checks,
        "frozen_holdout_read": False,
    }
    return audit, selected_states, source_rows


class APIError(RuntimeError):
    def __init__(self, role: str, attempts: int, elapsed_seconds: float, reason: str):
        super().__init__(reason)
        self.role = role
        self.attempts = attempts
        self.elapsed_seconds = elapsed_seconds
        self.reason = reason


def chat_completion(role: str, settings: Mapping[str, Any], messages: Sequence[Mapping[str, str]], *,
                    temperature: float, top_p: float, max_tokens: int, seed: int,
                    timeout_seconds: int, retries: int) -> dict[str, Any]:
    url = str(settings["base_url"]).rstrip("/") + "/chat/completions"
    key_env = str(settings.get("api_key_env", ""))
    key = os.environ.get(key_env, "") if key_env else ""
    payload = {
        "model": str(settings["model"]),
        "messages": list(messages),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens),
        "seed": int(seed),
    }
    if "enable_thinking" in settings:
        payload["chat_template_kwargs"] = {"enable_thinking": bool(settings["enable_thinking"])}
    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    last_error = "unknown"
    total_elapsed = 0.0
    for attempt in range(retries):
        request = urllib.request.Request(url, data=encoded, headers=headers, method="POST")
        started = time.perf_counter()
        try:
            with direct_loopback_opener().open(request, timeout=timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
            total_elapsed += time.perf_counter() - started
            choice = body["choices"][0]
            content = choice.get("message", {}).get("content")
            if not isinstance(content, str):
                raise APIError("completion content was not text")
            usage = body.get("usage") or {}
            return {
                "content": content,
                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "completion_tokens": int(usage.get("completion_tokens", 0)),
                "latency_seconds": total_elapsed,
                "finish_reason": choice.get("finish_reason"),
            }
        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code}"
        except Exception as exc:
            last_error = type(exc).__name__
        total_elapsed += time.perf_counter() - started
        if attempt + 1 < retries:
            sleep_started = time.perf_counter()
            time.sleep(min(2 ** attempt, 4))
            total_elapsed += time.perf_counter() - sleep_started
    raise APIError(role, retries, total_elapsed, f"generation failed after {retries} attempt(s): {last_error}")


def answer_gold(ground_truth: Any) -> str:
    value = ground_truth.get("answer") if isinstance(ground_truth, Mapping) else ground_truth
    match = re.fullmatch(r"\s*([A-D])\s*", str(value).upper())
    if not match:
        raise ValueError("source ground_truth answer could not be parsed")
    return match.group(1)


def hidden_facts_for_row(row: Mapping[str, Any]) -> list[str]:
    value = (row.get("extra_info") or {}).get("atomic_facts") or []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        value = [value]
    return [str(fact) for fact in value if str(fact).strip()]


def normalized_text(value: str) -> str:
    return " ".join(value.casefold().split())


def repeated_question(state: PublicState, question: str) -> bool:
    target = normalized_text(question)
    old_questions = [
        message["content"].split(":", 1)[1]
        for message in state.dialogue
        if message["role"] == "assistant" and message["content"].lstrip().lower().startswith("question:")
    ]
    return target in {normalized_text(question_text) for question_text in old_questions}


def safe_public_fact_values(state: PublicState, facts: Sequence[str]) -> list[str]:
    # Facts already present in the visible dialogue are no longer hidden.
    return [fact for fact in facts if len(fact.strip()) >= 12 and fact.strip() not in state.public_text]


def run_action(action: Action, state: PublicState, facts: Sequence[str], *,
               config: Mapping[str, Any], rollout_seed: int) -> dict[str, Any]:
    inference = config["inference"]
    roles = config["roles"]
    trace: list[dict[str, Any]] = []
    answer: str | None = None
    question: str | None = None
    patient_reply: str | None = None
    ask_novel_text: bool | None = None
    ask_unknown: bool | None = None
    failure_reason: str | None = None
    calls = Counter()
    attempts = Counter()
    usage = Counter()
    token_usage_complete = True
    latencies: list[float] = []
    finish_reasons: list[str | None] = []
    try:
        if action is Action.ANSWER:
            messages = build_base_messages(state, Action.ANSWER)
            assert_no_privileged_payload(messages, public_state=state,
                                         unrevealed_facts=safe_public_fact_values(state, facts))
            call = chat_completion(
                "base", roles["base"], messages, temperature=inference["temperature"], top_p=inference["top_p"],
                max_tokens=inference["answer_max_tokens"], seed=rollout_seed,
                timeout_seconds=inference["timeout_seconds"], retries=inference["api_retries"],
            )
            calls["base"] += 1
            attempts["base"] += 1
            trace.append({"role": "base", "stage": "initial_answer", "output": call["content"],
                          "prompt_tokens": call["prompt_tokens"], "completion_tokens": call["completion_tokens"],
                          "latency_seconds": call["latency_seconds"], "finish_reason": call["finish_reason"]})
            usage.update(prompt=call["prompt_tokens"], completion=call["completion_tokens"])
            latencies.append(call["latency_seconds"])
            finish_reasons.append(call["finish_reason"])
            answer = parse_answer(call["content"])

        elif action is Action.ASK:
            messages = build_base_messages(state, Action.ASK)
            assert_no_privileged_payload(messages, public_state=state,
                                         unrevealed_facts=safe_public_fact_values(state, facts))
            ask_call = chat_completion(
                "base", roles["base"], messages, temperature=inference["temperature"], top_p=inference["top_p"],
                max_tokens=inference["ask_max_tokens"], seed=rollout_seed,
                timeout_seconds=inference["timeout_seconds"], retries=inference["api_retries"],
            )
            calls["base"] += 1
            attempts["base"] += 1
            trace.append({"role": "base", "stage": "ask", "output": ask_call["content"],
                          "prompt_tokens": ask_call["prompt_tokens"], "completion_tokens": ask_call["completion_tokens"],
                          "latency_seconds": ask_call["latency_seconds"], "finish_reason": ask_call["finish_reason"]})
            usage.update(prompt=ask_call["prompt_tokens"], completion=ask_call["completion_tokens"])
            latencies.append(ask_call["latency_seconds"])
            finish_reasons.append(ask_call["finish_reason"])
            question = parse_question(ask_call["content"])
            if repeated_question(state, question):
                raise ValueError("repeated_ask_question")

            patient_messages = build_patient_messages(facts, question)
            patient_call = chat_completion(
                "patient", roles["patient"], patient_messages, temperature=inference["patient_temperature"],
                top_p=inference["top_p"], max_tokens=inference["patient_max_tokens"],
                seed=rollout_seed, timeout_seconds=inference["timeout_seconds"],
                retries=inference["api_retries"],
            )
            calls["patient"] += 1
            attempts["patient"] += 1
            patient_reply = patient_call["content"].strip()
            if not patient_reply:
                raise ValueError("empty_patient_response")
            prior_public = normalized_text(state.public_text)
            ask_novel_text = normalized_text(patient_reply) not in prior_public
            ask_unknown = "cannot answer this question" in patient_reply.casefold() or patient_reply.strip().casefold() == "unknown"
            trace.append({"role": "patient", "stage": "patient_reply", "output": patient_reply,
                          "prompt_tokens": patient_call["prompt_tokens"], "completion_tokens": patient_call["completion_tokens"],
                          "latency_seconds": patient_call["latency_seconds"], "finish_reason": patient_call["finish_reason"],
                          "is_unknown": ask_unknown, "novel_text_vs_public_state": ask_novel_text})
            usage.update(prompt=patient_call["prompt_tokens"], completion=patient_call["completion_tokens"])
            latencies.append(patient_call["latency_seconds"])
            finish_reasons.append(patient_call["finish_reason"])

            continued_state = extend_after_ask(state, question, patient_reply)
            continue_messages = build_base_messages(continued_state, Action.ANSWER, stage="after_ask")
            assert_no_privileged_payload(continue_messages, public_state=continued_state,
                                         unrevealed_facts=safe_public_fact_values(continued_state, facts))
            final_call = chat_completion(
                "base", roles["base"], continue_messages, temperature=inference["temperature"], top_p=inference["top_p"],
                max_tokens=inference["answer_max_tokens"], seed=rollout_seed,
                timeout_seconds=inference["timeout_seconds"], retries=inference["api_retries"],
            )
            calls["base"] += 1
            attempts["base"] += 1
            trace.append({"role": "base", "stage": "terminal_answer_after_ask", "output": final_call["content"],
                          "prompt_tokens": final_call["prompt_tokens"], "completion_tokens": final_call["completion_tokens"],
                          "latency_seconds": final_call["latency_seconds"], "finish_reason": final_call["finish_reason"]})
            usage.update(prompt=final_call["prompt_tokens"], completion=final_call["completion_tokens"])
            latencies.append(final_call["latency_seconds"])
            finish_reasons.append(final_call["finish_reason"])
            answer = parse_answer(final_call["content"])

        elif action is Action.DEFER:
            messages = build_defer_messages(state)
            assert_no_privileged_payload(messages, public_state=state,
                                         unrevealed_facts=safe_public_fact_values(state, facts))
            call = chat_completion(
                "expert", roles["expert"], messages, temperature=inference["temperature"], top_p=inference["top_p"],
                max_tokens=inference["defer_max_tokens"], seed=rollout_seed,
                timeout_seconds=inference["timeout_seconds"], retries=inference["api_retries"],
            )
            calls["expert"] += 1
            attempts["expert"] += 1
            trace.append({"role": "expert", "stage": "defer_final_answer", "output": call["content"],
                          "prompt_tokens": call["prompt_tokens"], "completion_tokens": call["completion_tokens"],
                          "latency_seconds": call["latency_seconds"], "finish_reason": call["finish_reason"]})
            usage.update(prompt=call["prompt_tokens"], completion=call["completion_tokens"])
            latencies.append(call["latency_seconds"])
            finish_reasons.append(call["finish_reason"])
            answer = parse_answer(call["content"], expert=True)
        else:  # pragma: no cover - enum exhaustiveness guard
            raise ValueError(f"unsupported action {action}")
    except APIError as exc:
        token_usage_complete = False
        attempts[exc.role] += exc.attempts
        latencies.append(exc.elapsed_seconds)
        trace.append({"role": exc.role, "stage": "api_failure", "error": exc.reason,
                      "attempts": exc.attempts, "latency_seconds": exc.elapsed_seconds,
                      "token_usage_available": False})
        failure_reason = f"{exc.role}_api_failure"
    except Exception as exc:
        failure_reason = str(exc) if isinstance(exc, ValueError) else type(exc).__name__

    valid = answer is not None and failure_reason is None
    return {
        "action": action.value,
        "rollout_seed": int(rollout_seed),
        "continuation_policy_id": config["policy"]["continuation_policy_id"],
        "final_answer": answer,
        "valid": valid,
        "failure_reason": failure_reason,
        "ask_count": int(action is Action.ASK),
        "defer_count": int(action is Action.DEFER),
        "ask_question": question,
        "patient_response": patient_reply,
        "ask_response_nonempty": bool(patient_reply),
        "ask_response_novel_text": ask_novel_text,
        "ask_response_unknown": ask_unknown,
        "api_call_counts": dict(calls),
        "api_attempt_counts": dict(attempts),
        "token_usage_complete": token_usage_complete,
        "prompt_tokens": int(usage["prompt"]),
        "completion_tokens": int(usage["completion"]),
        "total_tokens": int(usage["prompt"] + usage["completion"]),
        "latency_seconds": float(sum(latencies)),
        "finish_reasons": finish_reasons,
        "trace": trace,
    }


def run_selected_states(config: Mapping[str, Any], states: Sequence[Mapping[str, Any]],
                        source_rows: Mapping[int, Mapping[str, Any]], output_path: Path) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing trajectory file: {output_path}")
    action_counts = Counter()
    result_counts = Counter()
    with output_path.open("w", encoding="utf-8") as handle:
        for state_row in states:
            source_index = int(state_row["source_row_index"])
            source_row = source_rows[source_index]
            facts = hidden_facts_for_row(source_row)
            state = public_state_from_row(state_row, max_asks=int(config["policy"]["max_asks_per_episode"]))
            action_results: list[dict[str, Any]] = []
            for action in ACTION_ORDER:
                if action not in state.legal_actions():
                    result = {
                        "action": action.value, "valid": False,
                        "failure_reason": "illegal_action_masked_before_model_call",
                        "ask_count": 0, "defer_count": 0, "final_answer": None,
                        "continuation_policy_id": config["policy"]["continuation_policy_id"],
                        "rollout_seed": None, "trace": [], "prompt_tokens": 0,
                        "completion_tokens": 0, "total_tokens": 0, "latency_seconds": 0.0,
                        "api_call_counts": {}, "ask_question": None, "patient_response": None,
                        "api_attempt_counts": {}, "token_usage_complete": True,
                        "ask_response_nonempty": False, "ask_response_novel_text": None,
                        "ask_response_unknown": None, "finish_reasons": [],
                    }
                else:
                    seed_material = f"{config['seed']}:{state.state_id}:{action.value}:0".encode("utf-8")
                    rollout_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:4], "big") % 2**31
                    result = run_action(action, state, facts, config=config, rollout_seed=rollout_seed)
                action_results.append(result)

            # The answer key is first read only after all three counterfactual branches have
            # completed. It is never passed to role prompts and is never serialized.
            ground_truth = answer_gold(source_row.get("ground_truth"))
            for result in action_results:
                ask_cost = float(config["costs"]["ask"]) * int(result["ask_count"])
                defer_cost = float(config["costs"]["defer"]) * int(result["defer_count"])
                correctness = None
                if result["valid"]:
                    correctness = float(result["final_answer"] == ground_truth)
                    return_components = action_net_return(
                        correctness,
                        ask_count=result["ask_count"],
                        defer_count=result["defer_count"],
                        ask_cost=float(config["costs"]["ask"]),
                        defer_cost=float(config["costs"]["defer"]),
                    )
                else:
                    return_components = {"accuracy_reward": None, "return": None}

                record = {
                    "record_type": "v2_counterfactual_rollout",
                    "case_id": state.case_id,
                    "source_row_index": source_index,
                    "state_index": state.state_index,
                    "state_id": state.state_id,
                    "public_state_digest": state.digest,
                    "public_state": state.visible_messages,
                    "action": result["action"],
                    "rollout_seed": result["rollout_seed"],
                    "continuation_policy_id": result["continuation_policy_id"],
                    "final_answer": result["final_answer"],
                    "final_accuracy": correctness,
                    "ask_count": result["ask_count"],
                    "defer_count": result["defer_count"],
                    "ask_question": result["ask_question"],
                    "patient_response": result["patient_response"],
                    "ask_response_nonempty": result["ask_response_nonempty"],
                    "ask_response_novel_text": result["ask_response_novel_text"],
                    "ask_response_unknown": result["ask_response_unknown"],
                    "cost_components": {
                        "ask_cost": ask_cost,
                        "defer_cost": defer_cost,
                        "expert_calls": int(result["api_call_counts"].get("expert", 0)),
                        "expert_tokens": int(sum(
                            int(call.get("prompt_tokens", 0)) + int(call.get("completion_tokens", 0))
                            for call in result["trace"] if call["role"] == "expert"
                        )),
                    },
                    "accuracy_reward": return_components["accuracy_reward"],
                    "return": return_components["return"],
                    "return_mean": return_components["return"],
                    "return_std": None,
                    "sample_count": 1,
                    "valid_sample_count": int(bool(result["valid"])),
                    "valid": bool(result["valid"]),
                    "failure_reason": result["failure_reason"],
                    "prompt_tokens": result["prompt_tokens"],
                    "completion_tokens": result["completion_tokens"],
                    "total_tokens": result["total_tokens"],
                    "latency_seconds": result["latency_seconds"],
                    "api_call_counts": result["api_call_counts"],
                    "api_attempt_counts": result["api_attempt_counts"],
                    "token_usage_complete": result["token_usage_complete"],
                    "finish_reasons": result["finish_reasons"],
                    "trace": result["trace"],
                }
                key = (state.case_id, state.state_id, result["action"])
                if key in done:
                    raise ValueError("duplicate case/state/action tuple")
                done.add(key)
                action_counts[result["action"]] += 1
                result_counts["valid"] += int(bool(result["valid"]))
                result_counts["invalid"] += int(not bool(result["valid"]))
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
    return {
        "attempted_rollouts": sum(action_counts.values()),
        "by_action": dict(action_counts),
        "valid_rollouts": result_counts["valid"],
        "invalid_rollouts": result_counts["invalid"],
        "trajectory_file": str(output_path),
    }


def run_role_smoke(config: Mapping[str, Any], state_row: Mapping[str, Any],
                   source_row: Mapping[str, Any], output_path: Path) -> dict[str, Any]:
    """Exercise Base ANSWER, Base ASK -> Patient -> Base, and Expert DEFER once.

    This service smoke never reads or scores Gold and does not enter the pilot data file.
    """
    state = public_state_from_row(
        state_row, max_asks=int(config["policy"]["max_asks_per_episode"])
    )
    facts = hidden_facts_for_row(source_row)
    results = {}
    for action in (Action.ANSWER, Action.ASK, Action.DEFER):
        seed_material = f"{config['seed']}:service-smoke:{state.state_id}:{action.value}".encode("utf-8")
        rollout_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:4], "big") % 2**31
        results[action.value] = run_action(action, state, facts, config=config, rollout_seed=rollout_seed)

    ask = results["ASK"]
    checks = {
        "base_answer_parsed": bool(results["ANSWER"]["valid"]),
        "expert_defer_parsed": bool(results["DEFER"]["valid"]),
        "ask_question_parsed": bool(ask["ask_question"]),
        "patient_reply_nonempty": bool(ask["ask_response_nonempty"]),
        "patient_reply_received_once": int(ask["api_call_counts"].get("patient", 0)) == 1,
        "base_answer_after_ask_parsed": bool(ask["valid"] and ask["final_answer"]),
    }
    smoke = {
        "record_type": "v2_service_role_smoke",
        "case_id": state.case_id,
        "state_id": state.state_id,
        "public_state_digest": state.digest,
        "continuation_policy_id": config["policy"]["continuation_policy_id"],
        "services": {
            role: {key: settings.get(key) for key in ("base_url", "model", "model_revision", "shared_service")}
            for role, settings in config["roles"].items()
        },
        "checks": checks,
        "passed": all(checks.values()),
        "gold_read": False,
        "actions": results,
    }
    output_path.write_text(json.dumps(smoke, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"passed": smoke["passed"], "checks": checks, "smoke_file": str(output_path)}


def make_manifest(config_path: Path, config: Mapping[str, Any], cases: int, *, execute: bool) -> dict[str, Any]:
    audit, selected_states, source_rows = audit_and_select(config, cases)
    endpoint_roles = {role: endpoint_status(role, settings) for role, settings in config["roles"].items()
                      if role in ("base", "expert")}
    # Patient shares the Base service but remains a separate prompt and role.
    endpoint_roles["patient"] = {
        **endpoint_roles["base"], "role": "patient", "shares_endpoint_with": "base",
    }
    required_endpoints_ok = all(item["reachable"] and item["model_match"]
                                for item in endpoint_roles.values())
    return {
        "schema_version": 1,
        "experiment": "EA-ATPO V2 three-action paired counterfactual pilot",
        "created_at_utc": utc_now(),
        "status": "preflight_ready" if required_endpoints_ok else "blocked_endpoint_preflight",
        "mode": "execute" if execute else "preflight_only",
        "project_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_DIR,
            capture_output=True, text=True, check=False,
        ).stdout.strip() or None,
        "project_branch": subprocess.run(
            ["git", "branch", "--show-current"], cwd=PROJECT_DIR,
            capture_output=True, text=True, check=False,
        ).stdout.strip() or None,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "data_audit": audit,
        "selection_seed": int(config["seed"]),
        "selected_case_ids": sorted({str(row["case_id"]) for row in selected_states}),
        "selected_state_count": len(selected_states),
        "data_sha256": {
            name: sha256_file(PROJECT_DIR / config["data"][name])
            for name in ("public_states", "case_split", "source_jsonl")
        },
        "models": config["roles"],
        "endpoint_preflight": endpoint_roles,
        "gpu_inventory_read_only": gpu_inventory(),
        "model_execution_started": False,
        "frozen_holdout_read": False,
        "source_rows_loaded_for_selected_train_cases": len(source_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=V2_DIR / "configs" / "phase1_counterfactual.json")
    parser.add_argument("--cases", type=int, default=10)
    parser.add_argument("--preflight", action="store_true", help="audit inputs and configured services; do not generate")
    parser.add_argument("--execute", action="store_true", help="run real model calls after all gates pass")
    parser.add_argument("--smoke", action="store_true", help="run one paired state across the three real service roles; no Gold scoring")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.preflight == args.execute:
        parser.error("choose exactly one of --preflight or --execute")
    if args.smoke and not args.execute:
        parser.error("--smoke requires --execute")
    if args.smoke:
        cases_requested = 1
    else:
        if not 10 <= args.cases <= 20:
            parser.error("--cases must be between 10 and 20 for the full pilot")
        cases_requested = args.cases

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = args.output_dir or (V2_DIR / "outputs" / f"phase1_{stamp}")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = make_manifest(config_path, config, cases_requested, execute=args.execute)
    manifest_path = output_dir / "run_manifest.json"
    if args.preflight:
        manifest["status"] = "preflight_ready" if manifest["status"] == "preflight_ready" else "blocked_endpoint_preflight"
        manifest["model_execution_started"] = False
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"status": manifest["status"], "selected_cases": cases_requested,
                          "selected_states": manifest["selected_state_count"],
                          "output_dir": str(output_dir)}, indent=2))
        return

    if not all(item["reachable"] and item["model_match"]
               for role, item in manifest["endpoint_preflight"].items() if role in ("base", "expert")):
        manifest["status"] = "blocked_endpoint_preflight"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        raise SystemExit("configured Base/Expert endpoints are unavailable or model aliases mismatch; no rollouts started")

    audit, selected_states, source_rows = audit_and_select(config, cases_requested)
    manifest["status"] = "running"
    manifest["model_execution_started"] = True
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        if args.smoke:
            first_state = min(selected_states, key=lambda row: int(row["state_index"]))
            summary = run_role_smoke(
                config, first_state, source_rows[int(first_state["source_row_index"])],
                output_dir / "service_role_smoke.json",
            )
            manifest["status"] = "completed" if summary["passed"] else "role_smoke_failed"
            manifest["role_smoke_summary"] = summary
        else:
            summary = run_selected_states(config, selected_states, source_rows, output_dir / "pilot_trajectories.jsonl")
            manifest["status"] = "completed"
            manifest["rollout_summary"] = summary
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failure_class"] = type(exc).__name__
        manifest["failure_message"] = str(exc)
        raise
    finally:
        manifest["finished_at_utc"] = utc_now()
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest.get("role_smoke_summary", manifest.get("rollout_summary", {})), indent=2))
    if args.smoke and manifest.get("status") != "completed":
        raise SystemExit("real-service role smoke failed; full pilot not started")


if __name__ == "__main__":
    main()
