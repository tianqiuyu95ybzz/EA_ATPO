#!/usr/bin/env python3
"""Compare post-CONSULT decision policies on frozen Pilot states and advice."""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import random
import re
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp


AUDIT_DIR = Path(__file__).resolve().parent
PILOT_DIR = AUDIT_DIR.parent
PILOT_TRAJECTORIES = PILOT_DIR / "outputs/full/pilot_trajectories.jsonl"
OUT_DIR = AUDIT_DIR / "outputs/advice_utilization"
INPUTS_PATH = OUT_DIR / "frozen_consult_inputs.jsonl"
TRAJECTORIES_PATH = OUT_DIR / "advice_utilization_trajectories.jsonl"
MANIFEST_PATH = OUT_DIR / "run_manifest.json"
SUMMARY_PATH = OUT_DIR / "advice_utilization_summary.csv"
BASE_URL = "http://127.0.0.1:30133/v1"
MODEL_ALIAS = "qwen3-8b-advice-base"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
SEED = 20261105
STATE_COUNT = 200
BOOTSTRAP_REPEATS = 10_000
LIVE_METHODS = ("CONSULT_NAIVE", "CONSULT_RETAIN", "CONSULT_CRITICAL")
METHODS = ("PILOT_ANSWER", *LIVE_METHODS)

BASE_SYSTEM = """You are a medical multiple-choice benchmark solver. Use only the visible case, options, prior public dialogue, and any consultation advice included in the user message. Never infer or request hidden atomic facts, benchmark answers, answer explanations, or future dialogue. Return exactly one option A-D in the required Final Answer field. Do not output private chain-of-thought; any requested evidence note must be brief and based on visible information."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def parse_answer(text: str) -> str | None:
    matches = re.findall(r"Final\s+Answer\s*:\s*([A-D])\b", text, re.IGNORECASE)
    if len(matches) == 1:
        return matches[0].upper()
    if not matches:
        single = re.fullmatch(r"\s*([A-D])\s*", text, re.IGNORECASE)
        if single:
            return single.group(1).upper()
    return None


def parse_decision(text: str) -> str | None:
    match = re.search(r"^Decision\s*:\s*(KEEP|UPDATE)\b", text, re.IGNORECASE | re.MULTILINE)
    return match.group(1).upper() if match else None


def public_messages(public_state: list[dict[str, Any]]) -> list[dict[str, str]]:
    messages = []
    for msg in public_state:
        role = str(msg.get("role", ""))
        if role not in ("user", "assistant"):
            continue
        messages.append({"role": role, "content": str(msg.get("content", ""))})
    if not messages or not any(msg["role"] == "user" for msg in messages):
        raise ValueError("Pilot state does not contain visible user case content")
    return messages


def extract_consult_text(row: dict[str, Any], actor: str, field: str, fallback: Any) -> str:
    values = [
        str(event.get(field, "")).strip()
        for event in row.get("trace", [])
        if event.get("actor") == actor and event.get("action") == "CONSULT" and event.get(field)
    ]
    if values:
        return "\n".join(values)
    if isinstance(fallback, list):
        return "\n".join(str(value).strip() for value in fallback if str(value).strip())
    return str(fallback or "").strip()


def choose_frozen_sample(seed: int, state_count: int) -> list[dict[str, Any]]:
    trajectories = [x for x in read_jsonl(PILOT_TRAJECTORIES) if x.get("record_type") == "trajectory"]
    by_pair: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in trajectories:
        by_pair[(str(row["case_id"]), str(row["state_id"]), int(row["rollout_index"]))][str(row["action"])] = row
    by_state: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for (case_id, state_id, rollout_index), actions in by_pair.items():
        consult = actions.get("CONSULT")
        answer = actions.get("ANSWER")
        if not consult or not answer:
            continue
        if not consult.get("valid") or not str(consult.get("expert_advice", "")).strip():
            continue
        if not answer.get("valid") or not str(answer.get("final_answer", "")).strip():
            continue
        by_state[(case_id, state_id)].append({"consult": consult, "answer": answer, "rollout_index": rollout_index})
    state_keys = sorted(by_state)
    if len(state_keys) < state_count:
        raise ValueError(f"only {len(state_keys)} states have a valid CONSULT+ANSWER pair; requested {state_count}")
    rng = random.Random(seed)
    sampled_state_keys = sorted(rng.sample(state_keys, state_count))
    sample = []
    for sample_index, state_key in enumerate(sampled_state_keys):
        pair = rng.choice(by_state[state_key])
        consult, answer = pair["consult"], pair["answer"]
        messages = public_messages(consult["public_state"])
        consult_question = extract_consult_text(consult, "base", "consult_question", consult.get("consult_question"))
        expert_advice = extract_consult_text(consult, "expert", "response", consult.get("expert_advice"))
        if not consult_question or not expert_advice:
            raise ValueError(f"missing saved question/advice in eligible rollout {state_key}/{pair['rollout_index']}")
        # The model request will be built from these visible messages only. Gold
        # is retained separately for the evaluator and never passed to API calls.
        sample.append({
            "sample_id": f"consult-decision-{sample_index:03d}",
            "case_id": str(consult["case_id"]),
            "state_id": str(consult["state_id"]),
            "source_row_index": int(consult["source_row_index"]),
            "rollout_index": int(pair["rollout_index"]),
            "public_messages": messages,
            "public_state_sha256": sha256_bytes(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()),
            "consult_question": consult_question,
            "expert_advice": expert_advice,
            "expert_advice_sha256": sha256_bytes(expert_advice.encode()),
            "pilot_answer": str(answer["final_answer"]).strip().upper(),
            "pilot_answer_correct": bool(float(answer.get("outcome") or 0.0) == 1.0),
            "pilot_answer_valid": bool(answer.get("valid")),
            "pilot_answer_prompt_tokens": int(answer.get("base_prompt_tokens", 0)),
            "pilot_answer_completion_tokens": int(answer.get("base_completion_tokens", 0)),
            "pilot_answer_latency_s": float(answer.get("base_latency_s", 0.0)),
            "gold": str(consult.get("gold", "")).strip().upper(),
        })
    return sample


def make_messages(item: dict[str, Any], method: str) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": BASE_SYSTEM}]
    messages.extend(item["public_messages"])
    advice_context = (
        "Consult question asked to the Expert:\n" + item["consult_question"] +
        "\n\nExpert response (verbatim):\n" + item["expert_advice"]
    )
    if method == "CONSULT_NAIVE":
        messages.append({"role": "user", "content": advice_context + "\n\nRe-answer the original multiple-choice question now. Consider the case and this advice, then give one brief evidence basis and exactly one `Final Answer: A-D`."})
    elif method == "CONSULT_RETAIN":
        messages.append({"role": "assistant", "content": f"My prior decision before seeing this advice was: Final Answer: {item['pilot_answer']}. The Pilot did not save a verbal rationale for that decision."})
        messages.append({"role": "user", "content": advice_context + "\n\nReview the prior decision. You may keep it or update it after considering the advice; do not change it merely because an Expert spoke. Output `Decision: KEEP` or `Decision: UPDATE`, one brief basis, and exactly one `Final Answer: A-D`."})
    elif method == "CONSULT_CRITICAL":
        messages.append({"role": "assistant", "content": f"My prior decision before seeing this advice was: Final Answer: {item['pilot_answer']}. The Pilot did not save a verbal rationale for that decision."})
        messages.append({"role": "user", "content": advice_context + "\n\nAudit the advice against the visible case: state its main claim, identify what public case evidence supports or contradicts it, and mark claims not verifiable from this state as unverified rather than facts. Check that the advice addresses the original question. Then choose whether to keep or update the prior answer. Do not use any hidden facts or answer key. Output `Decision: KEEP` or `Decision: UPDATE`, one brief `Evidence Check` of at most 35 words, and exactly one `Final Answer: A-D`."})
    else:
        raise ValueError(f"unknown method {method}")
    return messages


async def request_one(session: aiohttp.ClientSession, messages: list[dict[str, str]], seed: int) -> dict[str, Any]:
    payload = {
        "model": MODEL_ALIAS,
        "messages": messages,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 128,
        "seed": seed,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    started = time.perf_counter()
    try:
        async with session.post(BASE_URL.rstrip("/") + "/chat/completions", json=payload) as response:
            response.raise_for_status()
            body = await response.json()
        raw = str(body["choices"][0]["message"].get("content", ""))
        usage = body.get("usage") or {}
        return {
            "raw_response": raw,
            "predicted": parse_answer(raw),
            "decision_label": parse_decision(raw),
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "latency_s": time.perf_counter() - started,
            "error": None,
        }
    except Exception as exc:
        return {
            "raw_response": "", "predicted": None, "decision_label": None,
            "prompt_tokens": 0, "completion_tokens": 0,
            "latency_s": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
        }


def cluster_bootstrap(rows: list[dict[str, Any]], stat_fn, seed: int, repeats: int = BOOTSTRAP_REPEATS) -> tuple[float, float]:
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case[str(row["case_id"])].append(row)
    cases = sorted(by_case)
    rng = random.Random(seed)
    values = []
    for _ in range(repeats):
        chosen = rng.choices(cases, k=len(cases))
        sampled = [row for case in chosen for row in by_case[case]]
        value = stat_fn(sampled)
        if value is not None:
            values.append(float(value))
    values.sort()
    return values[int(.025 * len(values))], values[min(len(values) - 1, int(.975 * len(values)))]


def accuracy(rows: list[dict[str, Any]], method: str) -> float:
    if method == "PILOT_ANSWER":
        return statistics.mean(float(bool(row["pilot_answer_correct"])) for row in rows)
    return statistics.mean(float(bool(row["methods"][method]["correct"])) for row in rows)


def conditional_rate(rows: list[dict[str, Any]], method: str, kind: str) -> float | None:
    if kind == "correction":
        eligible = [r for r in rows if not r["pilot_answer_correct"]]
        numerator = [r for r in eligible if (r["pilot_answer_correct"] if method == "PILOT_ANSWER" else r["methods"][method]["correct"])]
    elif kind == "degradation":
        eligible = [r for r in rows if r["pilot_answer_correct"]]
        numerator = [r for r in eligible if not (r["pilot_answer_correct"] if method == "PILOT_ANSWER" else r["methods"][method]["correct"])]
    else:
        raise ValueError(kind)
    return len(numerator) / len(eligible) if eligible else None


def answer_change_rate(rows: list[dict[str, Any]], method: str) -> float:
    if method == "PILOT_ANSWER":
        return 0.0
    valid_rows = [r for r in rows if r["methods"][method]["valid"]]
    if not valid_rows:
        return 0.0
    return statistics.mean(float(r["methods"][method]["predicted"] != r["pilot_answer"]) for r in valid_rows)


def write_summary(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    fields = ("metric", "group", "value", "ci95_low", "ci95_high", "n_states", "notes")
    summary: list[dict[str, Any]] = []

    def add(metric: str, group: str, value: Any, low: Any = "", high: Any = "", n: int | None = None, notes: str = "") -> None:
        summary.append({"metric": metric, "group": group, "value": value, "ci95_low": low, "ci95_high": high, "n_states": len(rows) if n is None else n, "notes": notes})

    for method in METHODS:
        acc = accuracy(rows, method)
        low, high = cluster_bootstrap(rows, lambda xs, m=method: accuracy(xs, m), SEED + 10)
        add("accuracy", method, acc, low, high, notes="Case-cluster bootstrap; invalid/API/parse failures score 0 and remain in denominator.")
        for kind, label in (("correction", "correction_rate_given_pilot_answer_wrong"), ("degradation", "degradation_rate_given_pilot_answer_correct")):
            rate = conditional_rate(rows, method, kind)
            low, high = cluster_bootstrap(rows, lambda xs, m=method, k=kind: conditional_rate(xs, m, k), SEED + (11 if kind == "correction" else 12))
            denom = sum(not r["pilot_answer_correct"] for r in rows) if kind == "correction" else sum(r["pilot_answer_correct"] for r in rows)
            add(label, method, rate if rate is not None else "", low, high, n=denom, notes="Denominator is Pilot paired ANSWER errors/successes; case-cluster bootstrap.")
        change = answer_change_rate(rows, method)
        change_low, change_high = cluster_bootstrap(rows, lambda xs, m=method: answer_change_rate(xs, m), SEED + 13)
        add("answer_change_rate_vs_pilot_answer", method, change, change_low, change_high, notes="Among valid parsed answers, final option differs from paired pre-consult Pilot answer; operational uptake proxy, not semantic advice adoption.")
        if method in ("CONSULT_RETAIN", "CONSULT_CRITICAL"):
            labels = [r["methods"][method]["decision_label"] for r in rows]
            n_labeled = sum(label in ("KEEP", "UPDATE") for label in labels)
            def update_rate(xs: list[dict[str, Any]], m: str = method) -> float | None:
                values = [r["methods"][m]["decision_label"] for r in xs]
                valid_labels = [x for x in values if x in ("KEEP", "UPDATE")]
                return sum(x == "UPDATE" for x in valid_labels) / len(valid_labels) if valid_labels else None
            update_low, update_high = cluster_bootstrap(rows, update_rate, SEED + 14)
            add("self_reported_update_rate", method, update_rate(rows) if update_rate(rows) is not None else "", update_low, update_high, n=n_labeled, notes="Model-emitted Decision: UPDATE among parseable KEEP/UPDATE labels; descriptive only.")
            add("decision_label_parse_rate", method, n_labeled / len(rows), notes="Fraction with a parseable KEEP/UPDATE declaration.")
        if method == "PILOT_ANSWER":
            add("mean_prompt_tokens", method, statistics.mean(float(r["pilot_answer_prompt_tokens"]) for r in rows), notes="Stored in the original paired Pilot answer branch.")
            add("mean_completion_tokens", method, statistics.mean(float(r["pilot_answer_completion_tokens"]) for r in rows), notes="Stored in the original paired Pilot answer branch.")
            add("mean_total_tokens", method, statistics.mean(float(r["pilot_answer_prompt_tokens"]) + float(r["pilot_answer_completion_tokens"]) for r in rows), notes="Stored in the original paired Pilot answer branch.")
            add("mean_latency_s", method, statistics.mean(float(r["pilot_answer_latency_s"]) for r in rows), notes="Stored in the original paired Pilot answer branch; timing is from the original batched service.")
            add("valid_rate", method, statistics.mean(float(r["pilot_answer_valid"]) for r in rows), notes="Invalid responses remain in accuracy denominator and score 0.")
        else:
            for metric, field in (("mean_prompt_tokens", "prompt_tokens"), ("mean_completion_tokens", "completion_tokens"), ("mean_total_tokens", "total_tokens"), ("mean_latency_s", "latency_s")):
                add(metric, method, statistics.mean(float(r["methods"][method][field]) for r in rows), notes="OpenAI-compatible vLLM API usage; one request per state.")
            add("valid_rate", method, statistics.mean(float(r["methods"][method]["valid"]) for r in rows), notes="Invalid response remains in accuracy denominator and scores 0.")

    pairs = (("CONSULT_NAIVE", "PILOT_ANSWER"), ("CONSULT_RETAIN", "PILOT_ANSWER"), ("CONSULT_CRITICAL", "PILOT_ANSWER"),
             ("CONSULT_RETAIN", "CONSULT_NAIVE"), ("CONSULT_CRITICAL", "CONSULT_NAIVE"), ("CONSULT_CRITICAL", "CONSULT_RETAIN"))
    for first, second in pairs:
        def delta(xs, a=first, b=second):
            a_vals = [float(r["methods"][a]["correct"]) for r in xs] if a != "PILOT_ANSWER" else [float(r["pilot_answer_correct"]) for r in xs]
            b_vals = [float(r["methods"][b]["correct"]) for r in xs] if b != "PILOT_ANSWER" else [float(r["pilot_answer_correct"]) for r in xs]
            return statistics.mean(x - y for x, y in zip(a_vals, b_vals))
        low, high = cluster_bootstrap(rows, delta, SEED + 20 + pairs.index((first, second)))
        add("paired_accuracy_delta", f"{first}_minus_{second}", delta(rows), low, high, notes="Paired state outcomes; bootstrap resamples whole case clusters.")

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    manifest.update({
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evaluated_states": len(rows),
        "evaluated_cases": len({r["case_id"] for r in rows}),
        "failure_counts": {
            "PILOT_ANSWER": sum(not r["pilot_answer_valid"] for r in rows),
            **{m: sum(not r["methods"][m]["valid"] for r in rows) for m in LIVE_METHODS},
        },
        "summary_sha256": sha256_file(SUMMARY_PATH),
        "trajectory_sha256": sha256_file(TRAJECTORIES_PATH),
    })
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


async def main_async(args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sample = choose_frozen_sample(SEED, STATE_COUNT)
    pilot_sha = sha256_file(PILOT_TRAJECTORIES)
    sample_bytes = b"".join((json.dumps(x, ensure_ascii=False) + "\n").encode() for x in sample)
    if INPUTS_PATH.exists():
        if INPUTS_PATH.read_bytes() != sample_bytes:
            raise RuntimeError("existing frozen inputs differ from deterministic selection")
    else:
        INPUTS_PATH.write_bytes(sample_bytes)

    manifest = {
        "status": "running",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experiment": "Post-consult decision mechanism comparison on existing Pilot public states and verbatim Pilot Expert advice.",
        "pilot_trajectory_path": str(PILOT_TRAJECTORIES.resolve()),
        "pilot_trajectory_sha256": pilot_sha,
        "sample_seed": SEED,
        "sampled_state_count": len(sample),
        "sampled_case_count": len({x["case_id"] for x in sample}),
        "sampled_state_ids_sha256": sha256_bytes("\n".join(x["state_id"] for x in sample).encode()),
        "inputs_path": str(INPUTS_PATH.resolve()),
        "inputs_sha256": sha256_file(INPUTS_PATH),
        "eligibility": "One randomly selected rollout per sampled case/state, requiring a valid saved CONSULT response and a valid paired Pilot ANSWER. One of 291 states has no eligible valid CONSULT advice; 290 states are eligible.",
        "models": {"base": {"repo_id": "Qwen/Qwen3-8B", "revision": MODEL_REVISION, "served_alias": MODEL_ALIAS}, "advice_source": {"repo_id": "Qwen/Qwen3-8B", "revision": MODEL_REVISION, "source": "Existing Pilot CONSULT trace; advice unchanged across methods."}},
        "methods": {
            "PILOT_ANSWER": "Existing paired Pilot direct ANSWER, no Expert advice; historical measured tokens/latency retained.",
            "CONSULT_NAIVE": "Fresh Qwen3-8B response to same public state plus verbatim consult question/advice; previous answer withheld.",
            "CONSULT_RETAIN": "Same state/advice plus paired Pilot answer; Base may keep or update, with a KEEP/UPDATE declaration.",
            "CONSULT_CRITICAL": "Same state/advice and prior answer; Base makes a short public-evidence consistency check before KEEP/UPDATE.",
        },
        "generation": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 128, "seed": SEED, "enable_thinking": False, "max_num_seqs": 1, "prefix_caching": False},
        "scoring": "Same exact single-letter Final Answer parser for all methods. Invalid/API/parse failures remain in denominator and score 0.",
        "gold_policy": "Gold is stored in evaluator sample only; it is not included in any Base request. atomic_facts and future Patient responses are not used or sent.",
        "prior_reasoning_limitation": "Pilot branch trace stores only the prior answer, not a verbal rationale or private chain-of-thought. Retain/Critical therefore receive the prior option only; Critical performs a new brief visible-evidence audit.",
        "advice_uptake_definition": "Answer-change rate relative to the paired Pilot answer is the cross-method operational uptake proxy. It is not a semantic adjudication that the advice caused or was followed by the answer change. Retain/Critical also report their self-declared UPDATE rate.",
        "token_latency_caveat": "Pilot_ANSWER usage/latency comes from the original batched Pilot service; new CONSULT variants run sequentially on a separate single-sequence service with prefix cache disabled. Compare costs as observed measurements, not controlled wall-clock throughput estimates.",
        "trajectory_path": str(TRAJECTORIES_PATH.resolve()),
    }
    if MANIFEST_PATH.exists():
        prior = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        for key in ("pilot_trajectory_sha256", "sample_seed", "inputs_sha256", "methods", "generation"):
            if prior.get(key) != manifest.get(key):
                raise RuntimeError(f"resume mismatch in {key}")
        manifest = prior
    else:
        # Persist the frozen run configuration before API calls so interruption
        # always leaves enough metadata for a verified resume.
        manifest["status"] = "resuming_without_initial_manifest" if TRAJECTORIES_PATH.exists() else "running"
        MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    inputs_by_id = {x["sample_id"]: x for x in sample}
    existing = read_jsonl(TRAJECTORIES_PATH) if TRAJECTORIES_PATH.exists() else []
    done = {(str(x["sample_id"]), str(x["method"])) for x in existing}
    if len(done) != len(existing):
        raise RuntimeError("duplicate sample/method rows in resumable trajectory output")
    for record in existing:
        sample_id, method = str(record["sample_id"]), str(record["method"])
        if sample_id not in inputs_by_id or method not in LIVE_METHODS:
            raise RuntimeError(f"unexpected resumable output key {(sample_id, method)}")
        item = inputs_by_id[sample_id]
        if record.get("public_state_sha256") != item["public_state_sha256"] or record.get("expert_advice_sha256") != item["expert_advice_sha256"]:
            raise RuntimeError(f"resumable public state/advice hash mismatch for {(sample_id, method)}")
        expected_messages = make_messages(item, method)
        expected_prompt_sha = sha256_bytes(json.dumps(expected_messages, ensure_ascii=False, sort_keys=True).encode())
        if record.get("prompt_sha256") != expected_prompt_sha:
            raise RuntimeError(f"resumable prompt mismatch for {(sample_id, method)}")
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(BASE_URL.rstrip("/") + "/models") as response:
            response.raise_for_status()
            aliases = [str(x.get("id")) for x in (await response.json()).get("data", [])]
        if MODEL_ALIAS not in aliases:
            raise RuntimeError(f"expected Base alias {MODEL_ALIAS}; found {aliases}")

        calls = [(sample_id, method) for sample_id in inputs_by_id for method in LIVE_METHODS]
        # Interleave methods within states. Prefix caching is disabled in the
        # service config; this order also avoids a fixed method/order effect.
        random.Random(SEED + 1).shuffle(calls)
        for sample_id, method in calls:
            if (sample_id, method) in done:
                continue
            item = inputs_by_id[sample_id]
            messages = make_messages(item, method)
            result = await request_one(session, messages, SEED)
            pred = result["predicted"]
            record = {
                "record_type": "advice_utilization_trajectory",
                "sample_id": sample_id, "case_id": item["case_id"], "state_id": item["state_id"],
                "source_row_index": item["source_row_index"], "rollout_index": item["rollout_index"],
                "method": method, "model_alias": MODEL_ALIAS, "model_revision": MODEL_REVISION,
                "public_state_sha256": item["public_state_sha256"], "expert_advice_sha256": item["expert_advice_sha256"],
                "prompt_sha256": sha256_bytes(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()),
                "messages": messages,
                "gold": item["gold"], "pilot_answer": item["pilot_answer"],
                "pilot_answer_correct": item["pilot_answer_correct"],
                "pilot_answer_tokens": item["pilot_answer_prompt_tokens"] + item["pilot_answer_completion_tokens"],
                "pilot_answer_latency_s": item["pilot_answer_latency_s"],
                "predicted": pred, "correct": bool(pred == item["gold"]),
                "valid": pred is not None and result["error"] is None,
                "decision_label": result["decision_label"],
                "raw_response": result["raw_response"], "error": result["error"],
                "prompt_tokens": result["prompt_tokens"], "completion_tokens": result["completion_tokens"],
                "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
                "latency_s": result["latency_s"],
            }
            with TRAJECTORIES_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
            existing.append(record)
            done.add((sample_id, method))
            if len(done) % 30 == 0:
                print(f"completed method-state calls {len(done)}/{len(calls)}", flush=True)

    expected = len(sample) * len(LIVE_METHODS)
    if len(done) != expected:
        raise RuntimeError(f"incomplete method-state calls: {len(done)}/{expected}; rerun with --resume")
    outputs = {(x["sample_id"], x["method"]): x for x in existing}
    wide = []
    for item in sample:
        row = {**item, "methods": {m: outputs[(item["sample_id"], m)] for m in LIVE_METHODS}}
        wide.append(row)
    write_summary(wide, manifest)
    print(json.dumps({"status": "completed", "states": len(sample), "cases": len({x['case_id'] for x in sample}), "summary": str(SUMMARY_PATH)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume completed method/state calls from JSONL.")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
