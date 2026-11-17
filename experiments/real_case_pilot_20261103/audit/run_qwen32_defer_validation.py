#!/usr/bin/env python3
"""Collect fresh, paired Qwen3-8B ANSWER / Qwen3-32B-AWQ DEFER rollouts.

Prompts and gold/facts remain in memory. JSONL records keep raw generations,
predictions and telemetry, but never serialize prompts, gold, or atomic facts.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import math
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
OUT = HERE / "outputs/qwen32_defer_validation_20261119_attempt01"
SELECTION = OUT / "case_selection.json"
BASE_URL = "http://127.0.0.1:30157/v1"
EXPERT_URL = "http://127.0.0.1:30156/v1"
BASE_ALIAS = "qwen3-8b-defer-validation-base"
EXPERT_ALIAS = "qwen3-32b-awq-defer-validation"
SMOKE_SEED = 20261120
SMOKE_N = 20
AWQ_REV = "0499c3ac83fdef8810b907a23894ba91e95eddd8"
BASE_REV = "b968826d9c46dd6066d109eabc6255188de91218"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "ea_atpo_stage"))
import run_direct_reliability_eval as direct  # noqa: E402
import run_l2d_remedi_confirmation as l2d  # noqa: E402
import run_medical_candidate_eval as candidate  # noqa: E402

_spec = importlib.util.spec_from_file_location("defer_protocol_validation", HERE / "run_medical_candidate_defer_dev_eval.py")
if _spec is None or _spec.loader is None:
    raise ImportError("DEFER protocol helper cannot be loaded")
defer_helper = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = defer_helper
_spec.loader.exec_module(defer_helper)


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def public_features(messages: list[dict[str, str]]) -> dict[str, float]:
    text = "\n".join(str(m.get("content", "")) for m in messages if m.get("role") != "system")
    words = re.findall(r"\b\w+\b", text, re.UNICODE)
    option_lines = re.findall(r"(?m)^\s*[A-D]\s*[).:]\s*(.*?)\s*$", text)
    lower = text.lower()
    terms = ("pain", "fever", "cough", "blood", "dose", "drug", "symptom", "diagnosis", "treatment", "history", "onset", "patient", "exam", "test", "surgery", "infection", "pressure", "heart", "lung", "kidney", "pregnan")
    return {
        "log_public_chars": math.log1p(len(text)),
        "log_public_words": math.log1p(len(words)),
        "option_count": float(len(option_lines)),
        "mean_option_chars": float(sum(map(len, option_lines)) / len(option_lines)) if option_lines else 0.0,
        "question_marks": float(text.count("?") + text.count("？")),
        "digit_count": float(sum(c.isdigit() for c in text)),
        "clinical_lexicon_count": float(sum(lower.count(term) for term in terms)),
    }


def jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def token_text(item: dict[str, Any]) -> str:
    raw = item.get("bytes")
    if isinstance(raw, list):
        try:
            return bytes(int(x) for x in raw).decode("utf-8", errors="replace")
        except (TypeError, ValueError):
            pass
    return str(item.get("token", "")).replace("Ġ", " ").replace("▁", " ")


def answer_token_confidence(logprobs: Any) -> dict[str, Any]:
    """Extract A-D token uncertainty after the generated Final Answer prefix."""
    if not isinstance(logprobs, dict) or not isinstance(logprobs.get("content"), list):
        return {"uncertainty_valid": False, "uncertainty_error": "missing_token_logprobs"}
    entries = logprobs["content"]
    text_parts: list[str] = []
    spans: list[tuple[int, int, dict[str, Any]]] = []
    cursor = 0
    for entry in entries:
        chunk = token_text(entry)
        text_parts.append(chunk)
        spans.append((cursor, cursor + len(chunk), entry))
        cursor += len(chunk)
    joined = "".join(text_parts)
    match = re.search(r"Final\s+Answer\s*:\s*([A-D])\b", joined, re.IGNORECASE)
    if not match:
        return {"uncertainty_valid": False, "uncertainty_error": "answer_token_not_found_in_logprobs"}
    letter_pos = match.start(1)
    chosen_entry = next((entry for start, end, entry in spans if start <= letter_pos < end), None)
    if chosen_entry is None:
        return {"uncertainty_valid": False, "uncertainty_error": "answer_token_alignment_failed"}
    letter_logs: dict[str, float] = {}
    for cand in chosen_entry.get("top_logprobs") or []:
        text = token_text(cand).strip().upper()
        if re.fullmatch(r"[A-D]", text):
            try:
                value = float(cand["logprob"])
            except (KeyError, TypeError, ValueError):
                continue
            letter_logs[text] = max(letter_logs.get(text, -math.inf), value)
    selected = match.group(1).upper()
    try:
        chosen_logprob = float(chosen_entry["logprob"])
        selected_text = token_text(chosen_entry).strip().upper()
        if selected_text == selected:
            letter_logs[selected] = max(letter_logs.get(selected, -math.inf), chosen_logprob)
    except (KeyError, TypeError, ValueError):
        chosen_logprob = None
    if len(letter_logs) < 2:
        return {"uncertainty_valid": False, "uncertainty_error": "fewer_than_two_option_logprobs", "selected_answer_logprob": chosen_logprob}
    peak = max(letter_logs.values())
    weights = {k: math.exp(v - peak) for k, v in letter_logs.items()}
    total = sum(weights.values())
    probs = {k: v / total for k, v in weights.items()}
    ordered = sorted(probs.values(), reverse=True)
    entropy = -sum(p * math.log(max(p, 1e-15)) for p in probs.values()) / math.log(len(probs))
    return {
        "uncertainty_valid": True,
        "uncertainty_error": None,
        "answer_option_probabilities": probs,
        "answer_entropy_normalized": entropy,
        "answer_probability_margin": ordered[0] - ordered[1],
        "selected_answer_logprob": chosen_logprob,
        "logprob_candidate_count": len(probs),
    }


async def health(session: aiohttp.ClientSession, url: str, alias: str) -> None:
    async with session.get(url + "/models") as response:
        response.raise_for_status()
        ids = [item.get("id") for item in (await response.json()).get("data", [])]
    if alias not in ids:
        raise RuntimeError(f"Expected {alias}; got {ids}")


async def infer(session: aiohttp.ClientSession, role: str, messages: list[dict[str, str]]) -> dict[str, Any]:
    is_base = role == "base"
    url, alias = (BASE_URL, BASE_ALIAS) if is_base else (EXPERT_URL, EXPERT_ALIAS)
    body: dict[str, Any] = {
        "model": alias,
        "messages": messages,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 64 if is_base else 128,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if is_base:
        body.update({"logprobs": True, "top_logprobs": 20})
    started = time.perf_counter()
    try:
        async with session.post(url + "/chat/completions", json=body) as response:
            response.raise_for_status()
            payload = await response.json()
        choice = payload["choices"][0]
        raw = str(choice.get("message", {}).get("content", ""))
        usage = payload.get("usage", {})
        if is_base:
            predicted = direct.parse_answer(raw)
            parse_error = None if predicted else "strict_base_answer_parse_failure"
            confidence = answer_token_confidence(choice.get("logprobs"))
        else:
            try:
                normalized = defer_helper.validate_expert_response(defer_helper.Action.DEFER, raw)
                match = re.fullmatch(r"Expert Final Answer:\s*([A-D])", normalized, re.IGNORECASE)
                predicted = match.group(1).upper() if match else None
                parse_error = None if predicted else "strict_expert_answer_parse_failure"
            except (ValueError, TypeError) as exc:
                predicted, parse_error = None, f"expert_protocol_parse_failure:{type(exc).__name__}"
            confidence = {}
        return {
            "predicted": predicted,
            "valid": predicted is not None,
            "parse_error": parse_error,
            "raw_response": raw,
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "latency_s": time.perf_counter() - started,
            "finish_reason": choice.get("finish_reason"),
            "response_sha256": sha_text(raw),
            **confidence,
        }
    except Exception as exc:
        return {
            "predicted": None, "valid": False,
            "parse_error": f"request_or_service_error:{type(exc).__name__}:{str(exc)[:160]}",
            "raw_response": None, "prompt_tokens": 0, "completion_tokens": 0,
            "latency_s": time.perf_counter() - started, "finish_reason": "error",
            "response_sha256": None,
            **({"uncertainty_valid": False, "uncertainty_error": "request_failed"} if is_base else {}),
        }


def build_case(frame: pd.DataFrame, spec: dict[str, Any], split: str) -> dict[str, Any]:
    ix = int(spec["source_row_index"])
    row = frame.iloc[ix].to_dict()
    digest, gold, facts, base_messages, defer_messages = l2d.case_key(row, ix)
    if digest != spec["case_hash"]:
        raise RuntimeError(f"Case hash changed after selection: {spec['case_id']}")
    ph = hashlib.sha256(json.dumps(base_messages, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    if ph != spec["prompt_hash"]:
        raise RuntimeError(f"Public prompt hash changed after selection: {spec['case_id']}")
    hidden = defer_helper.unrevealed_atomic_facts(
        defer_helper.public_state(row, ix)[0], facts
    )
    defer_helper.validate_expert_view(base_messages, atomic_facts=hidden, ground_truth=gold)
    defer_helper.validate_expert_view(defer_messages, atomic_facts=hidden, ground_truth=gold)
    return {
        "case_id": spec["case_id"], "case_hash": digest, "prompt_hash": ph,
        "source_row_index": ix, "split": split, "gold": gold,
        "base_messages": base_messages, "defer_messages": defer_messages,
        "features": public_features(base_messages),
    }


async def collect(cases: list[dict[str, Any]], outdir: Path, resume: bool) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    base_path, expert_path = outdir / "base_answer_trajectories.jsonl", outdir / "qwen32_defer_trajectories.jsonl"
    old_base = jsonl_rows(base_path) if resume else []
    old_expert = jsonl_rows(expert_path) if resume else []
    done_base = {r["case_hash"] for r in old_base}
    done_expert = {r["case_hash"] for r in old_expert}
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        await health(session, BASE_URL, BASE_ALIAS)
        await health(session, EXPERT_URL, EXPERT_ALIAS)
        for i, case in enumerate(cases, 1):
            if case["case_hash"] not in done_base:
                result = await infer(session, "base", case["base_messages"])
                row = {
                    "case_id": case["case_id"], "case_hash": case["case_hash"], "prompt_hash": case["prompt_hash"],
                    "split": case["split"], "model_role": "qwen3_8b_base_answer", "action": "ANSWER",
                    "predicted": result["predicted"], "correct": bool(result["predicted"] == case["gold"]),
                    "valid": result["valid"], "parse_error": result["parse_error"],
                    "raw_response": result["raw_response"], "response_sha256": result["response_sha256"],
                    "prompt_tokens": result["prompt_tokens"], "completion_tokens": result["completion_tokens"],
                    "latency_s": result["latency_s"], "finish_reason": result["finish_reason"],
                    "uncertainty_valid": result.get("uncertainty_valid", False),
                    "uncertainty_error": result.get("uncertainty_error"),
                    "answer_option_probabilities": result.get("answer_option_probabilities"),
                    "answer_entropy_normalized": result.get("answer_entropy_normalized"),
                    "answer_probability_margin": result.get("answer_probability_margin"),
                    "selected_answer_logprob": result.get("selected_answer_logprob"),
                    "logprob_candidate_count": result.get("logprob_candidate_count"),
                }
                append(base_path, row); old_base.append(row); done_base.add(case["case_hash"])
            if case["case_hash"] not in done_expert:
                result = await infer(session, "expert", case["defer_messages"])
                row = {
                    "case_id": case["case_id"], "case_hash": case["case_hash"], "prompt_hash": case["prompt_hash"],
                    "split": case["split"], "model_role": "qwen3_32b_awq_defer", "action": "DEFER",
                    "predicted": result["predicted"], "correct": bool(result["predicted"] == case["gold"]),
                    "valid": result["valid"], "parse_error": result["parse_error"],
                    "raw_response": result["raw_response"], "response_sha256": result["response_sha256"],
                    "prompt_tokens": result["prompt_tokens"], "completion_tokens": result["completion_tokens"],
                    "latency_s": result["latency_s"], "finish_reason": result["finish_reason"],
                }
                append(expert_path, row); old_expert.append(row); done_expert.add(case["case_hash"])
            if i % 10 == 0 or i == len(cases):
                print(f"{outdir.name}: paired cases {i}/{len(cases)}", flush=True)


def load_cases(smoke: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if smoke:
        dev_path = ROOT / "experiments/ea_atpo_v1_smoke/ea_atpo_v1_attempt09_stability/data/test.parquet"
        frame = pd.read_parquet(dev_path)
        cases: list[dict[str, Any]] = []
        for i in range(len(frame)):
            row = frame.iloc[i].to_dict()
            digest, _, _, _, _ = l2d.case_key(row, int(i))
            cases.append({"source_row_index": i, "case_hash": digest, "prompt_hash": hashlib.sha256(json.dumps(direct.build_messages(row), ensure_ascii=False, sort_keys=True).encode()).hexdigest(), "case_id": f"attempt09-smoke-row-{i}"})
        import random
        random.Random(SMOKE_SEED).shuffle(cases)
        cases = cases[: min(SMOKE_N, len(cases))]
        chosen = {r["case_hash"] for r in cases}
        built = []
        for spec in cases:
            case = build_case(frame, spec, "protocol_smoke")
            built.append(case)
        return built, {"smoke_data_sha256": sha_file(dev_path), "smoke_rows": len(frame), "smoke_selected": len(chosen), "smoke_seed": SMOKE_SEED}
    if not SELECTION.is_file():
        raise FileNotFoundError(f"Frozen case selection missing: {SELECTION}")
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    train_frame = pd.read_parquet(candidate.DATA_PATH.parent / "train.pilot.parquet")
    test_frame = pd.read_parquet(candidate.DATA_PATH)
    built = []
    for split in ("train", "validation"):
        built.extend(build_case(train_frame, spec, split) for spec in selection["rows"][split])
    built.extend(build_case(test_frame, spec, "test") for spec in selection["rows"]["test"])
    return built, {"selection_sha256": sha_file(SELECTION), "counts": selection["counts"], "source_data": selection["source_data"], "split_hashes": selection["split_hashes"]}


def write_manifest(cases: list[dict[str, Any]], meta: dict[str, Any], outdir: Path, smoke: bool) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "run_manifest.json"
    manifest = {
        "status": "ready_for_inference", "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experiment": "Qwen3-8B ANSWER paired with Qwen3-32B-AWQ DEFER; selective policy and case-isolated router evaluation",
        "smoke_only": smoke, "case_count": len(cases), "selection": meta,
        "models": {
            "base": {"repo_id": "Qwen/Qwen3-8B", "revision": BASE_REV, "alias": BASE_ALIAS, "endpoint": BASE_URL},
            "expert": {"repo_id": "Qwen/Qwen3-32B-AWQ", "revision": AWQ_REV, "alias": EXPERT_ALIAS, "endpoint": EXPERT_URL},
        },
        "service_manifests": {
            "base": str(OUT / "services/attempt03/base/service_manifest.json"),
            "expert": str(OUT / "services/attempt03/expert/service_manifest.json"),
        },
        "generation": {"temperature": 0.0, "top_p": 1.0, "enable_thinking": False, "base_max_tokens": 64, "expert_max_tokens": 128, "base_logprobs": True, "base_top_logprobs": 20},
        "predecision_features": ["public prompt lexical/length/option features", "Base answer option entropy and margin", "Base predicted answer one-hot"],
        "cost_grid_preregistered": [0.0, 0.05, 0.10, 0.20, 0.40, 0.60],
        "primary_defer_cost": 0.60,
        "utility_definition": "terminal_correctness - lambda_defer * defer_call_indicator; costs deducted once",
        "policy_fit_rule": "train cases fit coefficients; validation cases select thresholds and hyperparameters; test cases are not read by the fitting script",
        "oracle_policy": "post-hoc upper bound only; never used for fitting, threshold selection, or decisions",
        "raw_output_policy": "raw generated completions and service telemetry are stored; prompts, gold, and hidden atomic facts are not serialized",
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "runner_sha256": sha_file(Path(__file__).resolve()),
    }
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("selection") != meta or old.get("smoke_only") != smoke:
            raise RuntimeError("Run manifest exists with a different frozen selection; refusing overwrite")
        return
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


async def main_async(args: argparse.Namespace) -> None:
    cases, meta = load_cases(args.smoke)
    outdir = OUT / "smoke_20" if args.smoke else OUT / "paired_rollouts"
    write_manifest(cases, meta, outdir, args.smoke)
    await collect(cases, outdir, args.resume)
    # A compact joined trace used by fitting/analysis; contains no prompts or gold.
    base = {r["case_hash"]: r for r in jsonl_rows(outdir / "base_answer_trajectories.jsonl")}
    expert = {r["case_hash"]: r for r in jsonl_rows(outdir / "qwen32_defer_trajectories.jsonl")}
    rows = []
    for c in cases:
        b, e = base.get(c["case_hash"]), expert.get(c["case_hash"])
        if not b or not e:
            continue
        rows.append({
            "case_id": c["case_id"], "case_hash": c["case_hash"], "prompt_hash": c["prompt_hash"], "split": c["split"],
            "base_predicted": b.get("predicted"), "base_correct": bool(b.get("correct")), "base_valid": bool(b.get("valid")),
            "base_prompt_tokens": b.get("prompt_tokens", 0), "base_completion_tokens": b.get("completion_tokens", 0), "base_latency_s": b.get("latency_s", 0.0),
            "base_raw_sha256": b.get("response_sha256"), "base_uncertainty_valid": b.get("uncertainty_valid", False),
            "base_uncertainty_error": b.get("uncertainty_error"), "base_answer_option_probabilities": b.get("answer_option_probabilities"),
            "base_answer_entropy_normalized": b.get("answer_entropy_normalized"), "base_answer_probability_margin": b.get("answer_probability_margin"),
            "public_features": c["features"],
            "expert_predicted": e.get("predicted"), "expert_correct": bool(e.get("correct")), "expert_valid": bool(e.get("valid")),
            "expert_prompt_tokens": e.get("prompt_tokens", 0), "expert_completion_tokens": e.get("completion_tokens", 0), "expert_latency_s": e.get("latency_s", 0.0),
            "expert_parse_error": e.get("parse_error"), "expert_raw_sha256": e.get("response_sha256"),
        })
    joined = outdir / "paired_case_results.jsonl"
    with joined.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    for split in ("train", "validation", "test", "protocol_smoke"):
        split_rows = [r for r in rows if r["split"] == split]
        if not split_rows:
            continue
        split_path = outdir / f"paired_{split}.jsonl"
        with split_path.open("w", encoding="utf-8") as f:
            for row in split_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest_path = outdir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({
        "status": "complete" if len(rows) == len(cases) else "partial",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "paired_rows": len(rows), "paired_results_sha256": sha_file(joined),
        "split_result_sha256": {name: sha_file(outdir / f"paired_{name}.jsonl") for name in ("train", "validation", "test", "protocol_smoke") if (outdir / f"paired_{name}.jsonl").is_file()},
        "base_trajectories_sha256": sha_file(outdir / "base_answer_trajectories.jsonl"),
        "expert_trajectories_sha256": sha_file(outdir / "qwen32_defer_trajectories.jsonl"),
        "invalid_base": sum(not r["base_valid"] for r in rows), "invalid_expert": sum(not r["expert_valid"] for r in rows),
        "base_uncertainty_valid": sum(bool(r["base_uncertainty_valid"]) for r in rows),
    })
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "paired_rows": len(rows), "invalid_base": manifest["invalid_base"], "invalid_expert": manifest["invalid_expert"], "base_uncertainty_valid": manifest["base_uncertainty_valid"], "output": str(outdir)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run isolated 20-case protocol smoke")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
