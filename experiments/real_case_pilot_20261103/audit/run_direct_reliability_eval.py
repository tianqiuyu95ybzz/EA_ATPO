#!/usr/bin/env python3
"""Paired Qwen3-8B/Qwen3-32B direct-answer evaluation on the held-out ATPO test split."""

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
import pandas as pd


AUDIT_DIR = Path(__file__).resolve().parent
PILOT_DIR = AUDIT_DIR.parent
DATA_PATH = PILOT_DIR / "assets/test.pilot.parquet"
SEED = 20261104
CASE_COUNT = 200
SYSTEM = (
    "You are answering a medical multiple-choice benchmark question. Use only the case and options "
    "in the user message. Select the single best option. Do not ask follow-up questions. "
    "Output exactly: Final Answer: <one uppercase option letter A-D>. Do not include an explanation."
)
MODELS = (
    {"role": "base_8b", "alias": "qwen3-8b-audit", "base_url": "http://127.0.0.1:30131/v1"},
    {"role": "candidate_expert_32b", "alias": "qwen3-32b-candidate", "base_url": "http://127.0.0.1:30132/v1"},
)


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def to_plain(value: Any) -> Any:
    return value.tolist() if hasattr(value, "tolist") else value


def build_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    raw = row.get("prompt")
    if raw is None:
        raw = row.get("extra_info", {}).get("prompt")
    raw = to_plain(raw)
    if not isinstance(raw, list):
        raise ValueError(f"test row {row.get('index')} lacks a message list")
    messages = [{"role": "system", "content": SYSTEM}]
    for item in raw:
        role = str(item.get("role", ""))
        if role == "system":
            continue
        if role not in ("user", "assistant"):
            raise ValueError(f"unexpected public prompt role {role!r}")
        messages.append({"role": role, "content": str(item.get("content", ""))})
    if len(messages) < 2 or not any(m["role"] == "user" for m in messages):
        raise ValueError(f"test row {row.get('index')} has no user case content")
    return messages


def answer_letter(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("answer", "")).strip().upper()
    return str(value).strip().upper()


def parse_answer(content: str) -> str | None:
    matches = re.findall(r"Final\s+Answer\s*:\s*([A-D])\b", content, re.IGNORECASE)
    if len(matches) == 1:
        return matches[0].upper()
    if not matches:
        single = re.fullmatch(r"\s*([A-D])\s*", content, re.IGNORECASE)
        if single:
            return single.group(1).upper()
    return None


def bootstrap_ci(values: list[float], seed: int, repeats: int = 10000) -> tuple[float, float]:
    if len(values) < 2:
        value = values[0] if values else float("nan")
        return value, value
    rng = random.Random(seed)
    n = len(values)
    means = sorted(statistics.mean(rng.choices(values, k=n)) for _ in range(repeats))
    return means[int(.025 * repeats)], means[min(repeats - 1, int(.975 * repeats))]


async def call_model(session: aiohttp.ClientSession, model: dict[str, str], messages: list[dict[str, str]]) -> dict[str, Any]:
    payload = {
        "model": model["alias"],
        "messages": messages,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 64,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    url = model["base_url"].rstrip("/") + "/chat/completions"
    last_error = None
    for attempt in range(3):
        started = time.perf_counter()
        try:
            async with session.post(url, json=payload) as response:
                response.raise_for_status()
                body = await response.json()
            content = str(body["choices"][0]["message"].get("content", ""))
            usage = body.get("usage") or {}
            return {
                "raw_response": content,
                "predicted": parse_answer(content),
                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "completion_tokens": int(usage.get("completion_tokens", 0)),
                "latency_s": time.perf_counter() - started,
                "error": None,
            }
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
    return {"raw_response": "", "predicted": None, "prompt_tokens": 0, "completion_tokens": 0, "latency_s": time.perf_counter() - started, "error": last_error or "request_failed"}


def write_summary(records: list[dict[str, Any]], output_dir: Path, manifest: dict[str, Any], expected_roles: list[str]) -> None:
    by_case: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for x in records:
        by_case[str(x["case_id"])][str(x["model_role"])] = x
    cases = sorted(by_case)
    base_role = "base_8b"
    expert_role = "candidate_expert_32b"
    score_lists = {role: [float(by_case[c][role].get("correct", False)) for c in cases] for role in expected_roles}
    rng_seed = SEED + 30
    summary = []
    for role in expected_roles:
        vals = score_lists[role]
        low, high = bootstrap_ci(vals, rng_seed)
        role_rows = [by_case[c][role] for c in cases]
        summary.append({
            "metric": "accuracy", "group": role, "value": statistics.mean(vals), "ci95_low": low, "ci95_high": high,
            "n_cases": len(cases), "valid_rate": statistics.mean(float(bool(r.get("valid"))) for r in role_rows),
            "mean_total_tokens": statistics.mean(r.get("prompt_tokens", 0) + r.get("completion_tokens", 0) for r in role_rows),
            "mean_latency_s": statistics.mean(float(r.get("latency_s", 0)) for r in role_rows),
            "notes": "Case-level bootstrap; parse failures/API failures score 0 and remain in denominator."
        })
    base_error_cases = [c for c in cases if not by_case[c][base_role].get("correct", False)]
    if expert_role in expected_roles:
        paired_delta = [float(by_case[c][expert_role].get("correct", False)) - float(by_case[c][base_role].get("correct", False)) for c in cases]
        low, high = bootstrap_ci(paired_delta, rng_seed + 1)
        summary.append({"metric":"paired_accuracy_delta_candidate_minus_base","group":"candidate_expert_32b-base_8b","value":statistics.mean(paired_delta),"ci95_low":low,"ci95_high":high,"n_cases":len(cases),"valid_rate":"","mean_total_tokens":"","mean_latency_s":"","notes":"Paired case bootstrap; delta of correctness indicators."})
        conditional = [float(by_case[c][expert_role].get("correct", False)) for c in base_error_cases]
        low, high = bootstrap_ci(conditional, rng_seed + 2)
        summary.append({"metric":"candidate_accuracy_given_base_error","group":"candidate_expert_32b_on_base_8b_errors","value":statistics.mean(conditional) if conditional else "","ci95_low":low,"ci95_high":high,"n_cases":len(base_error_cases),"valid_rate":"","mean_total_tokens":"","mean_latency_s":"","notes":"Denominator is all sampled cases where base Qwen3-8B answered incorrectly."})
        corrected = sum(bool(by_case[c][expert_role].get("correct")) for c in base_error_cases)
        base_correct_cases = [c for c in cases if by_case[c][base_role].get("correct", False)]
        degraded = sum(not bool(by_case[c][expert_role].get("correct")) for c in base_correct_cases)
        summary.extend([
            {"metric":"candidate_correct_on_base_errors","group":"base_error_correction","value":corrected,"ci95_low":"","ci95_high":"","n_cases":len(base_error_cases),"valid_rate":"","mean_total_tokens":"","mean_latency_s":"","notes":"Candidate correct among baseline errors."},
            {"metric":"candidate_wrong_on_base_correct","group":"base_correct_degradation","value":degraded,"ci95_low":"","ci95_high":"","n_cases":len(base_correct_cases),"valid_rate":"","mean_total_tokens":"","mean_latency_s":"","notes":"Candidate wrong among baseline successes."},
        ])
    with (output_dir / "direct_reliability_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=("metric","group","value","ci95_low","ci95_high","n_cases","valid_rate","mean_total_tokens","mean_latency_s","notes"))
        writer.writeheader()
        writer.writerows(summary)
    manifest["status"] = "completed" if expert_role in expected_roles else "baseline_only_candidate_unavailable"
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest["evaluated_cases"] = len(cases)
    manifest["base_error_count"] = len(base_error_cases)
    manifest["paired_accuracy_delta"] = statistics.mean(paired_delta) if expert_role in expected_roles else None
    manifest["candidate_accuracy_given_base_error"] = statistics.mean(conditional) if expert_role in expected_roles and conditional else None
    manifest["trajectory_sha256"] = file_sha(output_dir / "direct_reliability_trajectories.jsonl")
    (output_dir / "expert_reliability_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


async def main_async(args: argparse.Namespace) -> None:
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(args.data)
    if len(frame) < args.cases:
        raise ValueError(f"requested {args.cases}, only {len(frame)} test rows available")
    indices = random.Random(args.seed).sample(range(len(frame)), args.cases)
    selected = []
    for source_i in indices:
        row = frame.iloc[source_i].to_dict()
        messages = build_messages(row)
        gold = answer_letter(row.get("ground_truth") or row.get("reward_model", {}).get("ground_truth"))
        if gold not in "ABCD":
            raise ValueError(f"invalid gold at test row {source_i}: {gold!r}")
        prompt_digest = hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        selected.append({"source_row_index": int(source_i), "case_id": f"test-row-{source_i}:item-{row.get('index', source_i)}", "gold": gold, "messages": messages, "prompt_sha256": prompt_digest})
    run_manifest_path = out / "expert_reliability_manifest.json"
    models=list(MODELS)
    if args.base_only:
        models=[{**MODELS[0],"alias":args.base_alias,"base_url":args.base_url}]
    manifest = {
        "status":"running", "created_at_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experiment":"paired direct-answer accuracy comparison on pinned held-out ATPO test split",
        "data_path":str(args.data.resolve()), "data_sha256":file_sha(args.data), "data_rows":len(frame),
        "case_count":args.cases, "sample_seed":args.seed, "sampled_source_rows":indices,
        "pilot_train_sample_disjoint_by_split":True,
        "models":models, "generation":{"temperature":0.0,"top_p":1.0,"max_tokens":64,"enable_thinking":False,"system_prompt_sha256":hashlib.sha256(SYSTEM.encode()).hexdigest()},
        "scoring":"exact single option letter; invalid format or API failure scores 0",
        "gold_context_policy":"Gold and atomic_facts are retained only in evaluator memory and are not included in API messages.",
        "trajectory_path":str((out/"direct_reliability_trajectories.jsonl").resolve()),
    }
    if run_manifest_path.exists() and args.resume:
        old=json.loads(run_manifest_path.read_text())
        for key in ("data_sha256","sampled_source_rows","models","generation"):
            if old.get(key)!=manifest.get(key): raise ValueError(f"resume mismatch in {key}")
    elif (out / "direct_reliability_trajectories.jsonl").exists():
        raise FileExistsError("trajectory output exists; pass --resume or choose a fresh output directory")
    run_manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf-8")
    trajectory_path=out/"direct_reliability_trajectories.jsonl"
    done=set()
    records=[]
    if args.resume and trajectory_path.exists():
        for line in trajectory_path.open(encoding="utf-8"):
            if line.strip():
                x=json.loads(line); records.append(x); done.add((x["case_id"],x["model_role"]))
    timeout=aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for model in models:
            health_url=model["base_url"].rstrip("/")+"/models"
            async with session.get(health_url) as response:
                response.raise_for_status()
                ids=[str(x.get("id")) for x in (await response.json()).get("data",[])]
            if model["alias"] not in ids:
                raise RuntimeError(f"expected alias {model['alias']} missing from {health_url}: {ids}")
            for item in selected:
                key=(item["case_id"],model["role"])
                if key in done: continue
                result=await call_model(session,model,item["messages"])
                record={
                    "record_type":"direct_answer_evaluation", "case_id":item["case_id"],
                    "source_row_index":item["source_row_index"], "model_role":model["role"],
                    "model_alias":model["alias"], "prompt_sha256":item["prompt_sha256"],
                    "gold":item["gold"], "predicted":result["predicted"],
                    "correct":bool(result["predicted"]==item["gold"]),
                    "valid":result["predicted"] is not None and result["error"] is None,
                    "raw_response":result["raw_response"], "error":result["error"],
                    "prompt_tokens":result["prompt_tokens"], "completion_tokens":result["completion_tokens"],
                    "latency_s":result["latency_s"],
                }
                with trajectory_path.open("a",encoding="utf-8") as f:
                    f.write(json.dumps(record,ensure_ascii=False)+"\n"); f.flush()
                records.append(record); done.add(key)
                if len(done)%20==0:
                    print(f"completed model-case pairs: {len(done)}/{2*len(selected)}",flush=True)
    if len({(x["case_id"],x["model_role"]) for x in records}) != len(models)*len(selected):
        raise RuntimeError("incomplete paired output; resume after resolving the service error")
    write_summary(records,out,manifest,[x["role"] for x in models])
    print(json.dumps({"status":manifest["status"],"case_count":len(selected),"rows":len(records),"base_errors":manifest["base_error_count"],"candidate_accuracy_given_base_error":manifest["candidate_accuracy_given_base_error"]},indent=2))


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--data",type=Path,default=DATA_PATH)
    parser.add_argument("--output-dir",type=Path,default=AUDIT_DIR/"outputs")
    parser.add_argument("--cases",type=int,default=CASE_COUNT)
    parser.add_argument("--seed",type=int,default=SEED)
    parser.add_argument("--base-only",action="store_true",help="Run the held-out baseline only and mark candidate comparison unavailable.")
    parser.add_argument("--base-alias",default=MODELS[0]["alias"])
    parser.add_argument("--base-url",default=MODELS[0]["base_url"])
    parser.add_argument("--resume",action="store_true")
    asyncio.run(main_async(parser.parse_args()))


if __name__=="__main__": main()
