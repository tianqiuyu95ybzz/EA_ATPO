#!/usr/bin/env python3
"""Fresh-case L2D DEFER comparison for Qwen3-32B-AWQ and ReMeDi."""

from __future__ import annotations

import argparse
import asyncio
import csv
import gc
import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

AUDIT_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(AUDIT_DIR))
import run_l2d_remedi_confirmation as helper  # noqa: E402

DATA_PATH = helper.DATA_PATH
DEV_PATH = helper.DEV_PATH
OUTPUT_DIR = AUDIT_DIR / "outputs/l2d_expert_screen_20261118_gpu6"
SEED = 20261118
CASE_COUNT = 100
MAX_NEW_TOKENS = 128
DEFER_COST = 0.60
BASE32_DIR = Path("/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/experiments/real_case_pilot_20261103/assets/models/Qwen3-32B-AWQ")
BASE32_LOCK = BASE32_DIR / "model_source_lock.json"
MODEL_ALIAS = "qwen3-32b-awq-l2d"
API_URL = "http://127.0.0.1:30145/v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_cases() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frame = pd.read_parquet(DATA_PATH)
    old_cases, _ = helper.build_cases()
    old_indices = {int(case["source_index"]) for case in old_cases}
    prior_indices = set(random.Random(helper.candidate_eval.SEED).sample(range(len(frame)), helper.candidate_eval.CASE_COUNT))
    dev_frame = pd.read_parquet(DEV_PATH)
    dev_hashes = {helper.case_key(row.to_dict(), int(idx))[0] for idx, row in dev_frame.iterrows()}
    order = list(range(len(frame)))
    random.Random(SEED).shuffle(order)
    chosen: list[dict[str, Any]] = []
    seen: set[str] = set()
    excluded_old = excluded_prior = excluded_dev = duplicate = 0
    for source_index in order:
        if source_index in prior_indices:
            excluded_prior += 1
            continue
        if source_index in old_indices:
            excluded_old += 1
            continue
        row = frame.iloc[source_index].to_dict()
        digest, gold, facts, answer_messages, defer_messages = helper.case_key(row, source_index)
        if digest in dev_hashes:
            excluded_dev += 1
            continue
        if digest in seen:
            duplicate += 1
            continue
        seen.add(digest)
        chosen.append({
            "source_index": source_index, "case_hash": digest, "gold": gold, "facts": facts,
            "answer_messages": answer_messages, "defer_messages": defer_messages,
        })
        if len(chosen) == CASE_COUNT:
            break
    if len(chosen) != CASE_COUNT:
        raise RuntimeError(f"only selected {len(chosen)} unique cases")
    selection = {
        "data_sha256": sha256_file(DATA_PATH), "data_rows": len(frame),
        "attempt09_dev_data_sha256": sha256_file(DEV_PATH), "attempt09_dev_rows": len(dev_frame),
        "previous_200_case_rows_excluded": len(prior_indices),
        "previous_remedi_confirmation_rows_excluded": len(old_indices),
        "attempt09_dev_overlaps_excluded": excluded_dev,
        "duplicate_public_states_skipped": duplicate,
        "seed": SEED, "case_count": CASE_COUNT,
        "case_hashes_sha256": sha256_text("\n".join(case["case_hash"] for case in chosen)),
        "source_indices_sha256": sha256_text("\n".join(str(case["source_index"]) for case in chosen)),
    }
    return chosen, selection


async def call_32b(session: aiohttp.ClientSession, messages: list[dict[str, str]]) -> dict[str, Any]:
    payload = {
        "model": MODEL_ALIAS,
        "messages": messages,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": MAX_NEW_TOKENS,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    started = time.perf_counter()
    async with session.post(f"{API_URL}/chat/completions", json=payload) as response:
        response.raise_for_status()
        body = await response.json()
    choice = body["choices"][0]
    raw = str(choice["message"].get("content", ""))
    try:
        prediction = helper.parse_defer(raw)
        parse_error = None if prediction else "strict_answer_parse_failure"
    except ValueError:
        prediction, parse_error = None, "defer_protocol_validation_failure"
    usage = body.get("usage", {})
    return {
        "predicted": prediction, "valid": prediction is not None, "parse_error": parse_error,
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
        "latency_s": time.perf_counter() - started,
        "finish_reason": choice.get("finish_reason"),
    }


def bootstrap_ci(values: list[float], seed: int, n_repeats: int = 10000) -> tuple[float, float]:
    if len(values) <= 1:
        return (values[0], values[0]) if values else (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    means = sorted(statistics.mean(rng.choices(values, k=n)) for _ in range(n_repeats))
    return means[int(.025 * n_repeats)], means[min(n_repeats - 1, int(.975 * n_repeats))]


def append_record(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()


async def evaluate_api(cases: list[dict[str, Any]], records: list[dict[str, Any]], done: set[tuple[str, str]], limit: int) -> None:
    role = "qwen3_32b_awq_defer"
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(f"{API_URL}/models") as response:
            response.raise_for_status()
            ids = [item.get("id") for item in (await response.json()).get("data", [])]
        if MODEL_ALIAS not in ids:
            raise RuntimeError(f"model alias {MODEL_ALIAS} not served: {ids}")
        for index, case in enumerate(cases[:limit], 1):
            key = (case["case_hash"], role)
            if key in done:
                continue
            try:
                result = await call_32b(session, case["defer_messages"])
            except Exception as exc:
                result = {"predicted": None, "valid": False, "parse_error": type(exc).__name__, "prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0.0, "finish_reason": "error"}
            record = {
                "case_hash": case["case_hash"], "model_role": role, "action": "DEFER",
                "predicted": result["predicted"], "correct": bool(result["predicted"] == case["gold"]),
                "valid": result["valid"], "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"], "latency_s": result["latency_s"],
                "finish_reason": result["finish_reason"], "parse_error": result["parse_error"],
            }
            append_record(OUTPUT_DIR / "trajectories.jsonl", record)
            records.append(record)
            done.add(key)
            if index % 5 == 0 or index == limit:
                print(f"{role}: {index}/{limit}", flush=True)


async def check_api_ready() -> None:
    timeout = aiohttp.ClientTimeout(total=10)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(f"{API_URL}/models") as response:
            response.raise_for_status()
            ids = [item.get("id") for item in (await response.json()).get("data", [])]
    if MODEL_ALIAS not in ids:
        raise RuntimeError(f"model alias {MODEL_ALIAS} not served: {ids}")


def summarize(cases: list[dict[str, Any]], records: list[dict[str, Any]], manifest: dict[str, Any], limit: int) -> None:
    roles = ("always_answer_qwen3_8b", "base_qwen3_8b_defer", "qwen3_4b_remedi_defer", "qwen3_32b_awq_defer")
    by_case: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        by_case[record["case_hash"]][record["model_role"]] = record
    expected = [case["case_hash"] for case in cases[:limit]]
    complete = [case_hash for case_hash in expected if all(role in by_case[case_hash] for role in roles)]
    summary: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    base_role = roles[0]
    for case_hash in complete:
        values = by_case[case_hash]
        for role in roles[1:]:
            paired.append({
                "case_hash": case_hash, "comparison": role,
                "base_correct": bool(values[base_role]["correct"]),
                "candidate_correct": bool(values[role]["correct"]),
                "paired_delta": int(bool(values[role]["correct"])) - int(bool(values[base_role]["correct"])),
            })
    for role_index, role in enumerate(roles):
        rows = [by_case[case_hash][role] for case_hash in complete]
        outcomes = [float(row["correct"]) for row in rows]
        low, high = bootstrap_ci(outcomes, SEED + role_index)
        fields = {
            "accuracy": statistics.mean(outcomes) if outcomes else "",
            "valid_rate": statistics.mean(float(row["valid"]) for row in rows) if rows else "",
            "mean_total_tokens": statistics.mean(row["prompt_tokens"] + row["completion_tokens"] for row in rows) if rows else "",
            "mean_latency_s": statistics.mean(row["latency_s"] for row in rows) if rows else "",
            "net_value_cost_0_60": (statistics.mean(outcomes) - (0 if role == base_role else DEFER_COST)) if rows else "",
        }
        for metric, value in fields.items():
            summary.append({"metric": metric, "arm": role, "value": value,
                            "ci95_low": low if metric == "accuracy" else "",
                            "ci95_high": high if metric == "accuracy" else "", "n_cases": len(rows)})
    for role_index, role in enumerate(roles[1:], 1):
        deltas = [float(by_case[key][role]["correct"]) - float(by_case[key][base_role]["correct"]) for key in complete]
        low, high = bootstrap_ci(deltas, SEED + 20 + role_index)
        base_errors = [key for key in complete if not by_case[key][base_role]["correct"]]
        base_correct = [key for key in complete if by_case[key][base_role]["correct"]]
        summary.extend([
            {"metric": "paired_delta_vs_always_answer", "arm": role, "value": statistics.mean(deltas) if deltas else "", "ci95_low": low, "ci95_high": high, "n_cases": len(complete)},
            {"metric": "correct_on_always_answer_errors", "arm": role, "value": sum(bool(by_case[key][role]["correct"]) for key in base_errors), "ci95_low": "", "ci95_high": "", "n_cases": len(base_errors)},
            {"metric": "wrong_on_always_answer_correct", "arm": role, "value": sum(not bool(by_case[key][role]["correct"]) for key in base_correct), "ci95_low": "", "ci95_high": "", "n_cases": len(base_correct)},
        ])
    with (OUTPUT_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("metric", "arm", "value", "ci95_low", "ci95_high", "n_cases"))
        writer.writeheader(); writer.writerows(summary)
    with (OUTPUT_DIR / "paired_case_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("case_hash", "comparison", "base_correct", "candidate_correct", "paired_delta"))
        writer.writeheader(); writer.writerows(paired)
    manifest.update({
        "status": "completed" if len(complete) == CASE_COUNT else "partial",
        "complete_cases": len(complete), "complete_case_hashes_sha256": sha256_text("\n".join(complete)),
        "trajectory_sha256": sha256_file(OUTPUT_DIR / "trajectories.jsonl"),
        "summary_sha256": sha256_file(OUTPUT_DIR / "summary.csv"),
        "paired_case_results_sha256": sha256_file(OUTPUT_DIR / "paired_case_results.csv"),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    (OUTPUT_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-cases", type=int, default=CASE_COUNT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.limit_cases <= CASE_COUNT:
        raise ValueError("limit must be between 1 and 100")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "6":
        raise RuntimeError("the fresh-case local runs are pinned to physical GPU6")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(args.threads)
    torch.cuda.set_device(0)
    free, total = torch.cuda.mem_get_info()
    if free < 45 * 1024**3:
        raise RuntimeError(f"GPU6 safety gate: only {free / 1024**3:.1f} GiB free")
    cases, selection = load_cases()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trajectory_path = OUTPUT_DIR / "trajectories.jsonl"
    manifest_path = OUTPUT_DIR / "run_manifest.json"
    models = [
        {"role": "always_answer_qwen3_8b", "repo_id": "Qwen/Qwen3-8B", "revision": "b968826d9c46dd6066d109eabc6255188de91218", "path": str(helper.BASE_MODEL), "runtime": "Transformers", "action": "ANSWER"},
        {"role": "base_qwen3_8b_defer", "repo_id": "Qwen/Qwen3-8B", "revision": "b968826d9c46dd6066d109eabc6255188de91218", "path": str(helper.BASE_MODEL), "runtime": "Transformers", "action": "DEFER"},
        {"role": "qwen3_4b_remedi_defer", "repo_id": "VioletAmethyst/Qwen3-4B-ReMeDi", "revision": "dfa418ca1c3c0f8879afd6d356b88083f01ca032", "path": str(helper.REMEDI_SPEC["path"]), "runtime": "Transformers", "action": "DEFER"},
        {"role": "qwen3_32b_awq_defer", "repo_id": "Qwen/Qwen3-32B-AWQ", "revision": "0499c3ac83fdef8810b907a23894ba91e95eddd8", "path": str(BASE32_DIR), "source_lock_sha256": sha256_file(BASE32_LOCK), "runtime": "vLLM API", "action": "DEFER", "service_manifest": str(OUTPUT_DIR / "model_service_manifest.json")},
    ]
    prior_manifest_path = AUDIT_DIR / "outputs/medical_expert_candidates_20261116/run_manifest.json"
    prior_models = {}
    if prior_manifest_path.is_file():
        prior_models = {model.get("role"): model for model in json.loads(prior_manifest_path.read_text(encoding="utf-8")).get("models", [])}
    for model in models[:3]:
        model_path = Path(model["path"])
        role_key = "base_qwen3_8b" if model["repo_id"] == "Qwen/Qwen3-8B" else "qwen3_4b_remedi"
        model["config_sha256"] = sha256_file(model_path / "config.json")
        model["safetensors_index_sha256"] = sha256_file(model_path / "model.safetensors.index.json")
        model["weight_shards_sha256"] = prior_models.get(role_key, {}).get("weight_shards_sha256", {})
    model32_config = BASE32_DIR / "config.json"
    model32_index = BASE32_DIR / "model.safetensors.index.json"
    models[3]["config_sha256"] = sha256_file(model32_config)
    models[3]["safetensors_index_sha256"] = sha256_file(model32_index)
    manifest = {
        "status": "running", "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experiment": "fresh 100-case L2D DEFER candidate screen: Qwen3-8B, ReMeDi and Qwen3-32B-AWQ; no policy training",
        "selection": selection, "case_count": CASE_COUNT,
        "models": models, "generation": {"greedy": True, "max_new_tokens": MAX_NEW_TOKENS, "dtype_local": "bfloat16", "qwen3_enable_thinking": False},
        "runtime": {"torch": torch.__version__, "transformers": __import__("transformers").__version__, "gpu_global_id": 6, "gpu_name": torch.cuda.get_device_name(0), "gpu_free_at_runner_start_bytes": free, "gpu_total_bytes": total, "threads": args.threads},
        "gold_policy": "Gold and hidden atomic facts remain evaluator-only; only prediction and correctness indicators are serialized.",
        "defer_cost": DEFER_COST, "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
    }
    if args.resume:
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("selection", "models", "generation", "case_count"):
            if old.get(key) != manifest.get(key):
                raise RuntimeError(f"resume mismatch in {key}")
        manifest = old
    elif trajectory_path.exists() or manifest_path.exists():
        raise FileExistsError("output exists; pass --resume")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    asyncio.run(check_api_ready())
    records = [json.loads(line) for line in trajectory_path.read_text(encoding="utf-8").splitlines() if line.strip()] if args.resume and trajectory_path.exists() else []
    done = {(record["case_hash"], record["model_role"]) for record in records}
    local_specs = [
        ("always_answer_qwen3_8b", helper.BASE_MODEL, "answer_messages", "always_answer"),
        ("base_qwen3_8b_defer", helper.BASE_MODEL, "defer_messages", "DEFER"),
        ("qwen3_4b_remedi_defer", helper.REMEDI_SPEC["path"], "defer_messages", "DEFER"),
    ]
    for role, model_path, message_key, action in local_specs:
        pending = [case for case in cases[:args.limit_cases] if (case["case_hash"], role) not in done]
        if not pending:
            continue
        print(f"loading {role} on GPU6; pending={len(pending)}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="eager").to("cuda:0").eval()
        for index, case in enumerate(cases[:args.limit_cases], 1):
            key = (case["case_hash"], role)
            if key in done:
                continue
            result = helper.infer(model, tokenizer, case[message_key], action)
            record = {
                "case_hash": case["case_hash"], "model_role": role, "action": "ANSWER" if action == "always_answer" else "DEFER",
                "predicted": result["predicted"], "correct": bool(result["predicted"] == case["gold"]), "valid": result["valid"],
                "prompt_tokens": result["prompt_tokens"], "completion_tokens": result["completion_tokens"],
                "latency_s": result["latency_s"], "finish_reason": result["finish_reason"], "parse_error": result["error"],
            }
            append_record(trajectory_path, record)
            records.append(record); done.add(key)
            if index % 5 == 0 or index == args.limit_cases: print(f"{role}: {index}/{args.limit_cases}", flush=True)
        del model, tokenizer
        gc.collect(); torch.cuda.empty_cache()
    asyncio.run(evaluate_api(cases, records, done, args.limit_cases))
    summarize(cases, records, manifest, args.limit_cases)
    print(json.dumps({"status": manifest["status"], "complete_cases": manifest["complete_cases"], "output_dir": str(OUTPUT_DIR)}, indent=2))


if __name__ == "__main__":
    main()
