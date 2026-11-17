#!/usr/bin/env python3
"""Fresh 100-case DEFER comparison for previously shortlisted models."""

from __future__ import annotations

import argparse
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

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

AUDIT_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(AUDIT_DIR))
import run_l2d_expert_screen_32b as sample_builder  # noqa: E402
import run_l2d_remedi_confirmation as helper  # noqa: E402

OUTPUT_DIR = AUDIT_DIR / "outputs/l2d_prior_candidates_defer_20261118_gpu5"
CASE_COUNT = 100
MAX_NEW_TOKENS = 128
DEFER_COST = 0.60
SEED = 20261118


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def render_input(tokenizer: Any, messages: list[dict[str, str]], model_kind: str) -> dict[str, torch.Tensor]:
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_tensors": "pt",
    }
    if model_kind == "qwen3":
        kwargs["enable_thinking"] = False
    rendered = tokenizer.apply_chat_template(messages, **kwargs)
    if hasattr(rendered, "items"):
        tensors = {key: value for key, value in rendered.items() if isinstance(value, torch.Tensor)}
        if tensors:
            if "attention_mask" not in tensors:
                tensors["attention_mask"] = torch.ones_like(tensors["input_ids"])
            return tensors
    input_ids = rendered if isinstance(rendered, torch.Tensor) else torch.as_tensor(rendered)
    return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}


def infer(model: Any, tokenizer: Any, messages: list[dict[str, str]], model_kind: str, direct_answer: bool) -> dict[str, Any]:
    inputs = render_input(tokenizer, messages, model_kind)
    inputs = {key: value.to("cuda:0") for key, value in inputs.items()}
    prompt_tokens = int(inputs["input_ids"].shape[-1])
    generation: dict[str, Any] = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "num_beams": 1,
        "use_cache": True,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 50,
    }
    if tokenizer.eos_token_id is not None:
        generation["eos_token_id"] = tokenizer.eos_token_id
        generation["pad_token_id"] = tokenizer.eos_token_id
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, **generation)
    torch.cuda.synchronize()
    completion = output[0, prompt_tokens:]
    raw = tokenizer.decode(completion, skip_special_tokens=False)
    try:
        predicted = helper.direct.parse_answer(raw) if direct_answer else helper.parse_defer(raw)
        error = None if predicted else "strict_answer_parse_failure"
    except ValueError:
        predicted, error = None, "defer_protocol_validation_failure"
    ended = bool(tokenizer.eos_token_id is not None and completion.numel() and int(completion[-1]) == int(tokenizer.eos_token_id))
    return {
        "predicted": predicted,
        "valid": predicted is not None,
        "parse_error": error,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": int(completion.numel()),
        "latency_s": time.perf_counter() - started,
        "finish_reason": "stop" if ended else "length_or_model_stop",
    }


def bootstrap_ci(values: list[float], seed: int, repeats: int = 10000) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    n = len(values)
    means = sorted(statistics.mean(rng.choices(values, k=n)) for _ in range(repeats))
    return means[int(.025 * repeats)], means[min(repeats - 1, int(.975 * repeats))]


def write_record(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()


def summarize(cases: list[dict[str, Any]], records: list[dict[str, Any]], manifest: dict[str, Any], limit: int) -> None:
    roles = (
        "always_answer_qwen3_8b",
        "base_qwen3_8b_defer",
        "qwen3_4b_remedi_defer",
        "popit_qwen3_4b_medical_sft_defer",
        "huatuo_gpt_o1_7b_defer",
    )
    by_case: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        by_case[record["case_hash"]][record["model_role"]] = record
    expected = [case["case_hash"] for case in cases[:limit]]
    complete = [key for key in expected if all(role in by_case[key] for role in roles)]
    rows: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    for case_hash in complete:
        base = by_case[case_hash][roles[0]]
        for role in roles[1:]:
            candidate = by_case[case_hash][role]
            paired.append({
                "case_hash": case_hash,
                "comparison": role,
                "base_correct": bool(base["correct"]),
                "candidate_correct": bool(candidate["correct"]),
                "paired_delta": int(bool(candidate["correct"])) - int(bool(base["correct"])),
            })
    for role_index, role in enumerate(roles):
        arm_rows = [by_case[key][role] for key in complete]
        correct = [float(record["correct"]) for record in arm_rows]
        low, high = bootstrap_ci(correct, SEED + role_index)
        values = {
            "accuracy": statistics.mean(correct) if correct else "",
            "valid_rate": statistics.mean(float(record["valid"]) for record in arm_rows) if arm_rows else "",
            "mean_total_tokens": statistics.mean(record["prompt_tokens"] + record["completion_tokens"] for record in arm_rows) if arm_rows else "",
            "mean_latency_s": statistics.mean(record["latency_s"] for record in arm_rows) if arm_rows else "",
            "net_value_cost_0_60": statistics.mean(correct) - (0.0 if role == roles[0] else DEFER_COST) if arm_rows else "",
        }
        for metric, value in values.items():
            rows.append({"metric": metric, "arm": role, "value": value,
                         "ci95_low": low if metric == "accuracy" else "",
                         "ci95_high": high if metric == "accuracy" else "", "n_cases": len(arm_rows)})
    for role_index, role in enumerate(roles[1:], 1):
        deltas = [float(by_case[key][role]["correct"]) - float(by_case[key][roles[0]]["correct"]) for key in complete]
        low, high = bootstrap_ci(deltas, SEED + 20 + role_index)
        base_errors = [key for key in complete if not by_case[key][roles[0]]["correct"]]
        base_correct = [key for key in complete if by_case[key][roles[0]]["correct"]]
        correction = [float(by_case[key][role]["correct"]) for key in base_errors]
        degradation = [float(not by_case[key][role]["correct"]) for key in base_correct]
        correction_ci = bootstrap_ci(correction, SEED + 30 + role_index)
        degradation_ci = bootstrap_ci(degradation, SEED + 40 + role_index)
        rows.extend([
            {"metric": "paired_delta_vs_always_answer", "arm": role, "value": statistics.mean(deltas) if deltas else "", "ci95_low": low, "ci95_high": high, "n_cases": len(deltas)},
            {"metric": "correct_on_always_answer_errors", "arm": role, "value": sum(correction), "ci95_low": correction_ci[0], "ci95_high": correction_ci[1], "n_cases": len(correction)},
            {"metric": "wrong_on_always_answer_correct", "arm": role, "value": sum(degradation), "ci95_low": degradation_ci[0], "ci95_high": degradation_ci[1], "n_cases": len(degradation)},
        ])
    with (OUTPUT_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("metric", "arm", "value", "ci95_low", "ci95_high", "n_cases"))
        writer.writeheader(); writer.writerows(rows)
    with (OUTPUT_DIR / "paired_case_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("case_hash", "comparison", "base_correct", "candidate_correct", "paired_delta"))
        writer.writeheader(); writer.writerows(paired)
    manifest.update({
        "status": "completed" if len(complete) == CASE_COUNT else "partial",
        "complete_cases": len(complete),
        "complete_case_hashes_sha256": sha256_text("\n".join(complete)),
        "trajectory_sha256": sha256_file(OUTPUT_DIR / "trajectories.jsonl"),
        "summary_sha256": sha256_file(OUTPUT_DIR / "summary.csv"),
        "paired_case_results_sha256": sha256_file(OUTPUT_DIR / "paired_case_results.csv"),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    (OUTPUT_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    global OUTPUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-cases", type=int, default=CASE_COUNT)
    parser.add_argument("--gpu-index", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--min-free-gib", type=float, default=80.0)
    args = parser.parse_args()
    if not 1 <= args.limit_cases <= CASE_COUNT:
        raise ValueError("limit must be in [1, 100]")
    if args.gpu_index in (1, 7):
        raise RuntimeError("GPU1 and GPU7 are reserved and must not be used")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu_index):
        raise RuntimeError(f"set CUDA_VISIBLE_DEVICES={args.gpu_index}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    OUTPUT_DIR = args.output_dir.resolve()
    torch.set_num_threads(args.threads)
    torch.cuda.set_device(0)
    free, total = torch.cuda.mem_get_info()
    if free < args.min_free_gib * 1024**3:
        raise RuntimeError(f"GPU{args.gpu_index} safety gate: only {free / 1024**3:.1f} GiB free")

    cases, selection = sample_builder.load_cases()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trajectory = OUTPUT_DIR / "trajectories.jsonl"
    manifest_path = OUTPUT_DIR / "run_manifest.json"
    model_specs = [
        {"role": "always_answer_qwen3_8b", "name": "Qwen3-8B Always Answer", "path": str(helper.BASE_MODEL), "kind": "qwen3", "action": "ANSWER", "repo_id": "Qwen/Qwen3-8B", "revision": "b968826d9c46dd6066d109eabc6255188de91218"},
        {"role": "base_qwen3_8b_defer", "name": "Qwen3-8B DEFER", "path": str(helper.BASE_MODEL), "kind": "qwen3", "action": "DEFER", "repo_id": "Qwen/Qwen3-8B", "revision": "b968826d9c46dd6066d109eabc6255188de91218"},
    ]
    for role in ("qwen3_4b_remedi", "popit_qwen3_4b_medical_sft", "huatuo_gpt_o1_7b"):
        spec = next(item for item in helper.candidate_eval.MODEL_SPECS if item["role"] == role)
        model_specs.append({
            "role": f"{role}_defer", "name": spec["name"] + " DEFER", "path": str(spec["path"]),
            "kind": spec["kind"], "action": "DEFER", "repo_id": spec["repo_id"], "revision": spec["revision"],
        })
    for spec in model_specs:
        model_path = Path(spec["path"])
        if not (model_path / "config.json").is_file() or not (model_path / "model.safetensors.index.json").is_file():
            raise FileNotFoundError(f"incomplete local model snapshot: {spec['name']}")
        spec["config_sha256"] = sha256_file(model_path / "config.json")
        spec["safetensors_index_sha256"] = sha256_file(model_path / "model.safetensors.index.json")
    manifest = {
        "status": "running", "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experiment": "fresh 100-case DEFER comparison of previously shortlisted public medical candidates; no training",
        "selection": selection, "case_count": CASE_COUNT, "models": model_specs,
        "generation": {"greedy": True, "max_new_tokens": MAX_NEW_TOKENS, "dtype": "bfloat16", "qwen3_enable_thinking": False},
        "runtime": {"torch": torch.__version__, "transformers": __import__("transformers").__version__, "gpu_global_id": args.gpu_index, "gpu_name": torch.cuda.get_device_name(0), "initial_free_memory_bytes": free, "total_memory_bytes": total, "threads": args.threads, "minimum_free_gib_before_each_model": args.min_free_gib},
        "gold_policy": "Gold and hidden facts remain evaluator-only. They are never included in messages and are not serialized.",
        "scoring": "strict direct-answer parser for Always Answer; exact DEFER protocol for all expert models; parse failures remain incorrect in denominator.",
        "defer_cost": DEFER_COST, "code_commit": git_commit(), "runner_sha256": sha256_file(Path(__file__).resolve()),
    }
    if args.resume:
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("selection", "models", "generation", "case_count"):
            if old.get(key) != manifest.get(key):
                raise RuntimeError(f"resume mismatch in {key}")
        manifest = old
        manifest.setdefault("resume_events", []).append({
            "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "gpu_global_id": args.gpu_index,
            "minimum_free_gib_before_each_model": args.min_free_gib,
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "git_commit": git_commit(),
        })
        manifest["status"] = "running"
        manifest.setdefault("gpu_indices_used", []).append(args.gpu_index)
    elif trajectory.exists() or manifest_path.exists():
        raise FileExistsError("output exists; use --resume")
    else:
        manifest["gpu_indices_used"] = [args.gpu_index]
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    records = [json.loads(line) for line in trajectory.read_text(encoding="utf-8").splitlines() if line.strip()] if args.resume and trajectory.exists() else []
    done = {(row["case_hash"], row["model_role"]) for row in records}

    for spec in model_specs:
        model_path = Path(spec["path"])
        pending = [case for case in cases[:args.limit_cases] if (case["case_hash"], spec["role"]) not in done]
        if not pending:
            continue
        current_free, _ = torch.cuda.mem_get_info()
        if current_free < args.min_free_gib * 1024**3:
            manifest["status"] = "paused_resource_gate"
            manifest["blocked_before_model_role"] = spec["role"]
            manifest["free_memory_bytes_at_gate"] = current_free
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            raise RuntimeError(f"GPU{args.gpu_index} safety gate before {spec['role']}: only {current_free / 1024**3:.1f} GiB free")
        print(f"loading {spec['name']} on GPU{args.gpu_index}; pending={len(pending)}", flush=True)
        load_start = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True, torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True, attn_implementation="eager",
        ).to("cuda:0").eval()
        load_seconds = time.perf_counter() - load_start
        for index, case in enumerate(cases[:args.limit_cases], 1):
            key = (case["case_hash"], spec["role"])
            if key in done:
                continue
            result = infer(model, tokenizer, case["answer_messages"] if spec["action"] == "ANSWER" else case["defer_messages"], spec["kind"], spec["action"] == "ANSWER")
            record = {
                "case_hash": case["case_hash"], "model_role": spec["role"], "action": spec["action"],
                "predicted": result["predicted"], "correct": bool(result["predicted"] == case["gold"]),
                "valid": result["valid"], "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"], "latency_s": result["latency_s"],
                "finish_reason": result["finish_reason"], "parse_error": result["parse_error"],
            }
            write_record(trajectory, record)
            records.append(record); done.add(key)
            if index % 5 == 0 or index == args.limit_cases:
                print(f"{spec['role']}: {index}/{args.limit_cases}", flush=True)
        manifest.setdefault("model_load_seconds", {})[spec["role"]] = load_seconds
        manifest["peak_memory_allocated_bytes"] = max(int(manifest.get("peak_memory_allocated_bytes", 0)), int(torch.cuda.max_memory_allocated(0)))
        del model, tokenizer
        gc.collect(); torch.cuda.empty_cache()
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summarize(cases, records, manifest, args.limit_cases)
    print(json.dumps({"status": manifest["status"], "complete_cases": manifest["complete_cases"], "output_dir": str(OUTPUT_DIR)}, indent=2))


if __name__ == "__main__":
    main()
