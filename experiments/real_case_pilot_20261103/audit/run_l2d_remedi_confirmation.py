#!/usr/bin/env python3
"""Independent 100-case GPU confirmation of the L2D DEFER expert.

The evaluator keeps prompts, gold answers, and hidden atomic facts in memory.
Only hashes, predictions, correctness indicators, and aggregate telemetry are
written to the ignored output directory.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import os
import random
import re
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
PILOT_DIR = AUDIT_DIR.parent
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "ea_atpo_stage"))
sys.path.insert(0, str(AUDIT_DIR))
import run_direct_reliability_eval as direct  # noqa: E402
import run_medical_candidate_eval as candidate_eval  # noqa: E402

_defer_path = AUDIT_DIR / "run_medical_candidate_defer_dev_eval.py"
_spec = importlib.util.spec_from_file_location("l2d_defer_helpers", _defer_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot load DEFER protocol helpers from {_defer_path}")
_defer = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _defer
_spec.loader.exec_module(_defer)

Action = _defer.Action
build_expert_messages = _defer.build_expert_messages
parse_action = _defer.parse_action
public_state = _defer.public_state
unrevealed_atomic_facts = _defer.unrevealed_atomic_facts
validate_expert_response = _defer.validate_expert_response
validate_expert_view = _defer.validate_expert_view

DATA_PATH = candidate_eval.DATA_PATH
DEV_PATH = ROOT / "experiments/ea_atpo_v1_smoke/ea_atpo_v1_attempt09_stability/data/test.parquet"
BASE_MODEL = candidate_eval.BASE_MODEL_PATH
REMEDI_SPEC = next(spec for spec in candidate_eval.MODEL_SPECS if spec["role"] == "qwen3_4b_remedi")
OUTPUT_DIR = AUDIT_DIR / "outputs/l2d_remedi_confirmation_20261117"
SEED = 20261117
CASE_COUNT = 100
DEFER_COST = 0.60
MAX_NEW_TOKENS = 128


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def plain(value: Any) -> Any:
    return value.tolist() if hasattr(value, "tolist") else value


def row_gold(row: dict[str, Any]) -> str:
    raw = row.get("ground_truth") or row.get("reward_model", {}).get("ground_truth")
    gold = direct.answer_letter(raw)
    if gold not in "ABCD":
        raise ValueError("dataset row has an invalid answer key")
    return gold


def case_key(row: dict[str, Any], index: int) -> tuple[str, str, list[str], list[dict[str, str]], list[dict[str, str]]]:
    state, digest, gold, facts = public_state(row, index)
    defer_messages = build_expert_messages(state, Action.DEFER)
    answer_messages = direct.build_messages(row)
    hidden = unrevealed_atomic_facts(state, facts)
    validate_expert_view(defer_messages, atomic_facts=hidden, ground_truth=gold)
    validate_expert_view(answer_messages, atomic_facts=hidden, ground_truth=gold)
    return digest, gold, facts, answer_messages, defer_messages


def build_cases() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not DATA_PATH.is_file() or not DEV_PATH.is_file():
        raise FileNotFoundError(f"test or Attempt09 development parquet missing: {DATA_PATH}, {DEV_PATH}")
    frame = pd.read_parquet(DATA_PATH)
    dev_frame = pd.read_parquet(DEV_PATH)
    if len(frame) != 1268:
        raise ValueError(f"expected pinned 1268-row ATPO test split, got {len(frame)}")

    previous_indices = set(random.Random(candidate_eval.SEED).sample(range(len(frame)), candidate_eval.CASE_COUNT))
    dev_hashes: set[str] = set()
    for index, raw in dev_frame.iterrows():
        row = raw.to_dict()
        digest, _, _, _, _ = case_key(row, int(index))
        dev_hashes.add(digest)

    order = list(range(len(frame)))
    random.Random(SEED).shuffle(order)
    cases: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    excluded_prior = excluded_dev = duplicates = 0
    for source_index in order:
        if source_index in previous_indices:
            excluded_prior += 1
            continue
        row = frame.iloc[source_index].to_dict()
        digest, gold, facts, answer_messages, defer_messages = case_key(row, source_index)
        if digest in dev_hashes:
            excluded_dev += 1
            continue
        if digest in seen_hashes:
            duplicates += 1
            continue
        seen_hashes.add(digest)
        cases.append({
            "case_hash": digest,
            "source_index": source_index,
            "gold": gold,
            "facts": facts,
            "answer_messages": answer_messages,
            "defer_messages": defer_messages,
        })
        if len(cases) == CASE_COUNT:
            break
    if len(cases) != CASE_COUNT:
        raise ValueError(f"could select only {len(cases)} unique cases after exclusions")
    selection = {
        "data_sha256": sha256_file(DATA_PATH),
        "data_rows": len(frame),
        "attempt09_dev_data_sha256": sha256_file(DEV_PATH),
        "attempt09_dev_rows": len(dev_frame),
        "attempt09_dev_unique_public_states": len(dev_hashes),
        "prior_200_case_source_indices_excluded": len(previous_indices),
        "attempt09_public_state_hashes_excluded_if_overlapping": excluded_dev,
        "duplicate_public_states_skipped": duplicates,
        "seed": SEED,
        "selected_case_hashes_sha256": sha256_text("\n".join(c["case_hash"] for c in cases)),
        "selected_source_indices_sha256": sha256_text("\n".join(str(c["source_index"]) for c in cases)),
    }
    return cases, selection


def render_input(tokenizer: Any, messages: list[dict[str, str]]) -> dict[str, torch.Tensor]:
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=False,
    )
    if hasattr(rendered, "items"):
        tensors = {key: value for key, value in rendered.items() if isinstance(value, torch.Tensor)}
        if tensors:
            if "attention_mask" not in tensors:
                tensors["attention_mask"] = torch.ones_like(tensors["input_ids"])
            return tensors
    input_ids = rendered if isinstance(rendered, torch.Tensor) else torch.as_tensor(rendered)
    return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}


def parse_defer(raw: str) -> str | None:
    normalized = validate_expert_response(Action.DEFER, raw)
    match = re.fullmatch(r"Expert Final Answer:\s*([A-D])", normalized, re.IGNORECASE)
    return match.group(1).upper() if match else None


def infer(model: Any, tokenizer: Any, messages: list[dict[str, str]], arm: str) -> dict[str, Any]:
    device = next(model.parameters()).device
    inputs = render_input(tokenizer, messages)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    prompt_tokens = int(inputs["input_ids"].shape[-1])
    generation: dict[str, Any] = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "num_beams": 1,
        "use_cache": True,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 50,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if tokenizer.eos_token_id is not None:
        generation["eos_token_id"] = tokenizer.eos_token_id
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(**inputs, **generation)
    torch.cuda.synchronize()
    completion = generated[0, prompt_tokens:]
    raw = tokenizer.decode(completion, skip_special_tokens=False)
    try:
        predicted = direct.parse_answer(raw) if arm == "always_answer" else parse_defer(raw)
        error = None if predicted else "strict_answer_parse_failure"
    except ValueError:
        predicted, error = None, "defer_protocol_validation_failure"
    ended = bool(tokenizer.eos_token_id is not None and completion.numel() and int(completion[-1]) == int(tokenizer.eos_token_id))
    return {
        "predicted": predicted,
        "valid": predicted is not None,
        "error": error,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": int(completion.numel()),
        "latency_s": time.perf_counter() - started,
        "finish_reason": "stop" if ended else "length_or_model_stop",
    }


def load_done(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def bootstrap_ci(values: list[float], seed: int, repeats: int = 10000) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    n = len(values)
    means = sorted(statistics.mean(rng.choices(values, k=n)) for _ in range(repeats))
    return means[int(.025 * repeats)], means[min(repeats - 1, int(.975 * repeats))]


def summarize(records: list[dict[str, Any]], cases: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    arms = ("always_answer_qwen3_8b", "base_qwen3_8b_defer", "qwen3_4b_remedi_defer")
    by_case: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        by_case[record["case_hash"]][record["model_role"]] = record
    complete = [case["case_hash"] for case in cases if all(arm in by_case[case["case_hash"]] for arm in arms)]
    summary: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    for case_hash in complete:
        values = by_case[case_hash]
        base = values[arms[0]]
        for arm in arms[1:]:
            other = values[arm]
            paired.append({
                "case_hash": case_hash,
                "comparison": arm,
                "base_correct": bool(base["correct"]),
                "other_correct": bool(other["correct"]),
                "paired_delta": int(bool(other["correct"])) - int(bool(base["correct"])),
            })
    for arm_index, arm in enumerate(arms):
        rows = [by_case[key][arm] for key in complete]
        correct = [float(row["correct"]) for row in rows]
        low, high = bootstrap_ci(correct, SEED + arm_index)
        summary.extend([
            {"metric": "accuracy", "arm": arm, "value": statistics.mean(correct) if correct else "", "ci95_low": low, "ci95_high": high, "n_cases": len(rows)},
            {"metric": "valid_rate", "arm": arm, "value": statistics.mean(float(row["valid"]) for row in rows) if rows else "", "ci95_low": "", "ci95_high": "", "n_cases": len(rows)},
            {"metric": "mean_total_tokens", "arm": arm, "value": statistics.mean(row["prompt_tokens"] + row["completion_tokens"] for row in rows) if rows else "", "ci95_low": "", "ci95_high": "", "n_cases": len(rows)},
            {"metric": "mean_completion_tokens", "arm": arm, "value": statistics.mean(row["completion_tokens"] for row in rows) if rows else "", "ci95_low": "", "ci95_high": "", "n_cases": len(rows)},
            {"metric": "mean_generation_latency_s", "arm": arm, "value": statistics.mean(row["latency_s"] for row in rows) if rows else "", "ci95_low": "", "ci95_high": "", "n_cases": len(rows)},
            {"metric": "mean_net_value_cost_0_60", "arm": arm, "value": statistics.mean(correct) - (0.0 if arm == arms[0] else DEFER_COST) if rows else "", "ci95_low": "", "ci95_high": "", "n_cases": len(rows)},
        ])
    for arm_index, arm in enumerate(arms[1:], start=1):
        deltas = [float(by_case[key][arm]["correct"]) - float(by_case[key][arms[0]]["correct"]) for key in complete]
        low, high = bootstrap_ci(deltas, SEED + 10 + arm_index)
        summary.append({"metric": "paired_accuracy_delta_vs_always_answer", "arm": arm, "value": statistics.mean(deltas) if deltas else "", "ci95_low": low, "ci95_high": high, "n_cases": len(deltas)})
        corrected = [by_case[key][arm] for key in complete if not by_case[key][arms[0]]["correct"]]
        degraded = [by_case[key][arm] for key in complete if by_case[key][arms[0]]["correct"]]
        summary.append({"metric": "correct_on_always_answer_errors", "arm": arm, "value": sum(bool(row["correct"]) for row in corrected), "ci95_low": "", "ci95_high": "", "n_cases": len(corrected)})
        summary.append({"metric": "wrong_on_always_answer_correct", "arm": arm, "value": sum(not bool(row["correct"]) for row in degraded), "ci95_low": "", "ci95_high": "", "n_cases": len(degraded)})

    with (OUTPUT_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("metric", "arm", "value", "ci95_low", "ci95_high", "n_cases"))
        writer.writeheader()
        writer.writerows(summary)
    with (OUTPUT_DIR / "paired_case_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("case_hash", "comparison", "base_correct", "other_correct", "paired_delta"))
        writer.writeheader()
        writer.writerows(paired)
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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--gpu-index", type=int, default=3, help="Physical GPU index; expose only this GPU via CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    if not 1 <= args.limit_cases <= CASE_COUNT:
        raise ValueError(f"--limit-cases must be within [1, {CASE_COUNT}]")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu_index):
        raise RuntimeError(f"set CUDA_VISIBLE_DEVICES={args.gpu_index} so this run stays on the selected GPU")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.set_num_threads(args.threads)
    OUTPUT_DIR = args.output_dir.resolve()
    torch.cuda.set_device(0)
    free, total = torch.cuda.mem_get_info()
    if free < 40 * 1024**3:
        raise RuntimeError(f"GPU{args.gpu_index} safety gate: only {free / 1024**3:.1f} GiB free")
    if not (BASE_MODEL / "config.json").is_file() or not (Path(REMEDI_SPEC["path"]) / "config.json").is_file():
        raise FileNotFoundError("pinned Base or ReMeDi model snapshot is incomplete")

    cases, selection = build_cases()
    requested = cases[:args.limit_cases]
    output_path = OUTPUT_DIR / "trajectories.jsonl"
    manifest_path = OUTPUT_DIR / "run_manifest.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    models = [
        {"role": "always_answer_qwen3_8b", "name": "Qwen3-8B Always Answer", "path": str(BASE_MODEL), "message_key": "answer_messages", "action": "ANSWER", "repo_id": "Qwen/Qwen3-8B", "revision": "b968826d9c46dd6066d109eabc6255188de91218"},
        {"role": "base_qwen3_8b_defer", "name": "Qwen3-8B under DEFER contract", "path": str(BASE_MODEL), "message_key": "defer_messages", "action": "DEFER", "repo_id": "Qwen/Qwen3-8B", "revision": "b968826d9c46dd6066d109eabc6255188de91218"},
        {"role": "qwen3_4b_remedi_defer", "name": "Qwen3-4B-ReMeDi under DEFER contract", "path": str(REMEDI_SPEC["path"]), "message_key": "defer_messages", "action": "DEFER", "repo_id": REMEDI_SPEC["repo_id"], "revision": REMEDI_SPEC["revision"]},
    ]
    pinned_manifest_path = AUDIT_DIR / "outputs/medical_expert_candidates_20261116/run_manifest.json"
    pinned_models = {}
    if pinned_manifest_path.is_file():
        for model in json.loads(pinned_manifest_path.read_text(encoding="utf-8")).get("models", []):
            pinned_models[model.get("role")] = model
    manifest = {
        "status": "running",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experiment": "independent 100-case L2D DEFER confirmation; no training",
        "selection": selection,
        "case_count": CASE_COUNT,
        "models": [
            {
                **model,
                "config_sha256": sha256_file(Path(model["path"]) / "config.json"),
                "safetensors_index_sha256": sha256_file(Path(model["path"]) / "model.safetensors.index.json") if (Path(model["path"]) / "model.safetensors.index.json").is_file() else None,
                "weight_shards_sha256": pinned_models.get("base_qwen3_8b" if model["role"] != "qwen3_4b_remedi_defer" else "qwen3_4b_remedi", {}).get("weight_shards_sha256", {}),
            }
            for model in models
        ],
        "generation": {"sampling": "greedy", "do_sample": False, "max_new_tokens": MAX_NEW_TOKENS, "dtype": "bfloat16", "qwen3_enable_thinking": False},
        "runtime": {"framework": "Transformers", "torch": torch.__version__, "transformers": __import__("transformers").__version__, "gpu_global_id": args.gpu_index, "gpu_name": torch.cuda.get_device_name(0), "initial_free_memory_bytes": free, "total_memory_bytes": total, "threads": args.threads},
        "gold_policy": "Gold and hidden atomic facts are evaluator-only and never enter model messages; outputs store correctness booleans, not gold labels or prompts.",
        "scoring": "strict direct Final Answer parser for ANSWER; Attempt09 DEFER response validator and parser for DEFER; parse failures score incorrect.",
        "defer_cost": DEFER_COST,
        "code_commit": git_commit(),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
    }
    if args.resume:
        if not manifest_path.is_file():
            raise FileNotFoundError("--resume requires a previous manifest")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("selection", "models", "generation", "case_count"):
            if previous.get(key) != manifest.get(key):
                raise RuntimeError(f"resume mismatch in {key}")
        manifest = previous
        manifest["resume_started_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        manifest["status"] = "running"
    elif output_path.exists() or manifest_path.exists():
        raise FileExistsError("output exists; use --resume")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    records = load_done(output_path) if args.resume else []
    done = {(record["case_hash"], record["model_role"]) for record in records}
    for model_spec in models:
        model_path = Path(model_spec["path"])
        role_pending = [case for case in requested if (case["case_hash"], model_spec["role"]) not in done]
        if not role_pending:
            continue
        print(f"loading {model_spec['name']} on physical GPU{args.gpu_index}; pending={len(role_pending)}", flush=True)
        started = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        ).to("cuda:0").eval()
        load_seconds = time.perf_counter() - started
        print(f"loaded {model_spec['name']} in {load_seconds:.1f}s", flush=True)
        for index, case in enumerate(requested, 1):
            key = (case["case_hash"], model_spec["role"])
            if key in done:
                continue
            messages = case[model_spec["message_key"]]
            try:
                result = infer(model, tokenizer, messages, "always_answer" if model_spec["action"] == "ANSWER" else "DEFER")
            except Exception as exc:
                result = {"predicted": None, "valid": False, "error": f"{type(exc).__name__}", "prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0.0, "finish_reason": "error"}
            record = {
                "case_hash": case["case_hash"],
                "model_role": model_spec["role"],
                "action": model_spec["action"],
                "predicted": result["predicted"],
                "correct": bool(result["predicted"] == case["gold"]),
                "valid": bool(result["valid"]),
                "prompt_tokens": int(result["prompt_tokens"]),
                "completion_tokens": int(result["completion_tokens"]),
                "latency_s": float(result["latency_s"]),
                "finish_reason": result["finish_reason"],
                "parse_error": result["error"],
            }
            with output_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
            records.append(record)
            done.add(key)
            if index % 5 == 0 or index == len(requested):
                print(f"{model_spec['role']}: {index}/{len(requested)}", flush=True)
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        manifest.setdefault("model_load_seconds", {})[model_spec["role"]] = load_seconds
        manifest["peak_memory_allocated_bytes"] = max(int(manifest.get("peak_memory_allocated_bytes", 0)), int(torch.cuda.max_memory_allocated(0)))
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summarize(records, requested, manifest)
    print(json.dumps({"status": manifest["status"], "complete_cases": manifest["complete_cases"], "processed_cases": args.limit_cases, "output_dir": str(OUTPUT_DIR)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
