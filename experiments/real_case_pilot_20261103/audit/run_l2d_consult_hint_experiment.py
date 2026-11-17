#!/usr/bin/env python3
"""Paired 100-case small-hint CONSULT experiment for shortlisted models.

Gold and hidden facts stay in evaluator memory. The ignored output contains only
publicly generated question/hint text, answer letters, correctness, and telemetry.
"""

from __future__ import annotations

import argparse
import ast
import csv
import gc
import hashlib
import json
import random
import re
import resource
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

AUDIT_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(AUDIT_DIR))
import run_l2d_expert_screen_32b as sample_builder  # noqa: E402
import run_l2d_remedi_confirmation as helper  # noqa: E402

OUTPUT_DIR = AUDIT_DIR / "outputs/l2d_consult_hint_20261118_cpu"
BASE_MODEL = helper.candidate_eval.BASE_MODEL_PATH
SPECS = (
    {"role": "consult_base_hint", "name": "Qwen3-8B Base hint", "path": BASE_MODEL, "kind": "qwen3"},
    {"role": "consult_remedi_hint", "name": "Qwen3-4B-ReMeDi hint", "path": helper.REMEDI_SPEC["path"], "kind": "qwen3"},
    {"role": "consult_popit_hint", "name": "PopiT-Qwen3-4B-Medical-SFT hint", "path": helper.candidate_eval.MODEL_SPECS[3]["path"], "kind": "qwen3"},
)
CASE_COUNT = 100
SEED = 20261118
MAX_NEW_TOKENS = 48
DEVICE = "cpu"
QUESTION_ROLE = "consult_question"
ROLES = tuple(spec["role"] for spec in SPECS)

FIXED_CONSULT_QUESTION = "Which one concrete, visible clue in this case should the Base Agent notice? Give no interpretation or answer."
HINT_SYSTEM = (
    "You are a medical consultation assistant helping with a multiple-choice clinical case. Review only the "
    "visible case, options, dialogue, and the Base Agent's question. Point to one observable finding already "
    "in the case, without interpreting it. Do not name or imply a disease, diagnosis, medication, treatment, "
    "option, final answer, or unmentioned patient fact. Keep it to one short sentence of at most 15 words. "
    "Output exactly: Expert Hint: <one visible clue, without interpretation>."
)
CONTINUE_SYSTEM = (
    "You are the Base Agent answering a medical multiple-choice benchmark. Use only visible case information "
    "and the consultation hint. Treat the hint as fallible advice and verify it against the case. Do not use "
    "hidden facts or an answer key. Output exactly: Final Answer: <one uppercase option letter A-D>."
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def option_terms_for_case(case: dict[str, Any]) -> tuple[set[str], set[str]]:
    all_text = "\n".join(str(m.get("content", "")) for m in case["answer_messages"] if m.get("role") == "user")
    option_texts: list[str] = []
    match = re.search(r"(?is)\bOptions\s*:\s*(\{.*?\})\s*$", all_text)
    if match:
        try:
            parsed = ast.literal_eval(match.group(1))
            if isinstance(parsed, dict):
                option_texts = [str(x) for x in parsed.values()]
        except (SyntaxError, ValueError):
            option_texts = []
    if not option_texts:
        option_texts = re.findall(r"(?m)^\s*[A-D][).:]\s*(.+?)\s*$", all_text)
    stop = {"patient", "medical", "clinical", "history", "evidence", "option", "answer", "should", "would", "could", "visible", "current", "finding", "symptom", "question", "because", "without", "before", "during", "after", "normal", "abnormal", "management", "treatment"}
    def normalize(term: str) -> str:
        term = term.casefold()
        if term.endswith("ies") and len(term) > 5:
            return term[:-3] + "y"
        if term.endswith("s") and not term.endswith("ss") and len(term) > 4:
            return term[:-1]
        return term

    option_terms = {normalize(x) for text in option_texts for x in re.findall(r"[a-zA-Z][a-zA-Z'-]{6,}", text) if x.casefold() not in stop}
    context = all_text.split("Options:", 1)[0]
    context_terms = {normalize(x) for x in re.findall(r"[a-zA-Z][a-zA-Z'-]{6,}", context)}
    return option_terms, context_terms


def parse_hint(raw: str, case: dict[str, Any]) -> tuple[str | None, str | None]:
    match = re.fullmatch(r"\s*Expert\s+Hint\s*:\s*(.{5,600})\s*", raw, re.I | re.S)
    if not match:
        return None, "hint_format_failure"
    hint = match.group(1).strip()
    words = re.findall(r"\b[\w'-]+\b", hint)
    if len(words) > 15:
        return None, "hint_over_15_words"
    if re.search(
        r"\b(?:final answer|correct answer|answer is|choose option|select option|option\s+[A-D]|"
        r"I choose\s+[A-D]|I select\s+[A-D]|diagnos\w*|disease|disorder|syndrome|condition|"
        r"treatment|therapy|medication|prescribe|more likely|most likely|consistent with|suggests|indicates|"
        r"points? to|supports? a diagnosis|next step|management)\b",
        hint,
        re.I,
    ):
        return None, "interpretation_or_answer_leak"
    hint_terms = {x.casefold() for x in re.findall(r"[a-zA-Z][a-zA-Z'-]{6,}", hint)}
    hint_terms = {x[:-3] + "y" if x.endswith("ies") and len(x) > 5 else x[:-1] if x.endswith("s") and not x.endswith("ss") and len(x) > 4 else x for x in hint_terms}
    option_terms, context_terms = option_terms_for_case(case)
    if hint_terms & (option_terms - context_terms):
        return None, "option_text_overlap"
    return hint, None


def parse_answer(raw: str) -> str | None:
    matches = re.findall(r"Final\s+Answer\s*:\s*([A-D])\b", raw, re.I)
    if len(matches) == 1:
        return matches[0].upper()
    return None


def base_messages(messages: list[dict[str, str]], system: str) -> list[dict[str, str]]:
    result = [dict(x) for x in messages]
    result[0] = {"role": "system", "content": system}
    return result


def hint_messages(case: dict[str, Any], question: str) -> list[dict[str, str]]:
    messages = base_messages(case["answer_messages"], HINT_SYSTEM)
    messages.append({"role": "user", "content": f"Base Agent consult question: {question}"})
    return messages


def continuation_messages(case: dict[str, Any], question: str, hint: str) -> list[dict[str, str]]:
    messages = base_messages(case["answer_messages"], CONTINUE_SYSTEM)
    messages.append({"role": "assistant", "content": f"Consult Question: {question}"})
    messages.append({"role": "user", "content": f"Expert Hint: {hint}\n\nNow answer the original case."})
    return messages


def render_input(tokenizer: Any, messages: list[dict[str, str]]) -> dict[str, torch.Tensor]:
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", enable_thinking=False
    )
    if hasattr(rendered, "items"):
        tensors = {k: v for k, v in rendered.items() if isinstance(v, torch.Tensor)}
        if tensors:
            tensors.setdefault("attention_mask", torch.ones_like(tensors["input_ids"]))
            return tensors
    ids = rendered if isinstance(rendered, torch.Tensor) else torch.as_tensor(rendered)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def infer(model: Any, tokenizer: Any, messages: list[dict[str, str]], task: str, case: dict[str, Any]) -> dict[str, Any]:
    tensors = render_input(tokenizer, messages)
    tensors = {k: v.to(DEVICE) for k, v in tensors.items()}
    prompt_tokens = int(tensors["input_ids"].shape[-1])
    if DEVICE.startswith("cuda"):
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **tensors,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    if DEVICE.startswith("cuda"):
        torch.cuda.synchronize()
    completion = output[0, prompt_tokens:]
    raw = tokenizer.decode(completion, skip_special_tokens=True).strip()
    ended = bool(tokenizer.eos_token_id is not None and completion.numel() and int(completion[-1]) == tokenizer.eos_token_id)
    result: dict[str, Any] = {
        "raw_output": raw,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": int(completion.numel()),
        "latency_s": time.perf_counter() - started,
        "finish_reason": "stop" if ended else "length_or_model_stop",
        "valid": True,
        "error": None,
    }
    if task == "hint":
        result["parsed"], result["error"] = parse_hint(raw, case)
    else:
        result["parsed"] = parse_answer(raw)
        if result["parsed"] is None:
            result["error"] = "final_answer_format_failure"
    result["valid"] = result["parsed"] is not None
    return result


def append_record(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def bootstrap_ci(values: list[float], seed: int, n: int = 10000) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    draws = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(n))
    return draws[int(.025 * n)], draws[min(n - 1, int(.975 * n))]


def summary(cases: list[dict[str, Any]], records: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    by_case: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for rec in records:
        by_case[str(rec["case_hash"])][str(rec["record_type"] + ":" + rec["model_role"])] = rec
    rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    baseline = {case["case_hash"]: by_case[case["case_hash"]].get("baseline:always_answer", {}) for case in cases}
    base_correct = {case["case_hash"]: bool(baseline[case["case_hash"]].get("correct", False)) for case in cases}
    baseline_values = [float(base_correct[c["case_hash"]]) for c in cases]
    baseline_ci = bootstrap_ci(baseline_values, SEED + 99)
    rows.extend([
        {"metric": "accuracy", "method": "always_answer", "value": statistics.mean(baseline_values), "ci95_low": baseline_ci[0], "ci95_high": baseline_ci[1], "n": len(baseline_values)},
        {"metric": "mean_direct_answer_tokens", "method": "always_answer", "value": statistics.mean(int(baseline[c["case_hash"]].get("prompt_tokens", 0)) + int(baseline[c["case_hash"]].get("completion_tokens", 0)) for c in cases), "ci95_low": "", "ci95_high": "", "n": len(cases)},
        {"metric": "mean_direct_answer_latency_s", "method": "always_answer", "value": statistics.mean(float(baseline[c["case_hash"]].get("latency_s", 0.0)) for c in cases), "ci95_low": "", "ci95_high": "", "n": len(cases)},
    ])
    for i, role in enumerate(ROLES):
        final = [by_case[c["case_hash"]].get(f"continuation:{role}", {}) for c in cases]
        correct = [float(bool(x.get("correct", False))) for x in final]
        delta = [correct[j] - float(base_correct[c["case_hash"]]) for j, c in enumerate(cases)]
        acc_ci = bootstrap_ci(correct, SEED + i)
        delta_ci = bootstrap_ci(delta, SEED + 20 + i)
        hints = [by_case[c["case_hash"]].get(f"hint:{role}", {}) for c in cases]
        qs = [by_case[c["case_hash"]].get(f"question:{QUESTION_ROLE}", {}) for c in cases]
        base_errors = [j for j, c in enumerate(cases) if not base_correct[c["case_hash"]]]
        base_successes = [j for j, c in enumerate(cases) if base_correct[c["case_hash"]]]
        corrected = sum(correct[j] for j in base_errors)
        degraded = sum(1 - correct[j] for j in base_successes)
        correction_ci = bootstrap_ci([correct[j] for j in base_errors], SEED + 40 + i)
        degradation_ci = bootstrap_ci([1 - correct[j] for j in base_successes], SEED + 50 + i)
        branch_tokens = []
        branch_latency = []
        for j, (c, f) in enumerate(zip(cases, final)):
            q = by_case[c["case_hash"]].get(f"question:{QUESTION_ROLE}", {})
            h = by_case[c["case_hash"]].get(f"hint:{role}", {})
            branch_tokens.append(sum(int(x.get("prompt_tokens", 0)) + int(x.get("completion_tokens", 0)) for x in (q, h, f)))
            branch_latency.append(sum(float(x.get("latency_s", 0.0)) for x in (q, h, f)))
            paired_rows.append({
                "case_hash": c["case_hash"], "method": role,
                "baseline_correct": int(base_correct[c["case_hash"]]),
                "consult_correct": int(correct[j]),
                "paired_delta": int(correct[j] - float(base_correct[c["case_hash"]])),
                "question_valid": int(bool(q.get("valid"))), "hint_valid": int(bool(h.get("valid"))),
                "final_valid": int(bool(f.get("valid"))),
            })
        rows.extend([
            {"metric": "accuracy", "method": role, "value": statistics.mean(correct), "ci95_low": acc_ci[0], "ci95_high": acc_ci[1], "n": len(correct)},
            {"metric": "paired_delta_vs_answer", "method": role, "value": statistics.mean(delta), "ci95_low": delta_ci[0], "ci95_high": delta_ci[1], "n": len(delta)},
            {"metric": "question_valid_rate", "method": role, "value": statistics.mean(float(bool(x.get("valid"))) for x in qs), "ci95_low": "", "ci95_high": "", "n": len(qs)},
            {"metric": "hint_valid_rate", "method": role, "value": statistics.mean(float(bool(x.get("valid"))) for x in hints), "ci95_low": "", "ci95_high": "", "n": len(hints)},
            {"metric": "correct_on_answer_errors", "method": role, "value": corrected, "ci95_low": "", "ci95_high": "", "n": len(base_errors)},
            {"metric": "correction_rate_given_answer_error", "method": role, "value": statistics.mean([correct[j] for j in base_errors]) if base_errors else "", "ci95_low": correction_ci[0], "ci95_high": correction_ci[1], "n": len(base_errors)},
            {"metric": "wrong_on_answer_correct", "method": role, "value": degraded, "ci95_low": "", "ci95_high": "", "n": len(base_successes)},
            {"metric": "degradation_rate_given_answer_correct", "method": role, "value": statistics.mean([1-correct[j] for j in base_successes]) if base_successes else "", "ci95_low": degradation_ci[0], "ci95_high": degradation_ci[1], "n": len(base_successes)},
            {"metric": "mean_branch_tokens", "method": role, "value": statistics.mean(branch_tokens), "ci95_low": "", "ci95_high": "", "n": len(branch_tokens)},
            {"metric": "mean_branch_latency_s", "method": role, "value": statistics.mean(branch_latency), "ci95_low": "", "ci95_high": "", "n": len(branch_latency)},
            {"metric": "mean_direct_answer_tokens", "method": role, "value": statistics.mean(int(baseline[c["case_hash"]].get("prompt_tokens", 0)) + int(baseline[c["case_hash"]].get("completion_tokens", 0)) for c in cases), "ci95_low": "", "ci95_high": "", "n": len(cases)},
        ])
    with (OUTPUT_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=("metric", "method", "value", "ci95_low", "ci95_high", "n"))
        writer.writeheader(); writer.writerows(rows)
    with (OUTPUT_DIR / "paired_case_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=("case_hash", "method", "baseline_correct", "consult_correct", "paired_delta", "question_valid", "hint_valid", "final_valid"))
        writer.writeheader(); writer.writerows(paired_rows)
    manifest.update({
        "status": "completed" if len(records) >= len(cases) * (2 + 2 * len(ROLES)) else "partial",
        "trajectory_sha256": sha256_file(OUTPUT_DIR / "trajectories.jsonl"),
        "summary_sha256": sha256_file(OUTPUT_DIR / "summary.csv"),
        "paired_case_results_sha256": sha256_file(OUTPUT_DIR / "paired_case_results.csv"),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    (OUTPUT_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    global OUTPUT_DIR, DEVICE
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--gpu-index", type=int, default=5)
    parser.add_argument("--limit-cases", type=int, default=CASE_COUNT)
    parser.add_argument("--offset", type=int, default=0, help="start within the frozen 100-case candidate sample")
    parser.add_argument("--min-free-gib", type=float, default=70.0)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    if args.device == "cuda" and args.gpu_index in (1, 7):
        raise ValueError("GPU1 and GPU7 are excluded by the project resource policy")
    if not 1 <= args.limit_cases <= CASE_COUNT or args.offset < 0 or args.offset + args.limit_cases > CASE_COUNT:
        raise ValueError("offset and limit-cases must select within the frozen 100-case sample")
    DEVICE = "cpu" if args.device == "cpu" else "cuda:0"
    torch.set_num_threads(args.threads)
    free = total = 0
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        torch.cuda.set_device(0)
        free, total = torch.cuda.mem_get_info()
        if free < args.min_free_gib * 1024**3:
            raise RuntimeError(f"GPU{args.gpu_index} safety gate: {free/1024**3:.1f} GiB free")

    cases, selection = sample_builder.load_cases()
    cases = cases[args.offset:args.offset + args.limit_cases]
    selection["run_offset_in_candidate_sample"] = args.offset
    selection["run_case_count"] = len(cases)
    selection["run_case_hashes_sha256"] = sha256_text("\n".join(c["case_hash"] for c in cases))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = output_dir / "trajectories.jsonl"
    manifest_path = output_dir / "run_manifest.json"
    model_specs = []
    for spec in SPECS:
        path = Path(spec["path"])
        if not (path / "config.json").is_file() or not (path / "model.safetensors.index.json").is_file():
            raise FileNotFoundError(f"incomplete model snapshot: {spec['name']} at {path}")
        model_specs.append({**spec, "path": str(path), "config_sha256": sha256_file(path / "config.json"), "index_sha256": sha256_file(path / "model.safetensors.index.json")})
    manifest = {
        "status": "running", "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experiment": "exploratory paired small-hint CONSULT pilot using prior shortlist models and one fixed public-state question",
        "selection": selection, "case_count": len(cases), "models": model_specs,
        "generation": {"greedy": True, "dtype": "bfloat16", "max_new_tokens": MAX_NEW_TOKENS, "qwen3_thinking": False},
        "runtime": {"torch": torch.__version__, "transformers": __import__("transformers").__version__, "device": DEVICE, "gpu_index": args.gpu_index if args.device == "cuda" else None, "gpu_name": torch.cuda.get_device_name(0) if args.device == "cuda" else None, "initial_free_bytes": free if args.device == "cuda" else None, "total_bytes": total if args.device == "cuda" else None, "min_free_gib_before_model_load": args.min_free_gib if args.device == "cuda" else None, "threads": args.threads},
        "protocol": {"consult_question": FIXED_CONSULT_QUESTION, "consult_question_sha256": sha256_text(FIXED_CONSULT_QUESTION), "question_source": "same fixed question for all states and experts; not learned/generated", "hint_system_sha256": sha256_text(HINT_SYSTEM), "continuation_system_sha256": sha256_text(CONTINUE_SYSTEM), "hint_max_words": 15, "option_text_overlap_check": True, "explicit_answer_and_interpretation_leak_check": True},
        "gold_policy": "Gold and hidden atomic facts remain evaluator-only and are not serialized or sent to inference.",
        "baseline": {"role": "always_answer", "model": "Qwen3-8B", "generated_on_the_same_cases": True},
        "code_commit": git_commit(), "runner_sha256": sha256_file(Path(__file__).resolve()),
    }
    if args.resume:
        if not manifest_path.is_file():
            raise FileNotFoundError("resume requested but manifest is missing")
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("selection", "case_count", "models", "generation"):
            if old.get(key) != manifest.get(key):
                raise RuntimeError(f"resume mismatch in {key}")
        manifest = old
        manifest.setdefault("resume_events", []).append({"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "gpu_index": args.gpu_index, "free_bytes": free})
    elif trajectory_path.exists() or manifest_path.exists():
        raise FileExistsError("output exists; pass --resume or use a new --output-dir")
    (manifest_path).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    OUTPUT_DIR = output_dir
    records = read_jsonl(trajectory_path) if args.resume and trajectory_path.exists() else []
    done = {(x["record_type"], x["model_role"], x["case_hash"]) for x in records}
    question_map = {x["case_hash"]: x for x in records if x["record_type"] == "question"}
    hint_map = {(x["case_hash"], x["model_role"]): x for x in records if x["record_type"] == "hint"}

    def save(record: dict[str, Any]) -> None:
        append_record(trajectory_path, record)
        records.append(record)
        done.add((record["record_type"], record["model_role"], record["case_hash"]))
        if record["record_type"] == "question":
            question_map[record["case_hash"]] = record
        elif record["record_type"] == "hint":
            hint_map[(record["case_hash"], record["model_role"])] = record

    def load_model(spec: dict[str, Any]) -> tuple[Any, Any, float]:
        now_free = 0
        if args.device == "cuda":
            now_free, _ = torch.cuda.mem_get_info()
        if args.device == "cuda" and now_free < args.min_free_gib * 1024**3:
            manifest["status"] = "paused_resource_gate"
            manifest["blocked_before_model_role"] = spec["role"]
            manifest["free_bytes_at_gate"] = now_free
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            raise RuntimeError(f"GPU{args.gpu_index} safety gate before {spec['role']}: {now_free/1024**3:.1f} GiB free")
        path = Path(spec["path"])
        print(f"loading {spec['name']} on {DEVICE}", flush=True)
        started = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="eager").to(DEVICE).eval()
        if model.generation_config is not None:
            model.generation_config.do_sample = False
            model.generation_config.temperature = None
            model.generation_config.top_p = None
            model.generation_config.top_k = None
        if args.device == "cuda":
            torch.cuda.synchronize()
            post_free, _ = torch.cuda.mem_get_info()
        else:
            post_free = 0
        manifest.setdefault("model_load_seconds", {})[spec["role"]] = time.perf_counter() - started
        if args.device == "cuda":
            manifest.setdefault("free_after_load_bytes", {})[spec["role"]] = post_free
            manifest["peak_allocated_bytes"] = max(int(manifest.get("peak_allocated_bytes", 0)), int(torch.cuda.max_memory_allocated(0)))
        else:
            manifest["peak_rss_kib"] = max(int(manifest.get("peak_rss_kib", 0)), int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        if args.device == "cuda" and post_free < 45 * 1024**3:
            del model, tokenizer; gc.collect(); torch.cuda.empty_cache()
            raise RuntimeError(f"post-load reserve too low on GPU{args.gpu_index}: {post_free/1024**3:.1f} GiB")
        return model, tokenizer, post_free / 1024**3 if args.device == "cuda" else 0.0

    base_spec = next(x for x in model_specs if x["role"] == "consult_base_hint")
    model, tokenizer, _ = load_model(base_spec)
    for i, case in enumerate(cases, 1):
        key = ("baseline", "always_answer", case["case_hash"])
        if key in done: continue
        result = infer(model, tokenizer, case["answer_messages"], "answer", case)
        save({"record_type": "baseline", "model_role": "always_answer", "case_hash": case["case_hash"], "valid": result["valid"], "predicted": result["parsed"], "correct": bool(result["parsed"] == case["gold"]), "prompt_tokens": result["prompt_tokens"], "completion_tokens": result["completion_tokens"], "latency_s": result["latency_s"], "finish_reason": result["finish_reason"], "error": result["error"]})
        if i % 10 == 0: print(f"Always Answer {i}/{len(cases)}", flush=True)
    for case in cases:
        if ("question", QUESTION_ROLE, case["case_hash"]) in done:
            continue
        save({"record_type": "question", "model_role": QUESTION_ROLE, "case_hash": case["case_hash"], "valid": True, "parsed": FIXED_CONSULT_QUESTION, "question_source": "fixed_protocol", "prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0.0, "finish_reason": "not_applicable", "error": None})
    for i, case in enumerate(cases, 1):
        key = ("hint", "consult_base_hint", case["case_hash"])
        if key in done: continue
        q = question_map.get(case["case_hash"], {})
        if not q.get("valid"):
            save({"record_type": "hint", "model_role": "consult_base_hint", "case_hash": case["case_hash"], "valid": False, "raw_output": "", "prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0.0, "finish_reason": "not_run", "error": "consult_question_invalid"})
            continue
        result = infer(model, tokenizer, hint_messages(case, str(q["parsed"])), "hint", case)
        save({"record_type": "hint", "model_role": "consult_base_hint", "case_hash": case["case_hash"], "valid": result["valid"], "hint": result["parsed"], "raw_output": result["raw_output"], "hint_sha256": sha256_text(str(result["parsed"] or result["raw_output"])), "prompt_tokens": result["prompt_tokens"], "completion_tokens": result["completion_tokens"], "latency_s": result["latency_s"], "finish_reason": result["finish_reason"], "error": result["error"]})
        if i % 10 == 0: print(f"Base hints {i}/{len(cases)}", flush=True)
    del model, tokenizer; gc.collect()
    if args.device == "cuda": torch.cuda.empty_cache()

    for spec in [x for x in model_specs if x["role"] != "consult_base_hint"]:
        model, tokenizer, _ = load_model(spec)
        for i, case in enumerate(cases, 1):
            key = ("hint", spec["role"], case["case_hash"])
            if key in done: continue
            q = question_map.get(case["case_hash"], {})
            if not q.get("valid"):
                save({"record_type": "hint", "model_role": spec["role"], "case_hash": case["case_hash"], "valid": False, "hint": None, "raw_output": "", "prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0.0, "finish_reason": "not_run", "error": "consult_question_invalid"})
                continue
            result = infer(model, tokenizer, hint_messages(case, str(q["parsed"])), "hint", case)
            save({"record_type": "hint", "model_role": spec["role"], "case_hash": case["case_hash"], "valid": result["valid"], "hint": result["parsed"], "raw_output": result["raw_output"], "hint_sha256": sha256_text(str(result["parsed"] or result["raw_output"])), "prompt_tokens": result["prompt_tokens"], "completion_tokens": result["completion_tokens"], "latency_s": result["latency_s"], "finish_reason": result["finish_reason"], "error": result["error"]})
            if i % 10 == 0: print(f"{spec['role']} {i}/{len(cases)}", flush=True)
        if args.device == "cuda":
            manifest["peak_allocated_bytes"] = max(int(manifest.get("peak_allocated_bytes", 0)), int(torch.cuda.max_memory_allocated(0)))
        else:
            manifest["peak_rss_kib"] = max(int(manifest.get("peak_rss_kib", 0)), int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        del model, tokenizer; gc.collect()
        if args.device == "cuda": torch.cuda.empty_cache()

    model, tokenizer, _ = load_model(base_spec)
    for spec in model_specs:
        role = spec["role"]
        for i, case in enumerate(cases, 1):
            key = ("continuation", role, case["case_hash"])
            if key in done: continue
            q = question_map.get(case["case_hash"], {})
            h = hint_map.get((case["case_hash"], role), {})
            if not q.get("valid") or not h.get("valid"):
                save({"record_type": "continuation", "model_role": role, "case_hash": case["case_hash"], "valid": False, "predicted": None, "correct": False, "prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0.0, "finish_reason": "not_run", "error": "question_or_hint_invalid"})
                continue
            result = infer(model, tokenizer, continuation_messages(case, str(q["parsed"]), str(h["hint"])), "answer", case)
            save({"record_type": "continuation", "model_role": role, "case_hash": case["case_hash"], "valid": result["valid"], "predicted": result["parsed"], "correct": bool(result["parsed"] == case["gold"]), "prompt_tokens": result["prompt_tokens"], "completion_tokens": result["completion_tokens"], "latency_s": result["latency_s"], "finish_reason": result["finish_reason"], "error": result["error"]})
            if i % 10 == 0: print(f"{role} continuations {i}/{len(cases)}", flush=True)
    if args.device == "cuda":
        manifest["peak_allocated_bytes"] = max(int(manifest.get("peak_allocated_bytes", 0)), int(torch.cuda.max_memory_allocated(0)))
    else:
        manifest["peak_rss_kib"] = max(int(manifest.get("peak_rss_kib", 0)), int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    del model, tokenizer; gc.collect()
    if args.device == "cuda": torch.cuda.empty_cache()
    summary(cases, records, manifest)
    print(json.dumps({"status": manifest["status"], "output_dir": str(output_dir), "records": len(records), "peak_allocated_gib": manifest.get("peak_allocated_bytes", 0) / 1024**3 if args.device == "cuda" else None, "peak_rss_gib": manifest.get("peak_rss_kib", 0) / 1024**2 if args.device == "cpu" else None}, indent=2))


if __name__ == "__main__":
    main()
