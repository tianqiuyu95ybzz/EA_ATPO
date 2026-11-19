#!/usr/bin/env python3
"""Greedy validation evaluation for a raw Qwen3-32B + LoRA adapter."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/expert_sft"
for key, value in {
    "HF_HOME": EXP / "cache/hf_home",
    "HF_HUB_CACHE": EXP / "cache/hf_home/hub",
    "HF_DATASETS_CACHE": EXP / "cache/datasets",
    "XDG_CACHE_HOME": EXP / "cache/xdg",
    "TMPDIR": EXP / "cache/tmp",
    "TRITON_CACHE_DIR": EXP / "cache/triton",
    "TORCH_HOME": EXP / "cache/torch",
}.items():
    os.environ.setdefault(key, str(value))

import torch  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    names = ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin")
    for file in sorted(path / name for name in names if (path / name).is_file()):
        digest.update(file.relative_to(path).as_posix().encode("utf-8"))
        with file.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def gpu_snapshot() -> list[dict[str, Any]]:
    raw = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True)
    result = []
    for line in raw.strip().splitlines():
        fields = [x.strip() for x in line.split(",")]
        result.append({"index": int(fields[0]), "name": fields[1], "uuid": fields[2], "total_mib": int(fields[3]), "used_mib": int(fields[4]), "free_mib": int(fields[5]), "utilization_pct": int(fields[6])})
    return result


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def answer_from_target(text: str) -> str:
    match = re.fullmatch(r"\s*Expert Final Answer:\s*([A-D])\s*", text, re.I)
    if not match:
        raise ValueError("SFT data target is not exactly one Expert Final Answer letter")
    return match.group(1).upper()


def parse_expert_answer(text: str) -> str | None:
    found = re.findall(r"Expert Final Answer:\s*([A-D])\b", str(text), re.I)
    return found[0].upper() if len(found) == 1 else None


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def evaluate_one(adapter: Path, rows: list[dict[str, Any]], outdir: Path, max_cases: int | None, split: str, data_path: Path) -> dict[str, Any]:
    config = json.loads((EXP / "configs/domain_sft.json").read_text(encoding="utf-8"))
    model_path = ROOT / config["model"]["local_path"]
    if (outdir / "predictions.jsonl").exists() or (outdir / "summary.json").exists():
        raise FileExistsError(f"Refusing to overwrite prior Expert evaluation: {outdir}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        attn_implementation="sdpa", trust_remote_code=False,
    )
    model = PeftModel.from_pretrained(model, adapter, is_trainable=False).to("cuda")
    model.eval()
    model.config.use_cache = True
    torch.cuda.reset_peak_memory_stats()
    selected = rows[:max_cases] if max_cases else rows
    outdir.mkdir(parents=True, exist_ok=True)
    results = []
    started = time.time()
    for row in selected:
        # Generation receives only the public system/user/history messages. The
        # assistant target is read later by the evaluator, never passed to model.
        input_messages = row["messages"][:-1]
        prompt = tokenizer.apply_chat_template(
            input_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        tokens = {k: v.to("cuda") for k, v in tokens.items()}
        n_prompt = int(tokens["input_ids"].shape[-1])
        t0 = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **tokens, do_sample=False, max_new_tokens=64,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
        torch.cuda.synchronize()
        latency = time.perf_counter() - t0
        continuation = output[0, n_prompt:]
        raw = tokenizer.decode(continuation, skip_special_tokens=True).strip()
        predicted = parse_expert_answer(raw)
        # Gold is first accessed after generation for offline scoring only.
        gold = answer_from_target(row["messages"][-1]["content"])
        results.append({
            "case_id": row["case_id"], "split": split,
            "case_hash": row["case_hash"], "prompt_hash": row["prompt_hash"],
            "predicted": predicted, "valid": predicted is not None,
            "gold": gold, "correct": bool(predicted == gold),
            "prompt_tokens": n_prompt, "completion_tokens": int(continuation.shape[-1]),
            "latency_s": round(latency, 6), "raw_output_sha256": sha256_text(raw), "raw_output": raw,
        })
    path = outdir / "predictions.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    n = len(results)
    valid = sum(r["valid"] for r in results)
    correct = sum(r["correct"] for r in results)
    summary = {
        "status": "completed",
        "git_commit": git_commit(),
        "model_repo_id": config["model"]["repo_id"],
        "model_revision": config["model"]["revision"],
        "base_model_path": str(model_path.relative_to(ROOT)),
        "adapter": str(adapter.relative_to(ROOT) if adapter.is_relative_to(ROOT) else adapter),
        "adapter_tree_sha256": tree_sha256(adapter),
        "tokenizer_path": str(model_path.relative_to(ROOT)),
        "chat_template_sha256": hashlib.sha256(str(tokenizer.chat_template).encode("utf-8")).hexdigest(),
        "split": split,
        "n": n,
        "accuracy_end_to_end": correct / n if n else None,
        "parse_success_rate": valid / n if n else None,
        "accuracy_valid_outputs_only": (correct / valid) if valid else None,
        "mean_prompt_tokens": statistics.mean(r["prompt_tokens"] for r in results) if n else None,
        "mean_completion_tokens": statistics.mean(r["completion_tokens"] for r in results) if n else None,
        "mean_latency_s": statistics.mean(r["latency_s"] for r in results) if n else None,
        "generation_seconds": round(time.time() - started, 3),
        "predictions_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "input_data_sha256": sha256_file(data_path),
        "inference_config": {"do_sample": False, "max_new_tokens": 64, "enable_thinking": False, "dtype": "bfloat16", "seed": int(config["seed"])},
        "gpu_snapshot": gpu_snapshot(),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1),
        "gold_or_hidden_facts_sent_to_model": False,
        "invalid_outputs_count_as_incorrect": True,
        "protocol": "Expert Final Answer: A-D; one occurrence required; no parser repair",
    }
    (outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    del model
    torch.cuda.empty_cache()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=EXP / "data/validation_sft.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--split", default="validation")
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A BF16 capable CUDA GPU is required for evaluation")
    summary = evaluate_one(args.adapter, load_rows(args.data), args.output_dir, args.max_cases, args.split, args.data)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
