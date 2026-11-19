#!/usr/bin/env python3
"""Run fixed Qwen3-8B Base on exactly the SFT Expert prompt/cases.

Base receives the same public messages as Expert, with the supervised target
removed. Option probabilities are scored at the common answer-prefix boundary
and used only as decision-time Base uncertainty features.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
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
    "HF_HOME": EXP / "cache/hf_home", "HF_HUB_CACHE": EXP / "cache/hf_home/hub",
    "HF_DATASETS_CACHE": EXP / "cache/datasets", "XDG_CACHE_HOME": EXP / "cache/xdg",
    "TMPDIR": EXP / "cache/tmp", "TRITON_CACHE_DIR": EXP / "cache/triton", "TORCH_HOME": EXP / "cache/torch",
}.items():
    os.environ.setdefault(key, str(value))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


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


def parse_answer(text: str) -> str | None:
    matches = re.findall(r"Expert Final Answer:\s*([A-D])\b", str(text), re.I)
    return matches[0].upper() if len(matches) == 1 else None


def gold_from_target(row: dict[str, Any]) -> str:
    match = re.fullmatch(r"\s*Expert Final Answer:\s*([A-D])\s*", row["messages"][-1]["content"], re.I)
    if not match:
        raise ValueError("SFT target protocol invalid")
    return match.group(1).upper()


def public_features(messages: list[dict[str, str]]) -> dict[str, float]:
    text = "\n".join(m["content"] for m in messages if m["role"] != "system")
    option_texts = re.findall(r"(?m)^\s*[A-D]\s*[).:：]\s*(.*?)\s*$", text)
    option_labels = set(re.findall(r"(?m)^\s*([A-D])\s*[).:：]\s*", text))
    for match in re.finditer(r"Options:\s*(\{[^\n]*\})", text, re.I):
        try:
            options = ast.literal_eval(match.group(1))
        except (ValueError, SyntaxError):
            continue
        if isinstance(options, dict):
            values = {str(k).strip().upper(): str(v) for k, v in options.items() if str(k).strip().upper() in "ABCD"}
            if len(values) > len(option_labels):
                option_labels = set(values)
                option_texts = list(values.values())
    words = re.findall(r"\b\w+\b", text, re.UNICODE)
    lower = text.lower()
    lexicon = ("pain", "fever", "cough", "blood", "dose", "drug", "symptom", "diagnosis", "treatment", "history", "onset", "patient", "exam", "test", "surgery", "infection", "pressure", "heart", "lung", "kidney", "pregnan")
    return {
        "log_public_chars": math.log1p(len(text)),
        "log_public_words": math.log1p(len(words)),
        "option_count": float(len(option_labels)),
        "mean_option_chars": float(sum(map(len, option_texts)) / len(option_texts)) if option_texts else 0.0,
        "question_marks": float(text.count("?") + text.count("？")),
        "digit_count": float(sum(c.isdigit() for c in text)),
        "clinical_lexicon_count": float(sum(lower.count(word) for word in lexicon)),
    }


def tokenized_candidate(tokenizer, messages, letter: str) -> list[int]:
    full = list(messages) + [{"role": "assistant", "content": f"Expert Final Answer: {letter}"}]
    rendered = tokenizer.apply_chat_template(full, tokenize=False, add_generation_prompt=False, enable_thinking=False)
    return tokenizer(rendered, add_special_tokens=False)["input_ids"]


def option_distribution(model, tokenizer, messages) -> dict[str, float]:
    candidate_ids = {letter: tokenized_candidate(tokenizer, messages, letter) for letter in "ABCD"}
    min_len = min(map(len, candidate_ids.values()))
    common = 0
    while common < min_len and len({ids[common] for ids in candidate_ids.values()}) == 1:
        common += 1
    if common == 0 or common >= min_len:
        raise RuntimeError("Unable to identify the answer-letter decision boundary")
    option_ids = {letter: ids[common] for letter, ids in candidate_ids.items()}
    if len(set(option_ids.values())) != 4:
        raise RuntimeError("The A-D candidates do not map to four distinct next tokens")
    context_ids = candidate_ids["A"][:common]
    input_ids = torch.tensor([context_ids], dtype=torch.long, device="cuda")
    attention = torch.ones_like(input_ids)
    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention, use_cache=False).logits[0, -1].float()
    values = torch.tensor([logits[option_ids[x]].item() for x in "ABCD"], dtype=torch.float64)
    probs = torch.softmax(values, dim=0).tolist()
    return {letter: float(prob) for letter, prob in zip("ABCD", probs, strict=True)}


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-cases", type=int)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 CUDA GPU required for Base evaluation")
    model_path = EXP / "assets/models/Qwen3-8B"
    if not (model_path / "model.safetensors.index.json").is_file():
        raise FileNotFoundError(f"Pinned Base checkpoint is missing or incomplete: {model_path}")
    if args.output.exists() or args.output.with_suffix(".summary.json").exists():
        raise FileExistsError(f"Refusing to overwrite prior Base evaluation: {args.output}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="sdpa", trust_remote_code=False).to("cuda")
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    rows = load_rows(args.data)
    selected = rows[:args.max_cases] if args.max_cases else rows
    results = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for row in selected:
        visible_messages = row["messages"][:-1]
        # Feature extraction uses the same model and visible context but ends at
        # the answer-choice boundary; no target/gold is included.
        probs = option_distribution(model, tokenizer, visible_messages)
        prompt = tokenizer.apply_chat_template(visible_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        t0 = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(**inputs, do_sample=False, max_new_tokens=64, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, use_cache=True)
        torch.cuda.synchronize()
        latency = time.perf_counter() - t0
        continuation = output[0, prompt_tokens:]
        raw = tokenizer.decode(continuation, skip_special_tokens=True).strip()
        predicted = parse_answer(raw)
        gold = gold_from_target(row)
        ordered = sorted(probs.values(), reverse=True)
        entropy = -sum(p * math.log(max(p, 1e-12)) for p in probs.values()) / math.log(4)
        results.append({
            "case_id": row["case_id"], "split": args.split,
            "case_hash": row["case_hash"], "prompt_hash": row["prompt_hash"],
            "predicted": predicted, "valid": predicted is not None,
            "gold": gold, "correct": bool(predicted == gold),
            "option_probabilities": probs,
            "base_entropy_normalized": entropy,
            "base_probability_margin": ordered[0] - ordered[1],
            "base_confidence": ordered[0],
            **public_features(visible_messages),
            "prompt_tokens": prompt_tokens, "completion_tokens": int(continuation.shape[-1]),
            "latency_s": round(latency, 6), "raw_output_sha256": sha256_text(raw), "raw_output": raw,
        })
    with args.output.open("w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    valid = sum(x["valid"] for x in results)
    correct = sum(x["correct"] for x in results)
    summary = {
        "status": "completed", "model": "Qwen/Qwen3-8B", "revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "git_commit": git_commit(), "data_sha256": sha256_file(args.data),
        "split": args.split, "n": len(results), "accuracy_end_to_end": correct / len(results) if results else None,
        "parse_success_rate": valid / len(results) if results else None,
        "mean_prompt_tokens": statistics.mean(x["prompt_tokens"] for x in results) if results else None,
        "mean_completion_tokens": statistics.mean(x["completion_tokens"] for x in results) if results else None,
        "mean_latency_s": statistics.mean(x["latency_s"] for x in results) if results else None,
        "predictions_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "generation_config": {"do_sample": False, "max_new_tokens": 64, "enable_thinking": False, "dtype": "bfloat16", "seed": 20261119},
        "gpu_snapshot": gpu_snapshot(),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1),
        "gold_or_hidden_facts_sent_to_model": False, "prompt_matches_expert_sft": True,
        "parse_failures_count_incorrect": True,
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
