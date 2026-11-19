#!/usr/bin/env python3
"""Greedy evaluation on identical public multi-turn states/cases."""
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

ROOT = Path(__file__).resolve().parents[4]
MULTI = ROOT / "experiments/expert_sft/multiturn_dialogue"
for key, rel in {
    "HF_HOME": ".cache/hf_home", "HF_HUB_CACHE": ".cache/hf_home/hub",
    "HF_DATASETS_CACHE": ".cache/datasets", "XDG_CACHE_HOME": ".cache/xdg",
    "TMPDIR": ".cache/tmp", "TRITON_CACHE_DIR": ".cache/triton", "TORCH_HOME": ".cache/torch",
}.items():
    os.environ.setdefault(key, str(MULTI / rel))

import torch  # noqa: E402


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def answer_target(row: dict[str, Any]) -> str:
    match = re.fullmatch(r"\s*Expert Final Answer:\s*([A-D])\s*", str(row["messages"][-1]["content"]), re.I)
    if not match:
        raise ValueError(f"Invalid evaluation target for case_id={row.get('case_id')}")
    return match.group(1).upper()


def parse_answer(raw: str) -> str | None:
    matches = re.findall(r"Expert Final Answer:\s*([A-D])\b", raw, re.I)
    return matches[0].upper() if len(matches) == 1 else None


def snapshot_gpu(gpu: int) -> dict[str, Any]:
    cols = subprocess.check_output([
        "nvidia-smi", f"--id={gpu}",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True).strip().split(",")
    pids = subprocess.check_output([
        "nvidia-smi", f"--id={gpu}", "--query-compute-apps=pid", "--format=csv,noheader,nounits",
    ], text=True).strip()
    return {
        "index": int(cols[0].strip()), "name": cols[1].strip(),
        "total_mib": int(cols[2]), "used_mib": int(cols[3]), "free_mib": int(cols[4]),
        "utilization_pct": int(cols[5]),
        "compute_pids": [x.strip() for x in pids.splitlines() if x.strip() and x.strip().upper() != "N/A"],
    }


def load_model(kind: str, model_path: Path, adapter: Path | None):
    if kind in {"awq", "awq_sft"}:
        # Required before Transformers constructs the AWQ quantizer/linears.
        import gptqmodel  # noqa: F401
    from peft import PeftModel
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    if kind in {"awq", "awq_sft"}:
        q = getattr(config, "quantization_config", None) or {}
        if str(q.get("quant_method", "")).lower() != "awq":
            raise RuntimeError("AWQ evaluation requested for a non-AWQ model")
        q["backend"] = "torch_awq"
        config.quantization_config = q
        model = AutoModelForCausalLM.from_pretrained(
            model_path, config=config, device_map={"": 0}, local_files_only=True,
            trust_remote_code=False, attn_implementation="sdpa",
        )
    else:
        if getattr(config, "quantization_config", None):
            raise RuntimeError("Dense/BF16 evaluation refuses quantized weights")
        dtype = torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(
            model_path, config=config, torch_dtype=dtype, device_map={"": 0},
            low_cpu_mem_usage=True, local_files_only=True, trust_remote_code=False,
            attn_implementation="sdpa",
        )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    model.eval()
    model.config.use_cache = True
    return model, tokenizer


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    if not rows:
        raise ValueError(f"No rows in {path}")
    ids = [(str(x.get("case_id")), str(x.get("state_id", x.get("prompt_hash", "single")))) for x in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Evaluation data has duplicate case/state identities")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("base8b", "awq", "awq_sft", "bf16_sft"), required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-revision", default="unknown")
    parser.add_argument("--model-source-lock", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--max-cases", type=int)
    args = parser.parse_args()
    if args.gpu in {1, 7}:
        raise SystemExit(f"GPU {args.gpu} is excluded")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(0)
    if args.kind in {"awq_sft", "bf16_sft"} and args.adapter is None:
        raise ValueError("SFT evaluation requires an adapter")
    outdir = args.output_dir.resolve()
    if MULTI.resolve() not in outdir.parents:
        raise ValueError(f"Evaluation output must stay under {MULTI}")
    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite prior evaluation: {outdir}")
    data_path = args.data.resolve()
    model_path = args.model_path.resolve()
    rows = load_rows(data_path)
    selected = rows[:args.max_cases] if args.max_cases else rows
    started = time.time()
    gpu_before = snapshot_gpu(args.gpu)
    if gpu_before["free_mib"] < 150000 or gpu_before["utilization_pct"] > 20 or gpu_before["compute_pids"]:
        raise RuntimeError(f"Evaluation GPU safety gate closed: {gpu_before}")
    torch.cuda.reset_peak_memory_stats()
    model, tokenizer = load_model(args.kind, model_path, args.adapter)
    gpu_after = snapshot_gpu(args.gpu)
    if gpu_after["free_mib"] < 32768:
        raise RuntimeError(f"Post-load GPU reserve too low: {gpu_after}")
    results = []
    for row in selected:
        visible = row["messages"][:-1]
        prompt = tokenizer.apply_chat_template(visible, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        encoded = {key: value.to("cuda:0") for key, value in encoded.items()}
        prompt_tokens = int(encoded["input_ids"].shape[-1])
        t0 = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **encoded, do_sample=False, max_new_tokens=64,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id, use_cache=True,
            )
        torch.cuda.synchronize()
        latency = time.perf_counter() - t0
        continuation = generated[0, prompt_tokens:]
        raw = tokenizer.decode(continuation, skip_special_tokens=True).strip()
        pred = parse_answer(raw)
        # Read the target only after generation; no target/gold reaches model input.
        gold = answer_target(row)
        results.append({
            "case_id": str(row["case_id"]), "state_id": row.get("state_id"),
            "split": args.split, "prompt_hash": row.get("prompt_sha256", row.get("prompt_hash")),
            "case_hash": row.get("case_hash"), "predicted": pred,
            "valid": pred is not None, "gold": gold, "correct": pred == gold,
            "prompt_tokens": prompt_tokens, "completion_tokens": int(continuation.shape[-1]),
            "latency_s": round(latency, 6), "raw_output": raw,
            "raw_output_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        })
    outdir.mkdir(parents=True, exist_ok=False)
    pred_path = outdir / "predictions.jsonl"
    with pred_path.open("x", encoding="utf-8") as stream:
        for row in results:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    n = len(results)
    summary = {
        "status": "completed", "kind": args.kind,
        "model_path": str(model_path), "model_revision": args.model_revision,
        "model_source_lock": str(args.model_source_lock.resolve()) if args.model_source_lock else None,
        "model_source_lock_sha256": sha_file(args.model_source_lock) if args.model_source_lock and args.model_source_lock.is_file() else None,
        "adapter_path": str(args.adapter.resolve()) if args.adapter else None,
        "model_config_sha256": sha_file(model_path / "config.json"),
        "adapter_config_sha256": sha_file(args.adapter / "adapter_config.json") if args.adapter and (args.adapter / "adapter_config.json").is_file() else None,
        "adapter_weights_sha256": sha_file(args.adapter / "adapter_model.safetensors") if args.adapter and (args.adapter / "adapter_model.safetensors").is_file() else None,
        "input_data_sha256": sha_file(data_path), "split": args.split,
        "n_examples": n, "n_cases": len({r["case_id"] for r in results}),
        "accuracy_end_to_end": sum(r["correct"] for r in results) / n if n else None,
        "parse_success_rate": sum(r["valid"] for r in results) / n if n else None,
        "mean_prompt_tokens": statistics.mean(r["prompt_tokens"] for r in results) if n else None,
        "mean_completion_tokens": statistics.mean(r["completion_tokens"] for r in results) if n else None,
        "mean_latency_s": statistics.mean(r["latency_s"] for r in results) if n else None,
        "generation_config": {"greedy": True, "max_new_tokens": 64, "enable_thinking": False},
        "gold_or_hidden_facts_sent_to_model": False,
        "invalid_output_scored_incorrect": True,
        "prompt_policy": "same fixed Expert system prompt and public case/dialogue for all compared models",
        "gpu_before_load": gpu_before, "gpu_after_load": gpu_after,
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1),
        "elapsed_seconds": round(time.time() - started, 3),
        "git_commit": git_commit(), "predictions_sha256": sha_file(pred_path),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
