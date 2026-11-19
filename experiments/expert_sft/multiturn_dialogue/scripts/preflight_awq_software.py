#!/usr/bin/env python3
"""CPU-only local AWQ config, tokenizer, and SFT data preflight.

This intentionally does not load model weights or touch CUDA. It records only
aggregate counts and hashes, never case IDs, prompts, answers, or patient text.
Run with the isolated `.venv_awq` environment after setup_awq_env.sh.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
MULTI = ROOT / "experiments/expert_sft/multiturn_dialogue"
for name, relative in {
    "HF_HOME": ".cache/hf_home",
    "HF_HUB_CACHE": ".cache/hf_home/hub",
    "HF_DATASETS_CACHE": ".cache/datasets",
    "XDG_CACHE_HOME": ".cache/xdg",
    "TMPDIR": ".cache/tmp",
}.items():
    os.environ.setdefault(name, str(MULTI / relative))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=MULTI / "configs/awq_lora_sft.json")
    parser.add_argument("--output", type=Path, required=True,
                        help="New aggregate JSON file under multiturn_dialogue/outputs")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_path = Path(config["model"]["local_path"])
    if not model_path.is_absolute():
        model_path = (ROOT / model_path).resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Configured local model directory missing: {config['model']['local_path']}")

    # GPTQModel installs the Transformers AWQ bridge; no model weights are loaded here.
    import gptqmodel  # noqa: F401
    from transformers import AutoConfig, AutoTokenizer

    model_config = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    model_type = model_config.model_type
    architectures = list(getattr(model_config, "architectures", []) or [])
    if model_type != config["model"]["expected_model_type"]:
        raise RuntimeError(f"Expected model_type={config['model']['expected_model_type']}, got {model_type}")
    if config["model"]["expected_architecture"] not in architectures:
        raise RuntimeError(f"Unexpected model architectures: {architectures}")
    quant = getattr(model_config, "quantization_config", None)
    if not isinstance(quant, dict) or str(quant.get("quant_method", "")).lower() != "awq":
        raise RuntimeError("Configured checkpoint does not contain an AWQ quantization config")
    quant["backend"] = "torch_awq"
    model_config.quantization_config = quant
    if model_config.quantization_config.get("backend") != "torch_awq":
        raise RuntimeError("Could not set the trainable torch_awq backend")

    lock_path = ROOT / config["model"]["source_lock"]
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    file_status = {}
    for item in lock["weight_files"] + lock["auxiliary_files"]:
        path = model_path / item["name"]
        file_status[item["name"]] = {
            "exists": path.is_file(),
            "size_matches_source_lock": path.is_file() and path.stat().st_size == item["size"],
        }
        if not file_status[item["name"]]["exists"] or not file_status[item["name"]]["size_matches_source_lock"]:
            raise RuntimeError(f"Model file missing or size mismatch: {item['name']}")
    # Verify hashes for small config/tokenizer assets; large model shards remain read-only and
    # were locked by the existing model source manifest.
    small_names = {"config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json"}
    for item in lock["auxiliary_files"]:
        if item["name"] in small_names:
            file_status[item["name"]]["sha256_matches_source_lock"] = sha256(model_path / item["name"]) == item["sha256"]
            if not file_status[item["name"]]["sha256_matches_source_lock"]:
                raise RuntimeError(f"Model asset hash mismatch: {item['name']}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    if not tokenizer.chat_template:
        raise RuntimeError("Tokenizer has no chat template")
    split_summary = {}
    for split in ("train", "validation", "internal_test"):
        dataset_path = ROOT / config["data"][split]
        rows = read_rows(dataset_path)
        prefix_matches = target_valid = 0
        max_tokens = 0
        for row in rows:
            messages = row.get("messages", [])
            if len(messages) != 3 or [message.get("role") for message in messages] != ["system", "user", "assistant"]:
                raise RuntimeError(f"Unexpected SFT message structure in split {split}")
            if re.fullmatch(r"Expert Final Answer: [A-D]", str(messages[-1].get("content", ""))) is None:
                raise RuntimeError(f"Invalid answer-only target in split {split}")
            prompt_text = tokenizer.apply_chat_template(
                messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            full_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
            )
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
            prefix_matches += full_ids[:len(prompt_ids)] == prompt_ids
            target_valid += len(full_ids) > len(prompt_ids)
            max_tokens = max(max_tokens, len(full_ids))
        if prefix_matches != len(rows) or target_valid != len(rows):
            raise RuntimeError(f"Prompt/target token preflight failed for split {split}")
        split_summary[split] = {
            "examples": len(rows),
            "prompt_prefix_matches": prefix_matches,
            "valid_answer_targets": target_valid,
            "max_full_sequence_tokens": max_tokens,
            "dataset_sha256": sha256(dataset_path),
        }

    packages = ("torch", "transformers", "peft", "accelerate", "gptqmodel", "optimum", "safetensors")
    result = {
        "status": "cpu_software_preflight_passed",
        "weights_loaded": False,
        "cuda_touched": False,
        "model_alias": config["model"]["repo_id"],
        "configured_local_model_path": config["model"]["local_path"],
        "huggingface_revision": lock.get("huggingface_revision"),
        "modelscope_revision": lock.get("modelscope_revision"),
        "model_type": model_type,
        "architectures": architectures,
        "quantization_config": {k: quant.get(k) for k in ("quant_method", "bits", "group_size", "version", "zero_point", "backend")},
        "model_files": file_status,
        "tokenizer": {
            "vocab_size": len(tokenizer),
            "chat_template_present": bool(tokenizer.chat_template),
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        },
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "splits": split_summary,
    }
    output = args.output.resolve()
    if MULTI.resolve() not in output.parents or "outputs" not in output.parts:
        raise ValueError("Output must be under multiturn_dialogue/outputs")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite preflight result: {output}")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(output.relative_to(ROOT)),
                      "splits": split_summary, "weights_loaded": False, "cuda_touched": False}, indent=2))


if __name__ == "__main__":
    main()
