#!/usr/bin/env python3
"""Assistant-only LoRA SFT for the pinned Qwen3-32B checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
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
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


class AnswerOnlyDataset(torch.utils.data.Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.items = []
        for row in rows:
            messages = row["messages"]
            prompt = tokenizer.apply_chat_template(
                messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            full = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
            )
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
            if full_ids[: len(prompt_ids)] != prompt_ids:
                raise ValueError(f"Template boundary changed for case {row['case_id']}")
            if len(full_ids) > self.max_length:
                raise ValueError(f"Case {row['case_id']} exceeds max_sequence_length={self.max_length}; refusing silent truncation")
            labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
            if not any(token != -100 for token in labels):
                raise ValueError(f"No assistant loss tokens for case {row['case_id']}")
            self.items.append({"input_ids": full_ids, "labels": labels})

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.items[index]


class AssistantOnlyCollator:
    def __init__(self, pad_id: int):
        self.pad_id = int(pad_id)

    def __call__(self, items: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(item["input_ids"]) for item in items)
        input_rows, label_rows, masks = [], [], []
        for item in items:
            pad = max_len - len(item["input_ids"])
            input_rows.append(item["input_ids"] + [self.pad_id] * pad)
            label_rows.append(item["labels"] + [-100] * pad)
            masks.append([1] * len(item["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_rows, dtype=torch.long),
            "labels": torch.tensor(label_rows, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }


class SafeLogCallback(TrainerCallback):
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        allowed = {k: float(v) for k, v in (logs or {}).items() if isinstance(v, (int, float))}
        record = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "step": int(state.global_step), "metrics": allowed}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def gpu_snapshot() -> dict[str, Any]:
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
    ).strip().splitlines()
    parsed = []
    for row in rows:
        cols = [x.strip() for x in row.split(",")]
        parsed.append({"index": int(cols[0]), "name": cols[1], "total_mib": int(cols[2]), "used_mib": int(cols[3]), "free_mib": int(cols[4]), "utilization_pct": int(cols[5])})
    return {"physical_gpu": int(os.environ.get("EA_ATPO_SFT_PHYSICAL_GPU", "5")), "gpus": parsed}


def assert_postload_gpu_reserve(required_free_mib: int) -> dict[str, Any]:
    physical_gpu = int(os.environ.get("EA_ATPO_SFT_PHYSICAL_GPU", "5"))
    line = subprocess.check_output(
        ["nvidia-smi", f"--id={physical_gpu}", "--query-gpu=index,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    index, free_mib, utilization = [int(x.strip()) for x in line.split(",")]
    if index != physical_gpu or free_mib < required_free_mib:
        raise RuntimeError(f"Post-load GPU reserve failed before optimizer updates: gpu={index}, free_mib={free_mib}, required_mib={required_free_mib}")
    app_pids = subprocess.check_output(
        ["nvidia-smi", f"--id={physical_gpu}", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    others = [x.strip() for x in app_pids.splitlines() if x.strip() and x.strip().upper() != "N/A" and int(x.strip()) != os.getpid()]
    if others:
        raise RuntimeError(f"Another compute process appeared on GPU {physical_gpu} before optimizer updates: {others}")
    return {"physical_gpu": index, "free_mib_after_load_and_gradient_preflight": free_mib, "utilization_pct": utilization, "required_reserve_mib": required_free_mib, "other_compute_pids": others}


def run_manifest(path: Path, data_path: Path, validation_path: Path, config_path: Path, output_dir: Path, smoke_steps: int | None, started: float) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = {
        "status": "completed",
        "started_at_utc": datetime.fromtimestamp(started, timezone.utc).isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_at_start": git_commit(),
        "model_repo_id": config["model"]["repo_id"],
        "model_revision": config["model"]["revision"],
        "model_local_path": config["model"]["local_path"],
        "data_sha256": {"train": sha256_file(data_path), "validation": sha256_file(validation_path), "config": sha256_file(config_path)},
        "optimizer_steps_requested": smoke_steps if smoke_steps else None,
        "smoke_mode": bool(smoke_steps),
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "accelerate")},
        "training_config": config["training"],
        "lora_config": config["lora"],
        "gpu_snapshot": gpu_snapshot(),
        "cuda": {"available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda, "bf16_supported": torch.cuda.is_bf16_supported()},
        "peak_cuda_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "peak_cuda_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1),
        "elapsed_seconds": round(time.time() - started, 2),
        "training_logs_contain_prompts_gold_or_hidden_facts": False,
        "expert_output_protocol": "Expert Final Answer: A-D",
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=EXP / "configs/domain_sft.json")
    parser.add_argument("--train", type=Path, default=EXP / "data/train_sft.jsonl")
    parser.add_argument("--validation", type=Path, default=EXP / "data/validation_sft.jsonl")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke-steps", type=int)
    parser.add_argument("--epochs", type=float)
    parser.add_argument("--skip-gradient-preflight", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Qwen3-32B SFT run")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Selected GPU/runtime does not report BF16 support")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output_dir = args.output_dir or (EXP / "checkpoints/qwen3_32b_domain_sft/smoke" if args.smoke_steps else EXP / "checkpoints/qwen3_32b_domain_sft/formal")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = EXP / "logs" / ("sft_smoke_training.jsonl" if args.smoke_steps else "sft_formal_training.jsonl")
    started = time.time()
    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))
    torch.cuda.reset_peak_memory_stats()

    model_path = ROOT / config["model"]["local_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_rows = read_jsonl(args.train)
    val_rows = read_jsonl(args.validation)
    train_data = AnswerOnlyDataset(train_rows, tokenizer, config["training"]["max_sequence_length"])
    val_data = AnswerOnlyDataset(val_rows, tokenizer, config["training"]["max_sequence_length"])

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        trust_remote_code=False,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lora = config["lora"]
    model = get_peft_model(model, LoraConfig(
        r=int(lora["r"]), lora_alpha=int(lora["alpha"]), lora_dropout=float(lora["dropout"]),
        target_modules=list(lora["target_modules"]), bias="none", task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()
    if not args.skip_gradient_preflight:
        batch = AssistantOnlyCollator(tokenizer.pad_token_id)([train_data[0]])
        batch = {key: value.to("cuda") for key, value in batch.items()}
        model.train()
        output = model(**batch)
        if not torch.isfinite(output.loss):
            raise RuntimeError("One-batch forward loss is not finite")
        output.loss.backward()
        trainable_grads = [p.grad.detach().float().norm() for p in model.parameters() if p.requires_grad and p.grad is not None]
        grad_norm = torch.stack(trainable_grads).norm().item() if trainable_grads else 0.0
        if not grad_norm or not torch.isfinite(torch.tensor(grad_norm)):
            raise RuntimeError("One-batch backward produced no finite trainable LoRA gradients")
        preflight = {
            "status": "passed", "forward_loss_finite": True, "lora_grad_norm": grad_norm,
            "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
            "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1),
            "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "input_examples": 1, "gold_or_hidden_facts_passed_to_model": False,
        }
        (output_dir / "single_batch_gradient_preflight.json").write_text(json.dumps(preflight, indent=2) + "\n", encoding="utf-8")
        model.zero_grad(set_to_none=True)
        del batch, output
        torch.cuda.empty_cache()

    reserve_check = assert_postload_gpu_reserve(int(config["resources"]["reserve_mib_after_load"]))
    (output_dir / "postload_gpu_safety_check.json").write_text(json.dumps(reserve_check, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    train_args = config["training"]
    kwargs = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": int(train_args["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(train_args["per_device_eval_batch_size"]),
        "gradient_accumulation_steps": int(train_args["gradient_accumulation_steps"]),
        "learning_rate": float(train_args["learning_rate"]),
        "num_train_epochs": float(args.epochs or train_args["epochs"]),
        "max_steps": int(args.smoke_steps or -1),
        "warmup_ratio": float(train_args["warmup_ratio"]),
        "weight_decay": float(train_args["weight_decay"]),
        "max_grad_norm": float(train_args["max_grad_norm"]),
        "logging_steps": 1 if args.smoke_steps else int(train_args["logging_steps"]),
        "save_strategy": "steps" if args.smoke_steps else "epoch",
        "save_steps": int(args.smoke_steps) if args.smoke_steps else None,
        "save_total_limit": None if not args.smoke_steps else 1,
        "bf16": True,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "optim": str(train_args["optimizer"]),
        "lr_scheduler_type": str(train_args["scheduler"]),
        "dataloader_num_workers": int(train_args["dataloader_num_workers"]),
        "report_to": [],
        "seed": int(config["seed"]),
        "data_seed": int(config["seed"]),
        "remove_unused_columns": False,
        "logging_first_step": True,
        "save_safetensors": True,
    }
    if args.smoke_steps:
        kwargs["eval_strategy"] = "no"
    else:
        kwargs["eval_strategy"] = "epoch"
        kwargs["eval_on_start"] = False
    signature = inspect.signature(TrainingArguments).parameters
    kwargs = {k: v for k, v in kwargs.items() if k in signature}
    training_args = TrainingArguments(**kwargs)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=None if args.smoke_steps else val_data,
        data_collator=AssistantOnlyCollator(tokenizer.pad_token_id),
        callbacks=[SafeLogCallback(log_path)],
    )
    result = trainer.train()
    trainer.save_model(str(output_dir / "final_adapter"))
    tokenizer.save_pretrained(output_dir / "final_adapter")
    trainer.save_state()
    (output_dir / "trainer_metrics.json").write_text(json.dumps({
        "train_result": result.metrics,
        "global_step": trainer.state.global_step,
        "trainable_parameters": sum(p.numel() for p in trainer.model.parameters() if p.requires_grad),
        "optimizer_step_count": int(trainer.state.global_step),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    run_manifest(output_dir / "run_manifest.json", args.train, args.validation, args.config, output_dir, args.smoke_steps, started)
    print(json.dumps({"status": "completed", "global_step": trainer.state.global_step, "output_dir": str(output_dir), "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
