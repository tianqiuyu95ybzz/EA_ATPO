#!/usr/bin/env python3
"""Assistant-only LoRA SFT for frozen public ATPO dialogue states.

AWQ is loaded through GPTQModel's trainable torch_awq backend. BF16 is a
separate fallback mode. This script never edits the source checkpoint and
stores every artifact below experiments/expert_sft/multiturn_dialogue.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
import shutil
import subprocess
import time
from datetime import datetime, timezone
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


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    if not rows:
        raise ValueError(f"Empty dataset: {path}")
    return rows


def visible_prompt(messages: list[dict[str, str]], tokenizer: Any) -> tuple[list[int], list[int]]:
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("Chat template prompt is not a prefix of the supervised sequence")
    return prompt_ids, full_ids


def encode_rows(rows: list[dict[str, Any]], tokenizer: Any, max_length: int) -> list[dict[str, Any]]:
    items = []
    for row in rows:
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) != 3:
            raise ValueError(f"Expected system/user/assistant example for {row.get('case_id')}")
        if [m.get("role") for m in messages] != ["system", "user", "assistant"]:
            raise ValueError(f"Unexpected dialogue role sequence for {row.get('case_id')}")
        target = str(messages[-1].get("content", ""))
        import re
        if re.fullmatch(r"Expert Final Answer: [A-D]", target) is None:
            raise ValueError(f"Invalid answer-only target for {row.get('case_id')}")
        prompt_ids, full_ids = visible_prompt(messages, tokenizer)
        if len(full_ids) > max_length:
            raise ValueError(f"Sequence length {len(full_ids)} exceeds {max_length}; refusing truncation")
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        if not any(token != -100 for token in labels):
            raise ValueError(f"No assistant loss tokens for {row.get('case_id')}")
        items.append({"input_ids": full_ids, "labels": labels, "case_id": row["case_id"]})
    return items


def make_model(config: dict[str, Any], model_path: Path, trainable: bool):
    if config["model"].get("quant_method") == "awq":
        # This import must precede Transformers model imports: GPTQModel installs
        # the AWQ loading bridge and the torch_awq training implementation.
        import gptqmodel  # noqa: F401
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_config = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    expected = config["model"]
    if model_config.model_type != expected["expected_model_type"]:
        raise RuntimeError(f"Wrong model_type: {model_config.model_type}")
    if expected.get("quant_method") == "awq":
        qconfig = getattr(model_config, "quantization_config", None) or {}
        if str(qconfig.get("quant_method", "")).lower() != "awq":
            raise RuntimeError("Selected model does not advertise AWQ quantization")
        qconfig["backend"] = "torch_awq"
        model_config.quantization_config = qconfig
        base = AutoModelForCausalLM.from_pretrained(
            model_path, config=model_config, device_map={"": 0},
            local_files_only=True, trust_remote_code=False, attn_implementation="sdpa",
        )
    else:
        qconfig = getattr(model_config, "quantization_config", None)
        if qconfig:
            raise RuntimeError("BF16 fallback refuses a quantized checkpoint")
        base = AutoModelForCausalLM.from_pretrained(
            model_path, config=model_config, torch_dtype=torch.bfloat16,
            device_map={"": 0}, low_cpu_mem_usage=True, local_files_only=True,
            trust_remote_code=False, attn_implementation="sdpa",
        )
    if trainable:
        base.config.use_cache = False
        lora = config["lora"]
        model = get_peft_model(base, LoraConfig(
            r=int(lora["r"]), lora_alpha=int(lora["alpha"]),
            lora_dropout=float(lora["dropout"]), target_modules=list(lora["target_modules"]),
            bias=str(lora.get("bias", "none")), task_type="CAUSAL_LM",
        ))
        if config["training"].get("gradient_checkpointing", True):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
        model.print_trainable_parameters()
        return model, tokenizer
    return base, tokenizer


def gpu_state(gpu: int) -> dict[str, Any]:
    raw = subprocess.check_output([
        "nvidia-smi", f"--id={gpu}",
        "--query-gpu=index,name,uuid,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True).strip()
    col = [x.strip() for x in raw.split(",")]
    pids_raw = subprocess.check_output([
        "nvidia-smi", f"--id={gpu}", "--query-compute-apps=pid", "--format=csv,noheader,nounits",
    ], text=True).strip()
    return {
        "index": int(col[0]), "name": col[1], "uuid": col[2],
        "total_mib": int(col[3]), "used_mib": int(col[4]), "free_mib": int(col[5]),
        "utilization_pct": int(col[6]),
        "compute_pids": [x.strip() for x in pids_raw.splitlines() if x.strip() and x.strip().upper() != "N/A"],
    }


def assert_gpu_gate(gpu: int, resources: dict[str, Any], after_load: bool = False) -> dict[str, Any]:
    if gpu in {1, 7}:
        raise RuntimeError(f"GPU {gpu} is excluded")
    state = gpu_state(gpu)
    if after_load:
        external = [pid for pid in state["compute_pids"] if pid != os.getpid()]
        if state["free_mib"] < int(resources["minimum_free_mib_after_load_and_preflight"]) or external:
            raise RuntimeError(f"Post-load safety gate failed: free={state['free_mib']} MiB, external_compute_pids={external}")
        state["external_compute_pids"] = external
        state["gate"] = "passed_after_load"
    else:
        if state["free_mib"] < int(resources["minimum_free_mib_before_load"]) or state["utilization_pct"] > int(resources["maximum_utilization_pct_before_load"]) or state["compute_pids"]:
            raise RuntimeError(f"Pre-load safety gate failed: {state}")
        state["gate"] = "passed_before_load"
    return state


def batch_one(item: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([item["input_ids"]], dtype=torch.long, device=device),
        "attention_mask": torch.ones((1, len(item["input_ids"])), dtype=torch.long, device=device),
        "labels": torch.tensor([item["labels"]], dtype=torch.long, device=device),
    }


def val_loss(model, data: list[dict[str, Any]], device: torch.device) -> float:
    model.eval()
    losses = []
    with torch.inference_mode():
        for item in data:
            batch = batch_one(item, device)
            loss = model(**batch).loss.float()
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite validation loss")
            losses.append(float(loss.item()))
    return sum(losses) / len(losses)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("awq", "bf16"), required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--smoke-steps", type=int)
    args = parser.parse_args()
    config_path = args.config or MULTI / "configs" / ("awq_lora_sft.json" if args.mode == "awq" else "bf16_lora_sft.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (args.mode == "awq") != (config["model"].get("quant_method") == "awq"):
        raise ValueError("Mode and model quantization config disagree")
    physical = int(args.gpu)
    if physical in {1, 7}:
        raise RuntimeError(f"GPU {physical} is excluded")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(0)
    resources = config["resources"]
    pre_gpu = assert_gpu_gate(physical, resources, after_load=False)
    model_path = Path(config["model"]["local_path"])
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"Missing model config: {model_path}")
    train_path = args.train or ROOT / config["data"]["train"]
    val_path = args.validation or ROOT / config["data"]["validation"]
    source_lock_path = Path(config["model"]["source_lock"]) if config["model"].get("source_lock") else None
    if source_lock_path is not None and not source_lock_path.is_absolute():
        source_lock_path = ROOT / source_lock_path
    outdir = args.output_dir or MULTI / "checkpoints" / ("awq_lora" if args.mode == "awq" else "bf16_lora") / ("smoke" if args.smoke_steps else "formal")
    outdir = outdir.resolve()
    if MULTI.resolve() not in outdir.parents:
        raise ValueError(f"Outputs must stay under {MULTI}")
    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite checkpoint output: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    (MULTI / "logs").mkdir(parents=True, exist_ok=True)

    seed = int(config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    model, tokenizer = make_model(config, model_path, trainable=True)
    gpu_after_load = assert_gpu_gate(physical, resources, after_load=True)
    train_rows = read_rows(train_path)
    val_rows = read_rows(val_path)
    train_data = encode_rows(train_rows, tokenizer, int(config["data"]["max_sequence_length"]))
    val_data = encode_rows(val_rows, tokenizer, int(config["data"]["max_sequence_length"]))
    device = torch.device("cuda:0")
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("LoRA injection left no trainable parameters")
    trainable_n = sum(p.numel() for p in trainable)
    model.train()

    # One real supervised forward/backward/update is always performed before
    # accepting a run, including before any formal epoch starts.
    optimizer = torch.optim.AdamW(trainable, lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
    steps_target = int(args.smoke_steps or 0)
    epochs_target = 1 if steps_target else int(args.epochs or config["training"]["epochs"])
    if epochs_target < 1 or (steps_target and steps_target < 1):
        raise ValueError("Steps and epochs must be positive")
    grad_accum = int(config["training"]["gradient_accumulation_steps"])
    steps_per_epoch = math.ceil(len(train_data) / grad_accum)
    total_steps = steps_target if steps_target else epochs_target * steps_per_epoch + 1
    warmup_steps = int(total_steps * float(config["training"].get("warmup_ratio", 0.0)))
    def lr_scale(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(1e-8, (step + 1) / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scale)
    pre = [p.detach().float().clone() for p in trainable]
    optimizer.zero_grad(set_to_none=True)
    first = batch_one(train_data[0], device)
    first_out = model(**first)
    if not torch.isfinite(first_out.loss):
        raise RuntimeError("One-batch forward loss is not finite")
    first_loss = float(first_out.loss.detach().item())
    first_out.loss.backward()
    grad_parts = [p.grad.detach().float().norm() for p in trainable if p.grad is not None]
    grad_norm = float(torch.stack(grad_parts).norm().item()) if grad_parts else 0.0
    if not grad_norm or not math.isfinite(grad_norm):
        raise RuntimeError("Backward pass produced zero/non-finite LoRA gradients")
    torch.nn.utils.clip_grad_norm_(trainable, float(config["training"]["max_grad_norm"]))
    optimizer.step()
    scheduler.step()
    update_norm = math.sqrt(sum(float((p.detach().float() - old).pow(2).sum().item()) for p, old in zip(trainable, pre, strict=True)))
    if not update_norm or not math.isfinite(update_norm):
        raise RuntimeError("Optimizer step did not change LoRA parameters")
    preflight = {
        "status": "passed", "mode": args.mode, "forward_loss": first_loss,
        "lora_gradient_global_norm": grad_norm, "lora_parameter_update_l2": update_norm,
        "trainable_parameters": trainable_n,
        "prompt_or_gold_written_to_log": False,
        "gold_or_hidden_atomic_facts_sent_as_model_input": False,
        "loss_mask": "all system/user/public-dialogue tokens masked; assistant answer target only",
        "optimizer": "AdamW",
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1),
    }
    (outdir / "single_batch_gradient_update_preflight.json").write_text(json.dumps(preflight, indent=2) + "\n", encoding="utf-8")
    optimizer.zero_grad(set_to_none=True)
    del first, first_out, pre
    torch.cuda.empty_cache()

    # The gradient preflight above includes a real optimizer update. Count it
    # as step 1 so a 10-step smoke really performs 10 optimizer updates total.
    update = 1
    records = []
    best_val = float("inf")
    for epoch in range(1, epochs_target + 1):
        model.train()
        order = list(range(len(train_data)))
        random.Random(seed + epoch).shuffle(order)
        epoch_losses: list[float] = []
        accum_count = 0
        for position, idx in enumerate(order):
            batch = batch_one(train_data[idx], device)
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at optimizer step {update + 1}")
            epoch_losses.append(float(loss.detach().item()))
            (loss / int(config["training"]["gradient_accumulation_steps"])).backward()
            accum_count += 1
            last_in_epoch = position == len(order) - 1
            if accum_count >= int(config["training"]["gradient_accumulation_steps"]) or last_in_epoch:
                torch.nn.utils.clip_grad_norm_(trainable, float(config["training"]["max_grad_norm"]))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update += 1
                accum_count = 0
                if steps_target and update >= steps_target:
                    break
        epoch_val = val_loss(model, val_data, device)
        checkpoint = outdir / f"epoch-{epoch:02d}"
        checkpoint.mkdir(parents=False, exist_ok=False)
        model.save_pretrained(checkpoint, safe_serialization=True)
        tokenizer.save_pretrained(checkpoint)
        record = {
            "epoch": epoch, "optimizer_steps_total": update,
            "train_assistant_token_loss_mean": sum(epoch_losses) / len(epoch_losses),
            "validation_assistant_token_loss_mean": epoch_val,
            "training_examples_seen": len(epoch_losses),
            "validation_examples": len(val_data),
            "learning_rate_after_epoch": scheduler.get_last_lr()[0],
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
            "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1),
        }
        records.append(record)
        with (MULTI / "logs" / f"{args.mode}_training_metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), **record}, ensure_ascii=False) + "\n")
        if epoch_val < best_val:
            best_val = epoch_val
        if steps_target:
            break

    # Smoke tests must also prove adapter serialization and exact reload.
    final_adapter = outdir / f"epoch-{records[-1]['epoch']:02d}"
    if steps_target:
        try:
            from peft import PeftModel, get_peft_model_state_dict
            from safetensors.torch import load_file
            saved_weights = final_adapter / "adapter_model.safetensors"
            saved_state = load_file(str(saved_weights), device="cpu")
            del model
            torch.cuda.empty_cache()
            reloaded_base, _ = make_model({**config, "training": {**config["training"], "gradient_checkpointing": False}}, model_path, trainable=False)
            reloaded = PeftModel.from_pretrained(reloaded_base, final_adapter, is_trainable=False)
            actual = get_peft_model_state_dict(reloaded)
            same_keys = set(actual) == set(saved_state)
            same_values = same_keys and all(torch.equal(actual[k].detach().cpu(), saved_state[k]) for k in saved_state)
            if not same_values:
                raise RuntimeError("Adapter reload differs from saved LoRA tensors")
            del reloaded, reloaded_base
            torch.cuda.empty_cache()
            preflight["adapter_save_reload_roundtrip"] = "passed"
            preflight["adapter_state_tensor_count"] = len(saved_state)
            (outdir / "single_batch_gradient_update_preflight.json").write_text(json.dumps(preflight, indent=2) + "\n", encoding="utf-8")
        except Exception as exc:
            preflight["adapter_save_reload_roundtrip"] = "failed"
            preflight["adapter_reload_error_class"] = type(exc).__name__
            preflight["adapter_reload_error"] = str(exc)[:1000]
            (outdir / "single_batch_gradient_update_preflight.json").write_text(json.dumps(preflight, indent=2) + "\n", encoding="utf-8")
            raise

    manifest = {
        "status": "completed", "mode": args.mode,
        "started_at_utc": datetime.fromtimestamp(started, timezone.utc).isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(), "config_sha256": sha_file(config_path),
        "data_sha256": {"train": sha_file(train_path), "validation": sha_file(val_path)},
        "model": config["model"], "model_local_path": str(model_path),
        "source_lock_sha256": sha_file(source_lock_path) if source_lock_path and source_lock_path.is_file() else None,
        "package_versions": {name: package_version(name) for name in ("torch", "transformers", "peft", "accelerate", "optimum", "safetensors")},
        "gptqmodel_version": package_version("gptqmodel") if args.mode == "awq" else None,
        "optimizer_steps": update, "epochs_requested": epochs_target,
        "scheduler": {"name": config["training"]["scheduler"], "warmup_steps": warmup_steps, "total_steps": total_steps, "final_lr": scheduler.get_last_lr()[0]},
        "training_examples": len(train_data), "validation_examples": len(val_data),
        "training_unique_cases": len({r["case_id"] for r in train_rows}),
        "validation_unique_cases": len({r["case_id"] for r in val_rows}),
        "preload_gpu": pre_gpu, "postload_gpu": gpu_after_load,
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1),
        "epoch_metrics": records,
        "selected_by": "minimum_validation_assistant_token_cross_entropy",
        "selected_checkpoint_epoch": min(records, key=lambda x: x["validation_assistant_token_loss_mean"])["epoch"],
        "gold_or_hidden_atomic_facts_sent_as_model_input": False,
        "training_logs_contain_case_text_or_gold": False,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    (outdir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not steps_target:
        selected = min(records, key=lambda x: x["validation_assistant_token_loss_mean"])
        selected_source = ROOT / selected["checkpoint"]
        selected_dir = outdir / "selected_by_val_loss"
        selected_dir.mkdir(exist_ok=False)
        for filename in (
            "adapter_config.json", "adapter_model.safetensors", "adapter_model.bin",
            "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
            "vocab.json", "merges.txt", "added_tokens.json",
        ):
            source = selected_source / filename
            if source.is_file():
                shutil.copy2(source, selected_dir / filename)
        if not (selected_dir / "adapter_config.json").is_file() or not (selected_dir / "adapter_model.safetensors").is_file():
            raise RuntimeError("Selected checkpoint copy is incomplete")
        selected_digest = hashlib.sha256()
        for file in sorted(p for p in selected_dir.rglob("*") if p.is_file()):
            selected_digest.update(file.relative_to(selected_dir).as_posix().encode())
            with file.open("rb") as stream:
                for block in iter(lambda: stream.read(1 << 20), b""):
                    selected_digest.update(block)
        lock = {
            "status": "selected_on_validation_only",
            "selected_checkpoint": str(selected_dir.relative_to(ROOT)),
            "selected_checkpoint_sha256": selected_digest.hexdigest(),
            "selection_metric": "minimum_validation_assistant_token_cross_entropy",
            "selection_value": selected["validation_assistant_token_loss_mean"],
            "selection_epoch": selected["epoch"],
            "validation_data_sha256": sha_file(val_path),
            "clean_holdout_read": False,
            "case_split_ids_or_gold_used_for_selection": False,
        }
        lock_path = outdir / "selected_checkpoint_lock.json"
        if lock_path.exists():
            raise FileExistsError(lock_path)
        lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifest["selected_checkpoint_lock"] = str(lock_path.relative_to(ROOT))
        (outdir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "mode": args.mode, "optimizer_steps": update, "selected_checkpoint_epoch": manifest["selected_checkpoint_epoch"], "manifest": str(outdir / "run_manifest.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
