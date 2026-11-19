#!/usr/bin/env python3
"""Freeze a checkpoint using validation-only, predeclared lexicographic rules."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/expert_sft"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluations", type=Path, default=EXP / "evaluations/validation")
    parser.add_argument("--formal-dir", type=Path, default=EXP / "checkpoints/qwen3_32b_domain_sft/formal")
    parser.add_argument("--output", type=Path, default=EXP / "checkpoints/qwen3_32b_domain_sft/selected")
    args = parser.parse_args()
    candidates = []
    for result in args.evaluations.glob("checkpoint-*/summary.json"):
        summary = json.loads(result.read_text(encoding="utf-8"))
        if summary.get("split") != "validation" or summary.get("status") != "completed":
            continue
        adapter = Path(summary["adapter"])
        if not adapter.is_absolute():
            adapter = ROOT / adapter
        # Adapter points to the evaluated checkpoint's final_adapter subdir.
        trainer_state_path = adapter / "trainer_state.json"
        val_loss = None
        if trainer_state_path.exists():
            state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
            evals = [x.get("eval_loss") for x in state.get("log_history", []) if isinstance(x.get("eval_loss"), (int, float))]
            val_loss = evals[-1] if evals else None
        else:
            train_log = adapter / "trainer_metrics.json"
            if train_log.exists():
                info = json.loads(train_log.read_text(encoding="utf-8"))
                val_loss = info.get("eval_loss")
        summary["validation_loss"] = val_loss
        candidates.append((summary, adapter))
    if not candidates:
        raise RuntimeError("No completed validation checkpoint evaluations found")
    # Fixed primary order: higher end-to-end validation accuracy, higher parse
    # success, lower validation loss, fewer completion tokens, lower latency.
    candidates.sort(key=lambda pair: (
        -(pair[0].get("accuracy_end_to_end") or 0.0),
        -(pair[0].get("parse_success_rate") or 0.0),
        pair[0].get("validation_loss") if pair[0].get("validation_loss") is not None else float("inf"),
        pair[0].get("mean_completion_tokens") if pair[0].get("mean_completion_tokens") is not None else float("inf"),
        pair[0].get("mean_latency_s") if pair[0].get("mean_latency_s") is not None else float("inf"),
    ))
    best, adapter = candidates[0]
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace frozen selected adapter: {args.output}")
    args.output.mkdir(parents=True)
    for name in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"):
        source = adapter / name
        if source.is_file():
            shutil.copy2(source, args.output / name)
    if not (args.output / "adapter_config.json").is_file() or not any(
        (args.output / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")
    ):
        raise RuntimeError(f"Selected checkpoint is missing adapter files: {adapter}")
    digest = hashlib.sha256()
    for file in sorted(p for p in args.output.rglob("*") if p.is_file()):
        digest.update(file.relative_to(args.output).as_posix().encode())
        with file.open("rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                digest.update(block)
    lock = {
        "selection_split": "validation_only",
        "selection_order": ["validation end-to-end accuracy desc", "parse success desc", "validation loss asc", "mean completion tokens asc", "mean latency asc"],
        "selected_adapter": str(args.output.relative_to(ROOT)),
        "selected_adapter_tree_sha256": digest.hexdigest(),
        "selected_checkpoint_validation_summary": best,
        "candidate_count": len(candidates),
        "test_data_read": False,
    }
    lock_path = args.output.parent / "selected_checkpoint_lock.json"
    lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected": str(args.output), "validation_accuracy": best["accuracy_end_to_end"], "validation_parse_success": best["parse_success_rate"], "candidate_count": len(candidates)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
