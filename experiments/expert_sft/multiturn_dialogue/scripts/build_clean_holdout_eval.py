#!/usr/bin/env python3
"""Materialize the frozen 400-case cohort only after the SFT checkpoint lock."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
MULTI = ROOT / "experiments/expert_sft/multiturn_dialogue"
EXP = ROOT / "experiments/expert_sft"
SELECTION = EXP / "data/fresh_test_selection.json"
PARQUET = EXP / "data/reconstructed_compat/train.pilot.parquet"
BUILDER_PATH = EXP / "scripts/build_sft_dataset.py"
FROZEN_SYSTEM = (
    "You are a medical multiple-choice diagnostic expert. Use only the visible case, "
    "answer options, and public conversation below. Do not assume hidden patient facts "
    "or use an answer key. Return exactly one line in this format: "
    "Expert Final Answer: <one uppercase option letter A-D>. Do not include reasoning."
)


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_builder():
    spec = importlib.util.spec_from_file_location("multiturn_eval_sft_builder", BUILDER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(BUILDER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, default=SELECTION)
    parser.add_argument("--parquet", type=Path, default=PARQUET)
    parser.add_argument("--checkpoint-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=MULTI / "data/clean_holdout_eval.jsonl")
    args = parser.parse_args()
    output = args.output.resolve()
    if MULTI.resolve() not in output.parents:
        raise ValueError(f"Holdout output must be under {MULTI}")
    if output.exists() or output.with_suffix(".manifest.json").exists():
        raise FileExistsError(f"Refusing to overwrite/open a prior clean holdout materialization: {output}")
    lock = json.loads(args.checkpoint_lock.read_text(encoding="utf-8"))
    if lock.get("status") != "selected_on_validation_only" or lock.get("clean_holdout_read") is not False:
        raise RuntimeError("SFT checkpoint lock must be validation-only and must state clean_holdout_read=false")
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if selection.get("status") != "frozen_before_sft_checkpoint_selection":
        raise RuntimeError("Fresh holdout was not frozen before checkpoint selection")
    if sha_file(args.parquet) != json.loads((EXP / "configs/domain_sft.json").read_text())["data"]["expected_source_sha256"]:
        raise RuntimeError("The frozen source Parquet hash changed")
    selected_rows = selection.get("rows", [])
    if len(selected_rows) != 400 or len({str(r["case_id"]) for r in selected_rows}) != 400:
        raise RuntimeError("Expected exactly 400 frozen, unique holdout cases")
    existing_train = {int(r["source_row_index"]) for name in ("train_sft.jsonl", "validation_sft.jsonl", "internal_test_sft.jsonl") for r in (json.loads(x) for x in (MULTI / "data" / name).read_text().splitlines() if x.strip())}
    if existing_train & {int(r["source_row_index"]) for r in selected_rows}:
        raise RuntimeError("Fresh holdout overlaps a multi-turn SFT case")

    spec = importlib.util.spec_from_file_location("multiturn_source_builder", BUILDER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(BUILDER_PATH)
    builder = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = builder
    spec.loader.exec_module(builder)
    frame = pd.read_parquet(args.parquet)
    hasher = builder.load_public_state_hasher()
    records = []
    seen_source = set()
    for item in selected_rows:
        index = int(item["source_row_index"])
        if index in seen_source or index < 0 or index >= len(frame):
            raise RuntimeError(f"Invalid or repeated source row {index}")
        seen_source.add(index)
        row = frame.iloc[index].to_dict()
        case_id = str(item["case_id"])
        if case_id != f"train-row-{index}":
            raise RuntimeError(f"Frozen case_id/source_row mismatch at index {index}")
        _, case_hash, _, _ = hasher(row, index)
        if case_hash != str(item["case_hash"]):
            raise RuntimeError(f"Frozen case hash mismatch at {case_id}")
        visible_messages = builder.messages_for_row(row, FROZEN_SYSTEM)
        # The public prompt may already contain multiple turns. Only its visible
        # messages are retained. Hidden atomic facts are never read here.
        prompt_hash = hashlib.sha256(json.dumps(visible_messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        gold = builder.extract_gold(row)
        visible_messages.append({"role": "assistant", "content": f"Expert Final Answer: {gold}"})
        records.append({
            "case_id": case_id, "source_row_index": index, "state_id": None,
            "split": "clean_holdout", "case_hash": case_hash,
            "prompt_sha256": prompt_hash, "messages": visible_messages,
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest = {
        "status": "materialized_after_validation_checkpoint_lock_for_final_evaluation",
        "selection_sha256": sha_file(args.selection), "source_parquet_sha256": sha_file(args.parquet),
        "checkpoint_lock_sha256": sha_file(args.checkpoint_lock), "case_count": len(records),
        "output_sha256": sha_file(output), "source_row_overlap_with_sft_cases": 0,
        "gold_read_for_training_or_checkpoint_selection": False,
        "gold_used_only_as_offline_evaluation_target": True,
        "hidden_atomic_facts_read": False, "hidden_atomic_facts_sent_to_model": False,
        "all_model_inputs_are_frozen_public_prompt_messages": True,
        "prompt_system_sha256": hashlib.sha256(FROZEN_SYSTEM.encode()).hexdigest(),
        "model_selection_uses_clean_holdout": False,
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "cases": len(records), "output_sha256": manifest["output_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
