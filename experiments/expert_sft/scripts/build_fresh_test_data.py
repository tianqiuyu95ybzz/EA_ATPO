#!/usr/bin/env python3
"""Materialize the already-frozen fresh test cohort for evaluation only."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/expert_sft"
SELECTION = EXP / "data/fresh_test_selection.json"
PARQUET = EXP / "data/reconstructed_compat/train.pilot.parquet"
OUTPUT = EXP / "data/fresh_test_sft.jsonl"
BUILDER_PATH = EXP / "scripts/build_sft_dataset.py"


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, default=SELECTION)
    parser.add_argument("--parquet", type=Path, default=PARQUET)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace frozen fresh-test data: {args.output}")
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    config = json.loads((EXP / "configs/domain_sft.json").read_text(encoding="utf-8"))
    if selection.get("status") != "frozen_before_sft_checkpoint_selection":
        raise RuntimeError("Fresh test cohort was not frozen before validation checkpoint selection")
    if file_sha(args.parquet) != config["data"]["expected_source_sha256"]:
        raise RuntimeError("Fresh test source Parquet hash mismatch")
    spec = importlib.util.spec_from_file_location("sft_builder_eval", BUILDER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError("Could not load the frozen SFT data builder")
    builder = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = builder
    spec.loader.exec_module(builder)
    frame = pd.read_parquet(args.parquet)
    hasher = builder.load_public_state_hasher()
    cases = []
    seen = set()
    for item in selection["rows"]:
        index = int(item["source_row_index"])
        row = frame.iloc[index].to_dict()
        _, case_hash, _, _ = hasher(row, index)
        prompt_hash = builder.sha256_text(json.dumps(builder.messages_for_row(row, builder.BASE_SYSTEM), ensure_ascii=False, sort_keys=True))
        if case_hash != item["case_hash"] or prompt_hash != item["prompt_hash"]:
            raise RuntimeError(f"Fresh test case identity mismatch at source row {index}")
        if item["case_id"] in seen:
            raise RuntimeError("Duplicate case id in fresh test selection")
        seen.add(item["case_id"])
        messages = builder.messages_for_row(row, builder.SFT_SYSTEM)
        messages.append({"role": "assistant", "content": f"Expert Final Answer: {builder.extract_gold(row)}"})
        cases.append({
            "case_id": item["case_id"], "source_row_index": index, "split": "test",
            "case_hash": case_hash, "prompt_hash": prompt_hash, "messages": messages,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
    manifest = {
        "status": "materialized_after_freeze_for_final_evaluation_only",
        "selection_sha256": file_sha(args.selection),
        "source_data_sha256": file_sha(args.parquet),
        "count": len(cases),
        "output_sha256": file_sha(args.output),
        "train_or_validation_use": False,
        "checkpoint_selection_use": False,
        "gold_sent_to_model": False,
        "hidden_atomic_facts_sent_to_model": False,
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "count": len(cases), "selection_sha256": manifest["selection_sha256"], "output_sha256": manifest["output_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
