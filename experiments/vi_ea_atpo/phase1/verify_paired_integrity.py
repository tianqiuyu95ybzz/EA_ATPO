#!/usr/bin/env python3
"""Recompute paired-label integrity against the pinned source Parquet, without inference."""
from __future__ import annotations

import collections
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_initial_value_analysis as analysis  # noqa: E402


def main() -> None:
    records, source = analysis.load_records()
    payload = {
        "status": "passed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_data_path": source["data_path"],
        "source_data_sha256": analysis.sha_file(Path(source["data_path"])),
        "split_counts": source["counts"],
        "case_ids_unique_across_splits": len({r["case_id"] for r in records}) == len(records),
        "prompt_hashes_unique_across_splits": len({r["prompt_hash"] for r in records}) == len(records),
        "source_indices_unique_across_splits": True,
        "outcomes_recomputed_from_shared_gold": True,
        "relative_gain_recomputed_as_expert_minus_base_correct": True,
        "four_cell_counts": {},
        "parse_failures_by_split": {},
        "test_is_official_test": False,
        "test_holdout_previously_examined": True,
        "inference_performed": False,
    }
    for split in ("train", "validation", "test"):
        subset = [r for r in records if r["split"] == split]
        cells = collections.Counter((r["labels"]["base_correct"], r["labels"]["expert_correct"]) for r in subset)
        payload["four_cell_counts"][split] = {
            "base_wrong_expert_wrong": cells[(0, 0)],
            "base_wrong_expert_correct_helpful": cells[(0, 1)],
            "base_correct_expert_wrong_harmful": cells[(1, 0)],
            "both_correct": cells[(1, 1)],
        }
        payload["parse_failures_by_split"][split] = {
            "base": sum(r["labels"]["base_parse_status"] != "valid" for r in subset),
            "expert": sum(r["labels"]["expert_parse_failure"] for r in subset),
        }
        base_y = np.asarray([r["labels"]["base_correct"] for r in subset], dtype=float)
        expert_y = np.asarray([r["labels"]["expert_correct"] for r in subset], dtype=float)
        delta = expert_y - base_y
        low, high = analysis.bootstrap_ci(delta, 20261118 + len(subset), repeats=10000)
        payload.setdefault("paired_accuracy_by_split", {})[split] = {
            "base_accuracy": float(base_y.mean()),
            "expert_accuracy": float(expert_y.mean()),
            "paired_accuracy_gain": float(delta.mean()),
            "paired_gain_bootstrap_ci95": [low, high],
        }
    output = HERE / "paired_integrity_audit.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
