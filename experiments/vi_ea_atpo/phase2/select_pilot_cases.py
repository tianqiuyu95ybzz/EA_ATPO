#!/usr/bin/env python3
"""Freeze 100 train and 50 validation cases after global historical exclusion."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import random
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PHASE2 = ROOT / "experiments/vi_ea_atpo/phase2"
SELECTOR_PATH = ROOT / "experiments/real_case_pilot_20261103/audit/select_qwen32_router_expansion_cases.py"
sys.path.insert(0, str(SELECTOR_PATH.parent))
TRAIN_COUNT = 100
VALIDATION_COUNT = 50
SAMPLE_SEED = 20261118


def load_selector():
    spec = importlib.util.spec_from_file_location("history_selector_vi", SELECTOR_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(SELECTOR_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=PHASE2)
    parser.add_argument("--seed", type=int, default=SAMPLE_SEED)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "pilot_cases.csv"
    manifest_path = out / "selection_manifest.json"
    if csv_path.exists() or manifest_path.exists():
        raise FileExistsError(f"selection is frozen; refusing to replace {csv_path}")

    selector = load_selector()
    selector.OUT = out
    cache_path = Path(tempfile.gettempdir()) / f"vi_ea_atpo_history_{os.getpid()}.json"
    if cache_path.exists():
        raise FileExistsError(cache_path)
    selector.CACHE = cache_path
    try:
        historical_hashes, train_indices, _test_indices, ambiguous_indices, counts = selector.collect_history()
        train = pd.read_parquet(selector.TRAIN_DATA)
        dev = pd.read_parquet(selector.DEV_DATA)
        dev_hashes = set()
        for ix in range(len(dev)):
            digest, prompt_hash = selector.row_identity(dev, ix)
            dev_hashes.update((digest, prompt_hash))
        historical_hashes |= dev_hashes

        order = list(range(len(train)))
        random.Random(args.seed).shuffle(order)
        selected = []
        rejected = {
            "historical_source_index": 0,
            "ambiguous_source_index": 0,
            "historical_hash_or_prompt": 0,
            "duplicate_within_new_selection": 0,
        }
        selected_hashes: set[str] = set()
        for ix in order:
            if ix in train_indices:
                rejected["historical_source_index"] += 1
                continue
            if ix in ambiguous_indices:
                rejected["ambiguous_source_index"] += 1
                continue
            digest, prompt_hash = selector.row_identity(train, ix)
            if digest in historical_hashes or prompt_hash in historical_hashes:
                rejected["historical_hash_or_prompt"] += 1
                continue
            if digest in selected_hashes or prompt_hash in selected_hashes:
                rejected["duplicate_within_new_selection"] += 1
                continue
            selected_hashes.update((digest, prompt_hash))
            selected.append({"source_row_index": int(ix), "case_hash": digest, "prompt_hash": prompt_hash})
            if len(selected) == TRAIN_COUNT + VALIDATION_COUNT:
                break
        if len(selected) != TRAIN_COUNT + VALIDATION_COUNT:
            raise RuntimeError(f"only {len(selected)} unseen training cases available")

        rows = []
        for split, group in (("train", selected[:TRAIN_COUNT]), ("validation", selected[TRAIN_COUNT:])):
            for ordinal, item in enumerate(group, 1):
                rows.append({
                    "case_id": item["case_hash"],
                    "split": split,
                    "source_row_index": item["source_row_index"],
                    "case_hash": item["case_hash"],
                    "prompt_hash": item["prompt_hash"],
                    "sample_ordinal": ordinal,
                })
        ids = [r["case_id"] for r in rows]
        prompts = [r["prompt_hash"] for r in rows]
        if len(set(ids)) != len(ids) or len(set(prompts)) != len(prompts):
            raise RuntimeError("duplicate case or prompt across frozen train/validation sample")

        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        manifest = {
            "experiment": "VI-EA-ATPO active information acquisition pilot case selection",
            "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "repo_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "sample_seed": args.seed,
            "source_train_path": str(selector.TRAIN_DATA),
            "source_train_sha256": selector.sha_file(selector.TRAIN_DATA),
            "source_train_rows": len(train),
            "attempt09_development_rows_excluded_by_identity": len(dev),
            "historical_scan_roots": [str(x) for x in selector.HISTORY_ROOTS],
            "historical_scan_counts": counts,
            "historical_unique_public_hash_count": len(historical_hashes),
            "historical_train_source_index_count": len(train_indices),
            "ambiguous_source_index_count": len(ambiguous_indices),
            "exclusion_counts": rejected,
            "selected": {"train": TRAIN_COUNT, "validation": VALIDATION_COUNT, "test": 0},
            "selection_rule": "uniform seeded sample from ATPO train rows after historical case-index/hash exclusion; no outcome labels or uncertainty used",
            "previous_attempt04_400_holdout_excluded": True,
            "official_test_used": False,
            "case_ids_sha256": sha_text("\n".join(ids)),
            "prompt_hashes_sha256": sha_text("\n".join(prompts)),
            "pilot_cases_csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
            "case_ids_are_prompt_hashes": True,
            "gold_or_atomic_facts_stored": False,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps({
            "status": "completed",
            "counts": manifest["selected"],
            "history_counts": counts,
            "history_exclusions": rejected,
            "output": str(csv_path),
        }, indent=2))
    finally:
        cache_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
