#!/usr/bin/env python3
"""Freeze an unexposed test cohort and case-isolated router train/validation split.

The selection artifact stores only row indexes and SHA-256 hashes. It never
stores prompt text, gold labels, or hidden atomic facts.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "ea_atpo_stage"))
import run_direct_reliability_eval as direct  # noqa: E402
import run_l2d_remedi_confirmation as l2d  # noqa: E402
import run_medical_candidate_eval as candidate  # noqa: E402

OUT = HERE / "outputs/qwen32_defer_validation_20261119_attempt01"
TEST_N, TRAIN_N, VAL_N = 200, 200, 100
SEED = 20261119
DEV_PATH = ROOT / "experiments/ea_atpo_v1_smoke/ea_atpo_v1_attempt09_stability/data/test.parquet"
TRAIN_DATA = candidate.DATA_PATH.parent / "train.pilot.parquet"


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def prompt_hash(row: dict[str, Any]) -> str:
    messages = direct.build_messages(row)
    return sha_text(json.dumps(messages, ensure_ascii=False, sort_keys=True))


def historical_exposure() -> tuple[set[str], set[int], int]:
    """Read trace metadata only; no response text is loaded into selection."""
    hashes: set[str] = set()
    source_rows: set[int] = set()
    scanned = 0
    for path in (HERE / "outputs").rglob("*.jsonl"):
        if OUT in path.parents:
            continue
        try:
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    scanned += 1
                    for key in ("case_hash", "public_state_sha256", "prompt_sha256"):
                        value = row.get(key)
                        if isinstance(value, str) and len(value) == 64:
                            hashes.add(value)
                    case_id = row.get("case_id")
                    if isinstance(case_id, str) and case_id.startswith("test-row-"):
                        try:
                            source_rows.add(int(case_id.split("-", 2)[2].split(":", 1)[0]))
                        except (ValueError, IndexError):
                            pass
                    if isinstance(row.get("source_row_index"), int) and row.get("model_role") in {
                        "base_8b", "base_qwen3_8b", "qwen3_8b", "qwen3_32b_awq_defer"
                    }:
                        source_rows.add(int(row["source_row_index"]))
        except (OSError, UnicodeDecodeError):
            continue
    # The 2026-11-04 frozen direct-answer cohort is excluded even if a trace
    # format change omitted its case hashes.
    test_frame = pd.read_parquet(candidate.DATA_PATH)
    source_rows.update(random.Random(candidate.SEED).sample(range(len(test_frame)), candidate.CASE_COUNT))
    return hashes, source_rows, scanned


def main() -> None:
    if OUT.exists() and (OUT / "case_selection.json").exists():
        raise FileExistsError(f"Selection is already frozen: {OUT / 'case_selection.json'}")
    if not DEV_PATH.is_file() or not TRAIN_DATA.is_file() or not candidate.DATA_PATH.is_file():
        raise FileNotFoundError("Pinned train/test/development parquet is missing")
    prior_hashes, prior_indices, scanned = historical_exposure()

    test_frame = pd.read_parquet(candidate.DATA_PATH)
    if len(test_frame) != 1268:
        raise RuntimeError(f"Expected locked ATPO test split with 1268 rows; found {len(test_frame)}")
    # Build hashes from the public-state representation to exclude previous
    # DEFER/CONSULT cohorts as well as the frozen direct-answer cohort.
    test_pool: list[dict[str, Any]] = []
    test_seen: set[str] = set()
    excluded_prior_indices = excluded_prior_hash = duplicate_test = 0
    order = list(range(len(test_frame)))
    random.Random(SEED).shuffle(order)
    for ix in order:
        row = test_frame.iloc[ix].to_dict()
        case_digest, _, _, _, _ = l2d.case_key(row, int(ix))
        ph = prompt_hash(row)
        if ix in prior_indices:
            excluded_prior_indices += 1
            continue
        if case_digest in prior_hashes or ph in prior_hashes:
            excluded_prior_hash += 1
            continue
        if case_digest in test_seen or ph in test_seen:
            duplicate_test += 1
            continue
        test_seen.add(case_digest)
        test_pool.append({"source_row_index": int(ix), "case_hash": case_digest, "prompt_hash": ph})
        if len(test_pool) == TEST_N:
            break
    if len(test_pool) != TEST_N:
        raise RuntimeError(f"Only {len(test_pool)} previously unused test cases are available")

    # Exclude the Attempt09 smoke/dev cohort from router fitting, then split by
    # case_id before any supervised fitting or threshold selection.
    dev_frame = pd.read_parquet(DEV_PATH)
    dev_hashes: set[str] = set()
    for ix, raw in dev_frame.iterrows():
        row = raw.to_dict()
        dev_hashes.add(prompt_hash(row))
    train_frame = pd.read_parquet(TRAIN_DATA)
    train_order = list(range(len(train_frame)))
    random.Random(SEED + 1).shuffle(train_order)
    test_prompt_hashes = {r["prompt_hash"] for r in test_pool}
    selected: list[dict[str, Any]] = []
    seen_train: set[str] = set()
    excluded_cross = duplicate_train = 0
    for ix in train_order:
        row = train_frame.iloc[ix].to_dict()
        ph = prompt_hash(row)
        if ph in test_prompt_hashes or ph in dev_hashes:
            excluded_cross += 1
            continue
        if ph in seen_train:
            duplicate_train += 1
            continue
        seen_train.add(ph)
        digest, _, _, _, _ = l2d.case_key(row, int(ix))
        if digest in prior_hashes or ph in prior_hashes:
            excluded_cross += 1
            continue
        selected.append({"source_row_index": int(ix), "case_hash": digest, "prompt_hash": ph})
        if len(selected) == TRAIN_N + VAL_N:
            break
    if len(selected) != TRAIN_N + VAL_N:
        raise RuntimeError(f"Only {len(selected)} fresh train/validation cases are available")
    train_rows = [{**r, "case_id": f"train-row-{r['source_row_index']}", "split": "train"} for r in selected[:TRAIN_N]]
    val_rows = [{**r, "case_id": f"train-row-{r['source_row_index']}", "split": "validation"} for r in selected[TRAIN_N:]]
    test_rows = [{**r, "case_id": f"test-row-{r['source_row_index']}", "split": "test"} for r in test_pool]

    all_prompt_hashes = [r["prompt_hash"] for r in train_rows + val_rows + test_rows]
    all_case_hashes = [r["case_hash"] for r in train_rows + val_rows + test_rows]
    if len(set(all_prompt_hashes)) != len(all_prompt_hashes) or len(set(all_case_hashes)) != len(all_case_hashes):
        raise RuntimeError("Case hash collision or case-level leakage across train/validation/test")
    OUT.mkdir(parents=True, exist_ok=True)
    selection = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED,
        "counts": {"train": TRAIN_N, "validation": VAL_N, "test": TEST_N},
        "source_data": {
            "train_path": str(TRAIN_DATA), "train_sha256": sha_file(TRAIN_DATA),
            "test_path": str(candidate.DATA_PATH), "test_sha256": sha_file(candidate.DATA_PATH),
            "test_rows": len(test_frame),
            "attempt09_dev_sha256": sha_file(DEV_PATH), "attempt09_dev_rows": len(dev_frame),
        },
        "exclusion_audit": {
            "historical_jsonl_rows_scanned": scanned,
            "historical_hashes_seen": len(prior_hashes),
            "historical_source_rows_excluded": len(prior_indices),
            "test_rows_excluded_by_historical_source_index": excluded_prior_indices,
            "test_rows_excluded_by_historical_public_hash": excluded_prior_hash,
            "duplicate_test_cases_skipped": duplicate_test,
            "train_candidates_excluded_for_history_or_dev_or_test_overlap": excluded_cross,
            "duplicate_train_candidates_skipped": duplicate_train,
        },
        "case_ids": {"train": [r["case_id"] for r in train_rows], "validation": [r["case_id"] for r in val_rows], "test": [r["case_id"] for r in test_rows]},
        "rows": {"train": train_rows, "validation": val_rows, "test": test_rows},
        "split_hashes": {
            "train_prompt_sha256": sha_text("\n".join(r["prompt_hash"] for r in train_rows)),
            "validation_prompt_sha256": sha_text("\n".join(r["prompt_hash"] for r in val_rows)),
            "test_prompt_sha256": sha_text("\n".join(r["prompt_hash"] for r in test_rows)),
            "all_case_ids_sha256": sha_text("\n".join(r["case_id"] for r in train_rows + val_rows + test_rows)),
        },
        "gold_or_hidden_fact_content_stored": False,
    }
    path = OUT / "case_selection.json"
    path.write_text(json.dumps(selection, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "frozen", "path": str(path), "counts": selection["counts"], "exclusion_audit": selection["exclusion_audit"], "split_hashes": selection["split_hashes"]}, indent=2))


if __name__ == "__main__":
    main()
