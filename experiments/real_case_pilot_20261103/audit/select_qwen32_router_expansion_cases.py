#!/usr/bin/env python3
"""Freeze a larger case-isolated cohort after excluding prior EA-ATPO exposure.

The scan reads only historical metadata identifiers/hashes for exclusion. The
selection artifact never contains prompt text, Gold, or hidden atomic facts.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
SIBLING = Path("/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO")
OUT = HERE / "outputs/qwen32_router_generalization_20261118_attempt04"
CACHE = OUT / "history_exclusion_cache.json"
TRAIN_DATA = SIBLING / "experiments/real_case_pilot_20261103/assets/train.pilot.parquet"
TEST_DATA = SIBLING / "experiments/real_case_pilot_20261103/assets/test.pilot.parquet"
DEV_DATA = ROOT / "experiments/ea_atpo_v1_smoke/ea_atpo_v1_attempt09_stability/data/test.parquet"
TRAIN_N, VAL_N, TEST_N = 1000, 300, 400
SEED = 20261128
HISTORY_ROOTS = [
    ROOT / "experiments",
    SIBLING / "experiments",
]
sys.path.insert(0, str(ROOT / "ea_atpo_stage"))
import run_direct_reliability_eval as direct  # noqa: E402
import run_l2d_remedi_confirmation as l2d  # noqa: E402


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def public_prompt_hash(row: dict[str, Any]) -> str:
    payload = json.dumps(direct.build_messages(row), ensure_ascii=False, sort_keys=True)
    return sha_text(payload)


def classify_history_split(record: dict[str, Any], path: Path) -> str | None:
    split = str(record.get("split", "")).lower()
    if split in {"train", "validation", "val"}:
        return "train"
    if split == "test":
        return "test"
    cid = str(record.get("case_id", ""))
    match = re.match(r"^(train|test)-row-(\d+)", cid)
    if match:
        return match.group(1)
    low = str(path).lower()
    if any(x in low for x in ("direct_eval", "medical_expert_candidates", "strong_expert", "relative_value", "stepwise_test", "final_test", "medical_candidate_defer")):
        return "test"
    if any(x in low for x in ("pilot_public_states", "/outputs/full/", "advice_utilization", "consult_quality", "/smoke/", "smoke_attempt", "attempt09_stability", "online_validation", "traces/train", "traces/validation")):
        return "train"
    return None


def collect_history() -> tuple[set[str], set[int], set[int], set[int], dict[str, int]]:
    if CACHE.is_file():
        cached = json.loads(CACHE.read_text(encoding="utf-8"))
        if cached.get("train_data_sha256") != sha_file(TRAIN_DATA) or cached.get("test_data_sha256") != sha_file(TEST_DATA):
            raise RuntimeError("History exclusion cache belongs to different source Parquet files")
        return (set(cached["hashes"]), set(cached["train_indices"]), set(cached["test_indices"]), set(cached["ambiguous_indices"]), cached["counts"])
    hashes: set[str] = set()
    train_indices: set[int] = set()
    test_indices: set[int] = set()
    ambiguous_indices: set[int] = set()
    history_counts = {"files": 0, "rows": 0, "selection_manifests": 0, "sample_selection_manifests": 0, "sample_selection_rows_excluded": 0}
    jsonl_files: set[Path] = set()
    for root in HISTORY_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.jsonl"):
            low = str(path).lower()
            if "/assets/" in low or "/.git/" in low or OUT in path.parents:
                continue
            jsonl_files.add(path)
    for file_number, path in enumerate(sorted(jsonl_files), 1):
        try:
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    history_counts["rows"] += 1
                    if not isinstance(row, dict):
                        continue
                    for key in ("case_hash", "public_state_sha256", "prompt_hash", "prompt_sha256", "prompt_sha"):
                        value = row.get(key)
                        if isinstance(value, str) and len(value) == 64:
                            hashes.add(value)
                    source_ix = row.get("source_row_index")
                    cid = str(row.get("case_id", ""))
                    match = re.match(r"^(?:train|test)-row-(\d+)", cid)
                    if match:
                        source_ix = int(match.group(1))
                    if isinstance(source_ix, int):
                        bucket = classify_history_split(row, path)
                        if bucket == "train":
                            train_indices.add(source_ix)
                        elif bucket == "test":
                            test_indices.add(source_ix)
                        else:
                            ambiguous_indices.add(source_ix)
        except (OSError, UnicodeDecodeError):
            continue
        if file_number % 10 == 0 or file_number == len(jsonl_files):
            print(f"history audit progress: {file_number}/{len(jsonl_files)} JSONL files", flush=True)
    history_counts["files"] = len(jsonl_files)

    # Frozen prior cohort manifests carry selected IDs even when an old trace
    # format did not include a source index on each response row.
    for root in HISTORY_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("case_selection.json"):
            if OUT in path.parents:
                continue
            try:
                selection = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            history_counts["selection_manifests"] += 1
            for split, bucket in (("train", "train"), ("validation", "train"), ("test", "test")):
                for spec in selection.get("rows", {}).get(split, []):
                    ix = spec.get("source_row_index")
                    if isinstance(ix, int):
                        (train_indices if bucket == "train" else test_indices).add(ix)
                    for key in ("case_hash", "prompt_hash"):
                        value = spec.get(key)
                        if isinstance(value, str) and len(value) == 64:
                            hashes.add(value)

    # Older Router analyses used sample_selection_manifest.json rather than
    # case_selection.json. Those manifests can hold the only surviving source
    # row list for prior validation/test cohorts, so inspect them explicitly.
    test_sha = sha_file(TEST_DATA)
    for root in HISTORY_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*selection*manifest*.json"):
            if OUT in path.parents or path.name == "case_selection.json":
                continue
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            source_hash = str(manifest.get("data_sha256", ""))
            source_path = str(manifest.get("source_data", ""))
            # Only map row numbers when the manifest points to this exact
            # parquet snapshot; row numbers from any other split are unrelated.
            if source_hash != test_sha or Path(source_path).name != TEST_DATA.name:
                continue
            history_counts["sample_selection_manifests"] += 1
            for key in ("validation_rows", "test_rows"):
                values = manifest.get(key, [])
                if not isinstance(values, list):
                    continue
                for item in values:
                    ix = item if isinstance(item, int) else item.get("source_row_index") if isinstance(item, dict) else None
                    if isinstance(ix, int):
                        test_indices.add(ix)
                        history_counts["sample_selection_rows_excluded"] += 1
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    cache = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train_data_sha256": sha_file(TRAIN_DATA), "test_data_sha256": test_sha,
        "hashes": sorted(hashes), "train_indices": sorted(train_indices),
        "test_indices": sorted(test_indices), "ambiguous_indices": sorted(ambiguous_indices),
        "counts": history_counts,
        "note": "Local exclusion metadata only; contains no prompt text, Gold, or hidden facts. Do not publish case indexes/hashes.",
    }
    CACHE.write_text(json.dumps(cache, ensure_ascii=False) + "\n", encoding="utf-8")
    return hashes, train_indices, test_indices, ambiguous_indices, history_counts


def row_identity(frame: pd.DataFrame, index: int) -> tuple[str, str]:
    row = frame.iloc[index].to_dict()
    case_hash, _, _, _, _ = l2d.case_key(row, index)
    return str(case_hash), public_prompt_hash(row)


def main() -> None:
    if (OUT / "case_selection.json").exists():
        raise FileExistsError(f"This selection is frozen; refusing to replace it: {OUT / 'case_selection.json'}")
    for path in (TRAIN_DATA, TEST_DATA, DEV_DATA):
        if not path.is_file():
            raise FileNotFoundError(path)
    hashes, old_train_indices, old_test_indices, ambiguous_indices, history_counts = collect_history()
    # Attempt09 development cases are kept outside all three new cohorts.
    dev = pd.read_parquet(DEV_DATA)
    dev_hashes = set()
    for ix, raw in dev.iterrows():
        digest, ph = row_identity(dev, int(ix))
        dev_hashes.update((digest, ph))
    hashes |= dev_hashes

    train = pd.read_parquet(TRAIN_DATA)
    test = pd.read_parquet(TEST_DATA)
    if len(train) < TRAIN_N + VAL_N or len(test) != 1268:
        raise RuntimeError(f"Unexpected parquet size: train={len(train)}, test={len(test)}")
    rng = random.Random(SEED)
    train_order = list(range(len(train)))
    rng.shuffle(train_order)

    selected_fresh: list[dict[str, Any]] = []
    selected_hashes: set[str] = set()
    excluded = {"historical_source_index": 0, "historical_hash": 0, "dev": len(dev), "duplicate": 0}
    for ix in train_order:
        if ix in old_train_indices or ix in ambiguous_indices:
            excluded["historical_source_index"] += 1
            continue
        digest, ph = row_identity(train, ix)
        if digest in hashes or ph in hashes:
            excluded["historical_hash"] += 1
            continue
        if digest in selected_hashes or ph in selected_hashes:
            excluded["duplicate"] += 1
            continue
        selected_hashes.update((digest, ph))
        selected_fresh.append({"source_row_index": ix, "case_hash": digest, "prompt_hash": ph})
        if len(selected_fresh) == TRAIN_N + VAL_N + TEST_N:
            break

    if len(selected_fresh) != TRAIN_N + VAL_N + TEST_N:
        raise RuntimeError(
            "Insufficient unseen cases after global history audit: "
            f"fresh cases from ATPO train split={len(selected_fresh)}/{TRAIN_N + VAL_N + TEST_N}; no inference was started."
        )

    rows: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for spec in selected_fresh[:TRAIN_N]:
        rows["train"].append({**spec, "case_id": f"train-row-{spec['source_row_index']}", "split": "train"})
    for spec in selected_fresh[TRAIN_N:TRAIN_N + VAL_N]:
        rows["validation"].append({**spec, "case_id": f"train-row-{spec['source_row_index']}", "split": "validation"})
    for spec in selected_fresh[TRAIN_N + VAL_N:]:
        rows["test"].append({**spec, "case_id": f"train-row-{spec['source_row_index']}", "split": "test"})

    all_cases = [r for split in ("train", "validation", "test") for r in rows[split]]
    for field in ("case_id", "case_hash", "prompt_hash"):
        if len({r[field] for r in all_cases}) != len(all_cases):
            raise RuntimeError(f"Duplicate {field} across new split boundaries")
    selection = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED,
        "counts": {name: len(rows[name]) for name in rows},
        "source_data": {
            "train_path": str(TRAIN_DATA), "train_sha256": sha_file(TRAIN_DATA), "train_rows": len(train),
            "official_test_path": str(TEST_DATA), "official_test_sha256": sha_file(TEST_DATA), "official_test_rows": len(test),
            "attempt09_dev_sha256": sha_file(DEV_DATA), "attempt09_dev_rows": len(dev),
        },
        "split_source": {"train": "ATPO train split", "validation": "ATPO train split", "test": "fresh case_id holdout from ATPO train split; historical ATPO test split was nearly exhausted by prior evaluations"},
        "source_parquet_split": {"train": "train", "validation": "train", "test": "train"},
        "history_audit": {
            **history_counts,
            "unique_historical_public_hashes": len(hashes),
            "historical_train_source_indices": len(old_train_indices),
            "historical_test_source_indices": len(old_test_indices),
            "ambiguous_source_indices_excluded_from_both": len(ambiguous_indices),
            "exclusion_counts": excluded,
            "scanned_project_roots": [str(p) for p in HISTORY_ROOTS],
        },
        "rows": rows,
        "split_hashes": {
            name: sha_text("\n".join(row["case_hash"] for row in rows[name])) for name in rows
        },
        "all_case_ids_sha256": sha_text("\n".join(row["case_id"] for row in all_cases)),
        "gold_or_hidden_fact_content_stored": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "case_selection.json"
    path.write_text(json.dumps(selection, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "frozen", "path": str(path), "counts": selection["counts"], "split_source": selection["split_source"], "history_audit": selection["history_audit"]}, indent=2))


if __name__ == "__main__":
    main()
