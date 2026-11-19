#!/usr/bin/env python3
"""Freeze deterministic 50-case batches drawn only from the existing train split."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from run_counterfactual_pilot import PROJECT_DIR, V2_DIR, stable_case_rank, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=V2_DIR / "configs/phase1_counterfactual.json")
    parser.add_argument("--previous-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    split = json.loads((PROJECT_DIR / config["data"]["case_split"]).read_text(encoding="utf-8"))
    train_ids = set(map(str, split["train"]))
    if len(train_ids) < 50:
        raise SystemExit(f"Only {len(train_ids)} frozen train cases are available")
    states = read_jsonl(PROJECT_DIR / config["data"]["public_states"])
    state_case_ids = {str(row["case_id"]) for row in states}
    if not train_ids.issubset(state_case_ids):
        raise SystemExit("At least one frozen train case has no public state")
    ranked_selected = sorted(train_ids, key=lambda cid: stable_case_rank(cid, int(config["seed"])))[:50]
    old = json.loads(args.previous_manifest.read_text(encoding="utf-8"))["selected_case_ids"]
    if set(ranked_selected[:len(old)]) != set(old):
        raise SystemExit("Existing pilot case set does not match the deterministic top-ranked training cases")
    # Retain the previously frozen pilot order as batch 1, then continue the same
    # hash-ranked train-only list with 40 new cases.
    old_set = set(old)
    selected = list(old) + [case_id for case_id in ranked_selected if case_id not in old_set]
    batches = [selected[start:start + 10] for start in range(0, 50, 10)]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for number, batch in enumerate(batches, 1):
        (args.output_dir / f"batch_{number:02d}_case_ids.json").write_text(
            json.dumps(batch, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "seed": int(config["seed"]), "selection_method": "sha256(seed:case_id) rank over frozen train split; preserve original pilot order in batch 1",
        "train_only": True, "selected_case_count": len(selected),
        "selected_unique_case_count": len(set(selected)), "selected_state_count": sum(1 for row in states if row["case_id"] in set(selected)),
        "batch_size_cases": 10, "batch_count": len(batches), "batches": batches,
        "previous_10_cases_are_batch_01": batches[0] == old,
        "validation_case_selected": bool(set(selected) & set(map(str, split["validation"]))),
        "internal_test_case_selected": bool(set(selected) & set(map(str, split["internal_test"]))),
        "frozen_holdout_opened": False,
        "case_split_sha256": hashlib.sha256((PROJECT_DIR / config["data"]["case_split"]).read_bytes()).hexdigest(),
        "public_states_sha256": hashlib.sha256((PROJECT_DIR / config["data"]["public_states"]).read_bytes()).hexdigest(),
    }
    if not manifest["previous_10_cases_are_batch_01"] or manifest["validation_case_selected"] or manifest["internal_test_case_selected"]:
        raise SystemExit("Selection validation failed")
    (args.output_dir / "case_batch_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "batches"}, indent=2))


if __name__ == "__main__":
    main()
