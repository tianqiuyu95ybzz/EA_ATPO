#!/usr/bin/env python3
"""Audit whether remaining official Train cases are genuinely unseen.

This audit reads case identifiers and public initial-message payloads only. It
does not open the 400-case Frozen Holdout, inspect labels, or generate model
predictions. Run from the Window A worktree root.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value: Any) -> str:
    blob = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Window A worktree root")
    parser.add_argument("--output", type=Path, help="optional aggregate JSON output")
    args = parser.parse_args()
    root = args.root.resolve()

    split_path = root / "experiments/ea_atpo_value_router_20261105/artifacts/case_split.json"
    selected_path = root / "experiments/ea_atpo_v2/recovery_bundle_20261120/case_batch_manifest.json"
    old_checkpoint_path = root / "experiments/ea_atpo_value_router_20261105/artifacts/checkpoint/router_checkpoint.json"
    old_targets_path = root / "experiments/ea_atpo_value_router_20261105/artifacts/state_action_targets.csv"
    historical_public_states_path = root / "experiments/real_case_pilot_20261103/outputs/full/pilot_public_states.jsonl"
    current_split_path = root / "experiments/ea_atpo_v2/router_training/outputs/real_q_router_20261120/case_split_35_train_15_dev.json"
    current_embedding_manifest_path = root / "experiments/ea_atpo_v2/router_training/outputs/real_q_router_20261120/qwen3_8b_frozen_public_embeddings.jsonl.manifest.json"

    required = [split_path, selected_path, old_checkpoint_path, old_targets_path,
                historical_public_states_path, current_split_path, current_embedding_manifest_path]
    missing = [str(p.relative_to(root)) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("audit inputs are missing: " + ", ".join(missing))

    official = json.loads(split_path.read_text(encoding="utf-8"))
    if official.get("unit") != "case_id":
        raise ValueError("official split unit must be case_id")
    official_train = set(map(str, official["train"]))
    if len(official_train) != 70:
        raise ValueError(f"expected the frozen 70-case official Train split, found {len(official_train)}")

    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    current_cases = set(map(str, (case for batch in selected["batches"] for case in batch)))
    if len(current_cases) != 50 or not current_cases.issubset(official_train):
        raise ValueError("current 50-case V2 cohort does not match official Train")
    candidates = official_train - current_cases

    current_split = json.loads(current_split_path.read_text(encoding="utf-8"))
    current_router_cases = set(map(str, current_split["train"])) | set(map(str, current_split["dev"]))
    if current_router_cases != current_cases:
        raise ValueError("current Router 35/15 split does not reconstruct the same 50-case cohort")
    embedding_manifest = json.loads(current_embedding_manifest_path.read_text(encoding="utf-8"))
    if embedding_manifest.get("case_split_audit", {}).get("case_count") != 50:
        raise ValueError("current embedding manifest does not attest a 50-case cohort")

    old = json.loads(old_checkpoint_path.read_text(encoding="utf-8"))
    old_state_ids = list(map(str, old.get("train_state_ids", [])))
    old_fit_cases = {case for case in official_train
                     if any(state_id.startswith(case + ":") for state_id in old_state_ids)}
    old_fit_meta = old.get("training", {})

    target_cases: set[str] = set()
    with old_targets_path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            cid = str(row.get("case_id", ""))
            if cid:
                target_cases.add(cid)

    historical_initial_hashes: dict[str, str] = {}
    with historical_public_states_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            cid = str(row.get("case_id", ""))
            if cid in official_train and cid not in historical_initial_hashes:
                # Hash only the originally public case presentation. Never emit its text.
                historical_initial_hashes[cid] = canonical_hash(row.get("initial_messages", []))
    hash_groups: dict[str, set[str]] = defaultdict(set)
    for cid, digest in historical_initial_hashes.items():
        hash_groups[digest].add(cid)
    candidate_duplicate_cases = set().union(*(
        group for group in hash_groups.values() if len(group) > 1 and group & candidates
    )) if any(len(group) > 1 and group & candidates for group in hash_groups.values()) else set()

    reasons = {
        "used_by_current_v2_router_train_or_dev": len(candidates & current_router_cases),
        "included_in_historical_v1_final_router_fit": len(candidates & old_fit_cases),
        "present_in_historical_v1_target_table": len(candidates & target_cases),
        "present_in_historical_public_state_bundle": len(candidates & set(historical_initial_hashes)),
        "candidate_cases_in_exact_initial_message_duplicate_groups": len(candidate_duplicate_cases),
    }
    eligible = len(candidates)
    # Any prior fit/target exposure disqualifies confirmatory use. The V1 router
    # fit explicitly used the full 100-case cohort, including all candidates.
    eligible -= len(candidates & old_fit_cases)
    status = "blocked_no_unseen_cases" if eligible == 0 else "eligible_cases_require_manual_review"
    result = {
        "schema_version": 1,
        "status": status,
        "audit_scope": "official Train only; Frozen Holdout not opened",
        "official_train_cases": len(official_train),
        "current_v2_q_router_cases": len(current_cases),
        "nominal_remaining_cases": len(candidates),
        "confirmed_unseen_cases": eligible,
        "exposure_counts": reasons,
        "historical_v1_router_fit": {
            "checkpoint_training_cases": old_fit_meta.get("training_cases"),
            "checkpoint_training_states": old_fit_meta.get("training_states"),
            "checkpoint_state_ids": len(old_state_ids),
            "candidate_cases_in_fit": len(candidates & old_fit_cases),
        },
        "historical_target_table": {
            "unique_cases": len(target_cases),
            "candidate_cases_present": len(candidates & target_cases),
        },
        "historical_public_state_bundle": {
            "official_train_cases_with_public_initial_state": len(set(historical_initial_hashes) & official_train),
            "candidate_cases_present": len(candidates & set(historical_initial_hashes)),
            "candidate_exact_duplicate_cases": len(candidate_duplicate_cases),
        },
        "hashes": {
            "official_split_sha256": sha256(split_path),
            "current_case_batch_manifest_sha256": sha256(selected_path),
            "current_router_split_sha256": sha256(current_split_path),
            "current_embedding_manifest_sha256": sha256(current_embedding_manifest_path),
            "historical_router_checkpoint_metadata_sha256": sha256(old_checkpoint_path),
            "historical_target_table_sha256": sha256(old_targets_path),
            "historical_public_states_sha256": sha256(historical_public_states_path),
        },
        "nonclaims": [
            "No new model inference or end-to-end confirmation episode was run.",
            "This audit does not establish performance on unseen cases.",
            "Frozen Holdout contents and outcomes were not read.",
        ],
    }
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        out = args.output if args.output.is_absolute() else root / args.output
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
