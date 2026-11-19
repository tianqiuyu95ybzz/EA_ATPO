#!/usr/bin/env python3
"""Merge five frozen V2 batches after enforcing case/state/action pairing invariants."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve()
V2_DIR = HERE.parents[1]
PROJECT_DIR = V2_DIR.parents[1]
ACTION_SET = {"ASK", "ANSWER", "DEFER"}
FORBIDDEN_KEYS = {"gold", "ground_truth", "atomic_facts", "answer_idx"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def assert_no_privileged_keys(value: Any, location: str = "root") -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key).casefold() in FORBIDDEN_KEYS:
                raise ValueError(f"privileged field serialized at {location}.{key}")
            assert_no_privileged_keys(nested, f"{location}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            assert_no_privileged_keys(nested, f"{location}[{index}]")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--case-batch-manifest", type=Path, required=True)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    selection = json.loads(args.case_batch_manifest.read_text(encoding="utf-8"))
    if selection.get("selected_case_count") != 50 or selection.get("train_only") is not True:
        raise ValueError("case-batch manifest is not the expected frozen 50-case training selection")
    split = json.loads((PROJECT_DIR / config["data"]["case_split"]).read_text(encoding="utf-8"))
    train_ids = set(map(str, split["train"]))
    trajectory_rows: list[dict[str, Any]] = []
    batch_manifests = []
    seen_case_sets = []
    for batch_index in range(1, 6):
        batch_dir = args.batch_root / f"batch_{batch_index:02d}"
        manifest_path = batch_dir / "run_manifest.json"
        trajectory_path = batch_dir / "pilot_trajectories.jsonl"
        q_path = batch_dir / "q_dataset.jsonl"
        analysis_path = batch_dir / "analysis/pilot_summary.json"
        if not all(path.is_file() for path in (manifest_path, trajectory_path, q_path, analysis_path)):
            raise FileNotFoundError(f"batch {batch_index:02d} lacks trajectory/Q/analysis/manifest artifacts")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "completed" or manifest.get("frozen_holdout_read") is not False:
            raise ValueError(f"batch {batch_index:02d} is not completed train-only execution")
        if manifest.get("config_sha256") != hashlib.sha256(args.config.read_bytes()).hexdigest():
            raise ValueError(f"batch {batch_index:02d} used a different config")
        rows = read_jsonl(trajectory_path)
        q_rows = read_jsonl(q_path)
        summary = json.loads(analysis_path.read_text(encoding="utf-8"))
        if not summary.get("all_rollouts_valid") or summary.get("q_dataset_valid_action_targets") != summary.get("q_dataset_total_possible_action_targets"):
            raise ValueError(f"batch {batch_index:02d} has invalid rollouts or incomplete Q targets")
        batch_cases = set(map(str, manifest["selected_case_ids"]))
        if len(batch_cases) != 10 or not batch_cases.issubset(train_ids):
            raise ValueError(f"batch {batch_index:02d} case set is not 10 unique train cases")
        seen_case_sets.append(batch_cases)
        trajectory_rows.extend(rows)
        batch_manifests.append({
            "batch_index": batch_index, "run_manifest": str(manifest_path.resolve()),
            "run_manifest_sha256": sha256(manifest_path),
            "trajectory_path": str(trajectory_path.resolve()), "trajectory_sha256": sha256(trajectory_path),
            "q_dataset_path": str(q_path.resolve()), "q_dataset_sha256": sha256(q_path),
            "analysis_path": str(analysis_path.resolve()), "analysis_sha256": sha256(analysis_path),
            "case_count": len(batch_cases), "state_count": summary["states"],
            "rollout_count": summary["rollouts"], "valid_q_targets": summary["q_dataset_valid_action_targets"],
        })
    if any(seen_case_sets[i] & seen_case_sets[j] for i in range(5) for j in range(i + 1, 5)):
        raise ValueError("case IDs overlap across batches")
    all_cases = set.union(*seen_case_sets)
    expected_cases = set(map(str, selection["batches"][0] + selection["batches"][1] + selection["batches"][2] + selection["batches"][3] + selection["batches"][4]))
    if all_cases != expected_cases or len(all_cases) != 50:
        raise ValueError("merged case set does not match frozen 50-case manifest")
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in trajectory_rows:
        assert_no_privileged_keys(row)
        key = (str(row["case_id"]), str(row["state_id"]))
        action = str(row["action"])
        if action not in ACTION_SET or action in grouped[key]:
            raise ValueError(f"duplicate/invalid action at {key}")
        grouped[key][action] = row
    valid_targets = 0
    for key, actions in grouped.items():
        if set(actions) != ACTION_SET:
            raise ValueError(f"incomplete action set at {key}")
        if len({row["public_state_digest"] for row in actions.values()}) != 1:
            raise ValueError(f"counterfactuals do not share one initial public state at {key}")
        if len({row["continuation_policy_id"] for row in actions.values()}) != 1:
            raise ValueError(f"continuation policy mismatch at {key}")
        for row in actions.values():
            if row.get("valid") and row.get("final_accuracy") is not None:
                cost = float(row["cost_components"]["ask_cost"]) + float(row["cost_components"]["defer_cost"])
                if not math.isclose(float(row["return"]), float(row["final_accuracy"]) - cost, abs_tol=1e-9):
                    raise ValueError(f"reward/cost accounting mismatch at {key}/{row['action']}")
                valid_targets += 1
    expected_states = int(selection["selected_state_count"])
    if len(grouped) != expected_states or len(trajectory_rows) != len(grouped) * 3:
        raise ValueError("state/rollout count does not match frozen selection")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    trajectory_output = args.output_dir / "pilot_trajectories.jsonl"
    with trajectory_output.open("w", encoding="utf-8") as handle:
        for row in trajectory_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "experiment": "EA-ATPO V2 50-case real counterfactual expansion",
        "status": "complete", "case_count": len(all_cases), "state_count": len(grouped),
        "rollout_count": len(trajectory_rows), "action_set": sorted(ACTION_SET),
        "all_states_paired": True, "all_rollouts_valid": all(bool(r.get("valid")) for r in trajectory_rows),
        "all_q_targets_valid": valid_targets == len(trajectory_rows),
        "valid_q_targets": valid_targets, "total_q_targets": len(trajectory_rows),
        "costs_included_once_in_return": True, "gold_or_hidden_facts_serialized": False,
        "case_ids_all_train_only": all_cases.issubset(train_ids), "frozen_holdout_opened": False,
        "case_batch_manifest": str(args.case_batch_manifest.resolve()),
        "case_batch_manifest_sha256": sha256(args.case_batch_manifest),
        "config": str(args.config.resolve()), "config_sha256": sha256(args.config),
        "trajectory_output": str(trajectory_output.resolve()),
        "trajectory_sha256": sha256(trajectory_output), "batches": batch_manifests,
    }
    (args.output_dir / "merged_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "batches"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
