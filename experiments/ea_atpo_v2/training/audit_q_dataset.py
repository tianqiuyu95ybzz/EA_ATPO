#!/usr/bin/env python3
"""Read-only schema, mask, split, and cost audit for a V2 Q dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from . import ACTION_NAMES
from .data import q_example_from_dict, read_jsonl


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def q_values(row: dict[str, Any]) -> list[Any]:
    source = row.get("Q_target", row)
    if isinstance(source, dict):
        return [source.get(f"Q_{action}", source.get(action)) for action in ACTION_NAMES]
    if isinstance(source, list) and len(source) == len(ACTION_NAMES):
        return source
    raise ValueError("Q_target must be a three-action mapping/list")


def _canonical_row(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def validate_merged_run_provenance(q_rows: list[dict[str, Any]], trajectories: list[dict[str, Any]],
                                   q_path: Path, trajectories_path: Path,
                                   merged_manifest_path: Path) -> dict[str, Any]:
    """Validate that a merged dataset is exactly the union of pinned, compatible batches."""
    manifest = json.loads(merged_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("merged run manifest is not complete")
    if any(manifest.get(name) is not True for name in
           ("all_states_paired", "all_rollouts_valid", "all_q_targets_valid", "costs_included_once_in_return", "case_ids_all_train_only")):
        raise ValueError("merged run manifest does not certify paired train-only data, valid labels, and one-time costs")
    if manifest.get("frozen_holdout_opened") is not False or manifest.get("gold_or_hidden_facts_serialized") is not False:
        raise ValueError("merged run manifest indicates protected data or hidden facts were exposed")
    if set(manifest.get("action_set") or []) != set(ACTION_NAMES):
        raise ValueError("merged run manifest action set does not match the V2 three-action protocol")

    expected_trajectory_path = Path(manifest.get("trajectory_output", "")).resolve()
    if expected_trajectory_path != trajectories_path.resolve():
        raise ValueError("merged manifest trajectory path does not match the audited trajectory file")
    if q_path.resolve() != expected_trajectory_path.parent / "q_dataset.jsonl":
        raise ValueError("merged Q dataset must be the q_dataset.jsonl sibling of the manifested trajectories")
    if manifest.get("trajectory_sha256") != sha256(trajectories_path):
        raise ValueError("merged trajectories changed after the merge manifest was produced")

    batch_entries = manifest.get("batches")
    if not isinstance(batch_entries, list) or not batch_entries:
        raise ValueError("merged manifest contains no source batches")
    merged_q_by_key = {}
    for row in q_rows:
        md = row.get("metadata") or {}
        key = (str(md.get("case_id", "")), str(md.get("state_id", "")))
        if not all(key) or key in merged_q_by_key:
            raise ValueError(f"merged Q dataset has missing/duplicate state identity: {key}")
        merged_q_by_key[key] = _canonical_row(row)
    merged_trajectory_counter = Counter(_canonical_row(row) for row in trajectories)

    from .evaluate_policies import validate_real_run_provenance

    source_q_by_key: dict[tuple[str, str], str] = {}
    source_trajectory_counter: Counter[str] = Counter()
    seen_batch_cases: set[str] = set()
    provenances = []
    for entry in batch_entries:
        run_manifest_path = Path(entry["run_manifest"]).resolve()
        batch_q_path = Path(entry["q_dataset_path"]).resolve()
        batch_trajectory_path = Path(entry["trajectory_path"]).resolve()
        if sha256(run_manifest_path) != entry.get("run_manifest_sha256"):
            raise ValueError(f"source run manifest checksum mismatch: {run_manifest_path}")
        if sha256(batch_q_path) != entry.get("q_dataset_sha256"):
            raise ValueError(f"source batch Q checksum mismatch: {batch_q_path}")
        if sha256(batch_trajectory_path) != entry.get("trajectory_sha256"):
            raise ValueError(f"source batch trajectory checksum mismatch: {batch_trajectory_path}")

        batch_q_rows = read_jsonl(batch_q_path)
        batch_trajectory_rows = read_jsonl(batch_trajectory_path)
        batch_cases = set()
        for row in batch_q_rows:
            md = row.get("metadata") or {}
            key = (str(md.get("case_id", "")), str(md.get("state_id", "")))
            if not all(key) or key in source_q_by_key:
                raise ValueError(f"duplicate/missing state across merged source batches: {key}")
            batch_cases.add(key[0])
            source_q_by_key[key] = _canonical_row(row)
        if seen_batch_cases & batch_cases:
            raise ValueError("case_id appears in more than one merged source batch")
        seen_batch_cases.update(batch_cases)
        source_trajectory_counter.update(_canonical_row(row) for row in batch_trajectory_rows)
        if (len(batch_cases) != int(entry.get("case_count", -1))
                or len(batch_q_rows) != int(entry.get("state_count", -1))
                or len(batch_trajectory_rows) != int(entry.get("rollout_count", -1))):
            raise ValueError("source batch counts differ from the merged manifest")

        provenance = validate_real_run_provenance(
            batch_trajectory_rows, batch_trajectory_path, run_manifest_path
        )
        provenances.append(provenance)

    if source_q_by_key != merged_q_by_key:
        raise ValueError("merged Q dataset is not exactly the union of the checksummed source batch Q datasets")
    if source_trajectory_counter != merged_trajectory_counter:
        raise ValueError("merged trajectories are not exactly the union of the checksummed source batch trajectories")
    if len({p["config_sha256"] for p in provenances}) != 1:
        raise ValueError("source batches use different run configs")
    if len({json.dumps(p["models"], sort_keys=True) for p in provenances}) != 1:
        raise ValueError("source batches use different pinned Base/Expert models")
    if len({p["continuation_policy_id"] for p in provenances}) != 1:
        raise ValueError("source batches mix continuation policies")
    if len({json.dumps(p["costs"], sort_keys=True) for p in provenances}) != 1:
        raise ValueError("source batches use different action costs")

    config_path = Path(manifest["config"]).resolve()
    if sha256(config_path) != manifest.get("config_sha256"):
        raise ValueError("merged run config checksum mismatch")
    batch_manifest_path = Path(manifest["case_batch_manifest"]).resolve()
    if sha256(batch_manifest_path) != manifest.get("case_batch_manifest_sha256"):
        raise ValueError("case-batch manifest checksum mismatch")
    valid_q_targets = sum(sum(bool(value) for value in (row.get("valid_mask") or {}).values()) for row in q_rows)
    if (int(manifest.get("case_count", -1)) != len(seen_batch_cases)
            or int(manifest.get("state_count", -1)) != len(merged_q_by_key)
            or int(manifest.get("rollout_count", -1)) != len(trajectories)
            or int(manifest.get("valid_q_targets", -1)) != valid_q_targets
            or int(manifest.get("total_q_targets", -1)) != valid_q_targets):
        raise ValueError("merged manifest aggregate counts differ from audited inputs")

    first = provenances[0]
    if first["config_sha256"] != manifest.get("config_sha256"):
        raise ValueError("source batch config does not match the merged manifest config")
    return {
        "run_manifest": str(merged_manifest_path.resolve()),
        "manifest_type": "checksummed_merged_batches",
        "config_sha256": first["config_sha256"],
        "models": first["models"],
        "continuation_policy_id": first["continuation_policy_id"],
        "costs": first["costs"],
        "source_batch_count": len(provenances),
        "source_run_manifests": [p["run_manifest"] for p in provenances],
        "all_source_batches_same_config_models_policy_costs": True,
        "merged_q_dataset_is_exact_source_union": True,
        "merged_trajectories_are_exact_source_union": True,
    }


def audit(q_path: Path, trajectories_path: Path, split_path: Path, *, expected_cases: int | None,
         expected_states: int | None, expected_valid_labels: int | None,
         run_manifest_path: Path | None = None) -> dict[str, Any]:
    q_rows = read_jsonl(q_path)
    trajectories = read_jsonl(trajectories_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if split.get("unit") != "case_id":
        raise ValueError("the frozen split must use case_id")
    if not q_rows or not trajectories:
        raise ValueError("Q dataset and source trajectories must both be non-empty")
    run_provenance = None
    if run_manifest_path is not None:
        manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if isinstance(manifest.get("batches"), list):
            run_provenance = validate_merged_run_provenance(
                q_rows, trajectories, q_path, trajectories_path, run_manifest_path
            )
        else:
            from .evaluate_policies import validate_real_run_provenance
            run_provenance = validate_real_run_provenance(trajectories, trajectories_path, run_manifest_path)
    q_examples = [q_example_from_dict(row) for row in q_rows]
    q_key_to_row = {(example.case_id, example.state_id): (example, row) for example, row in zip(q_examples, q_rows)}
    if len(q_key_to_row) != len(q_rows):
        raise ValueError("duplicate case_id/state_id in Q dataset")
    case_ids = {example.case_id for example in q_examples}
    state_keys = set(q_key_to_row)
    official_train = set(map(str, split["train"]))
    official_nontrain = set(map(str, split.get("validation", []))) | set(map(str, split.get("internal_test", [])))
    if not case_ids.issubset(official_train) or case_ids & official_nontrain:
        raise ValueError("Q dataset contains cases outside the frozen official train split")
    continuation_policy_ids = {str(example.raw_metadata.get("continuation_policy_id") or "") for example in q_examples}
    if "" in continuation_policy_ids or len(continuation_policy_ids) != 1:
        raise ValueError(f"Q dataset mixes or omits continuation_policy_id values: {sorted(continuation_policy_ids)}")

    rollout_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in trajectories:
        key = (str(row.get("case_id", "")), str(row.get("state_id", "")), str(row.get("action", "")))
        if key[2] not in ACTION_NAMES or not key[0] or not key[1]:
            raise ValueError(f"malformed raw trajectory identity/action: {key}")
        rollout_groups[key].append(row)
    duplicate_seed_groups = []
    mixed_policy_groups = []
    for key, rows in rollout_groups.items():
        seeds = [str(row["rollout_seed"]) for row in rows if row.get("rollout_seed") is not None]
        if len(seeds) != len(set(seeds)):
            duplicate_seed_groups.append(key)
        if {str(row.get("continuation_policy_id") or "") for row in rows} != continuation_policy_ids:
            mixed_policy_groups.append(key)
    if duplicate_seed_groups or mixed_policy_groups:
        raise ValueError(json.dumps({
            "duplicate_rollout_seed_groups": len(duplicate_seed_groups),
            "state_action_policy_mismatches": len(mixed_policy_groups),
        }))
    raw_state_keys = {(case, state) for case, state, _ in rollout_groups}
    if raw_state_keys != state_keys:
        raise ValueError(f"Q/raw state mismatch: q_only={len(state_keys-raw_state_keys)}, raw_only={len(raw_state_keys-state_keys)}")

    valid_labels = 0
    mask_mismatches = []
    sample_count_mismatches = []
    attempted_count_mismatches = []
    return_std_mismatches = []
    policy_metadata_mismatches = []
    source_rollout_count_mismatches = []
    target_mismatches = []
    cost_formula_mismatches = []
    configured_cost_mismatches = []
    cost_metadata_false = []
    boundary_errors = []
    per_case_state_counts = Counter(example.case_id for example in q_examples)
    raw_action_counts = Counter()
    valid_rollout_counts = Counter()
    attempted_rollout_counts = Counter()
    rollout_k_values = Counter()
    valid_rollout_total = 0
    invalid_rollout_total = 0
    for key, (example, row) in q_key_to_row.items():
        md = row.get("metadata") or {}
        if str(md.get("continuation_policy_id") or "") != next(iter(continuation_policy_ids)):
            policy_metadata_mismatches.append(key)
        expected_source_rows = sum(len(rollout_groups.get((key[0], key[1], action), [])) for action in ACTION_NAMES)
        if md.get("source_rollout_count") is not None and int(md["source_rollout_count"]) != expected_source_rows:
            source_rollout_count_mismatches.append((key, int(md["source_rollout_count"]), expected_source_rows))
        if md.get("costs_included_in_targets") is not True:
            cost_metadata_false.append(key)
        if md.get("feature_boundary") != "public_state_before_action_only" or md.get("gold_or_future_fields_in_features") is not False:
            boundary_errors.append(key)
        if list(row.get("valid_mask", {}).keys()) != list(ACTION_NAMES):
            raise ValueError(f"{key}: valid_mask key order does not match {ACTION_NAMES}")
        if [name for name in row.keys() if name.startswith("Q_")] != [f"Q_{a}" for a in ACTION_NAMES]:
            if not isinstance(row.get("Q_target"), dict):
                raise ValueError(f"{key}: Q target key order does not match {ACTION_NAMES}")
        targets = q_values(row)
        for index, action in enumerate(ACTION_NAMES):
            raw = rollout_groups.get((key[0], key[1], action), [])
            raw_action_counts[action] += len(raw)
            malformed_valid = [item for item in raw if item.get("valid") and (not isinstance(item.get("return"), (int, float)) or not math.isfinite(float(item.get("return", math.nan))))]
            if malformed_valid:
                raise ValueError(f"valid {action} rollout has missing/non-finite return for {key}")
            valid_raw = [item for item in raw if item.get("valid")]
            valid = bool(example.valid_mask[index])
            valid_labels += int(valid)
            valid_rollout_total += len(valid_raw)
            invalid_rollout_total += len(raw) - len(valid_raw)
            valid_rollout_counts[action] += len(valid_raw)
            attempted_rollout_counts[action] += sum(int(item.get("sample_count", 1)) for item in raw)
            if valid:
                rollout_k_values[int(example.sample_count[index])] += 1
            if valid != bool(valid_raw):
                mask_mismatches.append((key, action))
            if int(example.sample_count[index]) != len(valid_raw):
                sample_count_mismatches.append((key, action, int(example.sample_count[index]), len(valid_raw)))
            attempted = sum(int(item.get("sample_count", 1)) for item in raw)
            if int(example.attempted_sample_count[index]) != attempted:
                attempted_count_mismatches.append((key, action, int(example.attempted_sample_count[index]), attempted))
            return_values = [float(item["return"]) for item in valid_raw]
            expected_std = statistics.stdev(return_values) if len(return_values) >= 2 else None
            observed_std = example.return_std[index]
            if expected_std is None:
                if math.isfinite(float(observed_std)):
                    return_std_mismatches.append((key, action, float(observed_std), None))
            elif not math.isfinite(float(observed_std)) or abs(float(observed_std) - expected_std) > 1e-7:
                return_std_mismatches.append((key, action, float(observed_std), expected_std))
            if valid_raw:
                expected_q = statistics.fmean(return_values)
                if targets[index] is None or abs(float(targets[index]) - expected_q) > 1e-7:
                    target_mismatches.append((key, action, targets[index], expected_q))
                for item in valid_raw:
                    components = item.get("cost_components") or {}
                    if not isinstance(item.get("final_accuracy"), (int, float)):
                        raise ValueError(f"valid {action} rollout lacks raw correctness for {key}")
                    expected_ask_count = int(action == "ASK")
                    expected_defer_count = int(action == "DEFER")
                    expected_ask_cost = 0.05 * expected_ask_count
                    expected_defer_cost = 0.60 * expected_defer_count
                    if int(item.get("ask_count", -1)) != expected_ask_count or int(item.get("defer_count", -1)) != expected_defer_count:
                        configured_cost_mismatches.append((key, action, "action_counts"))
                    if abs(float(components["ask_cost"]) - expected_ask_cost) > 1e-7 or abs(float(components["defer_cost"]) - expected_defer_cost) > 1e-7:
                        configured_cost_mismatches.append((key, action, "cost_components"))
                    raw_net = float(item["final_accuracy"]) - float(components["ask_cost"]) - float(components["defer_cost"])
                    if abs(raw_net - float(item["return"])) > 1e-7:
                        cost_formula_mismatches.append((key, action, raw_net, item["return"]))
    if (cost_metadata_false or boundary_errors or mask_mismatches or sample_count_mismatches
            or attempted_count_mismatches or return_std_mismatches or policy_metadata_mismatches
            or source_rollout_count_mismatches or target_mismatches or cost_formula_mismatches
            or configured_cost_mismatches):
        raise ValueError(json.dumps({
            "cost_metadata_false": len(cost_metadata_false), "feature_boundary_errors": len(boundary_errors),
            "valid_mask_mismatches": len(mask_mismatches), "sample_count_mismatches": len(sample_count_mismatches),
            "attempted_count_mismatches": len(attempted_count_mismatches),
            "return_std_mismatches": len(return_std_mismatches),
            "continuation_policy_metadata_mismatches": len(policy_metadata_mismatches),
            "source_rollout_count_mismatches": len(source_rollout_count_mismatches),
            "q_target_mismatches": len(target_mismatches), "cost_formula_mismatches": len(cost_formula_mismatches),
            "configured_cost_mismatches": len(configured_cost_mismatches),
        }))
    if expected_cases is not None and len(case_ids) != expected_cases:
        raise ValueError(f"expected {expected_cases} cases, found {len(case_ids)}")
    if expected_states is not None and len(state_keys) != expected_states:
        raise ValueError(f"expected {expected_states} states, found {len(state_keys)}")
    if expected_valid_labels is not None and valid_labels != expected_valid_labels:
        raise ValueError(f"expected {expected_valid_labels} valid action labels, found {valid_labels}")
    if not any(raw_action_counts.values()) or sum(valid_rollout_counts.values()) != valid_rollout_total:
        raise ValueError("raw rollout/valid label accounting failed")
    if sum(attempted_rollout_counts.values()) < len(trajectories):
        raise ValueError("attempted sample count is smaller than the number of raw rollout rows")
    total_action_slots = len(state_keys) * len(ACTION_NAMES)
    missing_action_slots = total_action_slots - valid_labels
    return_std_missing_count = sum(not math.isfinite(float(example.return_std[i])) for example in q_examples for i in range(len(ACTION_NAMES)) if example.valid_mask[i])
    k_distribution = {str(k): int(v) for k, v in sorted(rollout_k_values.items())}
    k_is_one = bool(rollout_k_values) and set(rollout_k_values) == {1}
    return {
        "status": "passed",
        "scope": "read_only_real_data_integrity_audit",
        "q_dataset": {"path": str(q_path.resolve()), "sha256": sha256(q_path), "rows": len(q_rows)},
        "raw_trajectories": {"path": str(trajectories_path.resolve()), "sha256": sha256(trajectories_path), "rows": len(trajectories)},
        "case_split": {"path": str(split_path.resolve()), "sha256": sha256(split_path), "unit": split["unit"]},
        "action_order": list(ACTION_NAMES),
        "case_count": len(case_ids), "state_count": len(state_keys),
        "valid_action_target_count": valid_labels,
        "raw_rollout_count_by_action": dict(raw_action_counts),
        "valid_rollout_count_by_action": dict(valid_rollout_counts),
        "attempted_rollout_count_by_action": dict(attempted_rollout_counts),
        "valid_raw_rollout_count": valid_rollout_total,
        "invalid_raw_rollout_count": invalid_rollout_total,
        "rollouts_per_state_action_K_distribution": k_distribution,
        "all_valid_state_action_targets_are_K1": k_is_one,
        "K1_means_single_sample_return_not_stable_expected_Q": k_is_one,
        "return_std_missing_valid_action_count": return_std_missing_count,
        "missing_valid_action_target_slots": missing_action_slots,
        "duplicate_state_action_rollouts_present": any(len(rows) > 1 for rows in rollout_groups.values()),
        "continuation_policy_ids": sorted(continuation_policy_ids),
        "continuation_policy_mixed_across_dataset": False,
        "real_run_provenance": run_provenance,
        "state_count_by_case": dict(sorted(per_case_state_counts.items())),
        "all_cases_in_official_train": True,
        "official_validation_or_internal_test_used": False,
        "frozen_holdout_opened": False,
        "costs_included_in_q_targets": True,
        "raw_return_formula": "final_accuracy - cost_components.ask_cost - cost_components.defer_cost",
        "q_target_matches_mean_valid_raw_return": True,
        "mask_and_valid_sample_count_match_raw_rollouts": True,
        "attempted_sample_count_matches_raw_rollouts": True,
        "configured_costs": {"ASK": 0.05, "ANSWER": 0.0, "DEFER": 0.60},
        "feature_boundary": "public_state_before_action_only",
        "gold_or_future_fields_in_features": False,
        "return_std": "validated against sample standard deviation of valid raw returns; null for K=1",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-dataset", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--case-split", type=Path, default=Path(__file__).resolve().parents[3] / "experiments/ea_atpo_value_router_20261105/artifacts/case_split.json")
    parser.add_argument("--run-manifest", type=Path, help="manifest for the same run directory; verifies pinned Base/Expert and policy")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int)
    parser.add_argument("--expected-states", type=int)
    parser.add_argument("--expected-valid-labels", type=int)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(f"refusing to overwrite audit report: {args.report}")
    report = audit(args.q_dataset, args.trajectories, args.case_split,
                   expected_cases=args.expected_cases, expected_states=args.expected_states,
                   expected_valid_labels=args.expected_valid_labels,
                   run_manifest_path=args.run_manifest)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
