#!/usr/bin/env python3
"""Validate real V2 rollouts and create a separately costed Train-only dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ACTION_ORDER = ("ASK", "ANSWER", "DEFER")
NEW_COSTS = {"ASK": 0.01, "ANSWER": 0.0, "DEFER": 0.05}
OLD_COSTS = {"ASK": 0.05, "ANSWER": 0.0, "DEFER": 0.60}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: {exc}") from exc
    return rows


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def fixed_case_split(case_ids, seed=20261120):
    ordered = sorted(set(map(str, case_ids)), key=lambda cid: (hashlib.sha256(f"{seed}|{cid}".encode()).hexdigest(), cid))
    if len(ordered) != 50:
        raise ValueError(f"expected 50 cases; found {len(ordered)}")
    train, dev = sorted(ordered[:35]), sorted(ordered[35:])
    if len(set(train) & set(dev)) or len(train) != 35 or len(dev) != 15:
        raise AssertionError("invalid case-level split")
    return {"unit": "case_id", "seed": seed, "algorithm": f"sort by SHA256('{seed}|' + case_id), first 35 train, last 15 dev", "train": train, "dev": dev}


def close(a, b, tol=1e-9):
    return math.isfinite(float(a)) and math.isfinite(float(b)) and abs(float(a) - float(b)) <= tol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-source", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--official-case-split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {args.output_dir}")
    q_rows, trajectories = read_jsonl(args.q_source), read_jsonl(args.trajectories)
    official = json.loads(args.official_case_split.read_text(encoding="utf-8"))
    if official.get("unit") != "case_id":
        raise ValueError("official split unit must be case_id")
    official_train = set(map(str, official["train"]))
    official_other = set(map(str, official.get("validation", []))) | set(map(str, official.get("internal_test", [])))

    q_by_key = {}
    for row in q_rows:
        md = row.get("metadata") or {}
        case_id, state_id = str(md.get("case_id", "")), str(md.get("state_id", ""))
        key = (case_id, state_id)
        if not all(key) or key in q_by_key:
            raise ValueError(f"missing or duplicate Q state identity: {key}")
        if md.get("feature_boundary") != "public_state_before_action_only" or md.get("gold_or_future_fields_in_features") is not False:
            raise ValueError(f"privileged or unverified feature boundary: {key}")
        if md.get("costs_included_in_targets") is not True or row.get("valid_mask") != {a: True for a in ACTION_ORDER}:
            raise ValueError(f"unexpected cost/mask metadata for {key}")
        if any(int((row.get("valid_sample_count") or {}).get(a, 0)) != 1 for a in ACTION_ORDER):
            raise ValueError(f"expected K=1 for all actions at {key}")
        public_state = md.get("public_state")
        if not isinstance(public_state, list) or not public_state or any(m.get("role") not in ("user", "assistant") for m in public_state):
            raise ValueError(f"invalid public-only state at {key}")
        q_by_key[key] = row

    if len({k[0] for k in q_by_key}) != 50 or not {k[0] for k in q_by_key}.issubset(official_train):
        raise ValueError("the Q cases must be 50 cases wholly inside the official Train partition")
    if {k[0] for k in q_by_key} & official_other:
        raise ValueError("the source includes official Validation/Internal Test cases")

    grouped = defaultdict(dict)
    trajectory_count = Counter()
    old_formula_checked = 0
    resource_fields = Counter()
    for tr in trajectories:
        if tr.get("record_type") != "v2_counterfactual_rollout":
            raise ValueError("unexpected trajectory record type")
        action = str(tr.get("action", ""))
        key = (str(tr.get("case_id", "")), str(tr.get("state_id", "")))
        if action not in ACTION_ORDER or key not in q_by_key:
            raise ValueError(f"trajectory has unknown state/action: {key}/{action}")
        if action in grouped[key]:
            raise ValueError(f"duplicate rollout for {key}/{action}; this run expects exactly K=1")
        if tr.get("valid") is not True or tr.get("format_valid") is not True or tr.get("failure_reason") is not None:
            raise ValueError(f"invalid trajectory cannot create a Q label: {key}/{action}")
        correctness = float(tr["final_accuracy"])
        if correctness not in (0.0, 1.0) or not close(tr.get("accuracy_reward"), correctness):
            raise ValueError(f"invalid terminal correctness at {key}/{action}")
        ask_count, defer_count = int(tr.get("ask_count", 0)), int(tr.get("defer_count", 0))
        if ask_count < 0 or defer_count < 0:
            raise ValueError(f"negative action count at {key}/{action}")
        components = tr.get("cost_components") or {}
        for name in ("expert_calls", "expert_tokens"):
            resource_fields[name] += int(components.get(name, 0))
        old_ask = float(components.get("ask_cost", math.nan))
        old_defer = float(components.get("defer_cost", math.nan))
        if not close(old_ask, OLD_COSTS["ASK"] * ask_count) or not close(old_defer, OLD_COSTS["DEFER"] * defer_count):
            raise ValueError(f"old cost ledger does not reconcile with counts at {key}/{action}")
        old_return = correctness - old_ask - old_defer
        if not close(tr.get("return"), old_return):
            raise ValueError(f"stored old return is inconsistent at {key}/{action}")
        old_q = float(q_by_key[key][f"Q_{action}"])
        if not close(old_q, old_return):
            raise ValueError(f"old Q target does not match raw trajectory at {key}/{action}")
        new_q = correctness - NEW_COSTS["ASK"] * ask_count - NEW_COSTS["DEFER"] * defer_count
        grouped[key][action] = {"trajectory": tr, "old_q": old_q, "new_q": new_q, "correctness": correctness}
        trajectory_count[action] += 1
        old_formula_checked += 1

    if len(trajectories) != 438 or len(grouped) != len(q_by_key):
        raise ValueError(f"expected 146 states x 3 actions=438 rollouts; found {len(trajectories)} rows/{len(grouped)} paired states")

    new_q_rows, states_only = [], []
    action_acc, q_values = {a: [] for a in ACTION_ORDER}, {a: [] for a in ACTION_ORDER}
    ask_counts, defer_counts = Counter(), Counter()
    for key in sorted(q_by_key):
        old = q_by_key[key]
        md = old["metadata"]
        if set(grouped[key]) != set(ACTION_ORDER):
            raise ValueError(f"incomplete action coverage at {key}")
        out = dict(old)
        for action in ACTION_ORDER:
            item = grouped[key][action]
            out[f"Q_{action}"] = item["new_q"]
            out[f"Q_{action}_old"] = item["old_q"]
            action_acc[action].append(item["correctness"])
            q_values[action].append(item["new_q"])
            ask_counts[action] += int(item["trajectory"]["ask_count"])
            defer_counts[action] += int(item["trajectory"]["defer_count"])
        out["action_cost"] = {a: NEW_COSTS[a] for a in ACTION_ORDER}
        out["costs_included_in_targets"] = True
        out_md = dict(md)
        out_md.update({
            "costs_included_in_targets": True,
            "cost_version": "answer_0_ask_0.01_defer_0.05_20261120",
            "old_q_target_preserved_in_top_level_Q_action_old_fields": True,
            "target_definition": "terminal_correctness - 0.01*cumulative_ask_count - 0.05*cumulative_defer_count",
            "resource_costs_not_in_utility": ["expert_calls", "expert_tokens", "token_count", "latency_seconds"],
        })
        out["metadata"] = out_md
        new_q_rows.append(out)
        states_only.append({"case_id": key[0], "state_id": key[1], "public_state": md["public_state"]})

    split = fixed_case_split({k[0] for k in q_by_key})
    if set(split["train"]) | set(split["dev"]) != {k[0] for k in q_by_key}:
        raise AssertionError("split does not cover exactly the 50 observed Train cases")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    q_out, state_out = output / "q_dataset_cost_ask001_defer005.jsonl", output / "public_states_only.jsonl"
    write_jsonl(q_out, new_q_rows)
    write_jsonl(state_out, states_only)
    write_json(output / "case_split_35_train_15_dev.json", split)
    report = {
        "status": "complete",
        "source_q": {"path": str(args.q_source.resolve()), "sha256": sha256(args.q_source), "rows": len(q_rows)},
        "source_trajectories": {"path": str(args.trajectories.resolve()), "sha256": sha256(args.trajectories), "rows": len(trajectories)},
        "output_q": {"path": str(q_out.resolve()), "sha256": sha256(q_out), "rows": len(new_q_rows)},
        "public_state_file": {"path": str(state_out.resolve()), "sha256": sha256(state_out), "rows": len(states_only), "contains_only_case_state_public_text": True},
        "case_count": len(split["train"]) + len(split["dev"]), "state_count": len(new_q_rows),
        "train_case_count": len(split["train"]), "dev_case_count": len(split["dev"]),
        "official_split_unit": "case_id", "all_source_cases_official_train": True,
        "no_official_validation_internal_test_or_frozen_holdout_used": True,
        "action_order": list(ACTION_ORDER), "action_trajectory_counts": dict(trajectory_count),
        "valid_labels": sum(trajectory_count.values()), "label_masks_unchanged_and_all_true": True,
        "rollouts_per_state_action": "exactly 1 (K=1)",
        "old_costs": OLD_COSTS, "new_costs": NEW_COSTS,
        "old_target_formula": "terminal_correctness - 0.05*cumulative_ask_count - 0.60*cumulative_defer_count",
        "new_target_formula": "terminal_correctness - 0.01*cumulative_ask_count - 0.05*cumulative_defer_count",
        "old_return_and_q_reconciliation_checks": old_formula_checked,
        "new_costs_applied_exactly_once": True, "no_second_cost_deduction_at_policy_evaluation": True,
        "cost_rebuilt_from_raw_terminal_correctness_and_cumulative_action_counts": True,
        "resource_telemetry_not_utility_cost": ["expert_calls", "expert_tokens", "prompt_tokens", "completion_tokens", "latency_seconds"],
        "resource_totals": dict(resource_fields), "mean_accuracy_by_action": {a: sum(v)/len(v) for a,v in action_acc.items()},
        "mean_new_q_by_action": {a: sum(v)/len(v) for a,v in q_values.items()},
        "total_ask_count_by_action": dict(ask_counts), "total_defer_count_by_action": dict(defer_counts),
        "gold_or_hidden_facts_in_feature_file": False,
        "note": "The input public_state can contain prior patient replies already made public; no hidden facts, gold, future reply, final answer, correctness, or rollout result is copied into the feature file.",
    }
    write_json(output / "cost_recalculation_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
