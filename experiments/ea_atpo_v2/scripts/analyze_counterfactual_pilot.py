#!/usr/bin/env python3
"""Aggregate V2 real counterfactual rollouts without exposing answer keys or hidden facts."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


ACTION_ORDER = ("ASK", "ANSWER", "DEFER")
FORBIDDEN_KEYS = {"gold", "ground_truth", "atomic_facts", "answer_idx"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def assert_no_privileged_keys(value: Any, where: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).casefold() in FORBIDDEN_KEYS:
                raise ValueError(f"privileged field serialized at {where}.{key}")
            assert_no_privileged_keys(nested, f"{where}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            assert_no_privileged_keys(nested, f"{where}[{index}]")


def mean(values: Sequence[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--q-dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = read_jsonl(args.trajectories)
    q_rows = read_jsonl(args.q_dataset)
    if not rows:
        raise ValueError("trajectory file is empty")
    for row in rows:
        assert_no_privileged_keys(row)
    for row in q_rows:
        assert_no_privileged_keys(row)

    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (str(row["case_id"]), str(row["state_id"]))
        action = str(row["action"])
        if action not in ACTION_ORDER or action in grouped[key]:
            raise ValueError(f"duplicate or invalid action in state {key}: {action}")
        grouped[key][action] = row
    for key, actions in grouped.items():
        if set(actions) != set(ACTION_ORDER):
            raise ValueError(f"incomplete action coverage at state {key}: {sorted(actions)}")
        if len({row["public_state_digest"] for row in actions.values()}) != 1:
            raise ValueError(f"counterfactual branches do not share one public state at {key}")
        if len({row["continuation_policy_id"] for row in actions.values()}) != 1:
            raise ValueError(f"continuation policy mismatch at {key}")

    action_summary: list[dict[str, Any]] = []
    for action in ACTION_ORDER:
        selected = [row for row in rows if row["action"] == action]
        valid = [row for row in selected if row.get("valid") and row.get("return") is not None]
        costs = [
            float((row.get("cost_components") or {}).get("ask_cost", 0.0))
            + float((row.get("cost_components") or {}).get("defer_cost", 0.0))
            for row in selected
        ]
        unknown = [row for row in selected if row.get("ask_response_unknown")]
        informative_proxy = [
            row for row in selected
            if action == "ASK" and row.get("ask_response_nonempty")
            and not row.get("ask_response_unknown") and row.get("ask_response_novel_text")
        ]
        action_summary.append({
            "action": action,
            "rollouts": len(selected),
            "valid_rollouts": len(valid),
            "invalid_rollouts": len(selected) - len(valid),
            "format_valid_rate": mean([float(bool(row.get("format_valid"))) for row in selected]),
            "accuracy": mean([float(row["final_accuracy"]) for row in valid]),
            "mean_return": mean([float(row["return"]) for row in valid]),
            "total_tokens": sum(int(row.get("total_tokens", 0)) for row in selected),
            "mean_tokens": mean([float(row.get("total_tokens", 0)) for row in selected]),
            "mean_prompt_tokens": mean([float(row.get("prompt_tokens", 0)) for row in selected]),
            "mean_completion_tokens": mean([float(row.get("completion_tokens", 0)) for row in selected]),
            "total_latency_seconds": sum(float(row.get("latency_seconds", 0.0)) for row in selected),
            "mean_latency_seconds": mean([float(row.get("latency_seconds", 0.0)) for row in selected]),
            "expert_calls": sum(int((row.get("cost_components") or {}).get("expert_calls", 0)) for row in selected),
            "total_action_cost": sum(costs),
            "mean_action_cost": mean(costs),
            "ask_unknown_responses": len(unknown),
            "ask_information_obtained_proxy_count": len(informative_proxy),
        })

    paired_rows: list[dict[str, Any]] = []
    helpful = harmful = both_correct = both_wrong = 0
    ask_helpful = ask_harmful = 0
    for (case_id, state_id), actions in sorted(grouped.items()):
        answer = actions["ANSWER"]
        ask = actions["ASK"]
        defer = actions["DEFER"]
        a_ok = answer.get("final_accuracy")
        q_ok = ask.get("final_accuracy")
        d_ok = defer.get("final_accuracy")
        if d_ok == 1.0 and a_ok == 0.0:
            helpful += 1
        elif d_ok == 0.0 and a_ok == 1.0:
            harmful += 1
        elif d_ok == 1.0 and a_ok == 1.0:
            both_correct += 1
        elif d_ok == 0.0 and a_ok == 0.0:
            both_wrong += 1
        if q_ok == 1.0 and a_ok == 0.0:
            ask_helpful += 1
        elif q_ok == 0.0 and a_ok == 1.0:
            ask_harmful += 1
        paired_rows.append({
            "case_id": case_id,
            "state_id": state_id,
            "answer_correct": a_ok,
            "ask_correct": q_ok,
            "defer_correct": d_ok,
            "answer_return": answer.get("return"),
            "ask_return": ask.get("return"),
            "defer_return": defer.get("return"),
            "ask_minus_answer_return": (
                float(ask["return"]) - float(answer["return"])
                if ask.get("return") is not None and answer.get("return") is not None else None
            ),
            "defer_minus_answer_return": (
                float(defer["return"]) - float(answer["return"])
                if defer.get("return") is not None and answer.get("return") is not None else None
            ),
            "ask_unknown": ask.get("ask_response_unknown"),
            "ask_information_obtained_proxy": bool(
                ask.get("ask_response_nonempty") and not ask.get("ask_response_unknown")
                and ask.get("ask_response_novel_text")
            ),
        })

    state_count = len(grouped)
    ask_rows = [row for row in rows if row["action"] == "ASK"]
    transition_valid_count = sum(
        bool((row.get("ask_state_transition") or {}).get("structurally_valid"))
        for row in ask_rows
    )
    finish_reasons = Counter(
        str(reason) for row in rows for reason in row.get("finish_reasons", [])
    )
    failure_reasons = Counter(
        str(row.get("failure_reason")) for row in rows if row.get("failure_reason")
    )
    api_calls = Counter()
    for row in rows:
        api_calls.update({key: int(value) for key, value in (row.get("api_call_counts") or {}).items()})
    valid_action_targets = sum(
        int(bool(value)) for qrow in q_rows for value in (qrow.get("valid_mask") or {}).values()
    )
    total_action_targets = sum(len(qrow.get("valid_mask") or {}) for qrow in q_rows)
    public_prompt_anomalies = 0
    for row in rows:
        public_text = "\n".join(str(message.get("content", "")) for message in row.get("public_state", []))
        if "arteries" in public_text.casefold() and all(
            name.casefold() in public_text.casefold()
            for name in ("Cryptosporidium parvum", "Dientamoeba fragilis", "Entamoeba histolytica", "Giardia lamblia")
        ):
            public_prompt_anomalies += 1
    unique_anomalous_states = len({row["state_id"] for row in rows if (
        "arteries" in "\n".join(str(message.get("content", "")) for message in row.get("public_state", [])).casefold()
        and "cryptosporidium parvum" in "\n".join(str(message.get("content", "")) for message in row.get("public_state", [])).casefold()
    )})

    summary = {
        "experiment": "EA-ATPO V2 real three-action counterfactual pilot",
        "states": state_count,
        "unique_cases": len({case_id for case_id, _ in grouped}),
        "rollouts": len(rows),
        "actions_per_state": list(ACTION_ORDER),
        "all_states_have_all_three_actions": len(rows) == state_count * 3,
        "all_rollouts_valid": all(bool(row.get("valid")) for row in rows),
        "all_action_outputs_format_valid": all(bool(row.get("format_valid")) for row in rows),
        "ask_state_transitions_structurally_valid": transition_valid_count,
        "ask_state_transition_rate": transition_valid_count / len(ask_rows) if ask_rows else None,
        "api_calls_by_role": dict(api_calls),
        "finish_reasons": dict(finish_reasons),
        "failure_reasons": dict(failure_reasons),
        "paired_contingency_answer_vs_defer": {
            "base_correct_expert_correct": both_correct,
            "base_wrong_expert_correct_helpful": helpful,
            "base_correct_expert_wrong_harmful": harmful,
            "base_wrong_expert_wrong": both_wrong,
        },
        "defer_corrections": helpful,
        "defer_regressions": harmful,
        "ask_corrections_vs_answer": ask_helpful,
        "ask_regressions_vs_answer": ask_harmful,
        "mean_ask_minus_answer_return": mean([
            float(row["ask_minus_answer_return"]) for row in paired_rows
            if row["ask_minus_answer_return"] is not None
        ]),
        "mean_defer_minus_answer_return": mean([
            float(row["defer_minus_answer_return"]) for row in paired_rows
            if row["defer_minus_answer_return"] is not None
        ]),
        "ask_information_obtained_proxy_count": sum(bool(row["ask_information_obtained_proxy"]) for row in paired_rows),
        "ask_unknown_count": sum(bool(row["ask_unknown"]) for row in paired_rows),
        "q_dataset_states": len(q_rows),
        "q_dataset_valid_action_targets": valid_action_targets,
        "q_dataset_total_possible_action_targets": total_action_targets,
        "q_dataset_costs_already_included": all(bool(row["metadata"]["costs_included_in_targets"]) for row in q_rows),
        "public_source_anomaly_rows_within_rollouts": public_prompt_anomalies,
        "public_source_anomaly_unique_states": unique_anomalous_states,
        "privileged_source_fields_serialized": False,
        "ask_information_metric_note": "nonempty, non-unknown patient reply that is textually novel relative to prior public state; only a proxy, not semantic clinical validation",
        "action_summary": action_summary,
    }
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "action_summary.csv", action_summary, list(action_summary[0]))
    write_csv(out / "paired_state_results.csv", paired_rows, list(paired_rows[0]))
    (out / "pilot_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
