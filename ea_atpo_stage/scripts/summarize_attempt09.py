#!/usr/bin/env python3
"""Create auditable, privacy-screened summaries for EA-ATPO Attempt09."""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import pathlib
import re
from typing import Any


ANSI = re.compile(r"\x1b\[[0-9;]*m")
PII_PATTERNS = {
    "email_like": re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"),
    "phone_like": re.compile(r"(?<!\d)(?:\+?\d[\d .()/-]{8,}\d)(?!\d)"),
    "ssn_like": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
}
FORBIDDEN_TRACE_KEYS = {
    "gold",
    "ground_truth",
    "atomic_facts",
    "hidden_facts",
    "patient_response",
    "public_state_before",
    "public_state_after",
    "public_initial_state",
}
ACTION_NAMES = ("ANSWER", "ASK", "CONSULT", "DEFER", "INVALID")
METRIC_FIELDS = (
    "custom/num_sample",
    "custom/num_valid",
    "custom/num_correct",
    "custom/accuracy_of_valid",
    "actor/pg_loss",
    "critic/vf_loss",
    "actor/grad_norm",
    "critic/grad_norm",
    "critic/returns/mean",
    "critic/returns/min",
    "critic/returns/max",
    "critic/advantages/mean",
    "critic/advantages/min",
    "critic/advantages/max",
    "critic/score/mean",
    "critic/score/min",
    "critic/score/max",
    "critic/vf_explained_var",
    "perf/max_memory_allocated_gb",
    "perf/max_memory_reserved_gb",
    "timing_s/step",
)


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(*(collect_keys(v) for v in value.values()))
    if isinstance(value, list):
        return set().union(*(collect_keys(v) for v in value)) if value else set()
    return set()


def parse_step_metrics(log_path: pathlib.Path) -> dict[int, dict[str, float]]:
    parsed: dict[int, dict[str, float]] = {}
    for raw_line in log_path.read_text(errors="replace").splitlines():
        line = ANSI.sub("", raw_line)
        if " - static/q_value_variance_mean:" not in line:
            continue
        fields: dict[str, float] = {}
        for segment in line.split(" - "):
            if ":" not in segment:
                continue
            key, value = segment.split(":", 1)
            key = key.rsplit(")", 1)[-1].strip()
            try:
                fields[key] = float(value.strip())
            except ValueError:
                continue
        step = int(fields.get("training/global_step", 0))
        if step:
            parsed[step] = fields
    return parsed


def parse_validation_metric(log_path: pathlib.Path, key: str) -> float:
    text = ANSI.sub("", log_path.read_text(errors="replace"))
    match = re.search(re.escape(key) + r"['\"]?\s*:\s*([-+0-9.eE]+)", text)
    if not match:
        raise SystemExit(f"Could not find validation metric {key} in training log")
    return float(match.group(1))


def action_name(event: dict[str, Any]) -> str:
    action = event.get("action")
    return action if action in ACTION_NAMES[:-1] else "INVALID"


def expert_attempts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        attempt
        for row in rows
        for event in row.get("action_trace", [])
        for attempt in event.get("expert_attempts", [])
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    trace_dir = run_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    source_all = trace_dir / "all_rollouts.jsonl"
    source_train = trace_dir / "train.jsonl"
    source_val = trace_dir / "validation.jsonl"

    if source_all.exists():
        all_rows = read_jsonl(source_all)
    else:
        all_rows = read_jsonl(source_train)
        write_jsonl(source_all, all_rows)

    if len(all_rows) == 100:
        train_rows, val_rows = all_rows[:80], all_rows[80:]
    elif len(all_rows) == 80 and source_val.exists():
        train_rows, val_rows = all_rows, read_jsonl(source_val)
    else:
        raise SystemExit(f"Expected the captured 80 train + 20 validation traces; found {len(all_rows)}")
    write_jsonl(source_train, train_rows)
    write_jsonl(source_val, val_rows)

    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    train_batch = int(manifest["training_config"].get("train_batch_size", 2))
    trajectories_per_case = int(manifest["training_config"].get("M_trajectories", 4))
    rows_per_step = train_batch * trajectories_per_case
    metrics_path = run_dir / "logs/ea_atpo_v1_smoke_20261115T101426Z.log"
    redacted_log_path = run_dir / "logs/ea_atpo_v1_smoke_20261115T101426Z_redacted.log"
    raw_log = metrics_path.read_text(errors="replace")
    redacted_log = re.sub(r"/pfs/[^\s\"'<>]+", "<PFS_PATH>", raw_log)
    redacted_log = "\n".join(line.rstrip() for line in redacted_log.splitlines()) + "\n"
    redacted_log_path.write_text(redacted_log)
    metrics = parse_step_metrics(metrics_path)
    expected_steps = int(manifest["completion"].get("observed_optimizer_steps", 10))
    if sorted(metrics) != list(range(1, expected_steps + 1)):
        raise SystemExit(f"Expected metrics for steps 1..{expected_steps}; got {sorted(metrics)}")

    step_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    for step in range(1, expected_steps + 1):
        episode_rows = train_rows[(step - 1) * rows_per_step : step * rows_per_step]
        if len(episode_rows) != rows_per_step:
            raise SystemExit(f"Step {step} has {len(episode_rows)} trace rows, expected {rows_per_step}")
        episode_events = [event for row in episode_rows for event in row.get("action_trace", [])]
        action_counts = collections.Counter(action_name(event) for event in episode_events)
        first_action_counts = collections.Counter(action_name(row["action_trace"][0]) for row in episode_rows)
        eligible_count = sum(bool(row.get("evaluation", {}).get("training_eligible")) for row in episode_rows)
        correct_count = sum(bool(row.get("evaluation", {}).get("correct")) for row in episode_rows)
        failure_counts = collections.Counter(
            row.get("evaluation", {}).get("failure_category")
            for row in episode_rows
            if row.get("evaluation", {}).get("failure_category") not in (None, "none")
        )
        attempts = expert_attempts(episode_rows)
        row: dict[str, Any] = {
            "step": step,
            "rollout_slots": len(episode_rows),
            "ppo_eligible_trajectories": eligible_count,
            "protocol_or_environment_failures": failure_counts.get("protocol_error", 0),
            "policy_constraint_violations_retained": failure_counts.get("policy_constraint_violation", 0),
            "episodes_correct": correct_count,
            "episode_accuracy": correct_count / len(episode_rows),
            "expert_requests": sum(int(x.get("expert_request_count", 0)) for x in episode_events),
            "expert_attempts": len(attempts),
            "expert_retries": sum(max(int(x.get("attempt", 1)) - 1, 0) for x in attempts),
            "expert_parse_failures": sum(bool(x.get("parse_error")) for x in attempts),
            "expert_finish_reasons": ";".join(
                f"{name}:{count}" for name, count in sorted(collections.Counter(x.get("finish_reason") or "missing" for x in attempts).items())
            ),
            "consult_answer_leaks": sum(
                "leak" in str(x.get("parse_error", "")).lower()
                for event in episode_events if event.get("action") == "CONSULT"
                for x in event.get("expert_attempts", [])
            ),
            "defer_format_failures": sum(
                bool(x.get("parse_error"))
                for event in episode_events if event.get("action") == "DEFER"
                for x in event.get("expert_attempts", [])
            ),
            "mean_total_tokens_per_episode": sum(int(x.get("resource_usage", {}).get("total_tokens", 0)) for x in episode_rows) / len(episode_rows),
            "mean_serial_latency_s_per_episode": sum(float(x.get("resource_usage", {}).get("measured_serial_latency_s", 0)) for x in episode_rows) / len(episode_rows),
        }
        row.update({f"action_{name.lower()}_events": action_counts.get(name, 0) for name in ACTION_NAMES})
        row.update({f"first_{name.lower()}_actions": first_action_counts.get(name, 0) for name in ACTION_NAMES})
        row.update({field.replace("/", "_"): metrics[step].get(field) for field in METRIC_FIELDS})
        step_rows.append(row)
        for name in ACTION_NAMES:
            action_rows.append({
                "step": step,
                "action": name,
                "decision_events": action_counts.get(name, 0),
                "initial_decisions": first_action_counts.get(name, 0),
            })

    for path, records in (
        (run_dir / "ppo_step_metrics.csv", step_rows),
        (run_dir / "action_coverage.csv", action_rows),
    ):
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)

    all_attempts = expert_attempts(all_rows)
    finish_counts = collections.Counter(x.get("finish_reason") or "missing" for x in all_attempts)
    train_case_ids = {row.get("case_id_sha256") for row in train_rows}
    val_case_ids = {row.get("case_id_sha256") for row in val_rows}
    val_correct = sum(bool(row.get("evaluation", {}).get("correct")) for row in val_rows)
    val_tokens = sum(int(row.get("resource_usage", {}).get("total_tokens", 0)) for row in val_rows)
    val_latency = sum(float(row.get("resource_usage", {}).get("measured_serial_latency_s", 0)) for row in val_rows)
    raw_texts = [x.get("raw_response", "") for x in all_attempts]
    pii_matches = {
        name: sum(bool(pattern.search(text)) for text in raw_texts)
        for name, pattern in PII_PATTERNS.items()
    }
    forbidden_keys = sorted(collect_keys(all_rows) & FORBIDDEN_TRACE_KEYS)
    if forbidden_keys:
        raise SystemExit(f"Forbidden supervision/state keys present in trace schema: {forbidden_keys}")
    if any(pii_matches.values()):
        raise SystemExit(f"Potential PII patterns found in raw Expert responses: {pii_matches}")

    validation_summary = {
        "validation_cases": len(val_rows),
        "unique_hashed_cases": len(val_case_ids),
        "correct": val_correct,
        "accuracy": val_correct / len(val_rows),
        "parseable_answer_rate": parse_validation_metric(metrics_path, "val-aux/mediq/is_valid/mean@1"),
        "trainer_reported_accuracy": parse_validation_metric(metrics_path, "val-aux/mediq/is_correct/mean@1"),
        "mean_total_tokens": val_tokens / len(val_rows),
        "mean_measured_serial_latency_s": val_latency / len(val_rows),
        "expert_requests": sum(int(x.get("resource_usage", {}).get("expert_requests", 0)) for x in val_rows),
        "action_event_counts": dict(collections.Counter(action_name(e) for row in val_rows for e in row.get("action_trace", []))),
        "initial_action_counts": dict(collections.Counter(action_name(row["action_trace"][0]) for row in val_rows)),
        "mean_net_episode_reward": sum(float(x.get("evaluation", {}).get("net_episode_reward", 0)) for x in val_rows) / len(val_rows),
        "train_validation_case_hash_overlap": len(train_case_ids & val_case_ids),
    }
    (run_dir / "validation_summary.json").write_text(json.dumps(validation_summary, indent=2) + "\n")

    train_actions = collections.Counter(action_name(e) for row in train_rows for e in row.get("action_trace", []))
    first_train_actions = collections.Counter(action_name(row["action_trace"][0]) for row in train_rows)
    trace_paths = [
        source_all,
        source_train,
        source_val,
        run_dir / "ppo_step_metrics.csv",
        run_dir / "action_coverage.csv",
        run_dir / "validation_summary.json",
        redacted_log_path,
    ]
    completion = manifest.setdefault("completion", {})
    completion.update({
        "launcher_exit_code": 0,
        "observed_optimizer_steps": expected_steps,
        "out_of_memory": False,
        "validation_metrics_present": True,
        "validation_accuracy": validation_summary["accuracy"],
        "validation_correct_cases": val_correct,
        "validation_case_count": len(val_rows),
        "validation_parseable_answer_rate": validation_summary["parseable_answer_rate"],
        "training_eligible_trajectories": sum(bool(x.get("evaluation", {}).get("training_eligible")) for x in train_rows),
        "training_trajectory_count": len(train_rows),
        "natural_training_action_events": dict(train_actions),
        "natural_training_initial_actions": dict(first_train_actions),
        "expert_finish_reasons": dict(finish_counts),
        "expert_parse_failures": sum(bool(x.get("parse_error")) for x in all_attempts),
        "expert_retries": sum(max(int(x.get("attempt", 1)) - 1, 0) for x in all_attempts),
        "consult_answer_leaks": sum(
            "leak" in str(a.get("parse_error", "")).lower()
            for row in all_rows for event in row.get("action_trace", []) if event.get("action") == "CONSULT"
            for a in event.get("expert_attempts", [])
        ),
        "defer_format_failures": sum(
            bool(a.get("parse_error"))
            for row in all_rows for event in row.get("action_trace", []) if event.get("action") == "DEFER"
            for a in event.get("expert_attempts", [])
        ),
        "max_reported_memory_allocated_gb": max(float(x.get("perf_max_memory_allocated_gb", 0)) for x in step_rows),
        "max_reported_memory_reserved_gb": max(float(x.get("perf_max_memory_reserved_gb", 0)) for x in step_rows),
        "trace_privacy_audit": {
            "forbidden_gold_or_hidden_state_keys": forbidden_keys,
            "possible_pii_pattern_hits": pii_matches,
            "raw_expert_responses_saved": len(all_attempts),
        },
    })
    manifest["git_commit"] = "70015b3e9272becf5d73b3b4975a62d8d583651d"
    manifest["official_atpo_commit"] = "unavailable: nested checkout git rev-parse failed during preflight"
    manifest["run_dir"] = "experiments/ea_atpo_v1_smoke/ea_atpo_v1_attempt09_stability"
    manifest["log_path"] = "experiments/ea_atpo_v1_smoke/ea_atpo_v1_attempt09_stability/logs/ea_atpo_v1_smoke_20261115T101426Z.log"
    completion["log_path"] = manifest["log_path"]
    manifest["pushed_redacted_log_path"] = "experiments/ea_atpo_v1_smoke/ea_atpo_v1_attempt09_stability/logs/ea_atpo_v1_smoke_20261115T101426Z_redacted.log"
    manifest["model"]["path"] = "Qwen3-8B; cache path intentionally omitted from public manifest"
    manifest["data"]["train_parquet"]["path"] = "experiments/ea_atpo_v1_smoke/ea_atpo_v1_attempt09_stability/data/train.parquet"
    manifest["data"]["test_parquet"]["path"] = "experiments/ea_atpo_v1_smoke/ea_atpo_v1_attempt09_stability/data/test.parquet"
    manifest["gpu_snapshot"] = "NVIDIA B20Z GPUs 4 and 6; stable start gate: 110933 MiB and 110729 MiB free, each above the 105000 MiB threshold. Other GPU allocation details omitted."
    manifest["artifact_paths"] = [str(path.relative_to(run_dir)) for path in trace_paths]
    for path in trace_paths:
        completion.setdefault("artifact_sha256", {})[str(path.relative_to(run_dir))] = sha256(path)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
