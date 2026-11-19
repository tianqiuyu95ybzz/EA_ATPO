#!/usr/bin/env python3
"""Summarize per-batch and merged counterfactual outcomes without answer keys."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def action_metrics(summary: dict[str, Any], action: str) -> dict[str, Any]:
    return next(row for row in summary["action_summary"] if row["action"] == action)


def batch_row(label: str, directory: Path) -> dict[str, Any]:
    summary = json.loads((directory / "analysis/pilot_summary.json").read_text(encoding="utf-8"))
    rows = read_jsonl(directory / "pilot_trajectories.jsonl")
    manifest_path = directory / "run_manifest.json"
    if not manifest_path.is_file():
        manifest_path = directory / "merged_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    gpu_before = manifest.get("gpu_inventory_read_only", [])
    gpu_after = manifest.get("gpu_inventory_after_run_read_only", [])
    asks = [row for row in rows if row["action"] == "ASK"]
    retries = sum(int(row.get("ask_response_review_attempts", 0)) for row in asks)
    accepted = sum(bool(row.get("ask_response_review_accepted")) for row in asks)
    ask, answer, defer = (action_metrics(summary, x) for x in ("ASK", "ANSWER", "DEFER"))
    return {
        "batch": label, "unique_cases": summary["unique_cases"], "states": summary["states"],
        "rollouts": summary["rollouts"], "valid_rollouts": sum(bool(row.get("valid")) for row in rows),
        "invalid_rollouts": sum(not bool(row.get("valid")) for row in rows),
        "q_valid_targets": summary["q_dataset_valid_action_targets"],
        "q_total_targets": summary["q_dataset_total_possible_action_targets"],
        "ask_accuracy": ask["accuracy"], "answer_accuracy": answer["accuracy"], "defer_accuracy": defer["accuracy"],
        "ask_unknown_count": summary["ask_unknown_count"],
        "ask_unknown_rate": summary["ask_unknown_count"] / summary["states"],
        "unknown_review_attempts": retries, "unknown_review_accepted": accepted,
        "ask_information_proxy_count": summary["ask_information_obtained_proxy_count"],
        "defer_corrections": summary["defer_corrections"], "defer_regressions": summary["defer_regressions"],
        "ask_corrections": summary["ask_corrections_vs_answer"], "ask_regressions": summary["ask_regressions_vs_answer"],
        "mean_ask_return": ask["mean_return"], "mean_answer_return": answer["mean_return"],
        "mean_defer_return": defer["mean_return"],
        "ask_tokens": ask["total_tokens"], "answer_tokens": answer["total_tokens"], "defer_tokens": defer["total_tokens"],
        "ask_latency_s": ask["total_latency_seconds"], "answer_latency_s": answer["total_latency_seconds"],
        "defer_latency_s": defer["total_latency_seconds"],
        "gpu2_free_mib_before": next((int(x["memory_free_mib"]) for x in gpu_before if x["index"] == "2"), None),
        "gpu2_util_pct_before": next((int(x["utilization_pct"]) for x in gpu_before if x["index"] == "2"), None),
        "gpu2_free_mib_after": next((int(x["memory_free_mib"]) for x in gpu_after if x["index"] == "2"), None),
        "gpu2_util_pct_after": next((int(x["utilization_pct"]) for x in gpu_after if x["index"] == "2"), None),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    rows = [batch_row(f"batch_{i:02d}", args.batch_root / f"batch_{i:02d}") for i in range(1, 6)]
    merged = batch_row("all_50_cases", args.batch_root.parent / "merged")
    rows.append(merged)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    columns = list(rows[0])
    with (args.output_dir / "batch_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "batch_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
