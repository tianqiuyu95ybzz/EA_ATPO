#!/usr/bin/env python3
"""Pair Base8B, historical AWQ and SFT Expert with case-cluster bootstrap."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
MULTI = ROOT / "experiments/expert_sft/multiturn_dialogue"
COSTS = (0.0, 0.05, 0.10, 0.20, 0.40, 0.60)


def read_predictions(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    output = {}
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        state = str(row.get("state_id") if row.get("state_id") is not None else row.get("prompt_hash", "single"))
        key = str(row["case_id"]), state
        if key in output:
            raise ValueError(f"Duplicate prediction row in {path}: {key}")
        output[key] = row
    if not output:
        raise ValueError(f"Empty prediction file: {path}")
    return output


def cluster_bootstrap(case_rows: dict[str, list[dict[str, Any]]], metric, seed: int, n_boot: int = 10000) -> list[float]:
    cases = sorted(case_rows)
    rng = random.Random(seed)
    values = []
    for _ in range(n_boot):
        sample = [cases[rng.randrange(len(cases))] for _ in cases]
        records = [item for case in sample for item in case_rows[case]]
        values.append(float(metric(records)))
    values.sort()
    return [values[int(0.025 * n_boot)], values[min(n_boot - 1, int(0.975 * n_boot))]]


def accuracy(rows, column: str) -> float:
    return sum(bool(x[column]) for x in rows) / len(rows) if rows else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--awq", type=Path, required=True)
    parser.add_argument("--sft", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()
    outdir = args.output_dir.resolve()
    if MULTI.resolve() not in outdir.parents:
        raise ValueError(f"Output must stay under {MULTI}")
    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite comparison: {outdir}")
    named = {"base": read_predictions(args.base), "awq": read_predictions(args.awq), "sft": read_predictions(args.sft)}
    keysets = [set(x) for x in named.values()]
    if not all(x == keysets[0] for x in keysets[1:]):
        raise ValueError("Model predictions are not paired on identical case/state keys")
    paired = []
    for key in sorted(keysets[0]):
        b, a, s = (named[k][key] for k in ("base", "awq", "sft"))
        if len({str(x.get("gold")) for x in (b, a, s)}) != 1:
            raise ValueError(f"Gold differs across models for {key}")
        if len({str(x.get("prompt_hash")) for x in (b, a, s)}) != 1:
            raise ValueError(f"Public prompt hash differs across models for {key}")
        paired.append({
            "case_id": key[0], "state_id": key[1], "split": args.split,
            "gold": b["gold"], "base_correct": bool(b["correct"]),
            "awq_correct": bool(a["correct"]), "sft_correct": bool(s["correct"]),
            "base_valid": bool(b["valid"]), "awq_valid": bool(a["valid"]), "sft_valid": bool(s["valid"]),
            "base_tokens": int(b["prompt_tokens"]) + int(b["completion_tokens"]),
            "awq_tokens": int(a["prompt_tokens"]) + int(a["completion_tokens"]),
            "sft_tokens": int(s["prompt_tokens"]) + int(s["completion_tokens"]),
            "prompt_hash": b.get("prompt_hash"),
        })
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in paired:
        by_case[row["case_id"]].append(row)

    def summarize_rows(rows: list[dict[str, Any]], model: str) -> dict[str, Any]:
        corr = f"{model}_correct"
        base_wrong = [x for x in rows if not x["base_correct"]]
        base_right = [x for x in rows if x["base_correct"]]
        helped = sum(not x["base_correct"] and x[corr] for x in rows)
        harmed = sum(x["base_correct"] and not x[corr] for x in rows)
        diff_ci = cluster_bootstrap(by_case, lambda sample: accuracy(sample, corr) - accuracy(sample, "base_correct"), args.seed, args.bootstrap_samples)
        result = {
            "split": args.split, "model": model, "n_states": len(rows), "n_cases": len(by_case),
            "accuracy_state_weighted": accuracy(rows, corr),
            "accuracy_case_macro": sum(sum(x[corr] for x in by_case[case]) / len(by_case[case]) for case in by_case) / len(by_case),
            "parse_success_rate": sum(x[f"{model}_valid"] for x in rows) / len(rows),
            "helpful_states": helped, "harmful_states": harmed,
            "both_correct_states": sum(x["base_correct"] and x[corr] for x in rows),
            "both_wrong_states": sum(not x["base_correct"] and not x[corr] for x in rows),
            "expert_correction_rate_given_base_wrong": helped / len(base_wrong) if base_wrong else None,
            "base_correct_regression_rate": harmed / len(base_right) if base_right else None,
            "paired_accuracy_diff_vs_base": accuracy(rows, corr) - accuracy(rows, "base_correct"),
            "paired_accuracy_diff_case_bootstrap_95_low": diff_ci[0],
            "paired_accuracy_diff_case_bootstrap_95_high": diff_ci[1],
            "mean_tokens": sum(x[f"{model}_tokens"] for x in rows) / len(rows),
        }
        return result

    summaries = [summarize_rows(paired, model) for model in ("base", "awq", "sft")]
    oracle_cases = []
    for cost in COSTS:
        decisions = []
        for row in paired:
            use_expert = int(row["awq_correct"]) - int(row["base_correct"]) > cost
            correct = row["awq_correct"] if use_expert else row["base_correct"]
            decisions.append({**row, "selected_expert": use_expert, "policy_correct": correct, "utility": int(correct) - (cost if use_expert else 0.0)})
        by_oracle_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in decisions:
            by_oracle_case[row["case_id"]].append(row)
        ci = cluster_bootstrap(by_oracle_case, lambda sample: sum(r["utility"] for r in sample) / len(sample), args.seed, args.bootstrap_samples)
        oracle_cases.append({
            "split": args.split, "policy": "oracle_selective_defer_qwen32_awq",
            "defer_cost": cost, "accuracy": accuracy(decisions, "policy_correct"),
            "utility_accuracy_minus_cost": sum(x["utility"] for x in decisions) / len(decisions),
            "defer_rate": sum(x["selected_expert"] for x in decisions) / len(decisions),
            "mean_expert_cost": cost * sum(x["selected_expert"] for x in decisions) / len(decisions),
            "utility_case_bootstrap_95_low": ci[0], "utility_case_bootstrap_95_high": ci[1],
            "upper_bound_only_uses_gold_posthoc": True,
        })

    outdir.mkdir(parents=True, exist_ok=False)
    with (outdir / "paired_case_state_results.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired[0]))
        writer.writeheader(); writer.writerows(paired)
    with (outdir / "paired_model_summary.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader(); writer.writerows(summaries)
    with (outdir / "oracle_cost_sensitivity.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(oracle_cases[0]))
        writer.writeheader(); writer.writerows(oracle_cases)
    manifest = {
        "status": "paired_analysis_completed", "split": args.split,
        "input_sha256": {"base": hashlib.sha256(args.base.read_bytes()).hexdigest(), "awq": hashlib.sha256(args.awq.read_bytes()).hexdigest(), "sft": hashlib.sha256(args.sft.read_bytes()).hexdigest()},
        "cases": len(by_case), "paired_keys": len(paired), "bootstrap_unit": "case_id cluster; all states from a case are resampled together",
        "bootstrap_samples": args.bootstrap_samples, "seed": args.seed,
        "clean_holdout_touched": args.split == "clean_holdout",
        "oracle_uses_gold_for_analysis_only": True,
        "output_files": ["paired_case_state_results.csv", "paired_model_summary.csv", "oracle_cost_sensitivity.csv"],
    }
    (outdir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"split": args.split, "n_states": len(paired), "n_cases": len(by_case), "model_summaries": summaries, "oracle_costs": oracle_cases, "output_dir": str(outdir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
