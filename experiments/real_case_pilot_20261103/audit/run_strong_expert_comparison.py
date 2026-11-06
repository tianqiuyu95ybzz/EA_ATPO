#!/usr/bin/env python3
"""Run Qwen3-32B on the previously frozen 200-case Qwen3-8B test sample."""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import random
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp
import pandas as pd

import run_direct_reliability_eval as direct


AUDIT_DIR = Path(__file__).resolve().parent
PILOT_DIR = AUDIT_DIR.parent
OUT_DIR = AUDIT_DIR / "outputs/strong_expert_evaluation"
BASELINE_DIR = AUDIT_DIR / "outputs/direct_eval"
BASELINE_TRAJECTORIES = BASELINE_DIR / "direct_reliability_trajectories.jsonl"
BASELINE_MANIFEST = BASELINE_DIR / "expert_reliability_manifest.json"
CANDIDATE = {
    "role": "candidate_expert_32b",
    "alias": "qwen3-32b-candidate",
    "base_url": "http://127.0.0.1:30132/v1",
}
CANDIDATE_REVISION = "9216db5781bf21249d130ec9da846c4624c16137"
BASE_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
BOOTSTRAP_REPEATS = 10_000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def case_bootstrap(
    rows: list[dict[str, Any]], stat_fn, seed: int, repeats: int = BOOTSTRAP_REPEATS
) -> tuple[float, float]:
    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(str(row["case_id"]), []).append(row)
    cases = sorted(by_case)
    rng = random.Random(seed)
    values = []
    for _ in range(repeats):
        sampled = rng.choices(cases, k=len(cases))
        sample_rows = [row for case in sampled for row in by_case[case]]
        value = stat_fn(sample_rows)
        if value is not None:
            values.append(float(value))
    if not values:
        return float("nan"), float("nan")
    values.sort()
    return values[int(.025 * len(values))], values[min(len(values) - 1, int(.975 * len(values)))]


def mean_correct(rows: list[dict[str, Any]], field: str) -> float:
    return statistics.mean(float(bool(row.get(field))) for row in rows)


def ratio(rows: list[dict[str, Any]], numerator, denominator) -> float | None:
    den = sum(bool(denominator(row)) for row in rows)
    if not den:
        return None
    return sum(bool(numerator(row)) and bool(denominator(row)) for row in rows) / den


def write_results(base_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    by_base = {str(row["case_id"]): row for row in base_rows}
    by_candidate = {str(row["case_id"]): row for row in candidate_rows}
    if set(by_base) != set(by_candidate):
        raise RuntimeError("base and candidate case sets do not match")
    paired = []
    for case_id in sorted(by_base):
        b, c = by_base[case_id], by_candidate[case_id]
        if b["prompt_sha256"] != c["prompt_sha256"]:
            raise RuntimeError(f"prompt mismatch for {case_id}")
        paired.append({
            "case_id": case_id,
            "base_correct": bool(b.get("correct")),
            "candidate_correct": bool(c.get("correct")),
            "base_valid": bool(b.get("valid")),
            "candidate_valid": bool(c.get("valid")),
        })

    base_errors = lambda x: not x["base_correct"]
    base_correct = lambda x: x["base_correct"]
    candidate_correct = lambda x: x["candidate_correct"]
    paired_delta = lambda xs: statistics.mean(float(x["candidate_correct"]) - float(x["base_correct"]) for x in xs)
    correction = lambda xs: ratio(xs, candidate_correct, base_errors)
    degradation = lambda xs: ratio(xs, lambda x: not x["candidate_correct"], base_correct)

    summary: list[dict[str, Any]] = []

    def add(metric: str, group: str, value: float | int, low: float | str = "", high: float | str = "", n: int | None = None, notes: str = "") -> None:
        summary.append({
            "metric": metric, "group": group, "value": value,
            "ci95_low": low, "ci95_high": high,
            "n_cases": len(paired) if n is None else n,
            "notes": notes,
        })

    for role, field in (("base_8b", "base_correct"), ("candidate_expert_32b", "candidate_correct")):
        values = [float(row[field]) for row in paired]
        point = statistics.mean(values)
        low, high = case_bootstrap(paired, lambda xs, f=field: mean_correct(xs, f), direct.SEED + 30)
        corresponding = base_rows if role == "base_8b" else candidate_rows
        add("accuracy", role, point, low, high, notes="Case-level bootstrap; invalid/parse-failed response scores 0.")
        add("valid_rate", role, statistics.mean(float(bool(row.get("valid"))) for row in corresponding), notes="Invalid outputs remain in the accuracy denominator.")
        add("mean_total_tokens", role, statistics.mean(int(row.get("prompt_tokens", 0)) + int(row.get("completion_tokens", 0)) for row in corresponding), notes="Prompt plus completion tokens returned by vLLM.")
        add("mean_latency_s", role, statistics.mean(float(row.get("latency_s", 0)) for row in corresponding), notes="Client-observed per-request latency; requests run sequentially.")

    delta = paired_delta(paired)
    d_low, d_high = case_bootstrap(paired, paired_delta, direct.SEED + 31)
    add("paired_accuracy_delta_candidate_minus_base", "candidate_expert_32b-base_8b", delta, d_low, d_high, notes="Paired case bootstrap of correctness-indicator differences.")

    for metric_name, stat_fn, denominator in (
        ("candidate_accuracy_given_base_error", correction, base_errors),
        ("correction_rate_given_base_error", correction, base_errors),
        ("degradation_rate_given_base_correct", degradation, base_correct),
    ):
        point = stat_fn(paired)
        low, high = case_bootstrap(paired, stat_fn, direct.SEED + (32 if metric_name != "degradation_rate_given_base_correct" else 33))
        den = sum(bool(denominator(row)) for row in paired)
        add(metric_name, "paired_conditional", point if point is not None else "", low, high, n=den,
            notes=f"Conditional denominator: {den} baseline cases in the specified stratum.")

    cells = Counter(
        ("base_correct" if row["base_correct"] else "base_wrong",
         "candidate_correct" if row["candidate_correct"] else "candidate_wrong")
        for row in paired
    )
    for (base_cell, candidate_cell), count in sorted(cells.items()):
        add("paired_correctness_cell", f"{base_cell}__{candidate_cell}", count, n=len(paired), notes="Four-cell paired contingency table.")

    summary_path = OUT_DIR / "strong_expert_summary.csv"
    fields = ("metric", "group", "value", "ci95_low", "ci95_high", "n_cases", "notes")
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    paired_path = OUT_DIR / "paired_correctness.csv"
    with paired_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=tuple(paired[0]))
        writer.writeheader()
        writer.writerows(paired)

    manifest.update({
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evaluated_cases": len(paired),
        "base_error_cases": sum(not row["base_correct"] for row in paired),
        "base_correct_cases": sum(row["base_correct"] for row in paired),
        "paired_cell_counts": {f"{a}__{b}": n for (a, b), n in cells.items()},
        "candidate_accuracy": mean_correct(paired, "candidate_correct"),
        "base_accuracy": mean_correct(paired, "base_correct"),
        "paired_accuracy_delta": delta,
        "candidate_accuracy_given_base_error": correction(paired),
        "correction_rate_given_base_error": correction(paired),
        "degradation_rate_given_base_correct": degradation(paired),
        "trajectory_sha256": sha256(OUT_DIR / "qwen3_32b_direct_answers.jsonl"),
        "summary_sha256": sha256(summary_path),
        "paired_table_sha256": sha256(paired_path),
    })
    (OUT_DIR / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


async def main_async() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline_manifest = json.loads(BASELINE_MANIFEST.read_text(encoding="utf-8"))
    if baseline_manifest.get("case_count") != 200 or baseline_manifest.get("sample_seed") != direct.SEED:
        raise RuntimeError("existing Qwen3-8B baseline is not the frozen 200-case sample")
    if baseline_manifest.get("data_sha256") != sha256(direct.DATA_PATH):
        raise RuntimeError("test data hash differs from the saved baseline")
    base_rows = [x for x in load_jsonl(BASELINE_TRAJECTORIES) if x.get("record_type") == "direct_answer_evaluation" and x.get("model_role") == "base_8b"]
    if len(base_rows) != 200 or len({x["case_id"] for x in base_rows}) != 200:
        raise RuntimeError("expected 200 unique saved Qwen3-8B baseline rows")

    frame = pd.read_parquet(direct.DATA_PATH)
    indices = random.Random(direct.SEED).sample(range(len(frame)), direct.CASE_COUNT)
    selected = []
    for source_i in indices:
        row = frame.iloc[source_i].to_dict()
        messages = direct.build_messages(row)
        gold = direct.answer_letter(row.get("ground_truth") or row.get("reward_model", {}).get("ground_truth"))
        case_id = f"test-row-{source_i}:item-{row.get('index', source_i)}"
        digest = hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        selected.append({"case_id": case_id, "source_row_index": int(source_i), "gold": gold, "messages": messages, "prompt_sha256": digest})
    base_by_case = {str(row["case_id"]): row for row in base_rows}
    if {row["case_id"] for row in selected} != set(base_by_case):
        raise RuntimeError("reconstructed test sample does not match the saved baseline cases")
    for item in selected:
        if base_by_case[item["case_id"]].get("prompt_sha256") != item["prompt_sha256"] or base_by_case[item["case_id"]].get("gold") != item["gold"]:
            raise RuntimeError(f"baseline prompt/gold mismatch at {item['case_id']}")

    trajectory_path = OUT_DIR / "qwen3_32b_direct_answers.jsonl"
    manifest_path = OUT_DIR / "run_manifest.json"
    manifest = {
        "status": "running",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experiment": "Candidate Qwen3-32B direct answering on the exact saved Qwen3-8B 200-case sample.",
        "data_path": str(direct.DATA_PATH.resolve()),
        "data_sha256": sha256(direct.DATA_PATH),
        "case_count": len(selected),
        "sample_seed": direct.SEED,
        "sampled_source_rows": indices,
        "base_baseline_path": str(BASELINE_TRAJECTORIES.resolve()),
        "base_baseline_sha256": sha256(BASELINE_TRAJECTORIES),
        "models": [
            {"role": "base_8b", "repo_id": "Qwen/Qwen3-8B", "revision": BASE_REVISION, "result_source": "Previously completed independent 200-case evaluation; not rerun or overwritten."},
            {"role": "candidate_expert_32b", "repo_id": "Qwen/Qwen3-32B", "revision": CANDIDATE_REVISION, "served_alias": CANDIDATE["alias"], "base_url": CANDIDATE["base_url"]},
        ],
        "generation": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 64, "enable_thinking": False, "seed": None,
                       "system_prompt_sha256": hashlib.sha256(direct.SYSTEM.encode()).hexdigest(),
                       "same_as_saved_baseline": True},
        "scoring": "Exact single option letter, using the same parser as saved baseline; invalid response scores 0.",
        "gold_context_policy": "Gold is retained only by the evaluator and is not sent in any model request. Hidden atomic_facts are not used.",
        "token_usage_policy": "Usage returned by the OpenAI-compatible vLLM API.",
        "trajectory_path": str(trajectory_path.resolve()),
    }
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("data_sha256", "sampled_source_rows", "base_baseline_sha256", "generation", "models"):
            if prior.get(key) != manifest.get(key):
                raise RuntimeError(f"resume mismatch in {key}")
        manifest = prior
    elif trajectory_path.exists():
        raise FileExistsError("candidate trajectory exists without manifest")

    existing = load_jsonl(trajectory_path) if trajectory_path.exists() else []
    done = {str(x["case_id"]) for x in existing}
    if len(done) != len(existing):
        raise RuntimeError("duplicate candidate rows in resumable output")
    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        health = CANDIDATE["base_url"].rstrip("/") + "/models"
        async with session.get(health) as response:
            response.raise_for_status()
            ids = [str(x.get("id")) for x in (await response.json()).get("data", [])]
        if CANDIDATE["alias"] not in ids:
            raise RuntimeError(f"candidate alias missing from service: {ids}")
        for item in selected:
            if item["case_id"] in done:
                continue
            result = await direct.call_model(session, CANDIDATE, item["messages"])
            record = {
                "record_type": "direct_answer_evaluation",
                "case_id": item["case_id"], "source_row_index": item["source_row_index"],
                "model_role": CANDIDATE["role"], "model_alias": CANDIDATE["alias"],
                "model_revision": CANDIDATE_REVISION, "prompt_sha256": item["prompt_sha256"],
                "gold": item["gold"], "predicted": result["predicted"],
                "correct": bool(result["predicted"] == item["gold"]),
                "valid": result["predicted"] is not None and result["error"] is None,
                "raw_response": result["raw_response"], "error": result["error"],
                "prompt_tokens": result["prompt_tokens"], "completion_tokens": result["completion_tokens"],
                "latency_s": result["latency_s"],
            }
            with trajectory_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
            existing.append(record)
            done.add(item["case_id"])
            if len(done) % 20 == 0:
                print(f"candidate cases completed: {len(done)}/200", flush=True)

    if len(done) != 200:
        raise RuntimeError(f"candidate evaluation incomplete: {len(done)}/200; rerun to resume")
    candidate_rows = load_jsonl(trajectory_path)
    write_results(base_rows, candidate_rows, manifest)
    print(json.dumps({"status": "completed", "cases": 200, "summary": str(OUT_DIR / "strong_expert_summary.csv")}, indent=2))


if __name__ == "__main__":
    asyncio.run(main_async())
