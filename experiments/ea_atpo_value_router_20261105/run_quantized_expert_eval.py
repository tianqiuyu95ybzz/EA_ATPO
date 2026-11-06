#!/usr/bin/env python3
"""Evaluate a pinned official Qwen3-32B quantized Expert on the frozen 200 set."""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
import sys

import aiohttp
import pandas as pd


HERE = Path(__file__).resolve().parent
AUDIT_DIR = HERE.parent / "real_case_pilot_20261103/audit"
sys.path.insert(0, str(AUDIT_DIR))
import run_direct_reliability_eval as direct  # noqa: E402
import run_strong_expert_comparison as strong  # noqa: E402


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


async def run(args: argparse.Namespace) -> None:
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    baseline_manifest = json.loads(strong.BASELINE_MANIFEST.read_text(encoding="utf-8"))
    if baseline_manifest.get("case_count") != 200 or baseline_manifest.get("sample_seed") != direct.SEED:
        raise RuntimeError("saved Base baseline is not the frozen 200-case set")
    if baseline_manifest.get("data_sha256") != sha(direct.DATA_PATH):
        raise RuntimeError("test parquet hash differs from the saved Base baseline")
    base_rows = [x for x in strong.load_jsonl(strong.BASELINE_TRAJECTORIES)
                 if x.get("record_type") == "direct_answer_evaluation" and x.get("model_role") == "base_8b"]
    if len(base_rows) != 200 or len({x["case_id"] for x in base_rows}) != 200:
        raise RuntimeError("expected 200 frozen Qwen3-8B baseline rows")

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
    base_by_case = {str(x["case_id"]): x for x in base_rows}
    if {x["case_id"] for x in selected} != set(base_by_case):
        raise RuntimeError("case list differs from the saved baseline")
    for item in selected:
        base = base_by_case[item["case_id"]]
        if base.get("prompt_sha256") != item["prompt_sha256"] or base.get("gold") != item["gold"]:
            raise RuntimeError(f"saved baseline mismatch for {item['case_id']}")

    trajectory_path = out / "qwen3_32b_direct_answers.jsonl"
    manifest_path = out / "run_manifest.json"
    candidate = {"role": "candidate_expert_32b", "alias": args.alias, "base_url": args.base_url.rstrip("/") + "/v1"}
    manifest = {
        "status": "running",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experiment": "Pinned official quantized Qwen3-32B direct-answer comparison on frozen independent 200-case sample",
        "data_path": str(direct.DATA_PATH.resolve()), "data_sha256": sha(direct.DATA_PATH),
        "case_count": len(selected), "sample_seed": direct.SEED, "sampled_source_rows": indices,
        "base_baseline_path": str(strong.BASELINE_TRAJECTORIES.resolve()),
        "base_baseline_sha256": sha(strong.BASELINE_TRAJECTORIES),
        "models": [
            {"role": "base_8b", "repo_id": "Qwen/Qwen3-8B", "revision": strong.BASE_REVISION, "result_source": "Previously frozen paired baseline; not overwritten."},
            {"role": "candidate_expert_32b", "repo_id": args.repo_id, "revision": args.revision, "precision": args.precision, "served_alias": args.alias, "base_url": candidate["base_url"]},
        ],
        "generation": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 64, "enable_thinking": False,
                       "system_prompt_sha256": hashlib.sha256(direct.SYSTEM.encode()).hexdigest(),
                       "same_as_saved_baseline": True},
        "scoring": "same exact single-option parser as frozen Base evaluation; invalid response scores 0",
        "gold_context_policy": "Gold is used only by evaluator; no gold or atomic_facts is sent to model prompts.",
        "trajectory_path": str(trajectory_path.resolve()),
    }
    if manifest_path.exists():
        if not args.resume:
            raise FileExistsError("candidate run already exists; pass --resume to continue")
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("data_sha256", "sampled_source_rows", "base_baseline_sha256", "generation", "models"):
            if old.get(key) != manifest.get(key):
                raise RuntimeError(f"resume mismatch for {key}")
    elif trajectory_path.exists():
        raise FileExistsError("candidate trajectory exists without a matching manifest")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    existing = strong.load_jsonl(trajectory_path) if trajectory_path.exists() else []
    done = {str(x["case_id"]) for x in existing}
    if len(done) != len(existing):
        raise RuntimeError("duplicate candidate rows in resumable trajectory")
    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(candidate["base_url"] + "/models") as response:
            response.raise_for_status()
            ids = [str(x.get("id")) for x in (await response.json()).get("data", [])]
        if candidate["alias"] not in ids:
            raise RuntimeError(f"served candidate alias not available: {ids}")
        with trajectory_path.open("a", encoding="utf-8") as writer:
            for item in selected:
                if item["case_id"] in done:
                    continue
                result = await direct.call_model(session, candidate, item["messages"])
                record = {
                    "record_type": "direct_answer_evaluation", "case_id": item["case_id"],
                    "source_row_index": item["source_row_index"], "model_role": candidate["role"],
                    "model_alias": candidate["alias"], "model_repo_id": args.repo_id,
                    "model_revision": args.revision, "model_precision": args.precision,
                    "prompt_sha256": item["prompt_sha256"], "gold": item["gold"],
                    "predicted": result["predicted"], "correct": bool(result["predicted"] == item["gold"]),
                    "valid": result["predicted"] is not None and result["error"] is None,
                    "raw_response": result["raw_response"], "error": result["error"],
                    "prompt_tokens": result["prompt_tokens"], "completion_tokens": result["completion_tokens"],
                    "latency_s": result["latency_s"],
                }
                writer.write(json.dumps(record, ensure_ascii=False) + "\n"); writer.flush()
                existing.append(record); done.add(item["case_id"])
                if len(done) % 20 == 0:
                    print(f"candidate cases completed: {len(done)}/200", flush=True)
    if len(done) != 200:
        raise RuntimeError(f"incomplete candidate results: {len(done)}/200; rerun with --resume")

    strong.OUT_DIR = out
    paired_manifest = dict(manifest)
    # Reuse the established paired/cluster-bootstrap calculations but write
    # only inside this precision-specific output directory.
    strong.write_results(base_rows, existing, paired_manifest)
    paired_manifest.update({
        "status": "completed", "candidate_trajectory_sha256": sha(trajectory_path),
        "candidate_accuracy": float(sum(bool(x["correct"]) for x in existing) / len(existing)),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    strong.OUT_DIR = out
    (out / "run_manifest.json").write_text(json.dumps(paired_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (out / "strong_expert_summary.csv").open(encoding="utf-8", newline="") as stream:
        metric_rows = list(csv.DictReader(stream))
    by_key = {(row["metric"], row["group"]): row for row in metric_rows}
    def value(metric, group):
        return by_key.get((metric, group), {}).get("value", "")
    def confidence(metric, group):
        row = by_key.get((metric, group), {})
        low, high = row.get("ci95_low", ""), row.get("ci95_high", "")
        return f" ({float(low):.3f}–{float(high):.3f})" if low not in (None, "") and high not in (None, "") else ""
    report = [
        "# 官方 Qwen3-32B 量化 Expert 条件可靠性评估", "",
        f"模型：`{args.repo_id}`，revision `{args.revision}`，precision `{args.precision}`；共评估冻结的 200 个独立病例。", "",
        "候选 Expert 与 Qwen3-8B Base 使用完全相同的公开病例、提示词、答案解析和固定抽样。Gold 仅用于推理结束后的评分，atomic_facts 未进入模型提示。无效回答按 0 计入准确率。强模型在此是 LLM Expert，不是真实医生。", "",
        "## 结果", "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| Qwen3-8B Base accuracy | {float(value('accuracy','base_8b')):.3f}{confidence('accuracy','base_8b')} |",
        f"| Qwen3-32B candidate accuracy | {float(value('accuracy','candidate_expert_32b')):.3f}{confidence('accuracy','candidate_expert_32b')} |",
        f"| Paired accuracy difference, candidate − Base | {float(value('paired_accuracy_delta_candidate_minus_base','candidate_expert_32b-base_8b')):+.3f}{confidence('paired_accuracy_delta_candidate_minus_base','candidate_expert_32b-base_8b')} |",
        f"| Candidate accuracy conditioned on Base error | {float(value('candidate_accuracy_given_base_error','paired_conditional')):.3f}{confidence('candidate_accuracy_given_base_error','paired_conditional')} |",
        f"| Expert correction rate on Base errors | {float(value('correction_rate_given_base_error','paired_conditional')):.3f}{confidence('correction_rate_given_base_error','paired_conditional')} |",
        f"| Expert degradation rate on Base correct answers | {float(value('degradation_rate_given_base_correct','paired_conditional')):.3f}{confidence('degradation_rate_given_base_correct','paired_conditional')} |",
        "",
        "## 可复现文件", "",
        "- `qwen3_32b_direct_answers.jsonl`：200 条候选模型原始评估轨迹。",
        "- `paired_correctness.csv`：逐病例 Base/candidate 配对结果。",
        "- `strong_expert_summary.csv`：准确率、条件可靠性、配对置信区间、token 和延迟统计。",
        "- `run_manifest.json`、`model_service_manifest.json`：版本、样本、数据哈希和推理配置。",
    ]
    (out / "STRONG_EXPERT_EVALUATION.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "cases": 200, "summary": str(out / "strong_expert_summary.csv")}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--precision", required=True, choices=("FP8", "AWQ"))
    parser.add_argument("--alias", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:30132")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
