#!/usr/bin/env python3
"""Recompute frozen 8B/32B paired statistics from saved case-level outputs.

This analysis-only command does not call a model or alter the raw trajectories.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
AUDIT = PROJECT / "experiments/real_case_pilot_20261103/audit"
sys.path.insert(0, str(AUDIT))
import run_strong_expert_comparison as paired  # noqa: E402

OUT = HERE / "outputs/strong_expert_awq"
CANDIDATE_PATH = OUT / "qwen3_32b_direct_answers.jsonl"
REPORT_PATH = OUT / "STRONG_EXPERT_EVALUATION.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_summary() -> dict[tuple[str, str], dict[str, str]]:
    with (OUT / "strong_expert_summary.csv").open(newline="", encoding="utf-8") as stream:
        return {(row["metric"], row["group"]): row for row in csv.DictReader(stream)}


def fmt(summary: dict, metric: str, group: str, precision: int = 3) -> str:
    row = summary[(metric, group)]
    value = float(row["value"])
    low, high = row["ci95_low"], row["ci95_high"]
    if low not in (None, "") and high not in (None, ""):
        return f"{value:.{precision}f}（95% CI {float(low):.{precision}f}–{float(high):.{precision}f}）"
    return f"{value:.{precision}f}"


def main() -> None:
    base_rows = [
        row for row in paired.load_jsonl(paired.BASELINE_TRAJECTORIES)
        if row.get("record_type") == "direct_answer_evaluation" and row.get("model_role") == "base_8b"
    ]
    candidate_rows = paired.load_jsonl(CANDIDATE_PATH)
    manifest_path = OUT / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise RuntimeError("candidate run manifest is not completed")
    if manifest.get("candidate_trajectory_sha256") not in (None, sha256(CANDIDATE_PATH)):
        raise RuntimeError("candidate trajectory hash differs from its completed run manifest")
    if len(base_rows) != 200 or len(candidate_rows) != 200:
        raise RuntimeError(f"expected 200 paired rows; got Base={len(base_rows)}, candidate={len(candidate_rows)}")
    if manifest.get("data_sha256") != paired.sha256(paired.direct.DATA_PATH):
        raise RuntimeError("frozen test dataset hash differs from the candidate run manifest")
    base_by_id = {str(row["case_id"]): row for row in base_rows}
    candidate_by_id = {str(row["case_id"]): row for row in candidate_rows}
    if len(base_by_id) != 200 or len(candidate_by_id) != 200 or set(base_by_id) != set(candidate_by_id):
        raise RuntimeError("Base and candidate case IDs are not a matching unique 200-case set")
    for case_id, base in base_by_id.items():
        candidate = candidate_by_id[case_id]
        if base.get("prompt_sha256") != candidate.get("prompt_sha256") or base.get("gold") != candidate.get("gold"):
            raise RuntimeError(f"paired prompt or gold mismatch for {case_id}")
        if candidate.get("model_repo_id") != "Qwen/Qwen3-32B-AWQ" or candidate.get("model_revision") != "0499c3ac83fdef8810b907a23894ba91e95eddd8":
            raise RuntimeError(f"unexpected candidate model identity for {case_id}")
        for label, row in (("Base", base), ("candidate", candidate)):
            parsed_correct = bool(row.get("valid")) and row.get("predicted") == row.get("gold")
            if bool(row.get("correct")) != parsed_correct:
                raise RuntimeError(f"stored {label} correctness disagrees with parsed answer/gold for {case_id}")

    manifest["statistics_recomputed_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest["statistics_script"] = str(Path(__file__).resolve())
    manifest["statistics_note"] = "Conditional numerator is intersected with its baseline stratum; raw model trajectories unchanged."
    paired.OUT_DIR = OUT
    paired.write_results(base_rows, candidate_rows, manifest)

    summary = load_summary()
    for metric in ("candidate_accuracy_given_base_error", "correction_rate_given_base_error", "degradation_rate_given_base_correct"):
        value = float(summary[(metric, "paired_conditional")]["value"])
        low = float(summary[(metric, "paired_conditional")]["ci95_low"])
        high = float(summary[(metric, "paired_conditional")]["ci95_high"])
        if not (0.0 <= low <= value <= high <= 1.0):
            raise RuntimeError(f"invalid conditional estimate or CI for {metric}: {low}, {value}, {high}")

    cells = {
        row["group"]: int(float(row["value"]))
        for row in summary.values() if row["metric"] == "paired_correctness_cell"
    }
    report = [
        "# Qwen3-32B AWQ 强 Expert 条件可靠性评估", "",
        "候选模型使用 Qwen 官方 `Qwen/Qwen3-32B-AWQ` AWQ 权重，与先前冻结的 Qwen3-8B Base 在同一组 200 个独立病例上配对比较。两者的公开题目、提示、答案解析和病例顺序一致；Gold 只在生成后用于评分，atomic_facts 未发送给模型。无效输出保留并计为错误。Expert 是 LLM 模拟器，不是真实医生。", "",
        "## 结果", "",
        "| 指标 | Qwen3-8B Base | Qwen3-32B AWQ |",
        "|---|---:|---:|",
        f"| 总体准确率 | {fmt(summary, 'accuracy', 'base_8b')} | {fmt(summary, 'accuracy', 'candidate_expert_32b')} |",
        f"| 有效输出率 | {fmt(summary, 'valid_rate', 'base_8b')} | {fmt(summary, 'valid_rate', 'candidate_expert_32b')} |",
        f"| 平均 tokens/病例 | {fmt(summary, 'mean_total_tokens', 'base_8b', 2)} | {fmt(summary, 'mean_total_tokens', 'candidate_expert_32b', 2)} |",
        f"| 平均模型请求延迟（秒/病例） | {fmt(summary, 'mean_latency_s', 'base_8b', 3)} | {fmt(summary, 'mean_latency_s', 'candidate_expert_32b', 3)} |",
        "",
        f"32B−8B 配对准确率差：{fmt(summary, 'paired_accuracy_delta_candidate_minus_base', 'candidate_expert_32b-base_8b')}。病例级配对 bootstrap CI 包含 0，因此这 200 例没有证明总体准确率提升。", "",
        f"在 Base 错误的 {int(float(summary[('candidate_accuracy_given_base_error', 'paired_conditional')]['n_cases']))} 例中，32B 答对 {cells['base_wrong__candidate_correct']} 例，条件准确率/错误修正率为 {fmt(summary, 'candidate_accuracy_given_base_error', 'paired_conditional')}。",
        f"在 Base 正确的 {int(float(summary[('degradation_rate_given_base_correct', 'paired_conditional')]['n_cases']))} 例中，32B 将 {cells['base_correct__candidate_wrong']} 例改为错误，正确退化率为 {fmt(summary, 'degradation_rate_given_base_correct', 'paired_conditional')}。", "",
        "| Base / 32B 配对结果 | 病例数 |",
        "|---|---:|",
        f"| 两者都正确 | {cells['base_correct__candidate_correct']} |",
        f"| Base 正确、32B 错误 | {cells['base_correct__candidate_wrong']} |",
        f"| Base 错误、32B 正确 | {cells['base_wrong__candidate_correct']} |",
        f"| 两者都错误 | {cells['base_wrong__candidate_wrong']} |", "",
        "## 统计与复现", "",
        "准确率差和条件比例使用 10,000 次病例级 bootstrap；条件统计在每次重采样中分别使用 Base 错误或 Base 正确的病例作为分母。统计从已保存轨迹离线重算，没有重新调用模型，也没有更改原始 JSONL。", "",
        "- 原始 32B 输出：`qwen3_32b_direct_answers.jsonl`。",
        "- 配对病例表：`paired_correctness.csv`；汇总表：`strong_expert_summary.csv`。",
        "- 模型部署、数据与推理参数：`model_service_manifest.json`、`run_manifest.json`。",
    ]
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": "recomputed", "base_cases": len(base_rows), "candidate_cases": len(candidate_rows),
                      "paired_accuracy_delta": manifest["paired_accuracy_delta"],
                      "candidate_accuracy_given_base_error": manifest["candidate_accuracy_given_base_error"],
                      "degradation_rate_given_base_correct": manifest["degradation_rate_given_base_correct"],
                      "report": str(REPORT_PATH)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
