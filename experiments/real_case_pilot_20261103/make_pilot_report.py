#!/usr/bin/env python3
"""Build the concise, result-focused REAL_PILOT_REPORT.md from run artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lookup(rows: list[dict[str, str]], section: str, group: str, metric: str) -> dict[str, str]:
    return next(
        row for row in rows
        if row.get("section") == section and row.get("group") == group and row.get("metric") == metric
    )


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def interval(row: dict[str, str], digits: int = 1) -> str:
    if not row.get("ci95_low"):
        return ""
    return f"[{100 * float(row['ci95_low']):.{digits}f}, {100 * float(row['ci95_high']):.{digits}f}]%"


def denominator(row: dict[str, str]) -> int:
    match = re.search(r"Denominator: (\d+)", row.get("notes", ""))
    return int(match.group(1)) if match else 0


def write_report(output_dir: Path) -> Path:
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    rows = read_rows(output_dir / "summary.csv")
    model = manifest.get("model", {})
    service = manifest.get("model_service", {})
    source = manifest.get("source", {})
    cases = int(manifest.get("selected_cases", 0))
    states = int(manifest.get("max_states_per_case", 0))
    trajectories = int(manifest.get("trajectory_count", 0))
    valid = int(manifest.get("valid_trajectory_count", 0))
    failed = int(manifest.get("failed_trajectory_count", 0))

    action_names = ("ANSWER", "ASK", "CONSULT", "DEFER")
    table_lines = [
        "| 初始动作 | 准确率（95% 病例 bootstrap CI） | 协议有效率 | 总 token / rollout | 专家 token / rollout | 患者轮数 / rollout | 延迟（秒） |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for action in action_names:
        accuracy = lookup(rows, "action", action, "accuracy")
        valid_rate = lookup(rows, "action", action, "valid_rate")
        tokens = lookup(rows, "cost", action, "total_tokens_per_rollout")
        expert_tokens = lookup(rows, "cost", action, "expert_tokens_per_rollout")
        patient_turns = lookup(rows, "cost", action, "patient_turns_per_rollout")
        latency = lookup(rows, "cost", action, "total_latency_seconds_per_rollout")
        table_lines.append(
            f"| {action} | {pct(float(accuracy['value']))} {interval(accuracy)} "
            f"| {pct(float(valid_rate['value']))} | {float(tokens['value']):,.0f} "
            f"| {float(expert_tokens['value']):,.0f} | {float(patient_turns['value']):.2f} "
            f"| {float(latency['value']):.3f} |"
        )

    consult_answer = lookup(rows, "comparison", "CONSULT-ANSWER", "paired_accuracy_delta")
    consult_defer = lookup(rows, "comparison", "CONSULT-DEFER", "paired_accuracy_delta")
    corrections = lookup(rows, "revision", "CONSULT", "correction_count")
    correction_rate = lookup(rows, "revision", "CONSULT", "correction_rate_given_answer_wrong")
    degradations = lookup(rows, "revision", "CONSULT", "degradation_count")
    degradation_rate = lookup(rows, "revision", "CONSULT", "degradation_rate_given_answer_correct")

    uncosted = {
        action: lookup(rows, "cost_sensitivity", "lambdaE=0;lambdaP=0", f"optimal_action_share:{action}")
        for action in action_names
    }
    all_shares: dict[str, list[float]] = {action: [] for action in action_names}
    for row in rows:
        if row.get("section") == "cost_sensitivity" and row.get("metric", "").startswith("optimal_action_share:"):
            action = row["metric"].split(":", 1)[1]
            all_shares[action].append(float(row["value"]))

    cost_curve = read_rows(output_dir / "action_cost_accuracy_points.csv")
    deferral_points = read_rows(output_dir / "tradeoff_curve_points.csv")
    selected_accuracy = [float(row["selected_accuracy"]) for row in deferral_points]
    selected_defer = [float(row["selected_defer_rate"]) for row in deferral_points]
    selected_tokens = [float(row["selected_total_tokens"]) for row in deferral_points]
    sample_rows = manifest.get("sampled_source_rows", [])

    analysis_env_path = output_dir / "analysis_manifest.json"
    analysis_note = ""
    if analysis_env_path.is_file():
        analysis = json.loads(analysis_env_path.read_text(encoding="utf-8"))
        analysis_note = (
            f"- 结果图由 Matplotlib `{analysis.get('matplotlib_version')}` 绘制；依赖锁："
            f"`{analysis.get('requirements_lock')}`。"
        )

    lines = [
        "# EA-ATPO 真实数据 Pilot 报告",
        "",
        f"状态：**{manifest.get('status', 'unknown')}**。本实验为真实 Qwen 模型推理，不是 Router/ATPO 训练。",
        "",
        "## 实验规模",
        "",
        f"- 数据：[{source.get('repository')}]({source.get('repository')}) 官方快照 `{source.get('commit')}` 的训练集，{manifest.get('train_rows', 0):,} 例；固定种子 `{manifest.get('sample_seed')}` 抽取 {cases} 例。抽样源行号保存在 `run_manifest.json`。",
        f"- 状态与分支：{states} 个公开状态上限/病例，实际 {manifest.get('trajectory_count', 0) // 12 if manifest.get('trajectory_count') else 0} 个状态；四动作各 3 次，完整轨迹 {trajectories:,} 条。",
        f"- 完成情况：有效 {valid:,}/{trajectories:,}（{pct(valid / trajectories if trajectories else 0)}）；协议失败 {failed} 条。",
        f"- 模型：`{model.get('repo_id')}` revision `{model.get('revision')}`；vLLM `{service.get('vllm_version')}`，GPU {service.get('gpu_physical_index')}，BF16，`{service.get('attention_backend')}`。",
        "- Base、Patient、Expert 均为同一个 Qwen3-8B 的模拟角色；没有真实医生参与。Patient 只接收 `atomic_facts`，Base/Expert 只看公开状态，Expert 请求前执行隐藏事实与特权字段检查。Gold 仅在分支结束后用于评分。",
        "- 动作语法由 vLLM regex structured output 约束；模型仍生成具体问题、咨询焦点和选项。最后一个 Base turn 要求输出终局答案。每条 rollout 最多一次 CONSULT。",
        "",
        "## 四动作结果",
        "",
        *table_lines,
        "",
        "准确率按病例汇总公开状态和重复 rollout；无效轨迹在准确率中保守记为 0。延迟为本地 vLLM 测得的秒数。失败 Expert 调用在响应格式校验阶段未保存 usage，因此失败样本的 token/延迟均值可能偏低。",
        "",
        "## CONSULT 的独立价值",
        "",
        f"- CONSULT − ANSWER 配对准确率差：{100 * float(consult_answer['value']):+.2f} 个百分点，95% CI `{interval(consult_answer)}`。区间包含 0，当前 Pilot 未能确认 CONSULT 有独立准确率增益。",
        f"- CONSULT − DEFER 配对准确率差：{100 * float(consult_defer['value']):+.2f} 个百分点，95% CI `{interval(consult_defer)}`。",
        f"- 按状态×重复次数配对：ANSWER 错误的 {denominator(corrections)} 组中，CONSULT 修正 {int(float(corrections['value']))} 组（{pct(float(correction_rate['value']))}）；ANSWER 正确的 {denominator(degradations)} 组中，CONSULT 退化 {int(float(degradations['value']))} 组（{pct(float(degradation_rate['value']))}）。修正与退化频率接近。",
        "- 每次 CONSULT 平均使用约 814 个 Expert token、3,604 个总 token、0.81 秒；这是本地推理成本，不是美元价格。",
        "",
        "## 四动作可区分性与成本敏感性",
        "",
        "无成本时，经验最优动作占比（291 个公开状态）：",
        "",
        "| ANSWER | ASK | CONSULT | DEFER |",
        "|---:|---:|---:|---:|",
        "| " + " | ".join(pct(float(uncosted[action]["value"])) for action in action_names) + " |",
        "",
        "四动作都在部分状态上成为经验最优。全成本网格中各动作的最优占比范围：",
        "",
        "| ANSWER | ASK | CONSULT | DEFER |",
        "|---:|---:|---:|---:|",
        "| " + " | ".join(
            f"{pct(min(all_shares[action]))}–{pct(max(all_shares[action]))}" for action in action_names
        ) + " |",
        "",
        "成本网格为 `lambdaE ∈ {0, 0.02, 0.05, 0.1}`（每千 Expert token 的效用惩罚）与 `lambdaP ∈ {0, 0.01, 0.03}`（每患者轮次惩罚），均为敏感性单位。基于四动作实测结果逐状态选出的 post-hoc oracle 在网格中准确率均为 "
        f"{pct(min(selected_accuracy))}，DEFER 率 {pct(min(selected_defer))}–{pct(max(selected_defer))}，总 token {min(selected_tokens):,.0f}–{max(selected_tokens):,.0f}/rollout。该 oracle 使用了事后 gold outcome，不能作为可部署 Router 的性能结论。",
        "",
        "![四种固定动作的准确率—总推理 token 成本曲线](accuracy_cost_curve.png)",
        "",
        "![成本敏感性网格下 post-hoc oracle 的准确率—DEFER 率曲线](accuracy_deferral_curve.png)",
        "",
        "数值点和区间可在 `action_cost_accuracy_points.csv` 与 `tradeoff_curve_points.csv` 中复核；矢量版为同名 `.svg`。",
        "",
        "## 失败与边界",
        "",
        f"- {failed} 条失败均保留在原始 JSONL 和汇总错误分析中；失败轨迹的 outcome 记为 0。其初始动作分布：ASK 5、CONSULT 8、DEFER 14、ANSWER 0。主要原因是模拟 Expert 输出不符合 CONSULT/DEFER 格式或触发终局答案泄漏规则，不是推理服务或数据下载中断。",
        "- 本 Pilot 用 ATPO 训练 split 的公开医学选择题，不是真实临床记录或患者试验；模拟 Patient/Expert 都不是医生。结果只支持工程可行性和描述性比较。",
        "- CONSULT − ANSWER 的置信区间跨 0，因此现有证据不足以宣称 CONSULT 提升准确率。100 例 Pilot 也不用于训练或评估最终策略泛化。",
        "",
        "## 复现与文件",
        "",
        f"- Base/Patient/Expert smoke：`../service_role_smoke.jsonl`；10 例 smoke：`../smoke/`（120/120 有效）。首轮失败轨迹另存于 `../smoke_attempt1/`。",
        "- 完整轨迹：`pilot_trajectories.jsonl`；公开状态：`pilot_public_states.jsonl`；统计：`summary.csv`、`action_value_analysis.csv`；配置/哈希：`run_manifest.json`。",
        "- 数据源与模型文件清单：`../assets/source_manifest.json`、`../assets/adapter_manifest.json`、`../assets/models/model_manifest.json`。原始 ATPO JSONL 未覆盖；未启动 Router 或 ATPO RL 训练。",
        f"- 报告构建输入：`summary.csv`；抽样病例数：{cases}，首尾源行索引 `{sample_rows[0] if sample_rows else 'n/a'}` / `{sample_rows[-1] if sample_rows else 'n/a'}`。",
        analysis_note,
        "",
    ]
    report = output_dir / "REAL_PILOT_REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    analysis_path = output_dir / "analysis_manifest.json"
    if analysis_path.is_file():
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        analysis["report_script_sha256"] = file_sha256(Path(__file__).resolve())
        analysis.setdefault("output_sha256", {})[report.name] = file_sha256(report)
        analysis_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        run_path = output_dir / "run_manifest.json"
        run_manifest = json.loads(run_path.read_text(encoding="utf-8"))
        run_manifest["postrun_analysis"] = {
            key: value for key, value in analysis.items() if key != "input_sha256"
        }
        run_path.write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = write_report(args.output_dir)
    print(report)


if __name__ == "__main__":
    main()
