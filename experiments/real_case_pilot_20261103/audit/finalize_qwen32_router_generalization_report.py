#!/usr/bin/env python3
"""Finish Attempt04 reporting from its already materialized analysis tables.

This report finalizer deliberately does not read the held-out paired rollout or
recompute scores, thresholds, confidence intervals, or policies. It only reads
the CSVs and manifest produced by the one completed analysis pass, plus raw
response metadata to classify parser failures without printing responses.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
RUN = HERE / "outputs/qwen32_router_generalization_20261118_attempt04"
ANALYSIS = RUN / "analysis"
ROLLOUTS = RUN / "paired_rollouts"
REPORT = HERE / "QWEN32_ROUTER_GENERALIZATION_20261118_RESULTS.md"


def read_csv(name: str) -> list[dict[str, str]]:
    path = ANALYSIS / name
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(name: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No diagnostics to write: {name}")
    with (ANALYSIS / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def f(value: str | float, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def parser_failure_diagnostics() -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str, str, str, str]] = Counter()
    expert_prefix = re.compile(r"^Expert Final Answer:\s*(.*)$", re.IGNORECASE | re.DOTALL)
    for filename, model in (
        ("base_answer_trajectories.jsonl", "Base"),
        ("qwen32_defer_trajectories.jsonl", "Expert"),
    ):
        path = ROLLOUTS / filename
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row.get("valid"):
                    continue
                response = (row.get("raw_response") or "").strip()
                finish = str(row.get("finish_reason") or "missing")
                if model == "Expert":
                    match = expert_prefix.match(response)
                    rest = (match.group(1) if match else "").lstrip()
                    shape = "required_prefix_non_option_text" if match and rest and rest[0].upper() not in "ABCD" else "other_strict_format_failure"
                else:
                    shape = "strict_base_answer_parse_failure"
                counts[(model, str(row.get("split")), str(row.get("parse_error")), finish, shape)] += 1
    rows = [
        {"model": key[0], "split": key[1], "parse_error": key[2], "finish_reason": key[3], "response_shape": key[4], "count": count}
        for key, count in sorted(counts.items())
    ]
    write_csv("parser_failure_diagnostics.csv", rows)
    return rows


def main() -> None:
    manifest_path = ANALYSIS / "analysis_manifest.json"
    provenance_path = RUN / "execution_provenance.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("No completed first-pass test analysis manifest; refusing to finalize a report")
    if not provenance_path.is_file():
        raise FileNotFoundError("Attempt04 execution provenance is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed_once" or manifest.get("test_opened_after_policy_lock") is not True:
        raise RuntimeError("Expected the single completed, policy-locked analysis pass")
    if REPORT.exists():
        raise FileExistsError(f"Refusing to overwrite report: {REPORT}")

    labels = read_csv("train_validation_test_label_distribution.csv")
    paired = read_csv("paired_expert_summary.csv")
    cost_rows = read_csv("policy_cost_sensitivity.csv")
    comparisons = read_csv("paired_policy_comparisons.csv")
    diagnostics = read_csv("router_predictive_diagnostics.csv")
    errors = read_csv("relative_value_prediction_errors.csv")
    features = read_csv("feature_distribution.csv")
    failures = parser_failure_diagnostics()
    label_by_split = {row["split"]: row for row in labels}
    paired_by_split = {row["split"]: row for row in paired}
    test_pair = paired_by_split["test"]
    test_labels = label_by_split["test"]

    report = [
        "# Qwen3-32B-AWQ 专家优势 Router 泛化结果（Attempt04）", "",
        "> 本结果用于判断是否值得进入后续研究；当前结论不启动 PPO/ATPO RL。",
        "",
        "## 实验与数据边界", "",
        "- 本轮模型推理已完成：1,700 个病例配对，按 case_id 固定划分为 train 1,000、validation 300、test 400。三组均来自 ATPO train split 的新病例 holdout；测试组不是 ATPO 官方 test split。",
        "- Base 为 `Qwen/Qwen3-8B` revision `b968826d9c46dd6066d109eabc6255188de91218`；DEFER Expert 为 `Qwen/Qwen3-32B-AWQ` revision `0499c3ac83fdef8810b907a23894ba91e95eddd8`。temperature=0、top-p=1、禁用 thinking；提示和 parser 在病例评估前固定。",
        "- 训练标签使用配对结果 `Expert correct - Base correct`；输入只包含动作前公开特征、Base 选项概率/不确定性及 Base 预测选项。Gold 和 Expert 输出未进入特征。训练只读 train；模型/阈值只用 validation 选择；test 阈值冻结后才打开。",
        "- 预注册 DEFER 成本为 0、0.05、0.10、0.20、0.40、0.60；主成本为 0.60。净 Utility = 最终正确率 − 成本 × Expert 调用率，成本只扣一次。",
        "- Oracle Selective DEFER 使用事后配对正确性，仅表示理论上界，不用于训练或部署决策。",
        "- 输出目录中的 raw completion、轨迹和病例级表保存在 PFS，不推送到 GitHub；GitHub 只保留脱敏汇总、代码和哈希。",
        "",
        "## Expert 格式失败审计", "",
        "20 例 smoke 的 Base/Expert 解析失败均为 0。正式 1,700 例中，Base 有 3 条严格答案解析失败（均在 train）；Expert 有 66 条 DEFER parser 失败：train 38、validation 13、test 15。66 条均正常结束（finish_reason=stop），不是生成长度截断；Expert 使用了要求的 `Expert Final Answer:` 前缀，但前缀后输出以 N 开头的非 A-D 选项文字，因此无法通过固定的一选项 full-match parser。模型回复未做事后修复。",
        "",
        "这些回复在部署端意味着 DEFER 没有给出可评分答案，因此端到端正确性按 0 计；模型拟合使用的相对收益标签也按这一真实部署结果计算。下表同时给出 parser 覆盖率，以及仅在成功解析回复上的条件准确率，避免把 parser 失败误称为临床答错。",
        "",
        "| Split | N | Base 解析失败 | Expert 解析失败 | Expert parse coverage | Base accuracy（端到端） | Expert accuracy（端到端） | Expert accuracy（仅成功解析） |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("train", "validation", "test"):
        row = label_by_split[name]
        n = int(row["n"])
        invalid_base, invalid_expert = int(row["invalid_base"]), int(row["invalid_expert"])
        base_valid_n, expert_valid_n = n - invalid_base, n - invalid_expert
        base_cond = int(row["base_correct"]) / base_valid_n if base_valid_n else float("nan")
        expert_cond = int(row["expert_correct"]) / expert_valid_n if expert_valid_n else float("nan")
        report.append(
            f"| {name} | {n} | {invalid_base} | {invalid_expert} | {expert_valid_n/n:.3f} | {f(row['base_accuracy'])} | {f(row['expert_accuracy'])} | {expert_cond:.3f} ({row['expert_correct']}/{expert_valid_n}) |"
        )

    report += [
        "", "## 配对专家互补性", "",
        f"独立测试集 Base accuracy 为 {f(test_pair['base_accuracy'])} [{f(test_pair['base_accuracy_ci_low'])}, {f(test_pair['base_accuracy_ci_high'])}]；Expert 端到端 accuracy 为 {f(test_pair['expert_accuracy'])} [{f(test_pair['expert_accuracy_ci_low'])}, {f(test_pair['expert_accuracy_ci_high'])}]。配对差为 {f(test_pair['paired_accuracy_delta_expert_minus_base'])} [{f(test_pair['paired_delta_ci_low'])}, {f(test_pair['paired_delta_ci_high'])}]。",
        "",
        f"| Base / Expert | 正确 | 错误或无可解析答案 |", "|---|---:|---:|",
        f"| Base 正确 | {test_pair['base_right_expert_right']} | {test_pair['base_right_expert_wrong']} |",
        f"| Base 错误 | {test_pair['base_wrong_expert_right']} | {test_pair['base_wrong_expert_wrong']} |",
        "",
        f"在 206 个 Base 错误病例中，Expert 修正 72 个（{f(test_pair['expert_correction_rate_given_base_error'])}，病例级 95% CI [{f(test_pair['correction_ci_low'])}, {f(test_pair['correction_ci_high'])}]）；在 194 个 Base 正确病例中，Expert 退化 35 个（{f(test_pair['expert_degradation_rate_given_base_correct'])}，95% CI [{f(test_pair['degradation_ci_low'])}, {f(test_pair['degradation_ci_high'])}]）。测试组 385 条可解析 Expert 回复中答对 231 条，即条件准确率 0.600；全 pipeline 的 15 条格式失败仍计入端到端错误。",
        "",
        "## 预注册成本敏感性", "",
        "下表列出 Always Answer、Always Defer、Uncertainty、主 Logistic Relative Value Router 与 Oracle。完整所有模型/随机基线见 `outputs/.../analysis/policy_cost_sensitivity.csv`。",
        "",
        "| 成本 | 策略 | Accuracy | Utility | DEFER率 | 平均成本 | 平均总 Token |", "|---:|---|---:|---:|---:|---:|---:|",
    ]
    display = {"always_answer", "always_defer", "uncertainty_defer", "logistic_relative_value", "random_defer_budget_matched_realized", "oracle_selective_defer_upper_bound"}
    for row in cost_rows:
        if row["policy"] in display:
            report.append(f"| {float(row['cost']):.2f} | {row['policy']} | {f(row['accuracy'])} | {f(row['utility'])} | {f(row['defer_call_rate'])} | {f(row['mean_cost'])} | {f(row['mean_total_tokens'], 1)} |")

    report += [
        "", "### 主 Logistic Router 相对简单策略的病例配对 Utility（95% CI）", "",
        "| DEFER 成本 | 比较对象 | Utility 差 | 病例级 95% CI |", "|---:|---|---:|---:|",
    ]
    for row in comparisons:
        if row["policy"] == "logistic_relative_value" and row["comparator"] in {"always_answer", "uncertainty_defer", "random_defer_budget_matched_realized"}:
            report.append(f"| {float(row['cost']):.2f} | {row['comparator']} | {f(row['paired_utility_difference'])} | [{f(row['paired_ci_low'])}, {f(row['paired_ci_high'])}] |")

    report += [
        "", "### Simple Uncertainty Threshold 在中低成本下", "",
        "阈值只在 validation 选择。test 上成本 0.10 时，该策略 DEFER 5.25%，Accuracy 0.510、Utility 0.505；相对 Always Answer 的配对 Utility 差为 +0.0198（95% CI [+0.0070, +0.0348]）。成本 0.20 时 DEFER 3.5%，Accuracy 0.505、Utility 0.498；相对 Always Answer 为 +0.0130（95% CI [+0.0025, +0.0250]）。这是预注册成本点上的简单不确定性策略结果，需在新病例队列复现；它不能替代主成本 0.60 下的结论，也不能说明复杂 Learned Router 已学会优势。",
        "",
        "| 成本 | 不确定性阈值 vs budget-matched Random（Utility 差） | 病例级 95% CI |",
        "|---:|---:|---:|",
    ]
    for row in comparisons:
        if row["policy"] == "uncertainty_defer" and row["comparator"] == "random_defer_budget_matched_realized" and row["cost"] in {"0.1", "0.2"}:
            report.append(f"| {float(row['cost']):.2f} | {f(row['paired_utility_difference'])} | [{f(row['paired_ci_low'])}, {f(row['paired_ci_high'])}] |")

    report += [
        "", "## Router 识别专家优势的能力", "",
        "Test AUROC 用来衡量动作前分数能否排序出正向相对收益（Expert correct、Base wrong）；病例级 bootstrap CI 已在一次分析中计算。",
        "", "| 预决策分数 | AUROC | 95% CI |", "|---|---:|---:|",
    ]
    for row in diagnostics:
        if row["target"] == "positive_relative_advantage":
            report.append(f"| {row['model_or_feature']} | {f(row['test_auroc'])} | [{f(row['auc_ci_low'])}, {f(row['auc_ci_high'])}] |")

    report += [
        "", "## 主成本 0.60 的配对差异", "", "| Logistic Relative Value 与 | Utility 差 | 病例级 95% CI |", "|---|---:|---:|",
    ]
    for row in comparisons:
        if row["cost"] == "0.6" and row["policy"] == "logistic_relative_value" and row["comparator"] in {"always_answer", "uncertainty_defer", "random_defer_budget_matched_realized"}:
            report.append(f"| {row['comparator']} | {f(row['paired_utility_difference'])} | [{f(row['paired_ci_low'])}, {f(row['paired_ci_high'])}] |")

    report += [
        "", "## 价值预测误差与过拟合", "", "| Split | Router | MAE | MSE | Pearson r |", "|---|---|---:|---:|---:|",
    ]
    for row in errors:
        report.append(f"| {row['split']} | {row['model']} | {f(row['mae'])} | {f(row['mse'])} | {f(row['pearson'])} |")
    zero_mae = {
        name: (int(row["delta_minus_1"]) + int(row["delta_plus_1"])) / int(row["n"])
        for name, row in label_by_split.items()
    }
    option_count_rows = [row for row in features if row["feature"] in {"option_count", "mean_option_chars"}]
    max_smd = max(abs(float(row["test_vs_train_standardized_mean_difference"])) for row in features)
    report += [
        "", "## Router 泛化失败诊断", "",
        f"- **没有明显的整体特征漂移。** 决策特征 train/test 的最大标准化均值差为 {max_smd:.3f}；各 public-state 和 Base uncertainty 特征分布接近。",
        f"- **两个选项结构特征全为常数。** `option_count` 和 `mean_option_chars` 在 train/validation/test 全部为 0（{', '.join(row['feature'] for row in option_count_rows)}）；当前独立行格式提取器没有识别题目选项，导致这两项不提供任何区分信息。",
        f"- **相对收益标签由中性类主导。** Δ=-1/0/+1：train 105/740/155，validation 25/223/52，test 35/293/72。Δ=0 约占 73–74%，正优势占 15.5–18%；纠错事件分别为 155/522、52/167、72/206 个 Base 错误病例。",
        f"- **价值回归接近无信号。** Δ 的常数 0 预测基线 MAE train/validation/test 为 {zero_mae['train']:.3f}/{zero_mae['validation']:.3f}/{zero_mae['test']:.3f}；Router MAE 为约 0.287–0.323，test Pearson r 仅 0.007–0.046。训练的参数化价值模型并未优于简单零预测的 MAE。",
        "- **Base uncertainty 单独具有排序信号，但 Learned Router 没有有效利用。** test 上 uncertainty AUROC 为 0.710（正相对收益 95% CI [0.645, 0.771]）；Logistic/Ridge 约 0.57–0.59，MLP/历史 Router 约 0.50–0.52。简单 uncertainty threshold 在成本 0.10/0.20 有正配对 Utility，而主成本 0.60 下 Logistic Relative Value 选择零调用。",
        "",
        "类别比例、完整训练/验证/测试特征分布、纠错标签 AUROC、其他阈值、成本与策略完整对照见分析 CSV。", "",
        "## 研究结论", "",
        "1. **Expert 存在平均且配对可重复的互补性。** 本 test holdout 上 Expert 端到端准确率高于 Base，配对差区间不跨 0；这回答的是平均协作收益，不代表每个病例都适合 DEFER。",
        "2. **Oracle 显示有选择性 DEFER 的理论空间，但当前 Learned Router 未在主成本上兑现。** 在成本 0.60 下，验证集选出的主 Router 调用率为 0，因此 test 与 Always Answer 完全相同；它相对于 Always Answer、Uncertainty 和 budget-matched Random 均没有可证明的正 Utility 增益。",
        "3. **简单不确定性阈值在成本 0.10/0.20 有 held-out 正 Utility。** 两个预注册点相对 Always Answer 的配对置信区间高于 0，说明简单 Base 不确定性可能是可用的预决策信号；但当前 Learned Relative Value Router 没有复现这个收益。",
        "4. **不启动 PPO/ATPO RL。** Expert 的条件互补性成立，简单不确定性策略在较低成本点有初步正收益；但在主成本 0.60 下 Learned Router 不触发 DEFER，且未优于 Always Answer。下一阶段应在训练数据侧修复选项特征提取，并用新病例复现 uncertainty policy；后续预注册还需对每个策略分别匹配调用预算的 Random baseline，再讨论 Actor-Critic RL。本轮 test 不再用于调参。",
        "",
        "## 一次性分析完整性与限制", "",
        f"原一次性分析已在 policy lock 后打开 test，并生成全部表格及 manifest（policy SHA256 `{manifest['policy_sha256']}`；test SHA256 `{manifest['test_sha256']}`）。首次程序在最终 Markdown 组装阶段遇到 `auc_by_model` 初始化顺序错误；该错误发生在 CSV 和 analysis manifest 已写出之后。已修复分析源码中的顺序问题。本报告从已生成汇总 CSV 组装；额外读取 raw trajectories 中的 `valid`、`parse_error`、`finish_reason`、`raw_response` 字段来汇总解析失败类型，但不使用测试正确性标签、不重新拟合、不重选阈值，也不重算 test accuracy/bootstrap。",
        "",
        "Random 策略的现有实现按 Ridge Router 的 validation 调用率匹配，并非每个候选 Router 各自匹配预算。因此本轮 Random 对照不作为不确定性策略的严格等预算结论；后续复现需要按每个策略预先确定的 validation 调用率分别生成随机对照。",
        "",
        f"解析失败分组 CSV：`outputs/qwen32_router_generalization_20261118_attempt04/analysis/parser_failure_diagnostics.csv`。执行 provenance SHA256 `{hashlib.sha256(provenance_path.read_bytes()).hexdigest()}`。病例级测试结果、raw 轨迹、服务日志和 GPU telemetry 留在 PFS 的 ignored `outputs/`。", "",
    ]
    REPORT.write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"status": "report_written_from_saved_analysis_artifacts", "report": str(REPORT), "test_rows": manifest["test_rows"], "test_recomputed": False}, indent=2))


if __name__ == "__main__":
    main()
