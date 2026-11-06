#!/usr/bin/env python3
"""Summarize paired actual-interaction evaluation and generate thesis tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from router_core import file_sha256, load_jsonl, write_json


HERE = Path(__file__).resolve().parent
POLICY_ORDER = (
    "always_answer",
    "uncertainty_threshold",
    "fixed_cost_router",
    "learned_value_no_cost",
    "learned_value_router",
    "learned_value_higher_cost",
    "ablation_no_expert_reliability",
    "ablation_no_consult",
    "ablation_no_defer",
)
LAMBDA = {"ASK": 0.01, "CONSULT": 0.02, "DEFER": 0.02}
BOOTSTRAPS = 10000


def case_bootstrap_ci(values: list[float], seed: int, repeats: int = BOOTSTRAPS) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    if len(values) == 1:
        return values[0], values[0]
    rng = np.random.default_rng(seed)
    values_array = np.asarray(values, dtype=np.float64)
    # Chunk to keep memory bounded for large runs.
    means = []
    for start in range(0, repeats, 250):
        n = min(250, repeats - start)
        picks = rng.integers(0, len(values_array), size=(n, len(values_array)))
        means.extend(values_array[picks].mean(axis=1).tolist())
    means.sort()
    return float(means[int(0.025 * repeats)]), float(means[min(repeats - 1, int(0.975 * repeats))])


def mean_or_blank(values: list[float]) -> float | str:
    return float(np.mean(values)) if values else ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs")
    args = parser.parse_args()
    episode_path = args.output_dir / "policy_episodes.jsonl"
    manifest_path = args.output_dir / "run_manifest.json"
    rows = load_jsonl(episode_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["case_id"]), {})[str(row["policy"])] = row
    cases = sorted(grouped)
    policies = [p for p in POLICY_ORDER if any(p in by_policy for by_policy in grouped.values())]
    if not cases or not policies:
        raise RuntimeError("no policy episode records found")
    for case in cases:
        missing = [p for p in policies if p not in grouped[case]]
        if missing:
            raise ValueError(f"case {case} missing policy rows: {missing}")

    summaries = []
    trajectory_by_policy: dict[str, dict[str, dict[str, Any]]] = {}
    for pi, policy in enumerate(policies):
        by_case = {case: grouped[case][policy] for case in cases}
        trajectory_by_policy[policy] = by_case
        correctness = [float(bool(by_case[c].get("correctness"))) for c in cases]
        accuracy = float(np.mean(correctness))
        low, high = case_bootstrap_ci(correctness, 20261106 + pi)
        actions = Counter(str(by_case[c]["action"]) for c in cases)
        tokens = [float(by_case[c].get("total_tokens", 0)) for c in cases]
        delays = [float(by_case[c].get("model_inference_latency_s", 0)) for c in cases]
        wall = [float(by_case[c].get("elapsed_wall_s", 0)) for c in cases]
        expert_tokens = [float(by_case[c].get("expert_tokens", 0)) for c in cases]
        expert_calls = [float(by_case[c].get("expert_calls", sum(1 for t in (by_case[c].get("trace") or []) if t.get("actor") == "expert"))) for c in cases]
        patient_turns = [float(by_case[c].get("patient_turns", 0)) for c in cases]
        consult_tokens = [float(by_case[c].get("consult_expert_tokens", 0)) / 1000.0 for c in cases]
        defer_tokens = [float(by_case[c].get("defer_expert_tokens", 0)) / 1000.0 for c in cases]
        realized_utility = [
            correctness[i]
            - LAMBDA["ASK"] * patient_turns[i]
            - LAMBDA["CONSULT"] * consult_tokens[i]
            - LAMBDA["DEFER"] * defer_tokens[i]
            for i in range(len(cases))
        ]
        summaries.append({
            "policy": policy,
            "n_cases": len(cases),
            "accuracy": accuracy,
            "accuracy_ci95_low": low,
            "accuracy_ci95_high": high,
            "valid_rate": float(np.mean([float(bool(by_case[c].get("valid"))) for c in cases])),
            "ANSWER_rate": actions["ANSWER"] / len(cases),
            "ASK_rate": actions["ASK"] / len(cases),
            "CONSULT_rate": actions["CONSULT"] / len(cases),
            "DEFER_rate": actions["DEFER"] / len(cases),
            "expert_workload_calls_per_case": float(np.mean(expert_calls)),
            "expert_workload_total_calls": int(sum(expert_calls)),
            "expert_tokens_per_case": float(np.mean(expert_tokens)),
            "total_tokens_per_case": float(np.mean(tokens)),
            "total_tokens_all_cases": int(sum(tokens)),
            "inference_latency_s_per_case": float(np.mean(delays)),
            "inference_latency_s_all_cases": float(sum(delays)),
            "wall_latency_s_per_case": float(np.mean(wall)),
            "wall_latency_s_all_cases": float(sum(wall)),
            "patient_simulator_turns_per_case": float(np.mean(patient_turns)),
            "consult_expert_tokens_per_case": float(np.mean(consult_tokens) * 1000.0),
            "defer_expert_tokens_per_case": float(np.mean(defer_tokens) * 1000.0),
            "realized_utility_reference_cost": float(np.mean(realized_utility)),
            "action_counts": json.dumps({a: int(actions[a]) for a in ("ANSWER", "ASK", "CONSULT", "DEFER")}, sort_keys=True),
        })

    by_policy_summary = {r["policy"]: r for r in summaries}
    paired = []
    references = ["always_answer"]
    if "learned_value_router" in policies:
        references.append("learned_value_router")
    for ri, reference in enumerate(references):
        for pi, policy in enumerate(policies):
            if policy == reference:
                continue
            per_case_delta = [
                float(bool(trajectory_by_policy[policy][c].get("correctness")))
                - float(bool(trajectory_by_policy[reference][c].get("correctness")))
                for c in cases
            ]
            low, high = case_bootstrap_ci(per_case_delta, 20261200 + ri * 100 + pi)
            ref_correct_cases = [c for c in cases if bool(trajectory_by_policy[reference][c].get("correctness"))]
            ref_wrong_cases = [c for c in cases if not bool(trajectory_by_policy[reference][c].get("correctness"))]
            paired.append({
                "policy": policy,
                "reference": reference,
                "paired_accuracy_delta": float(np.mean(per_case_delta)),
                "ci95_low": low,
                "ci95_high": high,
                "n_cases": len(cases),
                "corrected_from_reference_errors": sum(bool(trajectory_by_policy[policy][c].get("correctness")) for c in ref_wrong_cases),
                "reference_error_denominator": len(ref_wrong_cases),
                "degraded_from_reference_correct": sum(not bool(trajectory_by_policy[policy][c].get("correctness")) for c in ref_correct_cases),
                "reference_correct_denominator": len(ref_correct_cases),
            })

    summary_csv = args.output_dir / "router_policy_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader(); writer.writerows(summaries)
    paired_csv = args.output_dir / "paired_baseline_comparisons.csv"
    with paired_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired[0]))
        writer.writeheader(); writer.writerows(paired)

    # Accuracy-cost curve uses observed system tokens and the reference policy
    # utility separately; costs are not subtracted a second time.
    curve = []
    for row in summaries:
        curve.append({
            "policy": row["policy"],
            "accuracy": row["accuracy"],
            "accuracy_ci95_low": row["accuracy_ci95_low"],
            "accuracy_ci95_high": row["accuracy_ci95_high"],
            "mean_total_tokens_per_case": row["total_tokens_per_case"],
            "mean_expert_tokens_per_case": row["expert_tokens_per_case"],
            "mean_expert_calls_per_case": row["expert_workload_calls_per_case"],
            "mean_inference_latency_s_per_case": row["inference_latency_s_per_case"],
            "realized_utility_reference_cost": row["realized_utility_reference_cost"],
        })
    curve_csv = args.output_dir / "accuracy_cost_curve.csv"
    with curve_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(curve[0]))
        writer.writeheader(); writer.writerows(curve)

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        for row in curve:
            ax.errorbar(row["mean_total_tokens_per_case"], row["accuracy"],
                        yerr=[[row["accuracy"] - row["accuracy_ci95_low"]], [row["accuracy_ci95_high"] - row["accuracy"]]],
                        marker="o", capsize=3, label=row["policy"])
            ax.annotate(row["policy"].replace("_", " "), (row["mean_total_tokens_per_case"], row["accuracy"]), xytext=(4, 3), textcoords="offset points", fontsize=7)
        ax.set_xlabel("Observed total tokens per case")
        ax.set_ylabel("Actual rollout accuracy")
        ax.set_title("Router policy accuracy–token cost")
        ax.grid(alpha=.25)
        fig.tight_layout()
        fig.savefig(args.output_dir / "accuracy_cost_curve.png", dpi=180)
        plt.close(fig)
    except Exception as exc:
        write_json(args.output_dir / "plot_error.json", {"error": f"{type(exc).__name__}: {exc}"})

    report_lines = [
        "# EA-ATPO 四动作价值 Router 实验报告",
        "",
        f"生成时间：{datetime.now(timezone.utc).isoformat(timespec='seconds')}。",
        "",
        "## 实验边界",
        "",
        f"在与强专家直接比较样本不重叠的 {len(cases)} 个官方 ATPO test 病例上运行了模型实际交互。Pilot 的 100 个病例按 `case_id` 划分训练/验证/内部测试，固定 Router 后才抽取外部 test。Base、Patient Simulator 和 Expert Simulator 都是 Qwen3-8B；Expert 是语言模型模拟器，不是真实医生。",
        "Router 决策只读取初始公开题干和公开对话文本。Gold 仅在 rollout 结束后评分；hidden atomic facts 仅进入 Patient Simulator；Router 不读取状态生成轨迹、事后正确性、未来患者回复或专家建议。",
        f"每个策略对每例都使用相同的公开初始状态，并运行其实际选择的动作。生成温度为 {manifest.get('generation', {}).get('temperature', '见运行清单')}；请求种子由模型和公开请求内容派生。同一病例/动作 rollout 只运行一次，并由选择该动作的策略共享，因此配对比较使用完全相同的输入。",
        "协议范围：本诊断只由各策略选择第一个动作；ASK/CONSULT 之后由 Base Agent continuation policy 选择后续动作。因此它不是 Router 逐轮参与的端到端策略结果。逐轮主评估见 `outputs/stepwise_test/STEPWISE_ROUTER_EXPERIMENT_REPORT.md`。",
        "",
        "## 实际交互策略结果",
        "",
        "| Policy | Accuracy (95% case bootstrap CI) | ASK | CONSULT | DEFER | Expert calls/case | Tokens/case | Inference s/case |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summaries:
        report_lines.append(
            f"| {r['policy']} | {r['accuracy']:.3f} ({r['accuracy_ci95_low']:.3f}–{r['accuracy_ci95_high']:.3f}) | "
            f"{r['ASK_rate']:.1%} | {r['CONSULT_rate']:.1%} | {r['DEFER_rate']:.1%} | "
            f"{r['expert_workload_calls_per_case']:.2f} | {r['total_tokens_per_case']:.1f} | {r['inference_latency_s_per_case']:.3f} |"
        )
    report_lines += [
        "",
        "## 与 Always Answer 的配对差异",
        "",
        "| Policy | Accuracy difference | 95% paired case bootstrap CI | 修正直答错误 | 退化直答正确 |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in paired:
        if r["reference"] != "always_answer":
            continue
        report_lines.append(
            f"| {r['policy']} | {r['paired_accuracy_delta']:+.3f} | {r['ci95_low']:+.3f}–{r['ci95_high']:+.3f} | "
            f"{r['corrected_from_reference_errors']}/{r['reference_error_denominator']} | {r['degraded_from_reference_correct']}/{r['reference_correct_denominator']} |"
        )
    if "learned_value_router" in policies:
        report_lines += [
            "",
            "## 与 Learned Value Router 的配对差异",
            "",
            "| Policy | Accuracy difference | 95% paired case bootstrap CI | 修正 Router 错误 | 退化 Router 正确 |",
            "|---|---:|---:|---:|---:|",
        ]
        for r in paired:
            if r["reference"] != "learned_value_router":
                continue
            report_lines.append(
                f"| {r['policy']} | {r['paired_accuracy_delta']:+.3f} | {r['ci95_low']:+.3f}–{r['ci95_high']:+.3f} | "
                f"{r['corrected_from_reference_errors']}/{r['reference_error_denominator']} | {r['degraded_from_reference_correct']}/{r['reference_correct_denominator']} |"
            )
    report_lines += [
        "",
        "## 方法与成本定义",
        "",
        "Outcome target 是每个公开状态/动作三个 Pilot rollout 的平均正确率；无效输出按 0 保留。价值模型为每个候选动作分别预测正确率、完整分支 patient simulator turns、Expert CONSULT tokens/1000、Expert DEFER tokens/1000。路由效用严格为 `Q(a,s) − λASK·patient_turns − λCONSULT·consult_tokens/1000 − λDEFER·defer_tokens/1000`，三个成本分量均从完整 rollout trace 统计，每项只扣一次。报告另列实际完整链路 token 与请求延迟。",
        "固定成本 Router 在全部 Pilot 训练数据上选出总体平均效用最高的单一动作。Uncertainty Threshold 仅用验证集选择阈值：由预测直答正确率得到不确定性，并在低信心时调用训练集上表现最好的固定非 ANSWER 动作。Learned Value Router 使用文本 TF-IDF 线性核岭回归预测每个动作的 Q 和成本；Expert reliability 特征是按病例交叉拟合的 Pilot DEFER 正确概率。",
        "Accuracy–Cost 曲线使用实际系统总 token 作为横轴，并同时输出专家 token、专家调用量和延迟。λ 是预注册的敏感性单位，不代表美元、医生劳动价格或临床效用测量。无成本、参考成本和高成本组都在同一监督目标/预测成本定义上只更换固定 λ。",
        "",
        "## 消融",
        "",
        "- `ablation_no_expert_reliability`：重新训练时去掉交叉拟合的专家可靠性输入特征。",
        "- `uncertainty_threshold`：不做四动作逐状态价值选择，改用验证集冻结的简单置信度阈值路由。",
        "- `ablation_no_consult`：动作菜单仅 ANSWER/ASK/DEFER。",
        "- `ablation_no_defer`：动作菜单仅 ANSWER/ASK/CONSULT。",
        "",
        "## 解释边界",
        "",
        "Router 监督数据来自 100 个病例、291 个状态、每状态每动作 3 次的历史 Qwen3-8B 模拟交互；状态级样本仍较少，外部病例准确率和成本差异需结合宽置信区间理解。专家可靠性特征预测的是同一 8B 模拟器在可见文本条件下的历史 DEFER 正确率，并非真实医生能力，也不是本次 CONSULT 建议质量的直接判别。",
        f"Router test 的多策略配对差异只包含选定分支的一次真实 rollout，不是四动作 oracle。由于相同 `(case,action)` 轨迹被多个策略共享，策略差异来自冻结决策选择；服务生成温度为 {manifest.get('generation', {}).get('temperature', '见运行清单')}。模型服务日志显示 vLLM prefix cache hit rate 非零，实测延迟包含该缓存带来的服务端加速，token 仍按完整 API prompt/completion usage 记录；部署服务缓存配置或负载不同会改变延迟。",
        "目前未启动 P5/P8 联合 ATPO RL。Qwen3-32B AWQ 权重恢复状态见 `outputs/STRONG_EXPERT_RECOVERY_REPORT.md`；在权重验证且专家评估跑完以前，没有 32B 准确率结果。本诊断只对应 8B 模拟 Expert。",
        "",
        "## 产物",
        "",
        "- `policy_episodes.jsonl`：每病例/策略的实际分支、动作、答案、Gold 评分、成本、完整轨迹。",
        "- `action_rollouts.jsonl`：去重后的病例/动作真实 rollout，支持断点续跑。",
        "- `router_policy_summary.csv`、`paired_baseline_comparisons.csv`、`accuracy_cost_curve.csv`。",
        "- `run_manifest.json`：样本行、数据哈希、模型 revision、配置和运行状态。",
    ]
    report_path = args.output_dir / "ROUTER_EXPERIMENT_REPORT.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    write_json(args.output_dir / "summary_manifest.json", {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "case_count": len(cases), "policies": policies,
        "policy_episode_sha256": file_sha256(episode_path),
        "summary_csv_sha256": file_sha256(summary_csv),
        "paired_csv_sha256": file_sha256(paired_csv),
        "accuracy_cost_csv_sha256": file_sha256(curve_csv),
    })
    print(f"wrote report for {len(cases)} cases and {len(policies)} policies")


if __name__ == "__main__":
    main()
