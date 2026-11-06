#!/usr/bin/env python3
"""Summarize case-paired, actual stepwise Router policy episodes."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from router_core import file_sha256

HERE = Path(__file__).resolve().parent
POLICY_ORDER = (
    "always_answer", "uncertainty_threshold", "fixed_cost_router",
    "learned_value_no_cost", "learned_value_router", "learned_value_higher_cost",
    "ablation_no_expert_reliability", "ablation_no_consult", "ablation_no_defer",
)
BOOTSTRAPS = 10000


def ci(values, seed):
    if len(values) == 1:
        return values[0], values[0]
    data = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = []
    for start in range(0, BOOTSTRAPS, 250):
        n = min(250, BOOTSTRAPS - start)
        means.extend(data[rng.integers(0, len(data), size=(n, len(data)))].mean(axis=1).tolist())
    means.sort()
    return float(means[int(0.025 * BOOTSTRAPS)]), float(means[min(BOOTSTRAPS - 1, int(0.975 * BOOTSTRAPS))])


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs/stepwise_test")
    args = parser.parse_args()
    episode_path = args.output_dir / "policy_episodes.jsonl"
    manifest_path = args.output_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise RuntimeError(f"run manifest status is {manifest.get('status')}, expected completed")
    rows = [json.loads(line) for line in episode_path.read_text(encoding="utf-8").splitlines() if line]
    grouped = {}
    for row in rows:
        grouped.setdefault(str(row["case_id"]), {})[str(row["policy"])] = row
    cases = sorted(grouped)
    policies = [p for p in POLICY_ORDER if any(p in x for x in grouped.values())]
    if not cases:
        raise RuntimeError("no stepwise policy episodes")
    for case in cases:
        missing = [p for p in policies if p not in grouped[case]]
        if missing:
            raise ValueError(f"case {case} is missing policy rows: {missing}")

    summaries = []
    by_policy = {}
    for pi, policy in enumerate(policies):
        selected = {case: grouped[case][policy] for case in cases}
        by_policy[policy] = selected
        correctness = [float(bool(selected[c]["correctness"])) for c in cases]
        accuracy_ci = ci(correctness, 20261120 + pi)
        n_decisions = sum(len(selected[c].get("decision_actions") or []) for c in cases)
        action_counts = Counter(a for c in cases for a in (selected[c].get("decision_actions") or []))
        patient_turns = [float(selected[c].get("patient_turns", 0)) for c in cases]
        consult_tokens = [float(selected[c].get("consult_expert_tokens", 0)) for c in cases]
        defer_tokens = [float(selected[c].get("defer_expert_tokens", 0)) for c in cases]
        utility = [correctness[i] - 0.01 * patient_turns[i] - 0.02 * consult_tokens[i] / 1000.0 - 0.02 * defer_tokens[i] / 1000.0 for i in range(len(cases))]
        total_tokens = [float(selected[c].get("total_prompt_tokens", 0) + selected[c].get("total_completion_tokens", 0)) for c in cases]
        inference = [float(selected[c].get("model_inference_latency_s", 0)) for c in cases]
        wall = [float(selected[c].get("elapsed_wall_s", 0)) for c in cases]
        expert_calls = [float(selected[c].get("expert_calls", 0)) for c in cases]
        summary = {
            "policy": policy,
            "n_cases": len(cases),
            "accuracy": float(np.mean(correctness)),
            "accuracy_ci95_low": accuracy_ci[0],
            "accuracy_ci95_high": accuracy_ci[1],
            "valid_rate": float(np.mean([float(bool(selected[c].get("valid"))) for c in cases])),
            "ASK_episode_rate": float(np.mean([selected[c].get("ASK_count", 0) > 0 for c in cases])),
            "CONSULT_episode_rate": float(np.mean([selected[c].get("CONSULT_count", 0) > 0 for c in cases])),
            "DEFER_episode_rate": float(np.mean([selected[c].get("DEFER_count", 0) > 0 for c in cases])),
            "ASK_decision_rate": action_counts["ASK"] / max(1, n_decisions),
            "CONSULT_decision_rate": action_counts["CONSULT"] / max(1, n_decisions),
            "DEFER_decision_rate": action_counts["DEFER"] / max(1, n_decisions),
            "decision_count_per_case": n_decisions / len(cases),
            "expert_calls_per_case": float(np.mean(expert_calls)),
            "total_expert_calls": int(sum(expert_calls)),
            "patient_turns_per_case": float(np.mean(patient_turns)),
            "consult_expert_tokens_per_case": float(np.mean(consult_tokens)),
            "defer_expert_tokens_per_case": float(np.mean(defer_tokens)),
            "total_tokens_per_case": float(np.mean(total_tokens)),
            "total_tokens_all_cases": int(sum(total_tokens)),
            "inference_latency_s_per_case": float(np.mean(inference)),
            "inference_latency_s_all_cases": float(sum(inference)),
            "wall_latency_s_per_case": float(np.mean(wall)),
            "wall_latency_s_all_cases": float(sum(wall)),
            "realized_utility_reference_cost": float(np.mean(utility)),
            "action_decision_counts": json.dumps({a: int(action_counts[a]) for a in ("ANSWER", "ASK", "CONSULT", "DEFER")}, sort_keys=True),
        }
        summaries.append(summary)

    paired = []
    references = [p for p in ("always_answer", "uncertainty_threshold", "fixed_cost_router", "learned_value_router") if p in policies]
    for ri, reference in enumerate(references):
        for pi, policy in enumerate(policies):
            if policy == reference:
                continue
            delta = [float(bool(by_policy[policy][c]["correctness"])) - float(bool(by_policy[reference][c]["correctness"])) for c in cases]
            low, high = ci(delta, 20261220 + ri * 100 + pi)
            wrong = [c for c in cases if not bool(by_policy[reference][c]["correctness"])]
            right = [c for c in cases if bool(by_policy[reference][c]["correctness"])]
            paired.append({
                "policy": policy,
                "reference": reference,
                "paired_accuracy_delta": float(np.mean(delta)),
                "ci95_low": low,
                "ci95_high": high,
                "n_cases": len(cases),
                "corrected_reference_errors": sum(bool(by_policy[policy][c]["correctness"]) for c in wrong),
                "reference_error_count": len(wrong),
                "degraded_reference_correct": sum(not bool(by_policy[policy][c]["correctness"]) for c in right),
                "reference_correct_count": len(right),
            })

    summary_path = args.output_dir / "router_policy_summary.csv"
    paired_path = args.output_dir / "paired_baseline_comparisons.csv"
    curve_path = args.output_dir / "accuracy_cost_curve.csv"
    write_csv(summary_path, summaries)
    write_csv(paired_path, paired)
    curve = [{"policy": r["policy"], "accuracy": r["accuracy"],
              "accuracy_ci95_low": r["accuracy_ci95_low"], "accuracy_ci95_high": r["accuracy_ci95_high"],
              "total_tokens_per_case": r["total_tokens_per_case"], "expert_calls_per_case": r["expert_calls_per_case"],
              "inference_latency_s_per_case": r["inference_latency_s_per_case"],
              "realized_utility_reference_cost": r["realized_utility_reference_cost"]} for r in summaries]
    write_csv(curve_path, curve)
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        for r in curve:
            ax.errorbar(r["total_tokens_per_case"], r["accuracy"],
                        yerr=[[r["accuracy"] - r["accuracy_ci95_low"]], [r["accuracy_ci95_high"] - r["accuracy"]]],
                        marker="o", capsize=3, label=r["policy"])
            ax.annotate(r["policy"].replace("_", " "), (r["total_tokens_per_case"], r["accuracy"]),
                        xytext=(4, 3), textcoords="offset points", fontsize=7)
        ax.set_xlabel("Observed total tokens per case")
        ax.set_ylabel("Actual stepwise rollout accuracy")
        ax.set_title("Stepwise Router accuracy–token cost")
        ax.grid(alpha=.25)
        fig.tight_layout()
        fig.savefig(args.output_dir / "accuracy_cost_curve.png", dpi=180)
        plt.close(fig)
    except Exception as exc:
        (args.output_dir / "plot_error.json").write_text(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, indent=2) + "\n")

    lines = [
        "# EA-ATPO 逐轮四动作 Router 实测报告", "",
        f"生成时间：{datetime.now(timezone.utc).isoformat(timespec='seconds')}。", "",
        "## 评估设计", "",
        f"在 {len(cases)} 个与强专家直接评估集不重叠的 ATPO 官方 test 病例上，对每种策略独立运行完整交互。每次公开状态更新后，Router 重新预测四个动作的 Q 和成本并作下一步选择；不把 continuation action 交给 Base Agent 自行决定。Gold 仅在 episode 结束后评分，hidden atomic_facts 仅交给 Patient Simulator，Expert 仅见公开状态及 CONSULT 问题。", 
        f"Base/Expert temperature={manifest['generation']['base_temperature']}，Patient temperature={manifest['generation']['patient_temperature']}；最多 {manifest['generation']['max_decisions']} 次动作，最多 1 次 CONSULT。每次生成请求的 seed 由模型、可见请求内容和 temperature 派生。所有 policy 使用相同 case_id 与初始公开状态。", "",
        "## 实际逐轮策略结果", "",
        "| Policy | Accuracy (95% case bootstrap CI) | Valid | Episode ASK/CONSULT/DEFER | Expert calls/case | Decisions/case | Tokens/case | Inference s/case |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summaries:
        lines.append(f"| {r['policy']} | {r['accuracy']:.3f} ({r['accuracy_ci95_low']:.3f}–{r['accuracy_ci95_high']:.3f}) | {r['valid_rate']:.1%} | {r['ASK_episode_rate']:.1%}/{r['CONSULT_episode_rate']:.1%}/{r['DEFER_episode_rate']:.1%} | {r['expert_calls_per_case']:.2f} | {r['decision_count_per_case']:.2f} | {r['total_tokens_per_case']:.1f} | {r['inference_latency_s_per_case']:.3f} |")
    for reference, title in (("always_answer", "Always Answer"), ("uncertainty_threshold", "Uncertainty Threshold"), ("fixed_cost_router", "Fixed-cost Router"), ("learned_value_router", "Learned Value Router")):
        if reference not in references:
            continue
        lines += ["", f"## 相对 {title} 的病例级配对差异", "",
                  "| Policy | Accuracy delta | 95% paired bootstrap CI | 修正参考策略错误 | 退化参考策略正确 |",
                  "|---|---:|---:|---:|---:|"]
        for r in paired:
            if r["reference"] == reference:
                lines.append(f"| {r['policy']} | {r['paired_accuracy_delta']:+.3f} | {r['ci95_low']:+.3f}–{r['ci95_high']:+.3f} | {r['corrected_reference_errors']}/{r['reference_error_count']} | {r['degraded_reference_correct']}/{r['reference_correct_count']} |")
    lines += ["", "## 方法、成本与消融", "",
              "Router utility 为 `Q − 0.01×患者模拟器轮数 − 0.02×CONSULT 专家 tokens/1000 − 0.02×DEFER 专家 tokens/1000`。分支完整 trace 记录成本，Q 与三项成本单独预测，每一项只扣一次。真实 Accuracy–Cost 横轴使用全链路 API usage 总 tokens；报告同时列出专家调用、完整调用延迟及 reference-cost utility。", 
              "固定成本策略在训练病例上选一个总体平均效用最高动作；阈值策略按 validation 冻结置信度阈值；无专家可靠性消融重新拟合对应模型；无 CONSULT/DEFER 消融从所有逐轮决策的候选动作中移除该动作。", "",
              "## 解释限制", "",
              "监督 Pilot 只有 100 个病例、291 个状态；这不足以支持高维深度价值模型。Q targets 来自 Pilot 中强制首动作后由 Base Agent continuation 的反事实 rollout；逐轮测试将该一步 Q 模型反复用于 Router policy，存在 continuation-policy 分布偏移，需在更大规模 on-policy Router rollout 中再训练验证。Uncertainty Threshold 的阈值只在 validation 上选择，随后用全 Pilot 开发数据重拟合的最终模型执行；因此其置信度尺度可能有校准偏移，但外部 test 未参与阈值或模型拟合。1800 条 episode 中 3 条 CONSULT advice 因输出最终选项而被协议拒绝；这些 rollout 保留、准确率记 0，成功生成的请求 token 和延迟仍纳入统计。Expert Simulator 与 Base/Patient Simulator 均为 Qwen3-8B，Expert 不是医生。病例 bootstrap 是病例级配对区间，未把一次 rollout 的模型随机性作为独立病例重复；按公开请求 hash seed 固定生成结果。vLLM 服务日志显示 prefix caching 生效，token 统计仍为完整 API prompt/completion usage，延迟反映当前缓存及 GPU 负载。",
              "旧的 `outputs/ROUTER_EXPERIMENT_REPORT.md` 是仅对起始状态选一次动作的诊断结果，ASK/CONSULT 后续动作由 Base Agent 续答；不能解读成逐轮 Router。此报告及本目录 `policy_episodes.jsonl` 才是逐轮 Router 主评估。", "",
              "## 产物", "",
              "- `policy_episodes.jsonl`：每病例/策略的逐轮路由动作、完整轨迹、答案和评分。",
              "- `router_policy_summary.csv`、`paired_baseline_comparisons.csv`、`accuracy_cost_curve.csv/png`。",
              "- `run_manifest.json`：病例、数据/checkpoint 哈希、生成参数和模型 revision。"]
    report_path = args.output_dir / "STEPWISE_ROUTER_EXPERIMENT_REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json = {"case_count": len(cases), "policy_episode_sha256": file_sha256(episode_path),
                  "summary_sha256": file_sha256(summary_path), "paired_sha256": file_sha256(paired_path),
                  "accuracy_cost_sha256": file_sha256(curve_path),
                  "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (args.output_dir / "summary_manifest.json").write_text(json.dumps(write_json, indent=2) + "\n", encoding="utf-8")
    print(f"wrote stepwise report for {len(cases)} cases and {len(policies)} policies")


if __name__ == "__main__":
    main()
