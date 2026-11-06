# EA-ATPO 核心方法阶段结论

## 当前结论

逐轮价值 Router、训练脚本、checkpoint、冻结配置、独立 200 病例实际交互评估和必要动作消融均已完成。当前实测没有证据证明四动作 Learned Value Router 优于简单策略。

| 方法 | Accuracy | 病例级 95% CI | Tokens/病例 | Expert calls/病例 | Reference-cost utility |
|---|---:|---:|---:|---:|---:|
| Always Answer | 48.5% | 41.5%–55.5% | 514 | 0.00 | 0.485 |
| Uncertainty Threshold | 47.0% | 40.0%–54.0% | 1,141 | 0.89 | 0.458 |
| Fixed-cost Router（DEFER） | 45.5% | 38.5%–52.5% | 622 | 1.00 | 0.443 |
| Learned Value Router | 45.5% | 38.5%–52.5% | 2,091 | 0.88 | 0.432 |
| Learned Value Router，高成本 | 47.0% | 40.0%–54.0% | 745 | 0.38 | 0.464 |

Learned Value Router 相对 Always Answer 的配对准确率差为 −3.0 个百分点，95% 病例 bootstrap CI 为 −7.0 至 +1.0 个百分点。区间跨 0，当前不能声称其提高了准确率。Reference-cost utility 也低于 Always Answer。

## 消融解读

- 去掉专家可靠性特征后，准确率从 45.5% 降至 43.5%；配对差 −2.0 个百分点，95% CI 为 −4.0 至 −0.5 个百分点。这个样本上显示该特征可能有帮助，但训练仅有 100 个病例，而且多项比较没有做多重检验校正，结论仍属初步。
- 去掉 CONSULT 后准确率为 46.5%，相对完整 Router +1.0 个百分点，配对 CI −2.5 至 +4.5 个百分点；去掉 DEFER 后同为 46.5%，配对 CI −1.5 至 +3.5 个百分点。没有证据表明 CONSULT 或 DEFER 在当前系统中提供独立准确率收益。
- 高成本组把调用压到 745 tokens/病例，准确率 47.0%，但仍未超过 Always Answer。收益主要是减少交互开销，而不是提高正确率。
- 1,800 条策略 episode 中 3 条 CONSULT 建议因生成最终选项而被协议拒绝，均保留为无效结果并计为错误；使用成功生成的 token 和延迟仍纳入成本。
- 阈值只在 validation 上选取，之后用全 Pilot 开发集拟合的最终模型执行；confidence calibration 可能有偏移。外部 test 未参与模型或阈值拟合。

## 对论文工作的建议

暂不启动 P5 Router RL 或 P8 联合 ATPO RL。现有 100 个训练病例、Qwen3-8B Pilot Expert 和一步反事实目标不足以支持稳定的在线价值学习；逐轮应用一步 Q 值还存在 continuation-policy 分布偏移。真实逐轮评估没有证明 Learned Router 优于 Always Answer，论文应如实报告负面结果和这些限制。

## 强 Expert 条件可靠性

Qwen3-32B-AWQ 已通过 vLLM 0.33.0 加载，并在冻结的 200 个独立病例上与 Qwen3-8B Base 配对评估。32B accuracy 为 48.0%（95% CI 41.0%–55.0%），8B 为 44.0%（37.0%–51.0%）；配对差 +4.0 个百分点（−2.5 至 +10.5），区间包含 0。32B 在 Base 错误的 112 例中答对 26 例，条件准确率/修正率 23.2%（95% CI 15.7%–31.4%）；在 Base 正确的 88 例中有 18 例退化为错，退化率 20.5%（12.4%–29.3%）。32B 请求平均延迟为 0.143 秒/例，8B 为 0.065 秒/例；两者平均 170.32 tokens/例。

这项直接回答评估只验证模型的条件正确率，不能证明 32B 在 CONSULT 建议生成或临床建议质量上可靠。按这 200 例结果，32B 是候选 LLM Expert，但没有证据表明它显著优于 8B，也不能称作真实医生。建议下一步先用该强模型重新收集更大规模的 CONSULT 交互和 Advice Utilization 数据，再决定是否更新 Router 目标与可靠性特征；不应把参数规模本身当成专家可靠性。

统计审计发现旧汇总的条件比例只限定了分母、没有把分子限制在同一条件组，导致条件值超过 1。已从原始 200 对 JSONL 离线重算病例级 bootstrap 与四格表；原始推理轨迹没有修改，正确结果见 `outputs/strong_expert_awq/STRONG_EXPERT_EVALUATION.md` 和 `strong_expert_summary.csv`。

## 可复现主产物

- 逐轮实测报告：`outputs/stepwise_test_v2/STEPWISE_ROUTER_EXPERIMENT_REPORT.md`
- 逐轮原始 JSONL：`outputs/stepwise_test_v2/policy_episodes.jsonl`
- 策略汇总与配对 CI：`outputs/stepwise_test_v2/router_policy_summary.csv`、`paired_baseline_comparisons.csv`
- 成本曲线：`outputs/stepwise_test_v2/accuracy_cost_curve.csv` 和 `.png`
- Router checkpoint：`artifacts/checkpoint/`；去可靠性特征消融：`artifacts/checkpoint_no_reliability/`
- 训练与评估入口：`run_training.sh`、`run_stepwise_router_eval.sh`
- 强 Expert 评估及配对病例表：`outputs/strong_expert_awq/STRONG_EXPERT_EVALUATION.md`、`paired_correctness.csv`
