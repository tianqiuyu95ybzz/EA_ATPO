# EA-ATPO 逐轮四动作 Router 实测报告

生成时间：2026-11-05T10:10:20+00:00。

## 评估设计

在 200 个与强专家直接评估集不重叠的 ATPO 官方 test 病例上，对每种策略独立运行完整交互。每次公开状态更新后，Router 重新预测四个动作的 Q 和成本并作下一步选择；不把 continuation action 交给 Base Agent 自行决定。Gold 仅在 episode 结束后评分，hidden atomic_facts 仅交给 Patient Simulator，Expert 仅见公开状态及 CONSULT 问题。
Base/Expert temperature=0.2，Patient temperature=0.0；最多 10 次动作，最多 1 次 CONSULT。每次生成请求的 seed 由模型、可见请求内容和 temperature 派生。所有 policy 使用相同 case_id 与初始公开状态。

## 实际逐轮策略结果

| Policy | Accuracy (95% case bootstrap CI) | Valid | Episode ASK/CONSULT/DEFER | Expert calls/case | Decisions/case | Tokens/case | Inference s/case |
|---|---:|---:|---:|---:|---:|---:|---:|
| always_answer | 0.485 (0.415–0.555) | 100.0% | 0.0%/0.0%/0.0% | 0.00 | 1.00 | 514.0 | 0.052 |
| uncertainty_threshold | 0.470 (0.400–0.540) | 100.0% | 0.0%/46.0%/42.5% | 0.89 | 1.46 | 1140.9 | 0.162 |
| fixed_cost_router | 0.455 (0.385–0.525) | 100.0% | 0.0%/0.0%/100.0% | 1.00 | 1.00 | 622.2 | 0.051 |
| learned_value_no_cost | 0.455 (0.385–0.525) | 100.0% | 76.0%/52.0%/58.0% | 1.10 | 3.83 | 3530.1 | 0.588 |
| learned_value_router | 0.455 (0.385–0.525) | 99.5% | 48.0%/35.5%/52.5% | 0.88 | 2.46 | 2090.7 | 0.334 |
| learned_value_higher_cost | 0.470 (0.400–0.540) | 100.0% | 3.5%/6.5%/31.0% | 0.38 | 1.18 | 744.7 | 0.084 |
| ablation_no_expert_reliability | 0.435 (0.365–0.505) | 99.5% | 45.5%/38.5%/54.5% | 0.93 | 2.54 | 2187.2 | 0.349 |
| ablation_no_consult | 0.465 (0.395–0.535) | 100.0% | 36.5%/0.0%/47.0% | 0.47 | 1.76 | 1266.2 | 0.182 |
| ablation_no_defer | 0.465 (0.395–0.535) | 99.5% | 68.0%/53.0%/0.0% | 0.53 | 3.56 | 3197.1 | 0.545 |

## 相对 Always Answer 的病例级配对差异

| Policy | Accuracy delta | 95% paired bootstrap CI | 修正参考策略错误 | 退化参考策略正确 |
|---|---:|---:|---:|---:|
| uncertainty_threshold | -0.015 | -0.050–+0.015 | 4/103 | 7/97 |
| fixed_cost_router | -0.030 | -0.065–+0.005 | 3/103 | 9/97 |
| learned_value_no_cost | -0.030 | -0.070–+0.010 | 6/103 | 12/97 |
| learned_value_router | -0.030 | -0.070–+0.010 | 6/103 | 12/97 |
| learned_value_higher_cost | -0.015 | -0.040–+0.005 | 1/103 | 4/97 |
| ablation_no_expert_reliability | -0.050 | -0.090–-0.010 | 4/103 | 14/97 |
| ablation_no_consult | -0.020 | -0.055–+0.015 | 4/103 | 8/97 |
| ablation_no_defer | -0.020 | -0.060–+0.020 | 6/103 | 10/97 |

## 相对 Uncertainty Threshold 的病例级配对差异

| Policy | Accuracy delta | 95% paired bootstrap CI | 修正参考策略错误 | 退化参考策略正确 |
|---|---:|---:|---:|---:|
| always_answer | +0.015 | -0.015–+0.045 | 7/106 | 4/94 |
| fixed_cost_router | -0.015 | -0.050–+0.020 | 5/106 | 8/94 |
| learned_value_no_cost | -0.015 | -0.060–+0.030 | 9/106 | 12/94 |
| learned_value_router | -0.015 | -0.060–+0.030 | 9/106 | 12/94 |
| learned_value_higher_cost | +0.000 | -0.030–+0.030 | 5/106 | 5/94 |
| ablation_no_expert_reliability | -0.035 | -0.080–+0.010 | 7/106 | 14/94 |
| ablation_no_consult | -0.005 | -0.040–+0.030 | 7/106 | 8/94 |
| ablation_no_defer | -0.005 | -0.050–+0.040 | 10/106 | 11/94 |

## 相对 Fixed-cost Router 的病例级配对差异

| Policy | Accuracy delta | 95% paired bootstrap CI | 修正参考策略错误 | 退化参考策略正确 |
|---|---:|---:|---:|---:|
| always_answer | +0.030 | +0.000–+0.065 | 9/109 | 3/91 |
| uncertainty_threshold | +0.015 | -0.020–+0.050 | 8/109 | 5/91 |
| learned_value_no_cost | +0.000 | -0.045–+0.045 | 10/109 | 10/91 |
| learned_value_router | +0.000 | -0.040–+0.040 | 8/109 | 8/91 |
| learned_value_higher_cost | +0.015 | -0.015–+0.045 | 6/109 | 3/91 |
| ablation_no_expert_reliability | -0.020 | -0.060–+0.020 | 7/109 | 11/91 |
| ablation_no_consult | +0.010 | -0.025–+0.045 | 7/109 | 5/91 |
| ablation_no_defer | +0.010 | -0.035–+0.055 | 11/109 | 9/91 |

## 相对 Learned Value Router 的病例级配对差异

| Policy | Accuracy delta | 95% paired bootstrap CI | 修正参考策略错误 | 退化参考策略正确 |
|---|---:|---:|---:|---:|
| always_answer | +0.030 | -0.010–+0.075 | 12/109 | 6/91 |
| uncertainty_threshold | +0.015 | -0.030–+0.060 | 12/109 | 9/91 |
| fixed_cost_router | +0.000 | -0.040–+0.040 | 8/109 | 8/91 |
| learned_value_no_cost | +0.000 | -0.040–+0.040 | 9/109 | 9/91 |
| learned_value_higher_cost | +0.015 | -0.020–+0.055 | 9/109 | 6/91 |
| ablation_no_expert_reliability | -0.020 | -0.040–-0.005 | 0/109 | 4/91 |
| ablation_no_consult | +0.010 | -0.025–+0.045 | 8/109 | 6/91 |
| ablation_no_defer | +0.010 | -0.015–+0.035 | 4/109 | 2/91 |

## 方法、成本与消融

Router utility 为 `Q − 0.01×患者模拟器轮数 − 0.02×CONSULT 专家 tokens/1000 − 0.02×DEFER 专家 tokens/1000`。分支完整 trace 记录成本，Q 与三项成本单独预测，每一项只扣一次。真实 Accuracy–Cost 横轴使用全链路 API usage 总 tokens；报告同时列出专家调用、完整调用延迟及 reference-cost utility。
固定成本策略在训练病例上选一个总体平均效用最高动作；阈值策略按 validation 冻结置信度阈值；无专家可靠性消融重新拟合对应模型；无 CONSULT/DEFER 消融从所有逐轮决策的候选动作中移除该动作。

## 解释限制

监督 Pilot 只有 100 个病例、291 个状态；这不足以支持高维深度价值模型。Q targets 来自 Pilot 中强制首动作后由 Base Agent continuation 的反事实 rollout；逐轮测试将该一步 Q 模型反复用于 Router policy，存在 continuation-policy 分布偏移，需在更大规模 on-policy Router rollout 中再训练验证。Uncertainty Threshold 的阈值只在 validation 上选择，随后用全 Pilot 开发数据重拟合的最终模型执行；因此其置信度尺度可能有校准偏移，但外部 test 未参与阈值或模型拟合。1800 条 episode 中 3 条 CONSULT advice 因输出最终选项而被协议拒绝；这些 rollout 保留、准确率记 0，成功生成的请求 token 和延迟仍纳入统计。Expert Simulator 与 Base/Patient Simulator 均为 Qwen3-8B，Expert 不是医生。病例 bootstrap 是病例级配对区间，未把一次 rollout 的模型随机性作为独立病例重复；按公开请求 hash seed 固定生成结果。vLLM 服务日志显示 prefix caching 生效，token 统计仍为完整 API prompt/completion usage，延迟反映当前缓存及 GPU 负载。
旧的 `outputs/ROUTER_EXPERIMENT_REPORT.md` 是仅对起始状态选一次动作的诊断结果，ASK/CONSULT 后续动作由 Base Agent 续答；不能解读成逐轮 Router。此报告及本目录 `policy_episodes.jsonl` 才是逐轮 Router 主评估。

## 产物

- `policy_episodes.jsonl`：每病例/策略的逐轮路由动作、完整轨迹、答案和评分。
- `router_policy_summary.csv`、`paired_baseline_comparisons.csv`、`accuracy_cost_curve.csv/png`。
- `run_manifest.json`：病例、数据/checkpoint 哈希、生成参数和模型 revision。
