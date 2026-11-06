# EA-ATPO 四动作价值 Router 实验报告

生成时间：2026-11-05T09:42:57+00:00。

## 实验边界

在与强专家直接比较样本不重叠的 200 个官方 ATPO test 病例上运行了模型实际交互。Pilot 的 100 个病例按 `case_id` 划分训练/验证/内部测试，固定 Router 后才抽取外部 test。Base、Patient Simulator 和 Expert Simulator 都是 Qwen3-8B；Expert 是语言模型模拟器，不是真实医生。
Router 决策只读取初始公开题干和公开对话文本。Gold 仅在 rollout 结束后评分；hidden atomic facts 仅进入 Patient Simulator；Router 不读取状态生成轨迹、事后正确性、未来患者回复或专家建议。
每个策略对每例都使用相同的公开初始状态，并运行其实际选择的动作。生成温度为 0.2；请求种子由模型和公开请求内容派生。同一病例/动作 rollout 只运行一次，并由选择该动作的策略共享，因此配对比较使用完全相同的输入。
协议范围：本诊断只由各策略选择第一个动作；ASK/CONSULT 之后由 Base Agent continuation policy 选择后续动作。因此它不是 Router 逐轮参与的端到端策略结果。逐轮主评估见 `outputs/stepwise_test/STEPWISE_ROUTER_EXPERIMENT_REPORT.md`。

## 实际交互策略结果

| Policy | Accuracy (95% case bootstrap CI) | ASK | CONSULT | DEFER | Expert calls/case | Tokens/case | Inference s/case |
|---|---:|---:|---:|---:|---:|---:|---:|
| always_answer | 0.485 (0.415–0.555) | 0.0% | 0.0% | 0.0% | 0.00 | 514.0 | 0.047 |
| uncertainty_threshold | 0.495 (0.425–0.565) | 0.0% | 46.0% | 0.0% | 0.49 | 2472.3 | 0.369 |
| fixed_cost_router | 0.455 (0.385–0.525) | 0.0% | 0.0% | 100.0% | 1.00 | 622.2 | 0.046 |
| learned_value_no_cost | 0.470 (0.400–0.540) | 54.0% | 22.0% | 17.5% | 0.81 | 4841.5 | 0.755 |
| learned_value_router | 0.460 (0.390–0.530) | 25.5% | 27.0% | 29.5% | 0.72 | 3340.2 | 0.511 |
| learned_value_higher_cost | 0.465 (0.395–0.535) | 1.0% | 6.5% | 28.0% | 0.35 | 779.2 | 0.089 |
| ablation_no_expert_reliability | 0.460 (0.390–0.530) | 23.5% | 28.5% | 30.5% | 0.76 | 3343.9 | 0.512 |
| ablation_no_consult | 0.480 (0.410–0.550) | 36.5% | 0.0% | 36.0% | 0.59 | 3037.0 | 0.455 |
| ablation_no_defer | 0.480 (0.410–0.550) | 34.0% | 37.5% | 0.0% | 0.60 | 4227.2 | 0.664 |

## 与 Always Answer 的配对差异

| Policy | Accuracy difference | 95% paired case bootstrap CI | 修正直答错误 | 退化直答正确 |
|---|---:|---:|---:|---:|
| uncertainty_threshold | +0.010 | -0.025–+0.045 | 7/103 | 5/97 |
| fixed_cost_router | -0.030 | -0.065–+0.000 | 3/103 | 9/97 |
| learned_value_no_cost | -0.015 | -0.060–+0.030 | 9/103 | 12/97 |
| learned_value_router | -0.025 | -0.065–+0.015 | 5/103 | 10/97 |
| learned_value_higher_cost | -0.020 | -0.045–+0.000 | 1/103 | 5/97 |
| ablation_no_expert_reliability | -0.025 | -0.065–+0.015 | 6/103 | 11/97 |
| ablation_no_consult | -0.005 | -0.040–+0.025 | 5/103 | 6/97 |
| ablation_no_defer | -0.005 | -0.040–+0.035 | 7/103 | 8/97 |

## 与 Learned Value Router 的配对差异

| Policy | Accuracy difference | 95% paired case bootstrap CI | 修正 Router 错误 | 退化 Router 正确 |
|---|---:|---:|---:|---:|
| always_answer | +0.025 | -0.010–+0.065 | 10/108 | 5/92 |
| uncertainty_threshold | +0.035 | -0.005–+0.075 | 12/108 | 5/92 |
| fixed_cost_router | -0.005 | -0.045–+0.035 | 8/108 | 9/92 |
| learned_value_no_cost | +0.010 | -0.025–+0.045 | 7/108 | 5/92 |
| learned_value_higher_cost | +0.005 | -0.025–+0.035 | 5/108 | 4/92 |
| ablation_no_expert_reliability | +0.000 | -0.015–+0.015 | 1/108 | 1/92 |
| ablation_no_consult | +0.020 | -0.010–+0.050 | 7/108 | 3/92 |
| ablation_no_defer | +0.020 | +0.000–+0.045 | 5/108 | 1/92 |

## 方法与成本定义

Outcome target 是每个公开状态/动作三个 Pilot rollout 的平均正确率；无效输出按 0 保留。价值模型为每个候选动作分别预测正确率、完整分支 patient simulator turns、Expert CONSULT tokens/1000、Expert DEFER tokens/1000。路由效用严格为 `Q(a,s) − λASK·patient_turns − λCONSULT·consult_tokens/1000 − λDEFER·defer_tokens/1000`，三个成本分量均从完整 rollout trace 统计，每项只扣一次。报告另列实际完整链路 token 与请求延迟。
固定成本 Router 在全部 Pilot 训练数据上选出总体平均效用最高的单一动作。Uncertainty Threshold 仅用验证集选择阈值：由预测直答正确率得到不确定性，并在低信心时调用训练集上表现最好的固定非 ANSWER 动作。Learned Value Router 使用文本 TF-IDF 线性核岭回归预测每个动作的 Q 和成本；Expert reliability 特征是按病例交叉拟合的 Pilot DEFER 正确概率。
Accuracy–Cost 曲线使用实际系统总 token 作为横轴，并同时输出专家 token、专家调用量和延迟。λ 是预注册的敏感性单位，不代表美元、医生劳动价格或临床效用测量。无成本、参考成本和高成本组都在同一监督目标/预测成本定义上只更换固定 λ。

## 消融

- `ablation_no_expert_reliability`：重新训练时去掉交叉拟合的专家可靠性输入特征。
- `uncertainty_threshold`：不做四动作逐状态价值选择，改用验证集冻结的简单置信度阈值路由。
- `ablation_no_consult`：动作菜单仅 ANSWER/ASK/DEFER。
- `ablation_no_defer`：动作菜单仅 ANSWER/ASK/CONSULT。

## 解释边界

Router 监督数据来自 100 个病例、291 个状态、每状态每动作 3 次的历史 Qwen3-8B 模拟交互；状态级样本仍较少，外部病例准确率和成本差异需结合宽置信区间理解。专家可靠性特征预测的是同一 8B 模拟器在可见文本条件下的历史 DEFER 正确率，并非真实医生能力，也不是本次 CONSULT 建议质量的直接判别。
Router test 的多策略配对差异只包含选定分支的一次真实 rollout，不是四动作 oracle。由于相同 `(case,action)` 轨迹被多个策略共享，策略差异来自冻结决策选择；服务生成温度为 0.2。模型服务日志显示 vLLM prefix cache hit rate 非零，实测延迟包含该缓存带来的服务端加速，token 仍按完整 API prompt/completion usage 记录；部署服务缓存配置或负载不同会改变延迟。
目前未启动 P5/P8 联合 ATPO RL。Qwen3-32B AWQ 权重恢复状态见 `outputs/STRONG_EXPERT_RECOVERY_REPORT.md`；在权重验证且专家评估跑完以前，没有 32B 准确率结果。本诊断只对应 8B 模拟 Expert。

## 产物

- `policy_episodes.jsonl`：每病例/策略的实际分支、动作、答案、Gold 评分、成本、完整轨迹。
- `action_rollouts.jsonl`：去重后的病例/动作真实 rollout，支持断点续跑。
- `router_policy_summary.csv`、`paired_baseline_comparisons.csv`、`accuracy_cost_curve.csv`。
- `run_manifest.json`：样本行、数据哈希、模型 revision、配置和运行状态。
