# Qwen3-32B-AWQ 专家优势 Router 泛化结果（Attempt04）

> 本结果用于判断是否值得进入后续研究；当前结论不启动 PPO/ATPO RL。

## 实验与数据边界

- 本轮模型推理已完成：1,700 个病例配对，按 case_id 固定划分为 train 1,000、validation 300、test 400。三组均来自 ATPO train split 的新病例 holdout；测试组不是 ATPO 官方 test split。
- Base 为 `Qwen/Qwen3-8B` revision `b968826d9c46dd6066d109eabc6255188de91218`；DEFER Expert 为 `Qwen/Qwen3-32B-AWQ` revision `0499c3ac83fdef8810b907a23894ba91e95eddd8`。temperature=0、top-p=1、禁用 thinking；提示和 parser 在病例评估前固定。
- 训练标签使用配对结果 `Expert correct - Base correct`；输入只包含动作前公开特征、Base 选项概率/不确定性及 Base 预测选项。Gold 和 Expert 输出未进入特征。训练只读 train；模型/阈值只用 validation 选择；test 阈值冻结后才打开。
- 预注册 DEFER 成本为 0、0.05、0.10、0.20、0.40、0.60；主成本为 0.60。净 Utility = 最终正确率 − 成本 × Expert 调用率，成本只扣一次。
- Oracle Selective DEFER 使用事后配对正确性，仅表示理论上界，不用于训练或部署决策。
- 输出目录中的 raw completion、轨迹和病例级表保存在 PFS，不推送到 GitHub；GitHub 只保留脱敏汇总、代码和哈希。

## Expert 格式失败审计

20 例 smoke 的 Base/Expert 解析失败均为 0。正式 1,700 例中，Base 有 3 条严格答案解析失败（均在 train）；Expert 有 66 条 DEFER parser 失败：train 38、validation 13、test 15。66 条均正常结束（finish_reason=stop），不是生成长度截断；Expert 使用了要求的 `Expert Final Answer:` 前缀，但前缀后输出以 N 开头的非 A-D 选项文字，因此无法通过固定的一选项 full-match parser。模型回复未做事后修复。

这些回复在部署端意味着 DEFER 没有给出可评分答案，因此端到端正确性按 0 计；模型拟合使用的相对收益标签也按这一真实部署结果计算。下表同时给出 parser 覆盖率，以及仅在成功解析回复上的条件准确率，避免把 parser 失败误称为临床答错。

| Split | N | Base 解析失败 | Expert 解析失败 | Expert parse coverage | Base accuracy（端到端） | Expert accuracy（端到端） | Expert accuracy（仅成功解析） |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 1000 | 3 | 38 | 0.962 | 0.478 | 0.528 | 0.549 (528/962) |
| validation | 300 | 0 | 13 | 0.957 | 0.443 | 0.533 | 0.557 (160/287) |
| test | 400 | 0 | 15 | 0.963 | 0.485 | 0.578 | 0.600 (231/385) |

## 配对专家互补性

独立测试集 Base accuracy 为 0.485 [0.435, 0.533]；Expert 端到端 accuracy 为 0.578 [0.527, 0.625]。配对差为 0.092 [0.045, 0.142]。

| Base / Expert | 正确 | 错误或无可解析答案 |
|---|---:|---:|
| Base 正确 | 159 | 35 |
| Base 错误 | 72 | 134 |

在 206 个 Base 错误病例中，Expert 修正 72 个（0.350，病例级 95% CI [0.284, 0.415]）；在 194 个 Base 正确病例中，Expert 退化 35 个（0.180，95% CI [0.126, 0.237]）。测试组 385 条可解析 Expert 回复中答对 231 条，即条件准确率 0.600；全 pipeline 的 15 条格式失败仍计入端到端错误。

## 预注册成本敏感性

下表列出 Always Answer、Always Defer、Uncertainty、主 Logistic Relative Value Router 与 Oracle。完整所有模型/随机基线见 `outputs/.../analysis/policy_cost_sensitivity.csv`。

| 成本 | 策略 | Accuracy | Utility | DEFER率 | 平均成本 | 平均总 Token |
|---:|---|---:|---:|---:|---:|---:|
| 0.00 | always_answer | 0.485 | 0.485 | 0.000 | 0.000 | 164.4 |
| 0.00 | always_defer | 0.578 | 0.578 | 1.000 | 0.000 | 239.0 |
| 0.00 | uncertainty_defer | 0.575 | 0.575 | 0.905 | 0.000 | 380.7 |
| 0.00 | logistic_relative_value | 0.565 | 0.565 | 0.907 | 0.000 | 380.9 |
| 0.00 | oracle_selective_defer_upper_bound | 0.665 | 0.665 | 0.180 | 0.000 | 208.0 |
| 0.00 | random_defer_budget_matched_realized | 0.570 | 0.570 | 0.917 | 0.000 | 384.1 |
| 0.05 | always_answer | 0.485 | 0.485 | 0.000 | 0.000 | 164.4 |
| 0.05 | always_defer | 0.578 | 0.527 | 1.000 | 0.050 | 239.0 |
| 0.05 | uncertainty_defer | 0.570 | 0.529 | 0.823 | 0.041 | 360.9 |
| 0.05 | logistic_relative_value | 0.565 | 0.520 | 0.907 | 0.045 | 380.9 |
| 0.05 | oracle_selective_defer_upper_bound | 0.665 | 0.656 | 0.180 | 0.009 | 208.0 |
| 0.05 | random_defer_budget_matched_realized | 0.570 | 0.524 | 0.917 | 0.046 | 384.1 |
| 0.10 | always_answer | 0.485 | 0.485 | 0.000 | 0.000 | 164.4 |
| 0.10 | always_defer | 0.578 | 0.478 | 1.000 | 0.100 | 239.0 |
| 0.10 | uncertainty_defer | 0.510 | 0.505 | 0.052 | 0.005 | 177.1 |
| 0.10 | logistic_relative_value | 0.482 | 0.480 | 0.025 | 0.003 | 170.3 |
| 0.10 | oracle_selective_defer_upper_bound | 0.665 | 0.647 | 0.180 | 0.018 | 208.0 |
| 0.10 | random_defer_budget_matched_realized | 0.517 | 0.498 | 0.193 | 0.019 | 210.6 |
| 0.20 | always_answer | 0.485 | 0.485 | 0.000 | 0.000 | 164.4 |
| 0.20 | always_defer | 0.578 | 0.378 | 1.000 | 0.200 | 239.0 |
| 0.20 | uncertainty_defer | 0.505 | 0.498 | 0.035 | 0.007 | 173.0 |
| 0.20 | logistic_relative_value | 0.482 | 0.477 | 0.025 | 0.005 | 170.3 |
| 0.20 | oracle_selective_defer_upper_bound | 0.665 | 0.629 | 0.180 | 0.036 | 208.0 |
| 0.20 | random_defer_budget_matched_realized | 0.487 | 0.482 | 0.028 | 0.006 | 171.0 |
| 0.40 | always_answer | 0.485 | 0.485 | 0.000 | 0.000 | 164.4 |
| 0.40 | always_defer | 0.578 | 0.177 | 1.000 | 0.400 | 239.0 |
| 0.40 | uncertainty_defer | 0.485 | 0.485 | 0.000 | 0.000 | 164.4 |
| 0.40 | logistic_relative_value | 0.485 | 0.485 | 0.000 | 0.000 | 164.4 |
| 0.40 | oracle_selective_defer_upper_bound | 0.665 | 0.593 | 0.180 | 0.072 | 208.0 |
| 0.40 | random_defer_budget_matched_realized | 0.485 | 0.485 | 0.000 | 0.000 | 164.4 |
| 0.60 | always_answer | 0.485 | 0.485 | 0.000 | 0.000 | 164.4 |
| 0.60 | always_defer | 0.578 | -0.022 | 1.000 | 0.600 | 239.0 |
| 0.60 | uncertainty_defer | 0.485 | 0.485 | 0.000 | 0.000 | 164.4 |
| 0.60 | logistic_relative_value | 0.485 | 0.485 | 0.000 | 0.000 | 164.4 |
| 0.60 | oracle_selective_defer_upper_bound | 0.665 | 0.557 | 0.180 | 0.108 | 208.0 |
| 0.60 | random_defer_budget_matched_realized | 0.485 | 0.485 | 0.000 | 0.000 | 164.4 |

### 主 Logistic Router 相对简单策略的病例配对 Utility（95% CI）

| DEFER 成本 | 比较对象 | Utility 差 | 病例级 95% CI |
|---:|---|---:|---:|
| 0.00 | always_answer | 0.080 | [0.033, 0.130] |
| 0.00 | uncertainty_defer | -0.010 | [-0.020, -0.003] |
| 0.00 | random_defer_budget_matched_realized | -0.005 | [-0.022, 0.013] |
| 0.05 | always_answer | 0.035 | [-0.013, 0.085] |
| 0.05 | uncertainty_defer | -0.009 | [-0.021, 0.002] |
| 0.05 | random_defer_budget_matched_realized | -0.004 | [-0.021, 0.012] |
| 0.10 | always_answer | -0.005 | [-0.017, 0.006] |
| 0.10 | uncertainty_defer | -0.025 | [-0.043, -0.008] |
| 0.10 | random_defer_budget_matched_realized | -0.018 | [-0.043, 0.006] |
| 0.20 | always_answer | -0.007 | [-0.020, 0.004] |
| 0.20 | uncertainty_defer | -0.020 | [-0.037, -0.005] |
| 0.20 | random_defer_budget_matched_realized | -0.004 | [-0.020, 0.010] |
| 0.40 | always_answer | 0.000 | [0.000, 0.000] |
| 0.40 | uncertainty_defer | 0.000 | [0.000, 0.000] |
| 0.40 | random_defer_budget_matched_realized | 0.000 | [0.000, 0.000] |
| 0.60 | always_answer | 0.000 | [0.000, 0.000] |
| 0.60 | uncertainty_defer | 0.000 | [0.000, 0.000] |
| 0.60 | random_defer_budget_matched_realized | 0.000 | [0.000, 0.000] |

### Simple Uncertainty Threshold 在中低成本下

阈值只在 validation 选择。test 上成本 0.10 时，该策略 DEFER 5.25%，Accuracy 0.510、Utility 0.505；相对 Always Answer 的配对 Utility 差为 +0.0198（95% CI [+0.0070, +0.0348]）。成本 0.20 时 DEFER 3.5%，Accuracy 0.505、Utility 0.498；相对 Always Answer 为 +0.0130（95% CI [+0.0025, +0.0250]）。这是预注册成本点上的简单不确定性策略结果，需在新病例队列复现；它不能替代主成本 0.60 下的结论，也不能说明复杂 Learned Router 已学会优势。

| 成本 | 不确定性阈值 vs budget-matched Random（Utility 差） | 病例级 95% CI |
|---:|---:|---:|
| 0.10 | 0.007 | [-0.018, 0.031] |
| 0.20 | 0.016 | [0.003, 0.031] |

## Router 识别专家优势的能力

Test AUROC 用来衡量动作前分数能否排序出正向相对收益（Expert correct、Base wrong）；病例级 bootstrap CI 已在一次分析中计算。

| 预决策分数 | AUROC | 95% CI |
|---|---:|---:|
| uncertainty_threshold | 0.710 | [0.645, 0.771] |
| logistic_correction | 0.590 | [0.512, 0.664] |
| logistic_relative_value | 0.573 | [0.502, 0.646] |
| ridge_relative_value | 0.578 | [0.500, 0.654] |
| mlp_relative_value | 0.504 | [0.430, 0.579] |
| historical_learned_router | 0.520 | [0.443, 0.598] |
| base_answer_uncertainty | 0.710 | [0.650, 0.769] |

## 主成本 0.60 的配对差异

| Logistic Relative Value 与 | Utility 差 | 病例级 95% CI |
|---|---:|---:|
| always_answer | 0.000 | [0.000, 0.000] |
| uncertainty_defer | 0.000 | [0.000, 0.000] |
| random_defer_budget_matched_realized | 0.000 | [0.000, 0.000] |

## 价值预测误差与过拟合

| Split | Router | MAE | MSE | Pearson r |
|---|---|---:|---:|---:|
| train | logistic_relative_value | 0.310 | 0.260 | 0.117 |
| train | ridge_relative_value | 0.291 | 0.254 | 0.113 |
| train | mlp_relative_value | 0.290 | 0.267 | -0.015 |
| train | historical_learned_router | 0.318 | 0.265 | -0.010 |
| validation | logistic_relative_value | 0.312 | 0.268 | 0.019 |
| validation | ridge_relative_value | 0.290 | 0.252 | 0.012 |
| validation | mlp_relative_value | 0.287 | 0.266 | -0.014 |
| validation | historical_learned_router | 0.310 | 0.253 | 0.044 |
| test | logistic_relative_value | 0.323 | 0.281 | 0.007 |
| test | ridge_relative_value | 0.296 | 0.261 | 0.046 |
| test | mlp_relative_value | 0.297 | 0.274 | 0.033 |
| test | historical_learned_router | 0.319 | 0.264 | 0.035 |

## Router 泛化失败诊断

- **没有明显的整体特征漂移。** 决策特征 train/test 的最大标准化均值差为 0.078；各 public-state 和 Base uncertainty 特征分布接近。
- **两个选项结构特征全为常数。** `option_count` 和 `mean_option_chars` 在 train/validation/test 全部为 0（option_count, mean_option_chars）；当前独立行格式提取器没有识别题目选项，导致这两项不提供任何区分信息。
- **相对收益标签由中性类主导。** Δ=-1/0/+1：train 105/740/155，validation 25/223/52，test 35/293/72。Δ=0 约占 73–74%，正优势占 15.5–18%；纠错事件分别为 155/522、52/167、72/206 个 Base 错误病例。
- **价值回归接近无信号。** Δ 的常数 0 预测基线 MAE train/validation/test 为 0.260/0.257/0.268；Router MAE 为约 0.287–0.323，test Pearson r 仅 0.007–0.046。训练的参数化价值模型并未优于简单零预测的 MAE。
- **Base uncertainty 单独具有排序信号，但 Learned Router 没有有效利用。** test 上 uncertainty AUROC 为 0.710（正相对收益 95% CI [0.645, 0.771]）；Logistic/Ridge 约 0.57–0.59，MLP/历史 Router 约 0.50–0.52。简单 uncertainty threshold 在成本 0.10/0.20 有正配对 Utility，而主成本 0.60 下 Logistic Relative Value 选择零调用。

类别比例、完整训练/验证/测试特征分布、纠错标签 AUROC、其他阈值、成本与策略完整对照见分析 CSV。

## 研究结论

1. **Expert 存在平均且配对可重复的互补性。** 本 test holdout 上 Expert 端到端准确率高于 Base，配对差区间不跨 0；这回答的是平均协作收益，不代表每个病例都适合 DEFER。
2. **Oracle 显示有选择性 DEFER 的理论空间，但当前 Learned Router 未在主成本上兑现。** 在成本 0.60 下，验证集选出的主 Router 调用率为 0，因此 test 与 Always Answer 完全相同；它相对于 Always Answer、Uncertainty 和 budget-matched Random 均没有可证明的正 Utility 增益。
3. **简单不确定性阈值在成本 0.10/0.20 有 held-out 正 Utility。** 两个预注册点相对 Always Answer 的配对置信区间高于 0，说明简单 Base 不确定性可能是可用的预决策信号；但当前 Learned Relative Value Router 没有复现这个收益。
4. **不启动 PPO/ATPO RL。** Expert 的条件互补性成立，简单不确定性策略在较低成本点有初步正收益；但在主成本 0.60 下 Learned Router 不触发 DEFER，且未优于 Always Answer。下一阶段应在训练数据侧修复选项特征提取，并用新病例复现 uncertainty policy；后续预注册还需对每个策略分别匹配调用预算的 Random baseline，再讨论 Actor-Critic RL。本轮 test 不再用于调参。

## 一次性分析完整性与限制

原一次性分析已在 policy lock 后打开 test，并生成全部表格及 manifest（policy SHA256 `facecda1505e7ed303ccbd7c35f4cfe7f6c67e767bcb7b3827faaa16b5228b04`；test SHA256 `944a4056105201b0f07f8cb0a0ca4b7fd14788d634bc75d0e5ddda0bcf07c969`）。首次程序在最终 Markdown 组装阶段遇到 `auc_by_model` 初始化顺序错误；该错误发生在 CSV 和 analysis manifest 已写出之后。已修复分析源码中的顺序问题。本报告从已生成汇总 CSV 组装；额外读取 raw trajectories 中的 `valid`、`parse_error`、`finish_reason`、`raw_response` 字段来汇总解析失败类型，但不使用测试正确性标签、不重新拟合、不重选阈值，也不重算 test accuracy/bootstrap。

Random 策略的现有实现按 Ridge Router 的 validation 调用率匹配，并非每个候选 Router 各自匹配预算。因此本轮 Random 对照不作为不确定性策略的严格等预算结论；后续复现需要按每个策略预先确定的 validation 调用率分别生成随机对照。

解析失败分组 CSV：`outputs/qwen32_router_generalization_20261118_attempt04/analysis/parser_failure_diagnostics.csv`。执行 provenance SHA256 `36a7e98d2850960d40cc2cf3001a01dd814429bca55b379da9c9a63c1babc640`。病例级测试结果、raw 轨迹、服务日志和 GPU telemetry 留在 PFS 的 ignored `outputs/`。
