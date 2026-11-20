# EA-ATPO V2 Window B — Router 泛化诊断与保守选择层

日期：2026-11-20

范围：现有 50 个训练病例中的固定 Train/Dev 划分；未读取窗口 A 新病例确认集、官方 Validation、Internal Test 或 Frozen Holdout。

## 结论

保留一个待后续独立验证的最小候选：**Embedding Relative Advantage + `margin=0.025`**。在旧 Dev 上，它相对 margin 0 的 Relative Advantage 平均 Utility 增加 0.00089，ASK 与 DEFER 率各下降约 1.5 个百分点，Accuracy 不变。该差异很小；margin 是在同一开发集上比较的，而且 checkpoint 也由这 15 个开发病例 early stopping 选择，不能视为确认性收益。

不建议用这个候选替换现有主方法 Q-guided Distillation。候选 Dev Utility 为 0.469，低于 Distillation 的 0.491；配对差为 -0.021，病例级 95% CI [-0.075, 0.043]。本结果不证明两者相同，也没有显示候选更好。保守层只减少 Relative Advantage 的调用，收益看起来主要来自少付成本，而非增加正确答案。

## 数据与实现审计

- 数据为 50 例、146 个公开状态、438 条有效动作轨迹。动作顺序为 `ASK / ANSWER / DEFER`，每个状态的三动作均有效。
- 固定划分按 `case_id`：35 例 / 103 状态 Train，15 例 / 43 状态 Dev；同一病例的状态没有跨集合。
- 当前对照使用已训练的冻结 Qwen3-8B 状态 embedding（4096 维，revision `b968826d9c46dd6066d109eabc6255188de91218`）及已保存的小模型 checkpoint，没有重新训练或调用模型服务。TF-IDF 词表和统计量在原训练流程中只对 35 个 Train 病例拟合。
- Embedding 审计边界是动作前公开状态。缓存只包含状态 embedding 与哈希；没有 Gold、hidden atomic facts、未来回复、最终答案、正确性或 rollout 结果。重算时逐条校验状态哈希和特征边界。
- Q 标签已由原始终局正确性及 ASK/DEFER 累计成本重建，成本为 ASK=0.01、ANSWER=0、DEFER=0.05，且各计一次。相对值是成本已包含的 `Q_action-Q_ANSWER`，选择层不再扣费。校验 `Q + action_cost` 可还原 0/1 正确性。
- 全部 438 条状态-动作标签都是 K=1，且使用同一个 continuation policy `ASK_ONCE_THEN_FORCE_BASE_ANSWER_PATIENT_REVIEW_V2`；没有混合 continuation policy。`return_std` 缺失是正确表示。它们是单次实现回报，不是低噪声的期望 Q；本诊断不估算单状态真实方差。
- 三个已保存训练 seed（17、42、2026）的 checkpoint 均成功重载。与已保存 Dev 分数最大绝对差为 `2.38e-7`；margin 0 的动作选择与原 Relative Advantage Dev 动作完全一致。

### 现有报告的 CI 复核

Q-guided Distillation 的点估计 0.491、Always ANSWER 的 0.400 均复现。原报告给出的 Utility 差 95% CI `[-0.014, 0.297]` 没有按三 seed 均值计算：旧 `cluster_bootstrap_delta` 用 `zip(policy_cases_by_seed, baseline_cases_by_seed)` 配对，而 learned policy 有三组 seed，Always ANSWER 只有一组，`zip` 因此只保留第一个 seed。复算时对每个病例先配对三个 seed 的策略结果与同一个 Always ANSWER 结果，再按病例聚类 bootstrap 5,000 次；三 seed 平均差为 `0.0906`，95% CI `[-0.0316, 0.2372]`。

旧 Relative Advantage 报告的名义正 Utility 区间也受此实现问题影响；修正后相对 Always ANSWER 的差为 `0.0686`，95% CI `[-0.0872, 0.2333]`。这些开发集区间仍是描述性结果：checkpoint 早停使用了同一批 Dev 病例。

### Base / Expert 四格分母

Dev 四格的分母是 **43 个状态**，不是 15 个病例；病例是 bootstrap 聚类单位。

| Base ANSWER | Expert DEFER | 状态数 | 状态比例 | 至少含一状态的病例数（15 例中） |
|---|---|---:|---:|---:|
| 正确 | 正确 | 11 | 25.6% | 6 |
| 正确 | 错误 | 6 | 14.0% | 3 |
| 错误 | 正确 | 9 | 20.9% | 4 |
| 错误 | 错误 | 17 | 39.5% | 7 |

四格病例数彼此可重叠，因为同一病例可有多个状态。Expert 在 Base 错误状态中纠正 9/26（34.6%），在 Base 正确状态中造成退化 6/17（35.3%）。ASK 相对 ANSWER：4 个状态纠正、2 个状态退化、37 个状态正确性不变。

## 泛化与优势可学习性

| 指标 | Train（35 例 / 103 状态） | Dev（15 例 / 43 状态） |
|---|---:|---:|
| Q-guided Distillation Accuracy / Utility | 0.581 / 0.568 | 0.507 / 0.491 |
| Q-guided ASK / ANSWER / DEFER 率 | 0.359 / 0.462 / 0.179 | 0.474 / 0.285 / 0.241 |
| Relative Advantage Accuracy / Utility | 0.581 / 0.561 | 0.493 / 0.469 |
| Relative Advantage ASK / ANSWER / DEFER 率 | 0.343 / 0.336 / 0.321 | 0.348 / 0.241 / 0.411 |
| Relative advantage MAE：ASK | 0.132（病例宏平均） | 0.309（病例宏平均，95% CI [0.232, 0.391]） |
| Relative advantage MAE：DEFER | 0.132（病例宏平均） | 0.465（病例宏平均，95% CI [0.327, 0.610]） |
| 正优势 AUROC：ASK / DEFER | 0.998 / 0.999 | 0.643 / 0.609 |

Q-guided Distillation 的 Train→Dev Utility 差为 0.078、Accuracy 差为 0.074；Relative Advantage 的对应差为 0.093 和 0.088。Relative Q Advantage 的 Train→Dev MAE 增量为 ASK 0.177、DEFER 0.332，显示优势值回归比策略表面准确率更明显地退化。

原绝对 Q 两层 MLP 的三动作 MAE：ASK/ANSWER/DEFER 在 Train 为 0.122/0.105/0.092，在 Dev 为 0.409/0.369/0.434；总体 MAE 从 0.106 升至 0.404，差距 0.298。Distillation head 输出策略 logits，不是数值 Q，因此不把 logits 解释为校准价值或报告其 Q MAE。

在已扣成本的真实标签中，ASK 相对 ANSWER 正优势为 Train 6/103（5.8%）、Dev 4/43（9.3%）；DEFER 正优势为 16/103（15.5%）、9/43（20.9%）。但 Relative Advantage 头在 Dev 的预测正值比例分别为 ASK 57.4%、DEFER 51.9%，远高于正标签比例。Train AUROC 接近 1、Dev 降至 0.643/0.609，且 MAE 大幅上升，说明训练集排序拟合没有迁移成可靠开发集优势预测。

Dev 四分位分箱也显示正向预测偏大：ASK 最高分箱平均预测净优势 +0.366，实测 +0.090，只有 20% 标签为正；DEFER 最高分箱预测 +0.299，实测 +0.050，只有 30% 为正。预测均值不等于校准概率，分箱仅 10 个状态，作为描述性误差提示。平均预测优势在 `|prediction|≤0.01` 附近的状态很少：ASK 2 个，其中 1 个符号错误；DEFER 2 个，2 个符号错误。此处样本过少，区间不能支持稳定判断。

### 误选动作的可观察原因

以下按 3 个 seed × 43 个 Dev 状态统计选择次数（129 次决策），不是 129 个独立病例；病例级 CI 仍按 `case_id` 聚类。这里“没有正优势”指该 K=1 成本后 Q 标签不高于 ANSWER。

| 策略 | ASK 选择 / 无正优势（其中正确性退化 / 不变） | DEFER 选择 / 无正优势（其中正确性退化 / 不变） |
|---|---:|---:|
| Q-guided Distillation | 61 / 55（3 / 52） | 30 / 18（1 / 17） |
| Relative Advantage，margin 0 | 46 / 39（3 / 36） | 52 / 39（5 / 34） |
| Conservative，margin 0.025 | 44 / 37（3 / 34） | 50 / 37（5 / 32） |

多数负优势选择没有改变终局正确性，只因 ASK/DEFER 成本而降低 Utility；少数 DEFER 负优势选择还将原本正确答案变错（Expert spoilage）。这与 ASK 调用过多和 DEFER 排序误差相符。现有汇总标签不足以可靠断言 ASK 失败来自患者模拟器还是提问策略；K=1 也无法把一次失败拆成稳定的信息价值估计。不能把“终局正确性未变”直接解释为患者回复没有临床信息。

## 相对价值保守选择层

复用现有 `mlp_relative` 头，不改变 embedding、checkpoint、训练标签或 ASK/ANSWER/DEFER 协议。Relative head 为 `4096→16→2`，输出 `[Q_ASK-Q_ANSWER, Q_DEFER-Q_ANSWER]`；Distillation head 为 `4096→16→3`，温度为 0.1。相对值已含注册成本。答案可用时它是保守回退；仅当预测净优势**严格大于 margin**时才选择 ASK/DEFER。所有边际值在同一预先列出的网格 `[0, 0.01, 0.025, 0.05]` 上报告。

选择层同时处理 `valid_mask`、每条决策路径最多一次 ASK 的预算，以及 ASK 后无新增信息时屏蔽再次 ASK。成本已在预测相对值里，选择层不再扣一次。当前日志是各公开状态的离线反事实标签，没有重放完整交互策略；因此 ASK 预算和“无新增信息”转换由可复用选择函数实现并做确定性不变量检查，但本对照不能估计一次 ASK 后的动态策略效果。

### Dev 对照

病例宏平均，先在病例内平均其状态，再对 15 个病例等权，并对三个固定 seed 求平均。所有成本网格都保留在 CSV 中。

| 方法 | Accuracy（病例级 95% CI） | Utility（病例级 95% CI） | ASK 率 | ANSWER 率 | DEFER 率 | Utility 相对 Distillation 的配对差（95% CI） |
|---|---:|---:|---:|---:|---:|---:|
| Embedding Q-guided Distillation | 0.507 [0.311, 0.707] | 0.491 [0.294, 0.679] | 0.474 | 0.285 | 0.241 | 基线 |
| Embedding Relative Advantage，margin 0 | 0.493 [0.300, 0.689] | 0.469 [0.274, 0.657] | 0.348 | 0.241 | 0.411 | -0.022 [-0.079, 0.043] |
| Conservative Relative Advantage，margin 0.01 | 0.493 [0.300, 0.689] | 0.469 [0.274, 0.658] | 0.341 | 0.263 | 0.396 | -0.021 [-0.078, 0.043] |
| **Conservative Relative Advantage，margin 0.025** | **0.493 [0.300, 0.689]** | **0.469 [0.275, 0.658]** | **0.333** | **0.270** | **0.396** | **-0.021 [-0.075, 0.043]** |
| Conservative Relative Advantage，margin 0.05 | 0.485 [0.304, 0.674] | 0.463 [0.278, 0.646] | 0.333 | 0.293 | 0.374 | -0.027 [-0.073, 0.021] |

Margin 0.025 相对 margin 0 的病例配对 Utility 差为 +0.00089，95% CI [0.00007, 0.00193]；Accuracy 差 0；ASK 与 DEFER 率各变化 -0.0148。很小的成本节省是在已经看过的 Dev 网格上观察到的，而且三个策略共用由同一 Dev 早停的 head。相对 Distillation，它没有更高 Utility，DEFER 率还高 15.6 个百分点。margin 0.05 虽然进一步降低 DEFER，但 Utility 比 margin 0 低 0.0054，Accuracy 低 0.0074。

因此 margin 0.025 只保留为**下一阶段待检候选**，不替换主方法，也不把这组 Dev 收益当成泛化提升。候选需要在未用于边际选择的新训练病例上重新拟合/选择，再按预注册流程做独立确认；本报告不使用窗口 A 确认数据作任何筛选。

## 复现

在此仓库根目录运行；`ROUTER_RUN_ROOT` 指向已存在的 50 例训练产物目录，输出路径必须是一个尚不存在的新目录。脚本仅 CPU 重载小型 head，不调用 GPU、大模型或在线服务。

```bash
export ROUTER_RUN_ROOT=/path/to/EA_ATPO_V1_router_training/experiments/ea_atpo_v2/router_training/outputs/real_q_router_20261120
export ROUTER_DIAGNOSTIC_OUTPUT=experiments/ea_atpo_v2/window_b_router_diagnostics/results_reproduction
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python -m experiments.ea_atpo_v2.window_b_router_diagnostics.run_diagnostic \
  --run-root "$ROUTER_RUN_ROOT" \
  --output-dir "$ROUTER_DIAGNOSTIC_OUTPUT"
```

输出 CSV/JSON 仅含汇总指标、数据与 checkpoint 哈希、配置和病例级 bootstrap 设定；不写病例 ID、状态文本或逐状态预测。正式训练报告、诊断代码和最终汇总文件位于本目录及 `results_20261120_final/`。
