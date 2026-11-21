# 指标与成本口径审计

## 统一定义

对某个状态和动作，终局正确性 `y ∈ {0,1}`，实际问诊次数为 `n_ASK`，专家调用次数为 `n_DEFER`：

```text
Utility = y − c_ASK × n_ASK − c_DEFER × n_DEFER
Net Gain(policy, baseline) = Utility(policy) − Utility(baseline)
```

Accuracy 是终局正确率，不扣成本。主 Router 成本配置为 `c_ANSWER=0, c_ASK=0.01, c_DEFER=0.05`。Window B 的预注册网格使用 ASK `{0, .01, .03, .05}` × DEFER `{0, .05, .10, .20, .40, .60}`。所有成本敏感结果都从原始终局正确性和动作计数计算。

V2 早期 rollout collector 曾使用 ASK `0.05`、DEFER `0.60`。后续 Router 标签将旧原始动作结果重算为 ASK `0.01`、DEFER `0.05`，历史原始文件没有覆盖。训练目标中已含成本，评估器以动作正确性及成本重建 Utility；没有再从净 Q 标签上扣一次成本。

## 公式与代码核查

- 成本标签生成检查 `valid_mask`、动作次序、决策前公开状态边界，回报按原始 correctness、ASK/DEFER 次数扣费：`prepare_costed_dataset.py`。
- Router 主评估计算：选择动作后读取同状态对应反事实 `final_accuracy`，Utility 按冻结成本计算；不把 `Q_target` 再当收益扣费：`router_training/scripts/train_evaluate.py`。
- Window B 复算脚本直接构造 `accuracy - [c_ASK, 0, c_DEFER]` 矩阵，Oracle 每状态选最高 Utility；同分在最优动作间分配相同概率。
- V3 Oracle 实验直接计算 `terminal_correctness - 0.01*cumulative_ASK - c_DEFER*cumulative_DEFER`。Frozen Router 动作对所有成本保持固定，没有按成本重新调阈值；Gold 仅在 Router 动作冻结后供 Oracle 评价/安全门使用。

## CPU 独立复算

新增 `scripts/recompute_windowb_metrics.py` 只读取旧 JSONL 轨迹，不加载模型、不拟合模型、不调用 GPU。复核得到：

| 项 | 重算结果 |
|---|---:|
| 轨迹行 / 有效行 | 438 / 438 |
| 病例 / 状态 | 50 / 146 |
| ASK / ANSWER / DEFER | 146 / 146 / 146 |
| 状态动作重复 | 0 |
| 原始 Return 与原始成本分量不一致 | 0 |
| 成本场景 × 策略行 | 24 × 4 = 96 |
| 与保存的 `cost_sensitivity.csv` 最大绝对差 | 0.0 |

重算 CI 使用 10,000 次 `numpy.default_rng`（PCG64）病例级重采样，复现 Window B 报告里的病例宏估计和区间。病例内所有状态先平均、每病例等权；CSV 同时保留 pooled-state 指标，不能把两种分母混称。

## 口径一致性

| 比较 | 样本 / 策略 | 指标口径 | 审计结论 |
|---|---|---|---|
| V2 Always ANSWER/ASK/DEFER/Retrospective Oracle | 50 个 Train 病例、146 状态 | 同状态三动作真实 rollout；case-macro Accuracy/Utility；另列 pooled-state | 相同样本；成本从 raw correctness 重算；Oracle 明确 hindsight |
| V2 Q-guided 与 Relative Advantage | 15 个 Dev 病例、43 状态；每个候选 3 seeds | 相同 paired action rows，case-macro 指标；seed 等权 | 与 Always 三种固定动作共用 43 状态；三 seed 预测保留，旧 CI 除外 |
| V3 Oracle | 同一 15 Dev 病例、43 状态 | case-macro；三 seed learned policy 在每病例内等权平均 | 同一数据，但 Gold safety gate 是不可部署上界 |

Uncertainty Threshold 在原始输出中没有可信 Base log-probability、entropy 或 confidence，因此不能与其他策略做有效的同口径实测。Static L2D 仅为基于训练标签/均值的启发式，不等于一个训练完成且经过独立测试的 Router。

## 关键观察

- 在 V2 Dev，Q-guided 的点估计高于三个固定动作；但病例聚类 CI 都包含 0。
- Relative Advantage MLP 的旧 Utility CI 看似略大于 0；这是 Bootstrap seed 截断错误导致的错觉。
- 在 V2 50-case Train cohort，cost `.01/.05` 下 Always ANSWER/ASK/DEFER 的 case-macro `(Accuracy, Utility)` 为 `(0.390, 0.390)`、`(0.387, 0.377)`、`(0.437, 0.387)`；Retrospective Oracle 为 `(0.597, 0.589)`。Oracle 增益仅作为观测标签上界。
- cost `.01/.60` 时 Always DEFER Utility 为 `-0.163`，Retrospective Oracle 为 `0.514`。Oracle 因为逐状态使用 outcome 选择，不能转译为实际策略可达到的收益。
