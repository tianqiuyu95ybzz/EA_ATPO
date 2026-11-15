# Attempt09 Critic 诊断

## 观测结果

恢复的 step10 Critic 可正常加载为 `Qwen3ForTokenClassification`，7.57B 参数，score head 形状为 `[1, 4096]`；probe 值有限。训练日志明确提示 `score.weight` 与 `score.bias` 是新初始化参数。checkpoint 能运行不等于 value function 已学好。

| 指标 | 10 步结果 |
|---|---:|
| Explained Variance | 10/10 为负；范围 -203.592 到 -0.560；均值 -28.845；step10 为 -2.158 |
| Value Loss | 0.0297–8.0967；均值 2.1717 |
| 每步 PPO 有效轨迹 | 通常 8；step9 为 7 |
| Return 均值 | -0.355 到 0.865；观测范围 -0.80 到 1.00 |
| Advantage 均值 | -0.274 到 0.648；观测范围 -2.279 到 2.991 |

EV 按代码定义为 `1 - Var(Return - Value) / (Var(Return) + 1e-5)`。从日志逐步重算后与记录最大绝对误差为 `1.71e-6`，公式实现正确。step4 的 Return 方差仅 0.0118，而残差方差为 2.409，EV 因而降至 -203.59；step10 的 Return 方差为 0.00949、残差方差为 0.0300，EV 仍为 -2.158。目标 Return 在这些 minibatch 中变化很小，微小的预测偏差也会显著拉低 EV。

## 最可能原因

1. **Critic 冷启动且更新次数少。** score head 从随机初始化开始，只有 10 次 PPO optimizer update。
2. **每步有效状态多样性很低。** 每步 8 条轨迹来自 2 个病例；整个 run 仅覆盖 20 个训练病例。80 条 rollout 不等于 80 个独立 Critic 状态。
3. **目标分布窄且变化大。** 训练后段 Return 多落在 0.75–1.0，方差约 0.01；预测残差仍大于目标方差。现有数据无法分辨这是初始化、样本数、树目标噪声各自贡献多少。

因此 Critic 当前不能作为可靠动作价值估计器，也不能据其数值解释 Router 或策略质量。这里是数据支持的诊断，不是单一原因的因果证明。

## Return、Advantage、GAE 与终止状态

EA loop 将 `gamma` 和 `lam` 从配置传给树搜索（`ea_agent_loop.py:1009-1011`）。当前 Advantage/Return 实际由树搜索预计算：树回传使用子节点 `action_reward + gamma * mdp_state_value`，父状态取候选子节点 Q 的均值；每个节点 `returns = mdp_state_value`，优势为一步边奖励加下一状态值减父状态值，再除以探索次数（`user_assistant_agent_loop.py:461-523`）。

MT trainer 发现 batch 已包含 `advantages` 和 `returns` 后直接使用，并跳过 `compute_advantage(..., lam=...)` 标准 GAE 路径（`mt_trainer.py:1303-1332`）。所以配置里的 `lam=1` 并没有在这次训练里执行 GAE；不应把这 10 步描述为 GAE Critic 结果。

终局节点 Critic value 被设为 0；终局正确性奖赏加在进入终局的动作边上（`ea_agent_loop.py:876-906`）。ANSWER/DEFER 终止；ASK/CONSULT 获得一次观察后由同一 Actor 继续。树回传使用的 Return 是树估值目标，不是简单的整 episode 奖励复制到每个 token。

## 成本核对

ASK/CONSULT/DEFER 负成本在动作边上各计一次，正确性奖赏与该边相加（`ea_agent_loop.py:543-550, 876-903`）。超预算动作不进行外部调用且成本归零，作为终止的策略约束违规（`ea_agent_loop.py:659-691`）。episode 报告值由终局正确性减累计资源成本构成；它没有再次送入预计算的 PPO Return。80 条轨迹的 episode 奖励、逐边奖励和成本恒等式检查均无失败。因此没有证据显示成本重复扣除或 GAE 重复计算。

## 结论

Critic 可加载、loss 有限，但 10/10 负 EV 且目标方差很低，故价值估计不可靠。当前无需改网络结构或 Loss；先保留该诊断，不将 Critic 当作已验证的价值函数。
