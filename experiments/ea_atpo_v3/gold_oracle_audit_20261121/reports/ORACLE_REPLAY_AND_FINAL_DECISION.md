# Gold 来源强化后的历史 Dev 重放与结论

这是历史 Dev **探索性子集重放**，并非独立验证。选择条件为唯一四选项匹配、Problem 片段为官方题目的精确子串、官方答案文本映射与 ATPO 键一致；这些条件用于来源一致性，并未使用模型输出。选择发生在原 Dev 结果已知之后，且仍无法证明被截短的问诊信息足以支持原题答案。共有 11 病例、31 状态、93 动作记录；从保存的冻结 Router 三个 seed 和原行动轨迹 CPU 重算，未训练或调用模型。仅用于检验原始结论对来源筛选的敏感性。

下面 Accuracy/Utility 是先病例内平均，再对病例和 Router seed 平均；ASK 成本 0.01，DEFER 成本仅扣一次。Oracle Conditional 需要事后 Base 正确性；Learned + Oracle 在 DEFER 时还用答案键事后保护 Base，二者均为不可部署上界。Always DEFER 指实际 Qwen Expert。保留/纠正及 DEFER precision 为状态和 seed 汇总条件比例；DEFER precision 定义为有用纠正/实际 DEFER 调用，而非 DEFER 回答自身正确率。Net Gain 是状态数纠正减退化，Learned 对 seed 取均值。

| 策略 | Accuracy | Utility c=.05 | Net Gain（状态数） | Base 保留 | Base 错误纠正 | DEFER precision | DEFER 提议率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Always ANSWER | .333 | .333 | 0 | 1.000 | .000 | 不适用 | .000 |
| Always ASK | .394 | .384 | 2 | .800 | .190 | 不适用 | .000 |
| Always DEFER（实际 Expert） | .470 | .420 | 4 | .500 | .429 | .290 | 1.000 |
| Oracle Conditional | 1.000 | .967 | 21 | 1.000 | 1.000 | 1.000 | .667 |
| Learned + 实际 Expert | .490 | .471 | 5 | .900 | .286 | .462 | .288 |
| Learned + Oracle 保护门 | .581 | .565 | 8 | .900 | .429 | 1.000 | .288 |

| DEFER 成本 | Always ANSWER | Always ASK | Always DEFER 实际 | Oracle Conditional | Learned 实际 | Learned Oracle 门 |
|---:|---:|---:|---:|---:|---:|---:|
| .05 | .333 | .384 | .420 | .967 | .471 | .565 |
| .10 | .333 | .384 | .370 | .933 | .456 | .554 |
| .20 | .333 | .384 | .270 | .867 | .428 | .533 |
| .60 | .333 | .384 | -.130 | .600 | .313 | .446 |

Learned + 实际 Expert 相对 Always ANSWER 的 case-cluster paired utility 差，成本 `.05/.10/.20/.60` 依次约为 `+.138 / +.123 / +.095 / -.020`，95% bootstrap CI 分别为 `[-.019,.324] / [-.033,.305] / [-.057,.268] / [-.168,.130]`，均含零。10,000 次重采样以 11 个 case_id 为单位；重复抽样不是新增病例，区间在小样本和事后子集下仅作探索。每个策略的保留率、纠正率、DEFER precision 及其条件比例区间、accuracy paired CI、逐 seed 指标与完整费用网格保存在 `data/historical_subset_replay_aggregate_v2.json`。Oracle 门的 precision=1 来自只在已知 Base 错误时执行，不能作为真实路由能力。

## 四项判断

1. 旧 Oracle 的答案键在 15 病例的候选官方选项映射上全部一致；11 例还有原题片段佐证，但两例候选不唯一、其余题目也经过改写/截短。其属于有证据支持的基准答案键，**不是已全面确证的完整病例 Gold 或医生认证**。
2. Learned Router + 真实 Expert 在旧 Dev 的低成本点估计高于 Always ANSWER，但 95% 区间含零；不能确认真正优于。`.60` 的点估计低于 Always ANSWER。
3. 新独立病例尚无冻结 Router、ASK 和 Expert 输出，独立有效 n=0；不能计算独立评估或宣称复现。
4. 尚不具备论文正式效果实验的充分证据。可报告工程实现、来源审计、开发集探索结果和明确的理论上界；需要冻结独立候选集、完成历史污染审计和预先注册测试、收集冻结策略实际预测，并按病例给出独立统计结果。

本轮没有正式 SFT、新 Router 拟合、Gold 合成或 PPO；没有打开原 400 例 Frozen Holdout。169 条历史不一致仍未确认。报告中使用过去已查看的 Dev 结果，不把它包装为新测试。
