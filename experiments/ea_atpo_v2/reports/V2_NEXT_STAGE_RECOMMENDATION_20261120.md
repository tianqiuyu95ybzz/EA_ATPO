# EA-ATPO V2 下一阶段建议（2026-11-20）

## 结论

1. **ASK Unknown 有可修复部分，但修复收益有限。** 原 23 条 Unknown 中 5 条已有可用事实，Patient 一次复核在全量 50 例中 111 次重试只接受 13 次；最终仍有 98/146（67.1%）ASK Unknown。
2. **当前训练样本出现 Expert 互补信号，但没有成本净收益证据。** 50 例的 146 个状态中，Expert 纠正 25、退化 18，准确率净增 4.8 个百分点。预设 DEFER 成本 0.60 下，DEFER 平均回报 -0.162，显著低于 ANSWER 的 0.390。该状态级样本来自训练病例，不能当作独立泛化结论。
3. **ASK 暂未显示净收益。** ASK 比 ANSWER 少答对 1 个状态，纠错 10、退化 11；平均回报差 -0.057。当前 Novel/非 Unknown 统计只是文本代理。
4. **数据接口现已齐备，但尚未训练模型。** Q Dataset 含 146 状态、438 个有效动作目标，成本只扣一次；每个状态/动作只有一次 rollout，目标噪声和模型复跑变化尚未充分估计。
5. **历史 200 例结果不与本轮横比。** 虽然模型 revision 与主要采样参数一致，提示、token 上限、seed、约束解码、parser 和病例集合不同。

## 推荐后续顺序

1. 先固定 V2 推理协议与模型服务 fingerprint；针对 20–30 个训练状态按不同 seed 重复 ASK / ANSWER / DEFER，量化相对 Q 标签的方差，不访问 validation、internal test 或 Frozen Holdout。
2. 检查并标注 3 个公开题干异常状态；保留原始轨迹，任何敏感性分析都同时报告“含/不含异常状态”，不覆盖主数据。
3. 在冻结 50 个训练病例上开展一次轻量 Q Regression 原型，按 `case_id` 训练/验证；使用既有 15 个 Validation 病例选阈值，15 个 Internal Test 病例只做一次冻结评估。汇报相对 ANSWER 的净收益、ASK/DEFER 频率、病例级置信区间和相对 Q 目标误差。
4. 只有当预决策特征能预测 Helpful/Harmful，且 Learned Policy 在未参与拟合的病例上稳定优于 Always Answer 与简单阈值后，才考虑 Actor 蒸馏或后续 RL。此阶段不启动 PPO/ATPO RL。

当前结论是：**可以进入小规模、离线、轻量 Q Regression 原型的准备阶段；尚不具备声称 Router 泛化有效或重启 RL 的证据。** 本轮没有执行 Q Regression、Actor 蒸馏、PPO 或 Expert SFT。
