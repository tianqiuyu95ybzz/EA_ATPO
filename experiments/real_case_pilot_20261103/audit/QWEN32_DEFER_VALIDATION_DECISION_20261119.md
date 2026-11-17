# Qwen3-32B-AWQ DEFER 验证阶段决策

日期：2026-11-18
依据：[配对验证报告](QWEN32_DEFER_VALIDATION_RESULTS_20261119.md)及其冻结的 200 例测试结果。

## 决定

**当前不启动 PPO/ATPO RL，也不把该 Learned Router 作为可部署策略。** 专家与 Base 在本 cohort 上有统计上为正的配对准确率差，但决策前可见特征未能稳定识别专家优势，且 Learned Router 在预注册成本网格中没有可靠超过简单基线。重启 RL 的门槛是这些条件同时成立；本轮未达到。

## 逐项判断

1. **专家是否互补？有初步证据，尚未证明可重复。** 200 例中 Qwen3-32B-AWQ DEFER 为 55.5%，Qwen3-8B Base 为 46.0%，配对差 +9.5 pp（95% CI +2.5–+16.5）。Expert 修正 35/108 个 Base 错误，同时破坏 16/92 个 Base 正确答案。只有一个新测试 cohort，不能称为重复验证。
2. **决策前特征能否预测专家优势？目前不能可靠预测。** 预决策 Ridge Router 的独立测试 Pearson r=0.054（95% CI −0.084–0.194），纠错 AUC=0.596（95% CI 0.489–0.697），区间包含无预测能力水平。
3. **Learned Router 是否可靠优于简单基线？没有。** λ=0 时它虽比 Always Answer 高 +0.070 Utility（95% CI +0.015–+0.125），但对 Uncertainty DEFER 的差异 +0.030（−0.020–+0.080）及 Random 期望的 +0.022（−0.014–+0.056）均不显著。λ=0.60 时低于 Always Answer 与 Uncertainty DEFER，各为 −0.017（95% CI −0.039–−0.003）。
4. **主成本下的选择性 DEFER 是否降低风险？没有。** λ=0.60 下调用率 2%，4 次接管没有修正 Base 错误且有 1 次正确退化；准确率从 46.0% 降为 45.5%。
5. **Oracle 是否说明有可部署收益？不说明。** λ=0.60 Oracle 上界为 63.5% Accuracy、0.530 Utility，但它用 Gold 事后选择专家调用，不属于模型能力或可部署策略。
6. **med_qwen 是否已评估？没有。** 保留在候选表中，等模型架构、基座、权重、模板和推理配置完整核验后另立实验；当前不报告任何性能值。

## 后续顺序

1. 将本报告、运行清单、权重指纹、阈值锁和原始输出归档；不改当前成本、提示、checkpoint 或测试标签。
2. 如继续确认互补性，先预注册另一个未使用病例 cohort，固定模型 revision、解析器、提示和 λ=0.60 主终点。将本次 Learned Router 完全冻结，用新 cohort 只做外部验证；不得在当前测试集或新测试集调阈值。
3. 只有当独立 cohort 再次显示正的 Base/Expert 配对增益，且冻结 Learned Router 相对 Always Answer、Uncertainty 和 Random 的成本效用优势有病例级配对证据，才进入 Router 方法迭代讨论。
4. 在上一步成立前，不启动 PPO/ATPO RL。若新 cohort 仍不能预测专家优势，则停止 Learned Router 主线，把结果作为“专家准确率提高但选择性路由暂不可学”的负面研究结论。
5. med_qwen 通过资产核验后单独评估；不与本次 Qwen3-32B-AWQ 测试结果合并。

没有真实医生参与本实验。这里的 “Expert” 始终指 Qwen3-32B-AWQ 语言模型模拟器。
