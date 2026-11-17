# Qwen3-32B-AWQ DEFER 验证计划（预注册）

**Attempt:** `qwen32_defer_validation_20261119_attempt01`
**计划状态：** 六阶段已执行完成；阶段决策为不启动 PPO/ATPO RL。预注册规则和阈值未因测试结果更改。
**目的：** 判断 Qwen3-32B-AWQ 在 DEFER 接管中是否与 Qwen3-8B Base 存在可预测、成本可接受的互补性。当前阶段不训练 PPO/ATPO RL。

## 六阶段安排

| 阶段 | 工作 | 通过条件 / 输出 | 本次结果 |
|---|---|---|
| 1. 环境、模型与数据锁定 | 固定 Qwen3-8B 与 Qwen3-32B-AWQ revision，核对本地权重及 tokenizer 哈希；冻结 train/validation/test 病例 ID | 运行清单、模型指纹、数据哈希和无病例重叠检查 | 完成；200/100/200 切分互斥，revision 与本地权重/tokenizer 哈希已保存 |
| 2. 协议 smoke | 在独立的 20 例开发集分别调用 Base ANSWER 与 Expert DEFER；保留 raw completion、token 数、finish reason、解析结果 | 20/20 格式有效；Base logprob 不确定性至少 16/20 可解析；否则停止 200 例实验 | 通过；Base/Expert 有效 20/20，不确定性有效 20/20 |
| 3. 200 例配对 DEFER | 同一公开病例分别由 Qwen3-8B ANSWER 与 Qwen3-32B-AWQ DEFER；gold 只在生成之后评分 | 总体准确率、Base/Expert 2×2 表、纠错率、正确退化率、病例级配对 bootstrap CI | 完成；Expert 55.5%，Base 46.0%，配对差 +9.5 pp（95% CI +2.5–+16.5） |
| 4. 选择性 DEFER 上界与简单基线 | 比较 Always Answer、Always DEFER、Random DEFER、Uncertainty DEFER 和 Oracle Selective DEFER | Oracle 只用于事后上界；不进入拟合、阈值选择或部署决策 | 完成；另报告随机策略单次固定种子结果和精确期望；Oracle 只作上界 |
| 5. 预决策可预测性、成本和风险—覆盖 | 训练 200 例、验证 100 例、独立测试 200 例；测试集只评估一次。比较线性相对价值 Router 与不确定性阈值 | 六档成本曲线、准确率/Utility/调用率/平均成本、病例级配对 CI、风险—覆盖和退化分析 | 完成；预决策预测未通过独立测试，主成本下风险筛选无收益；Learned 未可靠胜简单基线 |
| 6. 研究决策 | 汇总互补性、可预测性、Router 对简单基线的增益以及独立测试的不确定性 | 未通过预先定义的证据门槛，不启动 PPO/ATPO RL；只有另一个独立队列复现后才重新讨论 RL | No-Go：单一 cohort 未复现，且 Router 未可靠胜简单策略；暂停 RL |

## 固定病例与数据隔离

- 随机种子：`20261119`；Smoke 随机种子：`20261120`。
- Router 拟合集：ATPO train parquet 中 200 个病例；验证集：另外 100 个病例。
- 测试集：ATPO test parquet 中 200 个历史结果未使用病例。选样排除了已有直接回答/DEFER/CONSULT 轨迹中的公开状态哈希和已知测试行。
- 三个集合的 `case_id`、公开 prompt 哈希及公开状态哈希严格互斥；训练和验证病例也不进入 Attempt09 smoke 集。
- 原始 parquet 只读，不修改。输出存于独立的 ignored `outputs/qwen32_defer_validation_20261119_attempt01/` 目录。

## 主要比较与统计量

### 配对正确性

报告四格表：Base 对/Expert 对、Base 对/Expert 错、Base 错/Expert 对、Base 错/Expert 错。Expert 纠错率定义为 `Base 错且 Expert 对 / Base 错`；正确答案退化率定义为 `Base 对且 Expert 错 / Base 对`。准确率差、纠错率和退化率使用病例级配对 bootstrap（10,000 次，种子 `20261121`）。无效答案/解析失败按错误计入，不按有效样本筛选。

### 基线和 Oracle

- **Always Answer：** 每个病例使用 Base 答案。
- **Always DEFER：** 每个病例使用 Qwen3-32B-AWQ 的独立最终答案。
- **Random DEFER：** 使用 validation 上锁定的 Learned Router 调用预算，在测试病例上按固定种子 `20261122` 随机选同样数量；同时报告随机策略期望值。
- **Uncertainty DEFER：** 使用 Base 生成的答案选项 token logprob 熵；阈值在验证集按对应成本选取。
- **Learned Value Router：** Ridge 回归 `Expert correctness - Base correctness`，特征仅来自公开初始 prompt、Base 已生成的答案类别及其 logprob 不确定性。超参数和每档成本阈值均只用 train/validation 选择。
- **Oracle Selective DEFER：** 事后仅当 `Expert correctness - Base correctness > λ` 时选择 DEFER。由于看到了答案正确性，它是不可部署上界，禁止用于训练或任何测试时路由决策。

### 预注册成本

保留主设定 `λ=0.60`，并报告固定成本网格 `0, 0.05, 0.10, 0.20, 0.40, 0.60`。每档 Utility 定义为：

```text
Utility = final_accuracy - λ × expert_call_rate
mean_cost = λ × expert_call_rate
```

阈值在验证集按同一公式锁定；测试集不选成本、不调阈值。成本只扣一次。Accuracy、Utility、Expert 调用率、平均成本和 Token/延迟均按策略汇总。

### 风险—覆盖

`Base coverage` 定义为未转交给 Expert 的病例比例；同时报告保留 Base 案例的错误率和整个系统最终风险。对每个策略报告 Base 错误被转交的比例、Expert 修复错误的数量，以及被转交后从 Base 正确变成 Expert 错误的数量和比例。

## 资源与信息隔离

- 选定模型：Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`；Qwen3-32B-AWQ revision `0499c3ac83fdef8810b907a23894ba91e95eddd8`。
- 推理使用独立的 OpenAI-compatible vLLM 服务；Base 和 Expert 使用各自的角色提示与相同可见病例信息。Expert 只能依据公开病例独立给出最终选项。
- Gold 和 hidden atomic facts 不进入任何模型请求；它们仅在内存中用于请求视图校验及生成后的评分。
- 保留 raw model completion、prompt/completion token 数、finish reason、解析错误和延迟；不把 prompt、gold 或 hidden atomic facts 写入轨迹。
- GPU 2 共享部署；服务的 vLLM 显存比例分别限制为 0.25 和 0.12。缓存、临时文件和输出写入项目 PFS。

## 后续候选模型

`med_qwen`（ModelScope `zky001/med_qwen`）留在候选清单中。架构、基座、对话模板、配置和权重核验通过后，再以独立病例和独立报告评估；核验前不报告性能数值，也不与本次 Qwen3-32B-AWQ 结果混合。

## PPO/ATPO 重启门槛

本 Attempt 即使通过也不会自动启动 PPO。重新讨论 RL 至少需要：

1. 独立测试集 Base/Expert 净准确率差为正，配对 95% CI 下界高于 0，纠错与退化计数均披露；
2. 预决策特征在独立病例上能预测 Expert 优势；
3. Learned Router 相对 Uncertainty DEFER、Random DEFER 和 Always Answer 的 Utility 增益有配对证据，而不是只比 Oracle；
4. 上述互补性和 Router 增益在另一个独立病例队列复现；
5. med_qwen 若完成模型核验，作为单独专家候选报告，不改变本 Attempt 预注册规则。

## 执行结果补充（2026-11-18）

六阶段结果与具体统计见 [Qwen3-32B-AWQ DEFER 验证报告](QWEN32_DEFER_VALIDATION_RESULTS_20261119.md)，阶段决策见 [下一阶段决策](QWEN32_DEFER_VALIDATION_DECISION_20261119.md)。原始轨迹、模型/数据指纹、阈值锁、完整成本和风险 CSV 保存在共享 PFS 的 `outputs/qwen32_defer_validation_20261119_attempt01/`。最终分析版本为 `analysis_v4/`，与 `analysis/`、`analysis_v2/` 和 `analysis_v3/` 并存，没有覆盖历史输出。
