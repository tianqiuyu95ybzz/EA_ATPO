# EA-ATPO：专家条件互补性与成本可行性分析

更新日期：2026-11-17

## 结论摘要

- **目前没有通过独立测试的可部署 DEFER 路由器，也没有稳定获益的 DEFER 专家。** Qwen3-32B-AWQ 在既有 200 例上的**直接作答**有 +4.0 个百分点点估计，但病例配对 95% CI 为 −2.5 到 +10.5 个百分点；这不是 DEFER 结果。
- ReMeDi 和 PopiT 在 Base 错误病例中能改对一部分答案，但也会把更多或相近数量的 Base 正确答案改错。两个独立 100 例 DEFER 集合保持分开后，没有稳定净准确率提升。
- 事后 Oracle 显示有潜在选择空间，但它使用病例正确性挑选调用对象，不能作为可部署 Router 成绩。当前没有 Qwen3-32B 的 DEFER 结果，也没有用病例隔离训练/验证/测试完成的预决策 Router 结果。
- 最新 GPU 检查时 GPU3 空闲显存 10,513 MiB、利用率 100%；GPU7 空闲显存 66,540 MiB、利用率 100%。两者都未通过本实验的启动门槛。没有启动服务、抢占进程或启动 RL。

## 数据配对与统计口径

分别重算五组已有病例级结果，不把不同样本集合的准确率拼成单个估计。病例级 bootstrap 在每个来源 cohort 内以病例为单位重采样 10,000 次。四格表顺序为 Base 对/Expert 对、Base 对/Expert 错、Base 错/Expert 对、Base 错/Expert 错。无效/解析失败输出留在分母并计错。配对净准确率、Base 错误条件纠错率、Base 正确条件退化率的 bootstrap CI 均逐专家列在病例汇总 CSV 中（`delta_ci_*`、`correction_ci_*`、`degradation_ci_*`）。

两个 200 例直接评估共享相同的题目和提示，但其 Qwen3-8B 冻结输出相差一个病例：医疗候选筛选中为 87/200，32B 官方配对评估清单指定的 Base 结果为 88/200。为避免跨运行拼接，本报告将两个冻结比较分开，并在 32B 比较中沿用其原 manifest 指定的配对 Base 轨迹。

## 直接作答互补性

| 来源 cohort / Expert | n | Base→Expert 四格计数 | Base 准确率 | Expert 准确率 | 净差（病例配对 95% CI） | Base 错时纠错率 | Base 对时退化率 |
|---|---:|---|---:|---:|---:|---:|---:|
| 候选筛选 200 / HuatuoGPT-o1-7B | 200 | 13 / 74 / 10 / 103 | 43.5% | 11.5% | −32.0 pp [−40.0, −24.0] | 10/113 = 8.8% | 74/87 = 85.1% |
| 候选筛选 200 / Qwen3-4B-ReMeDi | 200 | 61 / 26 / 16 / 97 | 43.5% | 38.5% | −5.0 pp [−11.5, +1.0] | 16/113 = 14.2% | 26/87 = 29.9% |
| 候选筛选 200 / PopiT-Qwen3-4B | 200 | 58 / 29 / 20 / 93 | 43.5% | 39.0% | −4.5 pp [−11.5, +2.5] | 20/113 = 17.7% | 29/87 = 33.3% |
| 冻结 32B 配对 200 / Qwen3-32B-AWQ 直接回答 | 200 | 70 / 18 / 26 / 86 | 44.0% | 48.0% | +4.0 pp [−2.5, +10.5] | 26/112 = 23.2% | 18/88 = 20.5% |

32B 是直接回答中互补性最有希望的候选：26 个 Base 错误被改对、18 个 Base 正确被改错。但区间仍覆盖零，且上述比较没有使用 DEFER system prompt。因此现在只能称为**待验证的条件互补性信号**，不能称为 DEFER 优势。

## DEFER 配对结果

下表继续按来源 cohort 分开。20 例为开发集探索；两组 100 例是不同测试抽样。

| Cohort / Expert | n | Base→Expert 四格计数 | 净差（病例配对 95% CI） | Base 错时纠错率 | Base 对时退化率 | 有效输出率 |
|---|---:|---|---:|---:|---:|---:|
| 开发集 20 / Qwen3-8B DEFER prompt | 20 | 4 / 1 / 1 / 14 | 0.0 pp [−15, +15] | 1/15 = 6.7% | 1/5 = 20% | 100% |
| 开发集 20 / ReMeDi | 20 | 4 / 1 / 2 / 13 | +5.0 pp [−10, +25] | 2/15 = 13.3% | 1/5 = 20% | 100% |
| 开发集 20 / PopiT | 20 | 2 / 3 / 1 / 14 | −10.0 pp [−30, +10] | 1/15 = 6.7% | 3/5 = 60% | 100% |
| 开发集 20 / HuatuoGPT-o1 | 20 | 0 / 5 / 0 / 15 | −25.0 pp [−45, −10] | 0/15 | 5/5 = 100% | 0% |
| 独立确认 100 / Qwen3-8B DEFER prompt | 100 | 51 / 3 / 2 / 44 | −1.0 pp [−6, +3] | 2/46 = 4.3% | 3/54 = 5.6% | 100% |
| 独立确认 100 / ReMeDi | 100 | 35 / 19 / 5 / 41 | −14.0 pp [−23, −5] | 5/46 = 10.9% | 19/54 = 35.2% | 100% |
| 候选筛选 100 / Qwen3-8B DEFER prompt | 100 | 40 / 1 / 4 / 55 | +3.0 pp [−1, +8] | 4/59 = 6.8% | 1/41 = 2.4% | 99% |
| 候选筛选 100 / ReMeDi | 100 | 30 / 11 / 11 / 48 | 0.0 pp [−9, +9] | 11/59 = 18.6% | 11/41 = 26.8% | 100% |
| 候选筛选 100 / PopiT | 100 | 27 / 14 / 13 / 46 | −1.0 pp [−11, +9] | 13/59 = 22.0% | 14/41 = 34.1% | 99% |
| 候选筛选 100 / HuatuoGPT-o1 | 100 | 0 / 41 / 0 / 59 | −41.0 pp [−51, −32] | 0/59 | 41/41 = 100% | 0% |

ReMeDi 在 20 例开发集的 +1 例点估计没有复现：第一个独立 100 例 cohort 为 −14 pp，第二个独立 100 例 cohort 为 0 pp。PopiT 在第二个 cohort 为 −1 pp。Huatuo 的 DEFER 解析输出全部无效，不能把 0% 有效率解释为其医学知识能力估计。

## 成本敏感性与同预算上界

所有成本均为归一化效用单位，不是美元或医生收费。统一使用 `U = accuracy − λ × expert_call_rate`。DEFER 网格为 `λ={0,.15,.30,.45,.60,.75,.90,1.00}`；CONSULT 网格为 `λ={0,.05,.10,.20,.40,.60}`。Attempt09 原设定是 `λ_DEFER=.60`、`λ_CONSULT=.20`。每个动作只扣一次成本。

始终调用 Expert 的盈亏平衡条件为 `λ < p(Expert correct) − p(Base correct)`。已观测到的直接作答中，只有 Qwen3-32B-AWQ 的点估计给出正阈值（λ < .04），但它的配对区间跨零且不是 DEFER；ReMeDi/PopiT 直接回答和两组 ReMeDi/PopiT DEFER 的点估计均没有正的始终调用成本区间。20 例 ReMeDi 开发集的阈值点估计为 λ < .05，区间很宽，不能作为成本选择依据。

在原始 `λ_DEFER=.60` 下：

| 独立 cohort / Expert | Always Answer 准确率 / utility | Always Expert 准确率 / utility | 基于已观测输出的事后选择：调用率 / utility | 完美选择性 Oracle：调用率 / utility |
|---|---:|---:|---:|---:|
| 冻结 32B 直接回答 200 | 44.0% / .440 | 48.0% / −.120 | 13.0% / .492 | 56.0% / .664 |
| ReMeDi DEFER 确认 100 | 54.0% / .540 | 40.0% / −.200 | 5.0% / .560 | 46.0% / .724 |
| ReMeDi DEFER 候选筛选 100 | 41.0% / .410 | 41.0% / −.190 | 11.0% / .454 | 59.0% / .646 |
| PopiT DEFER 候选筛选 100 | 41.0% / .410 | 40.0% / −.200 | 13.0% / .462 | 59.0% / .646 |

“基于已观测输出的事后选择”仅在 `Base 错且已观测 Expert 对` 时调用，因而使用 Gold 事后正确性，**只是该 Expert 输出下的 Oracle 上界，不是路由器结果**。例如 32B 直接回答的 26 个纠错案例构成 13% 调用预算；如果随机调用同样比例，期望准确率为 44.52%、效用为 .367，而事后 Oracle 的准确率/效用为 57.0%/.492。两者差距说明价值取决于能否在决策前识别互补病例，不能用 Oracle 结果代替这项验证。

“完美选择性 Oracle”更强：假设知道 Base 哪些病例会错，且有完美 Expert，只在这些病例上 DEFER。它的调用率是 Base 错误率，`U=1−λ(1−p_Base)`。它在 λ=.60 时仍高于 Always Answer，和“完美 Expert 始终接管”不同；后者的效用是 `1−λ=.40`。二者均是理论上界，不是系统成绩。

所有成本曲线及其病例 bootstrap 区间见 [expert_cost_sensitivity.csv](experiments/real_case_pilot_20261103/audit/results/expert_complementarity_20261117/expert_cost_sensitivity.csv)。该文件列出成本、准确率、调用率、成本、净效用、始终调用、同预算随机调用、事后 Oracle 和完美选择性 Oracle。

## 预决策 Router：可行性尚未测得

当前 32B 结果只有既有测试集上的直接作答；没有该模型 DEFER 的 train/validation 结果，也没有可直接用于它的决策前不确定性字段。不能用现有 200 例测试集拟合规则再在同一批病例上报告性能。

已准备的独立采集协议是：ATPO train split 中 200 例训练、另 100 例验证，用验证集选择阈值；冻结的 200 例 test split 只做一次最终评估。预检得到 500 个唯一公开 prompt hash，train/validation 与 test 重叠为 0。训练目标为 `correct(32B-DEFER) − correct(Base-ANSWER)`，决策输入仅为初始公开 prompt 的长度、选项结构、标点/数字和固定词表计数。没有 Gold、隐藏 atomic facts、Expert 输出或事后结果作为 Router 输入。

不过 GPU 资源在服务启动前降到安全门槛以下，上述模型推理、阈值选择和独立 Router 测试**均未运行**。因此目前无法回答预决策特征能否预测专家优势。当前仅有一个轻量 Ridge 复现脚本作为待执行基线；若它无效，也不能据此断言校准后的 Base logits 一定无效。

## CONSULT：保留探索性边界

### 20 例 CONSULT 小提示

20 例探索中，固定 CONSULT 问题有效率 100%；这只是模板可解析，不代表问题质量。Base 生成提示的格式有效率为 85%，CONSULT 后准确率 45%（10/20 为 50% 的 ANSWER），配对差 −5 pp（95% CI −25 至 +15）；Base 错误中纠正 2/10，Base 正确中退化 3/10。ReMeDi、PopiT 提示有效率分别 30%、50%，最终准确率 15%、20%。无效提示分支成本可能被低估；这组结果不足以证明 CONSULT 无效，也不支持扩大同一提示/模型方案。

### 既有 200 状态 Advice Utilization 对照

同一批 200 个公开状态的 `CONSULT-Critical` 为 44.0%，原 Pilot ANSWER 为 43.5%，配对差 +0.5 pp（病例聚类 95% CI −5.9 至 +6.8）；错误修正率 13.3%（15/113），正确退化率 16.1%（14/87）。最终答案与原 ANSWER 不同的比例为 18%（36/200），只是答案变化的操作性代理，不能证明模型语义上采纳了建议。完整 CONSULT 分支平均约 1,756 tokens，对比 ANSWER 550 tokens；在 λ_CONSULT=.20 下效用为 .24，对比 ANSWER .435。Naive/Retain 准确率为 42%/40%。这些是 LLM 模拟专家结果，不是真人医生评价。

另有 150 条由同系列 Qwen3-8B 作为 Judge 的重叠探索性标注：20% 被标为建议正确且最终答对，33.3% 的咨询问题被判不清楚，6% 的建议与 Gold 冲突，2% 重复公开信息。它们不是独立医生标注，也不能作为在线 Router 特征。

**阶段决定：** 暂不扩大同样的 CONSULT 小提示实验。若后续需要继续，先把咨询问句改成可由公开病例回答、临床上可执行的问题，并用独立或医生标注检查建议质量，再启动一个预注册的较大配对样本；目前没有证据把 CONSULT 判为普遍无效。

## 明确回答与下一步

1. **哪个 Expert 有条件互补性？** Qwen3-32B-AWQ 直接作答显示最强的点估计信号，但统计不确定且尚未测试 DEFER。ReMeDi/PopiT 有个别纠错病例，但尚无预决策可识别的可利用子集。当前没有确认可部署的互补 Expert。
2. **哪些成本区间支持 DEFER？** 始终调用只在 `λ < Δ accuracy` 时可能优于 ANSWER。已测 DEFER 候选没有稳定的正阈值；Qwen3-32B 的 +.04 仅对应直接作答，不可转移到 DEFER。候选实际输出的事后选择 Oracle 在 λ=.60 有少量正上界，但不能用于部署结论。
3. **预决策 Router 是否预测到专家优势？** 未知；train/validation/test 专家标签和路由评估因 GPU 安全门槛未完成。
4. **CONSULT 是否值得扩大？** 当前不扩大同一模拟提示方案；20 例仍为探索性，不能宣告无效。优先修复问题清晰度和提示有效率。
5. **是否具备重启 EA-ATPO PPO 的前提？** 否。先拿到可复现 32B DEFER 和/或 med_qwen 权重，完成 200/100/200 病例隔离的 Router 验证；在验证集选阈值后，独立测试必须表现出成本下正净收益和可靠纠错，再讨论 PPO。

## 交付与复现

- 配对统计、病例级匿名结果、成本曲线、CONSULT 20 例匿名结果、哈希清单：[`results/expert_complementarity_20261117/`](experiments/real_case_pilot_20261103/audit/results/expert_complementarity_20261117/)。
- 离线重算脚本：[analyze_existing_expert_complementarity.py](experiments/real_case_pilot_20261103/audit/analyze_existing_expert_complementarity.py)。命令：`python3 experiments/real_case_pilot_20261103/audit/analyze_existing_expert_complementarity.py`。
- 32B DEFER 与训练/验证数据采集脚本：[run_qwen32_defer_complementarity.py](experiments/real_case_pilot_20261103/audit/run_qwen32_defer_complementarity.py)；安全启动：[run_qwen32_defer_complementarity.sh](experiments/real_case_pilot_20261103/audit/run_qwen32_defer_complementarity.sh)。
- 轻量 Router 拟合/验证阈值/独立测试脚本：[fit_public_feature_router.py](experiments/real_case_pilot_20261103/audit/fit_public_feature_router.py)。上述新模型脚本没有生成推理结果或 checkpoint；GPU 资源尝试状态见 [`qwen32_defer_resource_attempt.json`](experiments/real_case_pilot_20261103/audit/results/expert_complementarity_20261117/qwen32_defer_resource_attempt.json)。
- `med_qwen` 模型元数据与文件可用性核验：[MED_QWEN_EXPERT_EVALUATION.md](MED_QWEN_EXPERT_EVALUATION.md) 和 [`med_qwen_modelscope_verification.json`](experiments/real_case_pilot_20261103/audit/results/expert_complementarity_20261117/med_qwen_modelscope_verification.json)。
