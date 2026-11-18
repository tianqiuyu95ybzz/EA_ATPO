# Qwen3-32B-AWQ 专家优势 Router 泛化实验状态报告

**Attempt:** `qwen32_router_generalization_20261118_attempt04`

**日期：** 2026-11-18
**结果状态：** 预注册与病例划分已完成；模型实验因 GPU 资源门槛未执行。本报告不是性能结果报告。

## 研究问题与历史基线

历史的 200 例配对 DEFER 评估中，Qwen3-32B-AWQ 准确率为 55.5%，Qwen3-8B Base 为 46.0%；专家纠正 35/108 个 Base 错误，退化 16/92 个 Base 正确答案。此前 Learned Router 的优势识别 AUROC 为 0.596，独立测试准确率 45.5%，低于 Always Answer 的 46.0%。这些数据只用于提出本次泛化问题，不能代替新病例实验。

## 本次冻结设计

- 固定 Expert：`Qwen/Qwen3-32B-AWQ`，revision `0499c3ac83fdef8810b907a23894ba91e95eddd8`。
- 固定 Base：`Qwen/Qwen3-8B`，revision `b968826d9c46dd6066d109eabc6255188de91218`。
- 复用此前锁定的公开病例 prompt、解析器和推理设置：temperature 0、top-p 1、关闭 thinking、Base 最多 64 tokens、Expert 最多 128 tokens；Base 选项 logprobs top-20。
- Attempt02 的首版清单出现 34/400 官方 test source-row 历史重叠；Attempt03 完整排除后官方 test parquet 仅剩 8 个可用新病例。因此两个清单都在任何模型调用前作废。
- Attempt04 从 train split 的历史未暴露案例冻结 1,000 train /300 validation /400 test-source holdout，随机种子 `20261128`。这 400 例是 case_id 级独立 holdout，但仍来自 ATPO train split；结果不能称为官方 test split 的性能。集合按 `case_id`、case hash、prompt hash 隔离，并与历史 hash、历史 train source index 去重。
- 决策特征限定为公开病例长度/选项统计、Base 答案、Base 选项 logprob 熵/间隔和解析有效性。Gold、Expert 答案和事后正确性不进入 Router 特征。
- 计划比较不确定性阈值、Logistic Regression、小型单隐层 MLP、历史冻结 Router 及新配对相对收益 Router；监督目标包括专家纠错事件、相对收益 `expert_correct - base_correct`，以及扣除一次 DEFER 成本后的净优势。
- 成本网格预注册为 `0, 0.05, 0.10, 0.20, 0.40, 0.60`。阈值只由验证集确定；Oracle Selective DEFER 仅作为不可部署的事后上界。

完整方法与指标预注册见 [实验计划](QWEN32_ROUTER_GENERALIZATION_PLAN_20261118.md)。

## 本阶段实际完成和未完成

已完成全历史病例暴露审计和 1,700 例 Attempt04 队列冻结。扫描覆盖 84 个历史 JSONL/35,684 条记录、2 个旧 case-selection manifests 和 1 个旧 sample-selection manifest；已检查 case_id、prompt/case hash、源数据哈希与 source row index。Attempt04 清单 hash 为 `11d95c40d8c34bcbeae4dd8c18698fcee726f22db77b7f718d457d9527bc0cfa`。病例清单及历史排除 cache 保存在 PFS 的 Attempt04 `outputs` 目录，不上传 GitHub；仓库只保留不含病例 ID/hash 的选择脚本和协议文档。

最近的单次启动检查显示 GPU3 仅有 55.5 GiB 空闲、利用率 100%，未达到 Expert 门槛；启动脚本在加载模型前退出。前一快照的 GPU2/3/6/7 分别只有 111.8/56.9/56.4/56.3 GiB 空闲且均 100% 利用率。没有启动模型、改变模型配置或影响其他用户的进程。

因此，本次没有任何新的推理轨迹、模型拟合、测试集读取或策略性能统计。要求中的 Accuracy、纠错率、退化率、AUROC、成本/Utility、病例级置信区间和 paired case result 均尚无新实验数值。不会用历史数据、人工标签或概率模拟补齐。已准备采集、验证集拟合、测试集锁定分析和资源门槛脚本；三例真实病例的 case/prompt hash 构造与冻结清单相符，未调用模型。

资源与恢复条件详见 [GPU 资源阻塞记录](QWEN32_ROUTER_GENERALIZATION_RESOURCE_BLOCKER_20261118.md)。

## 当前结论

基于本阶段数据，无法判断 Router 是否能在独立病例上学习稳定的专家条件优势；不启动 PPO/ATPO RL。下一步必须在固定队列上完成真实 Base/Expert 配对推理，然后按锁定 train/validation/test 流程训练、选阈值并一次性评价测试集。
