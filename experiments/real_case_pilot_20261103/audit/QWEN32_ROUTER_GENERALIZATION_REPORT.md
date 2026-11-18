# Qwen3-32B-AWQ 专家优势 Router 泛化实验状态报告

**Attempt:** `qwen32_router_generalization_20261118_attempt04`

**日期：** 2026-11-18
**结果状态：** Attempt04 模型实验和一次性测试集分析已完成。下方更新的结果摘要和 [最终结果报告](QWEN32_ROUTER_GENERALIZATION_20261118_RESULTS.md) 为当前有效结论；较早的资源阻塞记录只说明实验启动前状态。

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

## 历史病例选择与资源阻塞

已完成全历史病例暴露审计和 1,700 例 Attempt04 队列冻结。扫描覆盖 84 个历史 JSONL/35,684 条记录、2 个旧 case-selection manifests 和 1 个旧 sample-selection manifest；已检查 case_id、prompt/case hash、源数据哈希与 source row index。Attempt04 清单 hash 为 `11d95c40d8c34bcbeae4dd8c18698fcee726f22db77b7f718d457d9527bc0cfa`。病例清单及历史排除 cache 保存在 PFS 的 Attempt04 `outputs` 目录，不上传 GitHub；仓库只保留不含病例 ID/hash 的选择脚本和协议文档。

Attempt02/03 在模型推理前分别因历史病例重叠和官方 test pool 剩余病例不足而作废。最初一次 GPU 快照也未达到安全门槛，故当时没有模型启动或推理；之后用户明确授权 GPU1/2/5，Attempt04 在 GPU1/5 完成。具体资源历史见 [资源记录](QWEN32_ROUTER_GENERALIZATION_RESOURCE_BLOCKER_20261118.md)。

## Attempt04 实际结果

- 20 例 smoke 通过，Base 和 Expert 解析失败均为 0；随后完成 1,700 例真实模型配对推理，train/validation/test 分别为 1,000/300/400，全部来自 ATPO train split 的全新 case_id holdout，不是官方 ATPO test split。
- Test：Base 48.5%；Qwen3-32B-AWQ DEFER 端到端 57.75%；配对差 +9.25 pp，病例级 95% CI [+4.49, +14.25] pp。Expert 修正 72/206 个 Base 错误；在 Base 正确病例中退化 35/194。
- Expert parser failure 为 66/1,700（train 38、validation 13、test 15）；全部正常结束但给出非 A-D 的答案文字，部署管线将其作为未答对，报告同时列出 parse coverage 和仅在有效解析回复上的条件准确率。
- 主成本 0.60 下 Logistic Relative Value Router 的 validation 选择为零调用，test 与 Always Answer 相同。Simple uncertainty threshold 在成本 0.10/0.20 的配对 Utility 相对 Always Answer 分别为 +0.0198 [+0.0070,+0.0348] 和 +0.0130 [+0.0025,+0.0250]；需用新病例复现。
- **研究决策：不启动 PPO/ATPO RL。** Expert 有平均配对互补性，简单 uncertainty policy 有成本条件下的初步信号，但 Learned Router 在预注册主成本没有可部署收益。Router AUROC、成本网格、配对置信区间、类别/特征分布和 MAE/MSE/Pearson 见 [最终结果报告](QWEN32_ROUTER_GENERALIZATION_20261118_RESULTS.md) 与 PFS 本地 analysis CSV。

首次分析器已写好一次性分析 CSV 和 manifest，但 Markdown 汇总遇到 `auc_by_model` 变量初始化顺序错误。已修复并仅从现有汇总 CSV 生成报告；没有重新训练、调阈值或重算 test 指标。PFS 中的 raw rollout、case-level CSV、日志和 GPU telemetry 未上传 GitHub。
