# EA-ATPO V3 三动作环境可行性审计

日期为 2026-11-28。先核对项目远端最新有效提交 `df31ad4f3530cfe7d04aa9005a311e69adde511b`。本轮只做官方资源、现有资产和接口审计。

**结论。** 两个作者仓库都提供了可复用的结构化患者环境。IMCS-21 的 DDP 包提供从 DISCOpen-MedBox 复用的规则患者，适合在许可及格式适配完成后做中文症状询问。DDXPlus 的官方 BASD 支持定病例初始化和逐证据转移，配合 `PATHOLOGY` 是最适合先做同状态三动作机制验证的资源。此次核实的三个数据源均未发现开箱即用的 EA-ATPO V3 ANSWER/ASK/DEFER 全流程。DDXPlus 所需增量是薄接口与评估端组合，无需重新构造患者、对话或 Gold。
[IMCS-21 论文第 5.5 节](https://arxiv.org/html/2204.08997#S5.SS5) · [DDXPlus 官方 BASD](https://github.com/mila-iqia/ddxplus/blob/69e8cf50a15ebb8979a57bb1f30ba5bb7f587f66/code/basd/env.py#L305-L408)

## 63 服务器与现有项目记录

- amax 用户工作目录下的 `EA_ATPO_V3/` 当前检出 `codex/ea-atpo-v3-migration-20261122`，HEAD 为 `f3fb006ebcfc464911d43520def073cbb95a0eda`。工作区状态为空。已保存的 `origin/codex/ea-atpo-v3-method-validation-20261128` 和远端 `ls-remote` 都返回 `df31ad4f3530cfe7d04aa9005a311e69adde511b`。审计从后者的独立本地分支开展，服务器原工作区保持在其恢复基线。
- 对用户工作目录做了限制深度的名称检索，核对项目内原登记的两个私有目录和 PFS 挂载。发现项目报告、运行脚本与模拟器原型，未定位到 IMCS-21 或 DDXPlus 原始包。登记的私有目录及旧 PFS 在本次检查中均不可见。其他位置是否尚有副本为 **UNKNOWN**。现场原始字节一致性为 **UNKNOWN**。
- 既有 GitHub 资料已经包含下载来源、IMCS-21 固定提交与文件指纹、DDXPlus v2 元数据指纹、旧数据准入审计以及 DDXPlus 50 例独立试验。历史英文 Validate 涉及 132,448 行，旧工程审计已记录官方 Train/Validate 精确重复。既有合成诊断原型与历史结果仍有效作为工程线索，不能视为本轮的官方 BASD 实测。
  [历史下载与 SHA-256](../gold_dataset_admission_20261121/SHA256_MANIFEST.json) · [既有实现及参数](../ddx_phase3_20261121/README.md) · [历史结果](../ddx_phase3_20261121/PHASE3_RESULT_REPORT.md) · [上轮项目版本](../method_validation_20261128/remedi_ask_answer_validation.md)

## 研究使用决策

1. **IMCS-21**。适合条件性 ASK 专项。作者代码能按索引复位病例并用结构化症状槽获取标注，但预处理格式不匹配，旧 manager 另有直接向 agent 传病种标签的接口。应复用患者端并限制公开字段。许可和临床标签来源仍有未知项。
2. **DDXPlus**。适合先做合成诊断三动作 Oracle 验证。使用同一病例和一致初态，ANSWER 和 ASK 后回答用同一个冻结 Base 与标签空间，DEFER 的 Gold 只在离线结束时可见。Base-Protecting 属事后上界。在精确标签评分下，它与 Standard Oracle 可能完全等效。不要把全部隐藏证据同时公开或将离线理想分数当真实专家能力。
3. **数据组合**。ReMeDi 保留中文历史行为与结构质量主线。DDXPlus 单独验证三动作机制。IMCS-21 在许可与适配过关后验证中文症状查询，当前两套专项数据无需同步启动。不同任务未共享评价尺度，因此不混合其 Q、准确率或临床质量结论。
4. **唯一下一步**。恢复已登记的 DDXPlus 英文 v2 最小验证包并核验 BASD 固定病例、可见初态、一证据 ASK 的边界。此后才进入受控同状态配对。此次未下载原始语料、未启动模拟器、未生成语料或 Gold、未用模型训练或推理、未使用树搜索、未读取 ReMeDi Test 或修改冻结数据。

## 交付

[IMCS-21 官方接口](IMCS21_AUDIT.md) · [DDXPlus 字段和 Oracle](DDXPLUS_AUDIT.md) · [许可与磁盘体量](DATA_LICENSES.md) · [三数据集对比矩阵](FEASIBILITY_MATRIX.md) · [下一步验收](NEXT_STEP.md)
