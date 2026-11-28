# 下一阶段唯一优先任务

**从已登记的受控备份恢复 DDXPlus 英文 v2 的 Validate ZIP 和两个元数据 JSON，在 amax 上对官方 BASD 做一病例、两次重置、一次 ASK 的接口验收。** 先核对官方版本和历史 SHA-256。此前已经保存过 v2 数据，优先查找经授权的既有副本。若无法恢复，记录具体阻塞，再决定是否批准获取官方 Validate 最小包。无须恢复整个语料或访问 Test。

按 `initialize_state(indices=[i])` 分别重置两个独立实例，启用发布的初始证据并断言公开 s0 相同。对其中一个实例只提交一个未揭示、依赖满足的证据码，记录投影差分。检查二元未问和已确认阴性的区别、多选单证据的值集合、其他证据保持不变，以及病种、差分、严重度、完整 target 和环境 reward 未进入 Base 可见状态。此验收需要在官方运行代码的使用范围获确认后进行。保持上游源码隔离，公共仓库仅记录接口、测试结果和项目独立代码。
[官方固定病例初始化](https://github.com/mila-iqia/ddxplus/blob/69e8cf50a15ebb8979a57bb1f30ba5bb7f587f66/code/basd/env.py#L305-L391) · [官方单证据 step](https://github.com/mila-iqia/ddxplus/blob/69e8cf50a15ebb8979a57bb1f30ba5bb7f587f66/code/basd/env.py#L394-L408) · [现有资产指纹](../gold_dataset_admission_20261121/SHA256_MANIFEST.json)

验收通过后，最小实现仅为调用现有 BASD 的受控公开状态适配，以及在评估端接既有 ANSWER、单步 ASK、Standard/Protecting Oracle 规则。沿用已核验的原始 Qwen3-8B、同 s0、同词表、同质量定义、同成本口径。每分支先覆盖一次再考虑树剪枝。只有同数据、同状态、同评价端点的收益可归入该数据集的 Q；有缺失动作的记录继续掩码。DDXPlus 的合成诊断精确标签收益、IMCS-21 的症状获取或十类准确率、ReMeDi 的槽或实体一致性分别报告。时间或成本可用共同计量字段，但不能消除任务质量量纲差异。

推荐组合是 **ReMeDi 作为中文行为与回复结构主线，DDXPlus 作为主要三动作机制验证，IMCS-21 作为可选中文 ASK 专项**。完成首个三动作机制验证无需同时使用 IMCS-21。IMCS-21 的后续准入取决于许可及字段适配。本次审计后，三动作树数据生产和 Q-guided 训练仍未准入。
[分工与状态矩阵](FEASIBILITY_MATRIX.md)
