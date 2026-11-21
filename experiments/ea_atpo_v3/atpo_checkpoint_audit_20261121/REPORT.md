# 真正 ATPO checkpoint 与既有 DDXPlus Pilot 审计

## 结论及模型来源

**未找到可证实经过官方数据 SFT、然后 ATPO 训练的 Qwen3-8B 最终 checkpoint。** 本轮没有用其他模型替代，也没有开展所要求的正确模型直接诊断或 8 轮交互实验；对应 Accuracy 均为不可得。不能以本结果回答该模型有多少诊断能力或启动正式 Router 训练。

| 环节 | 本地证据 | 判断 |
|---|---|---|
| 官方 SFT 数据 | `EA_ATPO_V1/ATPO-official/dataset/sft_training_data.jsonl` SHA-256 `772585c937180ab625833d220b972e89c41945cd20a824a8c2a3d50f9fc3be73`；与 [官方文件](https://github.com/Quark-Medical/ATPO/blob/af9a0d2000f3a305dffbb75f3a966eb018582e03/dataset/sft_training_data.jsonl) Git blob `8928c44624e26527f930f4e1bb8e49a269272dcc` 一致 | 数据来源核验通过，不能因此证明任何模型使用它训练 |
| 8B SFT checkpoint | 本地项目和已有记录未定位训练配置、日志及权重组成 | 缺失 |
| 8B ATPO 后续训练 | Original ATPO Attempt20 日志记录从原始 Qwen3-8B 路径直接初始化、LoRA rank=0、完成两步更新；该 Actor 已有 4 分片 HF 导出 | 有 ATPO 两步工程 smoke，但无 SFT 前置，**不是本轮指定模型** |
| 历史 SFT→ATPO | 报告描述 Qwen3-4B 的 SFT 和 20 步 ATPO，旧路径 `/kwkj-k8s/...` 当前未挂载 | 历史文字线索；非 8B，权重及原始日志不可核实，不能替代 |
| 先前 DDXPlus 推理 | `ddx_phase3_20261121/run_cpu_base.py` 中 `MODEL` 指向原始 Qwen3-8B、直接加载 AutoModelForCausalLM，无 Adapter；实际初始模型 5 个权重分片的 SHA 与项目锁一致 | 明确不是指定 checkpoint |

用户所要求的 Base 理应是 SFT 权重 + 后续训练结果；当前既无两阶段 8B 日志/权重，也无能确认正确装载 Adapter 的材料。`MODEL_SHA256SUMS` 包含旧 Pilot 原始 Base 与不合格候选 Attempt20 的分片和配置/词表摘要，不将后者误称最终模型。先前 tokenizer 与模型快照同时加载，使用模型原生 chat template，关闭 thinking；由于目标训练时 tokenizer 和 template 不存在，无法确认与目标训练一致。所有权重未上传。

## 旧 18% Pilot 的 CPU 复核

50/50 的行定位符和 Gold 与官方 validation 原始文件一致；旧 Base 9/50、18% 严格准确率重现，旧一次 ASK 2/50、4% 重现。50/50 输出非空（这是原始输出存在率，不是医学有效率），49/50 精确匹配 49 个官方名称。49 个官方名称逐个作为输入的解析单元测试全部通过；模型输出的**语义解析准确率无法确定**，需要独立语义标注，不应把 49/50 格式命中率称为解析准确率。唯一未匹配输出及所有逐病例原文在 `private/invalid_outputs.json` 和 `private/baseline_results.csv`；其是未列在官方 49 名称中的病名，是否同义需要独立判定，不改变原严格计分。

按 49 个官方类（未出现类 F1 按 0）macro-F1 为 **0.0989**；只对样本有支持的类 macro-F1 为 **0.1730**。每类真实/预测数在 `classification_report.json`，49×(49+不可解析) 聚合混淆矩阵在 `confusion_matrix.csv`。两者为汇总，非患者原始文本。50 条存档 ASK 重放的源状态、患者模拟器反馈和轨迹字段与保存版本一致；这只是存档一致性，**不是**对官方依赖规则的独立正确性验证。5 个依赖的根未在元数据出现，旧模拟器无法证明完整覆盖。原始直接诊断 prompt 仅 user 消息，列全部病种并要求单一名称；非 ATPO 训练的动作格式，24 token 限制也可能影响格式。旧 ASK 是脚本固定询问并重建可见状态后再次诊断，模型没有自主产生多轮问题，不能证明 ATPO 多轮能力。未找到病种映射错误或 Gold 进入 prompt 的证据；仅检查了现有脚本和逐条记录，并非排除所有实现问题。

## Oracle 与独立性

旧实验的 Gold 与 DDXPlus 官方 validation 行及 49 类条件元数据逐条匹配。Oracle A 和 B 对 50/50 返回正确官方标签，是由定义直接得出的合成病种分类上界，不代表医学真值或 Router 成绩。已知官方 train/validation 有 1,625 种相同完整行，且没有可靠患者 ID；样本级、患者级独立性未证实。DDXPlus 是合成数据，不应称为真实临床患者实验。

## 尚缺及下一步

需提供或定位 Qwen3-8B SFT 训练输出（数据版本、完整配置、日志、Base/Adapter 信息及哈希），以及从该输出继续训练的 ATPO 最终 checkpoint、训练日志、tokenizer/template 和加载规则；核实训练链后，才能固定原 50 行和 Gold、运行 A/B、产生模型真实输出并评价。如果仅有旧 4B 或两步未 SFT 的 8B，不符合本轮约束。当前 GPU 全部有其他计算任务；未中断任何进程，未训练、未用其他权重推理、未读取 400 例 Frozen Holdout 或 DDXPlus Test。之前 Pilot 保留。
