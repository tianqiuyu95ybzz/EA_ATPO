# med_qwen Expert 候选核验与评估

更新日期：2026-11-17

## 当前结论

`zky001/med_qwen` 是应保留在 EA-ATPO 候选清单中的公开医疗问答模型，但当前无法取得可加载的 checkpoint，因此**没有对它运行任何推理**。没有 Accuracy、纠错率、退化率或置信区间；不能把缺失实验记成 0%，也不能声称它比 Qwen3-32B-AWQ 好或差。

## ModelScope 元数据与可用性

已检查 [ModelScope 模型页](https://www.modelscope.cn/models/zky001/med_qwen) 与[官方模型元数据 API](https://modelscope.cn/api/v1/models/zky001/med_qwen)。API 返回：

- 架构标记为 `Qwen2ForCausalLM`，模型展示名为“千问医疗问答模型”。
- `BaseModel`、`Backbone`、`BaseModelRelation` 字段为空；模型卡没有说明具体基座版本、微调流程或训练数据。因此本报告不把它写成已核实的 Qwen/Qwen2 某个版本 SFT 模型。
- License 与 LicenseName 为空，许可条件需在获取权重前另行确认。
- 元数据给出 8 个 safetensors 分片，总计 15,442,693,432 bytes，另列 index 文件；SHA-256 元数据已经记录在 [`med_qwen_modelscope_verification.json`](experiments/real_case_pilot_20261103/audit/results/expert_complementarity_20261117/med_qwen_modelscope_verification.json)。
- 对话模板使用 `Human: ...\nAssistant: ...`，Assistant 结束标记为 `<|im_end|>`；它与当前 Qwen3 ChatML 模板不同，不能直接沿用 Qwen3 tokenizer 配置。

虽然模型元数据端点可访问，[模型文件树接口](https://modelscope.cn/api/v1/models/zky001/med_qwen/repo/files?Revision=master&Recursive=True) 返回 HTTP 404 / Code `10010200001`（获取模型失败）。ModelScope SDK 的 `snapshot_download` 也因该模型文件树请求失败，没有生成本地权重。当前目录缺少可验证的 config、tokenizer 和 safetensors 文件。仅凭 `Qwen2ForCausalLM` 架构名称不能证明 checkpoint 能通过 Transformers 或 vLLM 加载；推理兼容性尚未测试。

## 与现有 Qwen3-32B-AWQ 结果的区分

Qwen3-32B-AWQ 已锁定 Hugging Face revision `0499c3ac83fdef8810b907a23894ba91e95eddd8`，同一冻结 200 例上有直接作答结果：Base 为 44.0%，32B 为 48.0%，病例配对差 +4.0 pp（95% CI −2.5 至 +10.5）；Base 错误中的纠错率 23.2%（26/112），Base 正确中的退化率 20.5%（18/88）。这是**直接作答任务**，没有 DEFER prompt，不代表 Expert 接管能力。

本轮原计划在相同 200 个独立病例上补做 32B 的 DEFER 接管，并以 ATPO train split 的 200/100 例训练/验证数据拟合一个只看公开 prompt 特征的轻量 Router。GPU 资源在服务启动前降到安全门槛以下，未启动服务，所以 DEFER 和 Router 结果仍缺失。所有历史结果均保留。

## 可复现评估计划（权重可取后执行）

1. 固定并记录 ModelScope revision、文件 SHA-256、许可、架构/config、tokenizer 与 chat template；先做本地加载和单例协议检查。
2. 在同一冻结 200 例上分别运行 **直接回答** 和 **DEFER 接管**。两种提示、任务协议和准确率分开记录；Base 与 Expert 每条病例使用相同公开题干与选项，Gold 只在推理后评分，隐藏 atomic facts 不进入任何模型上下文。
3. 各任务分别汇报 Base/Expert 四格配对表、总体 Accuracy、有效输出率、纠错率、退化率、病例级配对 bootstrap CI、token/延迟与动作成本；无效格式保留在分母并计错。
4. 不根据测试结果调 prompt 或阈值。若可训练/验证权重可用，使用 train split 学习预决策价值，用独立 validation 选择阈值，只在固定 test split 评价；Router 输入不得包含 Gold、hidden facts、Expert 最终答案或事后正确性。
5. 将 med_qwen、Qwen3-32B-AWQ 和 Base 作为模拟模型分别报告。二者均不是临床医生，模型对照不能表述为真实医生实验。

当前没有启动 PPO/ATPO RL，也没有新增网络结构或 Loss。若文件树仍不可访问，下一步需要模型发布方提供可公开下载的 config/tokenizer/权重快照和许可信息；没有这些材料时不运行、不臆测结果。
