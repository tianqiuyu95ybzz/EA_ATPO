# Qwen3-32B-AWQ DEFER Router 泛化扩展预注册

**当前 Attempt:** `qwen32_router_generalization_20261118_attempt04`

**状态：** Attempt04 已完成 20 例协议 smoke、1,700 例配对推理、train/validation Router 拟合和一次性 400 例 test 分析。实际结果与限制见 [最终结果报告](QWEN32_ROUTER_GENERALIZATION_20261118_RESULTS.md)。Attempt02/03 在推理前作废，仅保留为病例历史审计记录。
**目标：** 在新病例上检验轻量 Router 是否能学习 Qwen3-32B-AWQ 相对 Qwen3-8B Base 的条件优势。不启动 PPO/ATPO RL，不修改原始 ATPO。

## 固定模型和协议

- Base：`Qwen/Qwen3-8B` revision `b968826d9c46dd6066d109eabc6255188de91218`。
- DEFER Expert：`Qwen/Qwen3-32B-AWQ` revision `0499c3ac83fdef8810b907a23894ba91e95eddd8`。
- 复用 Attempt01 的公开病例构造、角色提示、DEFER 格式校验/解析器和相同生成参数：temperature=0、top_p=1、thinking 关闭；Base 最多 64 tokens、Expert 最多 128 tokens；Base 收集 A–D token logprobs。
- 同病例向 Base 和 Expert 暴露相同题干/选项；Gold 仅由本地评分器在两个原始生成结束后比较，不写入 Router 特征、模型请求或模型原始输出。
- 测试 Expert 仍是语言模型模拟器，不称作真实医生。

## 新病例与拆分

- 当前 Attempt04 从 ATPO train split 的历史未暴露病例中抽取 **1,000 个训练病例、300 个验证病例、400 个病例级 holdout 测试病例**。三组全部来自 ATPO train parquet；测试集是新的 case_id holdout，不是 ATPO 官方 test split。
- 病例选择种子 `20261128`。`case_id`、case hash 和公开 prompt hash 在三个集合间互斥。
- 候选病例先排除 EA_ATPO 与 EA_ATPO_V1 两个项目中已记录的历史病例 ID、公开 prompt/hash、case hash 和历史 source row index；再排除 Attempt01 的 200/100/200 cohort、旧 Router 阈值验证病例、历史 Expert 候选筛选病例以及旧 smoke/dev 病例。
- 历史 Qwen32 Router/专家选择对官方 ATPO test split 的覆盖接近全量：Attempt03 全历史筛选后只有 8 个未暴露行，不足 400。为不复用历史阈值/模型选择病例，Attempt04 将 1,700 个严格未暴露的 ATPO train 记录按 `case_id` 隔离拆分；报告须把测试结论称为“train-source holdout”，不能称作官方 test split 结果。
- 若审计后某个 split 不足预注册样本数，停止推理并报告剩余数量；不缩小测试集、不重复使用旧病例，也不改成本/提示去迎合可用样本。
- 抽样清单只保存行号、哈希、case_id 和 split 元信息；不保存 Gold、atomic facts 或完整 prompt。

## 训练目标与模型

所有 Router 输入限于决策前可见特征：公开题干长度/选项统计、Base 生成选项、Base logprob 熵/间隔及解析有效性。不得输入 Gold、Expert 最终答案、正确性标签或未来信息。

病例级终局相对收益定义为 `Δ = expert_correct − base_correct ∈ {−1, 0, +1}`；纠错事件定义为 `base_correct=0 AND expert_correct=1`；退化事件定义为 `base_correct=1 AND expert_correct=0`。每档 DEFER 成本 `λ` 下净动作优势为 `Δ − λ`，推理时只扣一次调用成本。

固定比较项：

1. Uncertainty Threshold：Base 选项熵分数；阈值只在新验证集按 Utility 选择。
2. Logistic Regression：一套预测“专家纠错事件”的二分类模型；一套预测 Δ∈{−1,0,+1} 的三分类模型，以 `P(Δ=+1)−P(Δ=−1)` 形成相对价值。
3. 小型 MLP：15 维以内同一组预决策特征，单隐藏层 16 个单元，三分类预测 Δ；不增加大型模型或额外网络模块。
4. Attempt01 Learned Router：保留原有锁定 Ridge 系数，不重训；用新验证集统一选择该冻结分数的各成本阈值，作为历史 Router 对照。
5. 新 Router：比较预测纠错事件的 Logistic Regression、预测三类 `Δ` 的 Logistic Regression（**预注册主 Router**）、相对收益 Ridge、16 单元单隐层 MLP；以新验证集选择正则化和各成本阈值。
6. Cost-adjusted action value：以相对收益预测 `\hat{Δ}` 形成 `\hat{Δ}−λ`，成本只扣一次；保存每档成本的净收益和等价阈值，验证损失/净效用均只用 train/validation 选择。

训练只读取新训练 split；验证集用于超参数/阈值选择；测试 split 的标签、正确性和专家答案只在 Router 与全部阈值锁定后由一次性分析脚本读取。

## 成本、基线与指标

固定成本网格为 `λ ∈ {0, 0.05, 0.10, 0.20, 0.40, 0.60}`，主成本为 0.60。Utility 定义为 `final_accuracy − λ × defer_call_rate`，平均成本为 `λ × defer_call_rate`。不依据测试结果选择成本。

比较 Always Answer、Always DEFER、Random DEFER（按验证集 Expanded Ridge 调用预算匹配，并同时报告期望与固定种子实现值）、Uncertainty DEFER、各新 Learned Router、冻结 Attempt01 Learned Router 和 Oracle Selective DEFER。Oracle 仅作事后上界，不参与训练、验证选阈值或任何可部署决策。

报告 Accuracy、纠错率、正确答案退化率、Expert 调用率、平均成本、Utility、AUROC、相对收益的 MAE/MSE/Pearson、动作频率、类别比例、置信区间和病例级配对策略差异。使用病例级 paired bootstrap 10,000 次，seed=`20261130`。

分析新训练/验证/测试集合的预决策特征分布、相对收益标签分布和纠错/退化类别比例；检查不同模型的校准与预测误差，以及 Base 不确定性分数单独预测纠错事件/正净收益的能力。测试集仅作一次冻结后的最终诊断，不根据其结果调整模型或阈值。分析脚本要求先存在验证集锁定文件，并拒绝覆盖已生成的最终测试分析。

## 停止门槛

即使新 Router 表现更好，本阶段也不启动 PPO/ATPO RL。只有冻结策略在另一个新独立病例队列复现专家条件互补性，且 Learned Router 在新测试上相对 Always Answer、Uncertainty、Random 的 Utility 增益有病例级配对支持，才重新讨论 RL。若没有，则报告负面泛化结果并停止当前 Router 主线。

## 历史状态记录（2026-11-18，推理前）

全局历史审计覆盖 EA_ATPO 与 EA_ATPO_V1 两个工作区，检查 84 个历史 JSONL、35,684 条记录、2 个旧 `case_selection.json` 和 1 个旧 `sample_selection_manifest.json`。Attempt02 的首轮清单有 34/400 个 test source row 与旧选择重叠；Attempt03 在纳入完整旧 cohort 后，官方 test split 只有 8 个未暴露病例。两次均在推理前作废。Attempt04 从 train split 冻结 1,000/300/400，种子 `20261128`；`case_id`、case hash、prompt hash 均跨集合互斥，且无历史 hash/训练 source-index 重叠。清单 SHA256 为 `11d95c40d8c34bcbeae4dd8c18698fcee726f22db77b7f718d457d9527bc0cfa`，仅保存在共享 PFS 的 `outputs/qwen32_router_generalization_20261118_attempt04/case_selection.json`，不上传含病例标识的清单。

当时尚未调用模型；GPU3 仅有 56,783 MiB 空闲且利用率 100%，启动脚本在加载前退出。该资源阻塞后来解除。历史审计 cache 和病例清单仍只保留在 PFS。脚本和最终结果见 [复现说明](QWEN32_ROUTER_GENERALIZATION_REPRODUCE.md) 与 [最终结果报告](QWEN32_ROUTER_GENERALIZATION_20261118_RESULTS.md)。

## 执行结果补充（2026-11-18）

- User 授权使用 GPU1/2/5 后，固定 Qwen3-32B-AWQ 运行于 GPU1，Qwen3-8B 运行于 GPU5；Attempt02 服务目录绕过了 Attempt01 长 IPC socket 路径启动失败。GPU2 未使用。
- Smoke 为 20/20 配对病例，Base/Expert parser failures 均为 0。正式 1,700 例配对采集完成：Base invalid 3，Expert invalid 66。Expert invalid 按“部署时没有可评分答案”计入端到端失败；没有事后修复。
- 按原定 1,000/300/400 train-source holdout 划分。策略只在 train 拟合、validation 选阈值，test 在 policy lock 后一次性打开。模型、成本、病例级配对差异和 bootstrap CI 见最终报告及本地 analysis CSV。
- Test 中 Base accuracy 48.5%，Expert 端到端 accuracy 57.75%，配对提升 9.25 pp（病例级 95% CI 4.49–14.25 pp）。但主成本 0.60 的 Logistic Relative Value Router 在 validation 选出的调用率为 0，未优于 Always Answer；不启动 PPO/ATPO RL。简单 uncertainty threshold 在预注册成本 0.10/0.20 有正的配对 Utility 差，需在新病例复现。
- 原分析器已生成完整 CSV 和 manifest 后，在 Markdown 汇总阶段遇到 `auc_by_model` 初始化顺序 bug。已修复源码，并从既有汇总表生成报告；未重新拟合、选阈值或重算 test bootstrap。完整处理记录见最终报告。
