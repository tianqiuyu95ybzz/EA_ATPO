# EA-ATPO 真实数据实验预注册草案

日期：2026-11-03  
状态：真实数据运行前的内部锁定草案；不代表实验已经执行。

## 研究问题与假设

**问题：** 在专家推理成本相同的条件下，显式估计患者回答可用率、CONSULT 提示效用和 DEFER 接管正确率，能否改善四动作路由？

**主假设 H1：** 在验证集锁定的专家 token 预算下，可靠性条件路由的测试集准确率高于只使用 `Q_out(s,a)` 的 outcome-value gate。报告按病例配对的准确率差及 95% bootstrap CI；只有区间下界大于 0 才视为支持 H1。

**研究范围：** 公开医学选择题上的模拟对话和模拟专家评估，不是临床试验、医生效能比较或真实患者建议。

## 数据和隔离

1. 优先恢复官方 ATPO 的原始训练/测试划分并核对样本 ID、病例重复和文件哈希；历史交接记录中的 14,256/1,268 数量在恢复后重新核实。
2. 官方 test 完全锁定。只从官方 train 按病例 ID 固定切成 90% train、10% validation；同源改写题、重复病例必须在同一 split。
3. `atomic_facts` 仅可用于 patient simulator。agent、expert 和 gate 只能读取题干、选项、已公开问答/咨询记录、当前动作预算。
4. gold answer 仅在 rollout 结束后进入 outcome evaluator。任何 CONSULT 问句或提示都不得从答案、解释、答案选项位置或未来交互中摘取。
5. 所有切分、病例 ID 哈希、模型版本、提示词、随机种子、服务配置在首个 live rollout 前保存到 manifest；测试集不用于阈值、预算、checkpoint 或提示选择。

## 核心比较

- `Outcome-only gate`：使用公开状态估计各动作 `Q_out(s,a)`，不加入分通道可靠性特征。
- `+ patient availability`：增加 ASK 可回答概率。
- `+ CONSULT utility`：增加提示有根据且能改善后续决策的估计。
- `+ DEFER accuracy`：增加专家独立接管正确率估计。
- `Full reliability gate`：三路信号分别校准后联合路由。
- 参考项：ATPO/base agent、always ANSWER、always DEFER、规则路由。所有方法用同一病例、相同可见状态、同一成本定义。

CONSULT 的提示效用头与 DEFER 的终局正确率头必须分开训练和校准；不能把 expert 在一种任务上的可靠度直接复用到另一种任务。比较固定 gate 与 outcome-only gate，联合 ATPO 训练暂不纳入本轮主实验。

## 预算、指标和统计

- **主要预算：** validation 上锁定 `B = 0.20 × always-DEFER 平均专家 token 数/病例`。用 validation 选择每种 gate 的成本乘子；test 只使用冻结后的乘子和模型。若系统端点不返回 token usage，则先用 tokenizer 对请求/答复计数，并在运行前固定 tokenizer 版本。
- **主要终点：** locked test 上选择终局 ANSWER/DEFER 后的病例准确率；按病例进行 paired bootstrap，至少 10,000 次重采样，报告差值和 95% CI。
- **必要成本报告：** 专家 prompt/completion tokens、调用次数、墙钟延迟；patient 问题数和 UNKNOWN/拒答率。报告实际 test 成本偏离 B 的幅度，不按 test 重新调参。
- **必要安全/质量报告：** 无效格式率、CONSULT 泄漏终局答案率、unknown-as-negative 率、patient answerable/unanswerable/ambiguous 分布；不同难度和低可靠性分层上的准确率与校准误差。
- **校准：** availability、CONSULT utility、DEFER correctness 分别报告 Brier/ECE；校准方式与分箱数只在 validation 确定。
- **重复性：** 至少 3 个 rollout 随机种子；所有比较使用相同病例与配对种子。若成本/结论对种子高度敏感，报告分布并停止声称稳定增益。

## P4 数据采集 pilot 的门槛

恢复资产后，先在 train split 的 100 个病例上试跑，四动作各 1 条配对 rollout；检查服务失败、格式失败、泄漏、token 统计和 CONSULT 文本。修复协议问题后再扩大到 500 个以上病例，并做至少 3 个 rollout seeds。pilot 不用于最终测试结论。

一条 CONSULT 轨迹按此生成：

```text
公开状态 → base 输出一个具体 CONSULT 问句
         → expert 只读公开状态和该问句，输出短 Expert Advice
         → base 可继续 ASK / CONSULT（每病例最多一次）/ ANSWER / DEFER
         → rollout 结束后，evaluator 才读取 gold 并计算 outcome
```

保存 `sample_id/state_id`、公开状态、动作、CONSULT 问句和清洗后的提示、后续动作、最终答案、gold evaluator 标签、tokens、延迟、错误码。禁止把原始 `<think>` 写入公开状态或 JSONL trace。

## 失败条件与停止规则

- 若可靠性 gate 在锁定预算下的准确率差 CI 下界不大于 0，不支持 H1；按消融结果报告哪个信号失败，不事后更换主预算。
- 若只有某一路信号有增益，仅保留该信号作为后续候选；不能宣称三路联合有效。
- 若 unknown-as-negative、提示泄漏、无效率或分层校准明显恶化，停止该版本并先修协议/模型。
- 数据/模型/服务许可、病例级隔离或 prompt 泄漏无法验证时，不启动正式 test 评估。

## 咨询轨迹人工标注指南

先抽取至少 100 条 ASK 与 100 条 CONSULT；两个标注者独立标注，分歧仲裁。标注者读公开状态与回复，不看 gold 和隐藏 atomic facts。另由独立 outcome evaluator 在轨迹结束后用 gold 评 DEFER/最终答案。

| 对象 | 标签 | 操作定义 |
|---|---|---|
| patient 回答 | `answerable` / `unanswerable` / `ambiguous` | 直接回应问题；明确无相应事实/UNKNOWN；或答复含糊、矛盾、无法映射 |
| CONSULT 问句 | `focused` / `generic` / `answer-seeking` | 是否指向公开状态中的具体不确定点；是否只是泛问；是否索取隐藏答案/正确选项 |
| CONSULT 提示 | `grounded` / `unsupported` / `contradictory` | 是否由公开病例支持；是否引入状态外事实；是否与已公开事实矛盾 |
| 提示动作性 | `useful_next_step` / `non_actionable` / `premature_final_answer` | 是否给出可操作的鉴别或下一步证据；是否空泛；是否变相输出终局选项 |
| 协议与泄漏 | `pass` / `fail` + 原因码 | expert 输入是否出现 gold、隐藏事实、未来回复、私有 think；CONSULT 是否输出终局选项；DEFER 是否终止 |

“提示是否真的改善答案”不靠标注者猜测：以相同病例/随机种子下有提示与无提示的后续 rollout outcome 作配对评估；人工标签只评价证据依据和提示动作性。报告 Cohen's κ（类别）或 Krippendorff's α，并保存仲裁前后结果。
