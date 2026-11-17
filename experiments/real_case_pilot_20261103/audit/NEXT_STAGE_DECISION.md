# EA-ATPO 下一阶段决策

> **专家互补性成本审计补充（2026-11-17）：** 以下较早版本中的 pooled 100+100 指标只保留为历史描述，不作为当前配对推断。请按 cohort 阅读[最新分组审计](../../../EXPERT_CONDITIONED_COMPLEMENTARITY_COST_ANALYSIS.md)：ReMeDi 的两个 100 例 DEFER cohort 分别为 40% vs Base 54%、41% vs Base 41%。最新审计还将“完美 Expert 始终接管”和“知道 Base 错误时才调用的选择性 Oracle”分开；后者仅是不可部署上界。Qwen3-32B-AWQ 目前仍只有直接回答分数，没有 DEFER 结果。

> **最新状态（2026-11-18）：** 暂停 Router、PPO 和 ATPO RL，也不把 32B-AWQ 设为默认 Expert。新 100 例筛选中 Qwen3-8B Base-DEFER 为 44%，Always Answer 为 41%，ReMeDi 为 41%，PopiT 为 40%；未显示某个候选有稳定 DEFER 优势。跨两个独立 100 例集合，ReMeDi 为 40.5%，低于 Always Answer 的 47.5%，配对差 −7 pp（95% CI −13.5 至 −0.5）。Huatuo 候选在当前协议下全部解析失败。20 例 CONSULT 小提示探索实验中，提示合规率和准确率均未支持继续采用这些候选。Gold 直接生成的“正确 DEFER”只能作为理论上界，不是模型实验或训练样本。完整分析见 [`L2D_EXPERT_AND_CONSULT_HINT_ANALYSIS_2026-11-17.md`](L2D_EXPERT_AND_CONSULT_HINT_ANALYSIS_2026-11-17.md)。
>
> **历史状态（2026-11-06）：** 下文记录强 Expert 和 Router 实验完成前的阶段性决策。Qwen3-32B-AWQ 已完成冻结 200 例配对直接答题评估，四动作 Router 也已完成独立病例逐轮测试；强专家直接答题结果见 [`STRONG_EXPERT_EVALUATION.md`](../../ea_atpo_value_router_20261105/outputs/strong_expert_awq/STRONG_EXPERT_EVALUATION.md)。

## 决策

### 2026-11-18 更新

当前最合理的决策是**停止候选 Expert 扩容与联合训练，保留已有结果作为负面可行性证据**。现有证据不支持“医学对话微调就能带来 DEFER 可靠性”，也不支持把更大参数量当作可靠性的替代指标。Qwen3-32B-AWQ 仅有旧的直接答题筛选点估计（48% 对 8B 的 44%，配对区间跨 0），没有可解释的新 DEFER 病例结果；因此继续部署它目前不能回答 L2D 是否获益。

“DEFER 直接给正确答案”若指将 Gold 作为答案，只能做明示的 oracle 上界分析，不能称为真实模型表现、不能作为 Agent/Expert 输入，也不能进入正常 Router/PPO 训练。以当前 pooled ANSWER 准确率 47.5% 和 `λ_DEFER=0.60` 计算，完美 oracle 的净值 0.40 仍低于 ANSWER 的 0.475。若研究“高概率正确”的 DEFER，则应先对模型置信度做独立病例校准并报告 risk–coverage、Base 错误条件准确率和对 Base 正确答案的破坏率。

当前 CONSULT 小提示 pilot 也不支持把提示写成“保证不泄露答案”后就默认有用：20 例上 Base 提示准确率 45% 对直答 50%，ReMeDi/PopiT 提示分别只有 30%/50% 合规率，端到端结果更低。后续若继续，只应先研究可验证的证据线索提示和合规协议，并在新的独立病例上复验；本轮不扩成正式大样本，也不据此加入新训练模块。

下方 2026-11-06 的阶段分析与建议保留作历史记录；其中“恢复 32B BF16”及先前建议不再是当前行动项。

**暂不开始四动作 Router、P5 或 P8 / ATPO RL 训练。** 当前证据还不能说明模型可以可靠地区分“该咨询谁”与“该不该采用建议”；直接训练 Router 会把模拟 Expert 能力、建议质量、答案更新策略和动作成本混在一起。

| 问题 | 当前判断 | 依据 |
|---|---|---|
| CONSULT 是否有独立于 DEFER 的价值？ | **未建立，不能判定为没有。** | 既有 Pilot 上按病例聚类的 CONSULT−DEFER 配对差约 +0.17 pp，95% CI −4.78 至 +5.06 pp；无成本下仅 14/291 个状态是 CONSULT 的严格唯一最优（4.8%），DEFER 为 11/291（3.8%）。两个独立专家动作整体表现接近，但极少数状态可能有不同优势。 |
| 是否需要 Advice Reliability / Advice Utilization？ | **需要作为独立研究组件和评估层；不训练。** | 本次同一模拟建议下，Critical 的准确率只比直接 ANSWER 高 0.5 pp（95% CI −5.9 至 +6.8），但正确退化比 Naive 少 10.3 pp；它同时少修正了错误。单一准确率或答案改变率不足以指导路由，必须分别测建议可靠性和 Base 的恰当使用。 |
| 是否训练四动作 Router？ | **否。** | 候选 DEFER 筛选没有找到稳定优于 Base 的模型；Qwen3-32B-AWQ 只有旧的直接答题结果（48% 对 44%，配对区间跨 0），没有有效的新 DEFER 结果。新决策机制实验 200 状态/97 病例上的所有准确率差 CI 均跨 0；CONSULT 观测成本约为直接 ANSWER 的 3 倍 token，延迟也更高。 |
| EA-ATPO 贡献点是否调整？ | **调整研究叙事和方法分解。** | 将“动作选择”拆成两层：①信息获取/交接动作（ANSWER、ASK、CONSULT、DEFER）；②获得建议后，按建议证据可靠性和可见病例信息做保留、更新或升级处理。把“恰当采纳/拒绝建议”作为显式目标，而非默认 CONSULT 后继续回答就能带来收益。 |

这里的“Advice Reliability”是对专家来源或建议内容的可信度估计；“Advice Utilization”是 Base 在收到建议后如何决定保留、更新或请求更多信息。两者应分开测：更准的 Expert 不会自动保证 Base 能正确用它的建议，较高的采纳率也不会证明建议正确。

## 本轮结果怎样改变方法主张

既有 Pilot 的固定动作结果为 ANSWER 41.9%、ASK 45.6%、CONSULT 44.6%、DEFER 44.4%；CONSULT−ANSWER 为 +2.67 pp，病例级 CI −1.78 至 +7.33 pp。按病例聚类重算后，CONSULT−DEFER 的差异接近零，且区间宽。Pilot 中 ASK 后还能再咨询，而初始 CONSULT 已用完咨询预算；该资源差异也限制了跨动作的直接解释。

新的 200 状态后续决策实验补充了另一个事实：给 Base 一条 Qwen3-8B Expert 建议，并不会自动提高最终正确率。Naive 更常改答案，却更多破坏原本正确的答案；Critical 改答较少、退化较少，但它相较直答的准确率差仍不能与零区分。因而论文中的核心机制不宜写成“专家建议提高准确率”或“多模型协作优于单模型”；可检验的主张应改为：**在医疗问答动作策略中，建议质量与建议利用是两个独立决策；可见证据条件下的选择性更新值得作为一项方法假设单独评估。** 当前结果只支持提出此假设，尚未证明它提升临床决策表现。

## 研究文献与设计含义

1. [Kumaran et al., *Competing Biases underlie Overconfidence and Underconfidence in LLMs*, Nature Machine Intelligence (2026)](https://www.nature.com/articles/s42256-026-01217-9) 在多种模型和任务中报告了看见初始答案后更难改变、以及对相反建议过度敏感的现象。它支持将“保留正确答案”和“修正错误答案”作为并列指标；本项目还需要验证这种行为是否迁移到医疗题、Qwen3-8B 和当前咨询协议。
2. [Bo et al., *To Rely or Not to Rely?*, CHI (2025)](https://arxiv.org/abs/2412.15584) 测的是人类接受 LLM 建议的干预，不是一个 LLM 接受另一 LLM 建议；作者发现降低过度依赖并不必然改善恰当依赖。这提醒我们将采纳率和答案正确性分开报告。
3. [Ni, *Position Matters More Than Channel: Talking a Language Model Out of a Correct Answer*, arXiv preprint (2026)](https://arxiv.org/abs/2610.34809) 在 Qwen3.5-27B 与 MMLU-Pro 上操纵建议位置、来源和原回答；研究范围有限且仍是预印本，但提示“原答案是否呈现、建议放在何处、是否标注专家身份”本身会影响结果。下一轮需把这些因素拆成受控条件，不能把 Retain/Critical 的差异全归因于证据检查。
4. [Huang et al., *Large Language Models Cannot Self-Correct Reasoning Yet*, ICLR (2024)](https://openreview.net/forum?id=IkmD3fKBPQ) 聚焦无外部反馈的自我纠错，发现仅凭自我批评并不稳定。当前 CONSULT 有外部文本，但 Critical 仍依赖模型自评建议是否可信；后续应引入独立的证据来源或人工标注，不能只堆叠“再检查一次”的提示。

这些论文支持把适当依赖作为研究问题，**不证明 EA-ATPO 已有新颖性或本实验的 Critical prompt 有效**。需要明确与既有的适当依赖、自我纠错和多模型协作工作对照。更具体的潜在贡献是：在带有患者信息获取、可见证据边界和动作成本的医疗多智能体设置中，同时建模咨询动作价值与建议的选择性利用，并以“错误修正—正确退化”曲线衡量结果；仍需更强独立 Expert 和真实专家审阅来支撑。

## 建议的下一步顺序（2026-11-06 历史计划，已被后续结果取代）

1. **恢复原定 Qwen3-32B BF16。** 优先修复 Hub/代理传输，或从获授权的机器传入同一 revision 的完整权重并校验。暂不改成量化版本；若之后决定采用 FP8/AWQ，应作为独立模型条件记录，不能冒充 BF16 比较。
2. **先完成强专家 200 例配对比较。** Base 和候选 Expert 使用相同公开题目、提示、解析规则和生成参数，报告四格表、Base 错误条件准确率、修正和退化率、病例级 CI。只有模型实际回答后，才讨论 Expert 是否更可靠。
3. **冻结 CONSULT 建议质量标签。** 在模型不见 Gold 的条件下产生建议；事后由不知道 Base 最终答案的独立评审标注建议正确性、相关性、重复信息和可验证性。优先使用有资质的临床评审；若只能用另一个 LLM 评审，明确标成自动评审并抽样人工核验。
4. **做建议利用的受控分层实验。** 在同一状态分别控制建议正确/错误或不相关、支持/反对初始答案、是否展示原答案、建议位置和来源标签。报告正确建议采纳、错误建议拒绝、修正、退化、校准和成本，并对病例聚类 bootstrap。现有 Critical 与 Retain 提示一次改变了多项指令，下一轮需做消融。
5. **复现后再决定 Router。** 如果建议质量具有可测的独立差异，且某种利用策略在独立病例上降低净错误、保持正确答案并带来可接受成本，再训练一个可解释的 Advice Utilization gate。之后才评估它与 ANSWER/ASK/CONSULT/DEFER 动作选择是否联合学习有增益；P5/P8 保持暂停。

当前数据仍来自 ATPO 公开选择题和 Qwen3-8B Patient/Expert 模拟器，不是真实病历或医生实验。报告中的模拟建议不能称为临床专家意见。本决定阶段已完成，等待确认是否进入后续强专家恢复与受控验证；**本轮未启动训练**。
