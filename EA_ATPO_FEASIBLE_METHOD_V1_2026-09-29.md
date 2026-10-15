# EA-ATPO 可实现方法候选 v1（2026-09-29）

> 本文是当前约束下的首选**待验证方案**，不是已被实验证明“最优”的算法，也不代表下列训练已完成。研究对象是模拟医疗选择题问诊，不得外推为真实临床安全性结论。

## 1. 一句话目标与定位

以官方 ATPO 的多轮医疗问诊策略为基础，让 agent 从当前**公开信息状态**学习在 ANSWER、ASK、CONSULT、DEFER 间作序贯、成本敏感的选择。ACT 仅作 ASK/ANSWER 参考，不是主体算法。主方法采用两阶段：先改进 ATPO 的问诊训练，再冻结 agent、学习序贯专家接入策略；四动作联合 ATPO RL 只作为条件性扩展。

当前官方 baseline 是 Qwen3-4B SFT+ATPO step-20；模型大小和训练预算固定做方法消融。若 E1/E2 在该规模有效，再考虑官方 Qwen3-8B 配置复核；35B 专家模型不是 35B actor。所有规模比较应明确不是同算力结论。

## 2. 环境与动作语义

公开状态 `s_t`：题干/选项、患者已经回答的内容、公开专家建议、剩余轮次、剩余专家预算、已采用动作；绝不包括正确选项、隐藏 atomic_facts、专家未返回的内容或 actor `<think>` 私有推理。隐藏 atomic_facts 只给患者模拟器和训练时的奖励判定器。

- `ANSWER`：基础 agent 现在作终局答案，终止。
- `ASK`：agent 问患者一个问题，患者只依据 atomic_facts 回答，更新公开状态；之后重新决策。
- `CONSULT`：专家只看公开状态和具体咨询问题，给低工作量建议，不给最终选项；基础 agent 继续，专家预算减一。
- `DEFER`：专家只看公开状态，独立给终局答案；高工作量，终止。

主实验设每例最多一次 CONSULT，`b_t` 表示剩余 CONSULT 次数；可在后续任意轮 DEFER，仍须支付其高成本。若实验另设总专家预算，则单独加入总预算状态及合法动作掩码，不能混用两个预算定义。专家是模拟 LLM 专家，不能称为真实医师。CONSULT/DEFER 的输入长度、输出长度、延迟、调用次数均记录为成本；不能仅靠 prompt 名称区别两者。

## 3. Stage 0 / E0：先核实官方训练路径

官方论文给出 `Q=r+gamma V(next)`、树回传目标、turn-level advantage、PPO-style actor objective 与 critic MSE。现有代码在 `recipe/atpo/mt_reward_fn.py` 计算过程分数，但 `recipe/atpo/user_assistant_agent_loop.py::_evaluate_nodes()` 对非终止节点把 `action_reward` 设为 0.0。先用受控三轨迹和实际运行配置确认：过程分数有没有经过其他路径进入树 target、critic 或 actor；并核对终局奖励量纲。E0 未通过前不改官方代码或启动正式训练。

输出：最小轨迹手算与运行值对照、实际配置/数据哈希、checkpoint 与评估设置、E0 测试报告。现有未改动官方副本保持不动，在 `codes-repro` 工作。

## 4. Stage 1 / E1：Fact-aware Process-guided ATPO

保留 ATPO 的自适应树、actor/critic 架构及 PPO-style 目标；只修改由环境给出的 turn reward，使其确实参与树节点 `action_reward`、回传 `V_target` 和 advantage。提问回合：

`r_ask = w_n * new_fact_count_clipped - w_r * repeated_fact - w_u * unanswerable - w_t * turn_cost`。

新事实必须是患者**实际回复**提供、先前未公开的 fact；不能奖励仅因原病例隐藏事实列表里存在但患者没有说出的信息。先做简单可审计的 fact ID 匹配；不把 gold 选项用于选择“相关 fact”，也不把 fact ID 暴露给 agent。为防止机械刷 fact，限制每轮奖励上界，并保持终局诊断奖励的主导地位。权重只用训练/验证集选，先做权重敏感性曲线，不声称某一组是通用临床效用。

节点值与优势仍按 ATPO：`Q_t = r_t + gamma V_psi(s_{t+1})`；`A_t = r_t + gamma V_psi(s_{t+1}) - V_psi(s_t)`。critic 用树回传目标的 MSE；actor 用原 PPO-style clipped loss。**改变是奖励与由此产生的 target/advantage，不是凭空新增一个 actor loss。**

四组同预算消融：A 原官方 ATPO；B 仅接通已有 process_reward；C 仅新增事实信号；D B+C。至少 smoke 后再正式跑，比较准确率、平均轮数、新 fact 获取率、重复/无效提问率和 GPU 成本。若 B 已达到 D，不能声称 C 有独立贡献；若过程指标好但诊断变差，不通过 E1。

## 5. Stage 2 / E2：冻结基础 actor 的序贯协作决策

冻结 E1 最优的 actor；用公开状态编码器（首选冻结 actor 隐状态）加轻量四动作价值头。每个头在**采取动作前**预测长期 outcome `Q_out(s,a,b)`；分别记录成本 `Q_cost(s,a,b)`，推理时只扣一次：`U(s,a,b)=Q_out(s,a,b)-lambda_c Q_cost(s,a,b)`。仅当有可信的临床伤害分级标签时才引入 harm 项；当前选择题只评价准确率与资源成本，不假装有临床伤害真值。

为了避免“训练时只允许一次干预、测试时却每轮重新决策”的目标错位，定义带预算的序贯价值。`Q_out` 只累计终局诊断结果，`Q_cost` 累计整条后续轨迹的资源成本；E1 的事实塑形奖励**不混入** E2 的临床 outcome，否则会再次把成本/训练代理指标与最终收益混淆。给定当前门控策略 `pi`：

- `Q_out(s,ANSWER,b)=E[R_diagnosis from base answer now]`，`Q_out(s,DEFER,b)=E[R_diagnosis from expert terminal answer now]`；
- `Q_out(s,ASK,b)=gamma E[V_out^pi(s',b)]`，`Q_out(s,CONSULT,b)=gamma E[V_out^pi(s',b-1)]`；
- `Q_cost(s,a,b)=c(s,a)+gamma E[V_cost^pi(s',b')]`，终止动作无后续成本；
- `pi(s,b)=argmax_a [Q_out(s,a,b)-lambda_c Q_cost(s,a,b)]`，合法动作由 `b` 和轮次掩码决定。`V_out^pi` 与 `V_cost^pi` 分别是后续按该策略行动的期望 outcome 与成本。

训练时从同一病例/状态对四动作作配对 rollout；对于 ASK/CONSULT 采样患者/专家回应及后续轨迹。先用 base-only 继续策略收集初版数据，再在**训练集**上迭代一次“训练价值模型→用当前门控继续 rollout→重估目标”的 fitted policy improvement，最终版本由验证集选择，不能用测试集反复调整。记录每动作 return 方差、标签翻转率及 value gap；优先把采样预算给不确定且动作价值接近的状态，同时保持随机覆盖，防止只采难例的选择偏差。

首版损失：`L_value = sum_a w_a * Huber(Qhat_out(s,a,b), G_out(s,a,b))`，其中 `G_out` 是重复 rollout 的配对目标；成本头用相同方式拟合实际调用成本。可在验证后加辅助 pairwise ranking loss，让净收益差明显的动作更易排序；不将事后 argmax 标签当作无噪声真值。成本权重和选择阈值只在验证集校准；需要输出预测误差、校准误差和动作使用率。当前冻结表示+Ridge 的负结果是比较基线，不是最终网络已成功的证据。

推理流程：公开状态→四头预测预期 outcome/cost→掩码非法动作→选择最大净收益→执行动作→若未终止则用**新公开状态**重新决策。专家建议不能先生成再用于决定是否咨询；若咨询失败，应按预注册的失败策略与成本计入结果。DEFER 真正终止，不能把专家答案又交给 agent 重猜。

## 6. 数据、专家质量与有效性门槛

现有 500 状态/250 病例/每动作 3 rollout 数据是可行性集，约 485 完整状态；按病例拆分的 150 训练病例过小，冻结表示+Ridge/优势头在 98 完整测试状态尚未胜过 always ANSWER。先量化噪声与专家协议质量，不能仅凭 oracle 胜出数推出可学习性。主训练扩展按病例而非状态抽样，初始/后续状态分层；训练、验证、内部测试按病例隔离，官方最终测试集保持封存。新采样预算按 E0/E1 结果和先导学习曲线决定，不事先承诺固定规模。

抽检至少 100 条 CONSULT/DEFER：输入无 gold/atomic_facts/私有 think；CONSULT 不直接给最终选项；DEFER 不依赖 gold；判断建议是否公开信息可支持、有无事实幻觉。若专家在当前公开状态上的能力不超过 base 且 CONSULT 没有正向 uplift，就报告“专家设定不足”，先修协议或换更可靠专家，不靠调低成本强行制造优点。

评价：最终准确率、同专家调用预算下准确率、病例级净效用、CONSULT/DEFER 率、平均轮数、有效/重复/无效问诊、新 fact 获取率、专家 tokens/延迟、格式与泄漏率、训练/推理成本。对照 always ANSWER、官方 ATPO、E1 ATPO、always/随机 CONSULT、always DEFER、简单置信度规则、仅二元 CONSULT gate、当前离线 Ridge router；固定模型、测试集、患者/专家协议与预算。病例级 paired bootstrap CI；多随机种子。任何“最佳成本点”必须在验证集预先选定。

进入论文主结果的门槛：至少一个预注册合理预算档位在独立测试病例上超过 E1 base 与 always ANSWER，病例级 CI 支持正效益；CONSULT 或 DEFER 的使用有真实而非泄漏造成的增益；结果在初始/后续状态和不同专家成本下不过度崩塌。未通过则如实报告负结果、回查专家质量/标签噪声/状态表示，暂停联合 RL。

## 7. Stage 3 / E3：只在前两阶段成功后考虑

四动作正式进入 ATPO 的 macro-action 采样、树展开、终止条件、树备份、critic 与 actor PPO-style 更新。可用共享 actor backbone 加动作类型输出与条件文本生成，但必须同预算对照 Stage 2 固定门控；若 E3 无增益，主论文保留两阶段结果，不宣称联合优化成功。

## 8. 当前进度与立即下一步

- 已完成官方 ATPO baseline、清洁的四动作 500 状态采集和初版离线 router；当前 router 未胜过 always ANSWER。
- 已发现过程分数计算与树节点奖励设零之间的潜在断点；尚待 E0 动态验证。
- 本文冻结方法定义，不代表 E1/E2/E3 已训练。
- 立即执行 E0：受控轨迹验证奖励→树备份→critic/actor 路径，再决定 E1 最小补丁及实验预算。每一步保存配置、代码差异、病例划分、日志、checkpoint、指标和失败原因。

参考：ATPO https://arxiv.org/abs/2603.02216；SLTD https://arxiv.org/abs/2109.06312。
