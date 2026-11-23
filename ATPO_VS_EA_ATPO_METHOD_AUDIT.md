# ATPO 与 EA-ATPO 方法级代码审计

审计日期：2026-11-06

## 结论摘要

**是。按当前可复核实现，EA-ATPO 是 ATPO 的四动作外部协议与独立离线 Router 实验；尚未形成改变 ATPO Actor、Critic、树搜索或参数更新目标的联合算法。** 四动作行为可以在独立 rollout 中执行，Router 也有实际拟合出的 KRR 系数，但 Router 的拟合与 Base Agent 的反向传播没有连接。

原始 ATPO 的核心仍是候选树展开与启发式剪枝、树值回传、PPO Actor 更新和 clipped-MSE Critic 更新。EA 的已训练 Router 使用 NumPy 线性核岭回归，不是接入 ATPO Critic 的神经 Router。当前最小可辩护的下一步，是把四个动作改为同一 Actor 的 macro-action，在原有 PPO/树搜索中训练，并把终局诊断收益和资源成本合并成唯一、明确的回报路径；不需要再添大型网络或自定义 PPO 损失。

本次只读审计，没有修改源码，也没有启动训练。

## 审计范围与版本

- EA 工作区 Git HEAD：**ad9d0e31660b973a3e5ddfcadce6701f767e0f62**。审计时工作树已有未提交修改和未跟踪的 Router 实验文件；报告中的 EA 行号对应当时工作树内容。
- EA 工作区不含完整的原始 ATPO 训练源码。原始源码按只读方式克隆自官方仓库，审计版本为 commit **25af83688c06917f94245a4db06be7d0924c090c**。以下原始 ATPO 行号与链接都固定到该 commit。
- EA 的旧实验记录描述过 4B ATPO step-20 与 E1-B smoke，但记录也注明 checkpoint、日志未上传。因而历史训练只按交接文档报告，不视为本次能复核的训练产物。

## 1. 原始 ATPO 的实际训练方法

### 1.1 奖励：代码会算过程分，但标准训练路径实际主要优化终局结果

calculate_single_turn_reward() 计算：

\[
r_{\mathrm{process}} =
r_{\mathrm{format}}+r_{\mathrm{verifier}}+r_{\mathrm{effective}},
\qquad
r_{\mathrm{outcome}} =
r_{\mathrm{correctness}}.
\]

符合格式的回复得到默认 format_score=0.1；问题被 verifier 标记为 <Normal> 时得到 verify_score=0.4；患者给出非空且非“无法回答”的回复时再得到 effective_score=0.5。终局正确答案得到 correctness reward，错误答案为 0。正式 run.sh 把 correctness_score 覆盖为 3，YAML 默认值则是 1。无效终局会在树路径中使用 exceed_max_turn_penalty=-1。

- 过程/终局分的计算：[mt_reward_fn.py#L45-L154](https://github.com/Quark-Medical/ATPO/blob/25af83688c06917f94245a4db06be7d0924c090c/codes/recipe/atpo/mt_reward_fn.py#L45-L154)
- 奖励系数默认值：[ppo_trainer.yaml#L126-L131](https://github.com/Quark-Medical/ATPO/blob/25af83688c06917f94245a4db06be7d0924c090c/codes/recipe/atpo/config/ppo_trainer.yaml#L126-L131)
- 实际 launch 覆盖正确奖励为 3：[run.sh#L116-L123](https://github.com/Quark-Medical/ATPO/blob/25af83688c06917f94245a4db06be7d0924c090c/codes/recipe/atpo/run.sh#L116-L123)

但标准树搜索路径对非终止节点明确设 action_reward=0.0，终止的有效最终回答才接入 outcome reward，无效终止回答使用惩罚。树搜索没有把上述 process_reward 加到中间节点。[user_assistant_agent_loop.py#L374-L417](https://github.com/Quark-Medical/ATPO/blob/25af83688c06917f94245a4db06be7d0924c090c/codes/recipe/atpo/user_assistant_agent_loop.py#L374-L417)

独立 reward manager 也会计算并暂存 process_reward_tensor，但 return_dict=True 时返回给 trainer 的 reward_tensor 是 outcome_reward_tensor；过程分只进入额外统计信息，没有作为标准返回奖励。[mt_reward_manager.py#L30-L127](https://github.com/Quark-Medical/ATPO/blob/25af83688c06917f94245a4db06be7d0924c090c/codes/recipe/atpo/mt_reward_manager.py#L30-L127)

因此，准确表述是：**实现了过程奖励计算，但官方树训练的节点奖励路径没有使用它；当前可执行的树回报以终局正确性为主。** 仓库中的 EA patch 是一个默认关闭的拟议开关，不属于审计到的官方 commit。现有 smoke 测试还显示“患者无法回答”时，<Normal> 问题仍可能得到 0.5 过程分，所以不能把它直接等同于“取得了有诊断价值的信息”。[e1_process_reward_switch.patch#L1-L19](e1_process_reward_switch.patch)、[e0_test_atpo_reward_path.py#L103-L114](e0_test_atpo_reward_path.py#L103-L114)

### 1.2 树搜索、Critic 估值与回传

搜索循环最多收集 M 条完成轨迹；每个活动叶节点采样 N 个 Actor 候选，再通过模拟患者接口推进，Critic 为非终止状态估值。[user_assistant_agent_loop.py#L236-L334](https://github.com/Quark-Medical/ATPO/blob/25af83688c06917f94245a4db06be7d0924c090c/codes/recipe/atpo/user_assistant_agent_loop.py#L236-L334)

剪枝不是 PUCT/UCB，也不是按最大 Q 贪心保留。代码对同一父节点的候选计算 Q 方差和父 Critic 与子节点均值的差：

\[
S=0.3\,\widetilde{\operatorname{Var}}(Q_{\mathrm{child}})
  +0.7\,|V_{\mathrm{critic}}(s)-\operatorname{mean}(Q_{\mathrm{child}})|.
\]

当 S 低于阈值且随机数小于 0.9 时，随机保留一个子节点；否则保留全部子节点。[user_assistant_agent_loop.py#L336-L372](https://github.com/Quark-Medical/ATPO/blob/25af83688c06917f94245a4db06be7d0924c090c/codes/recipe/atpo/user_assistant_agent_loop.py#L336-L372)

节点 Q 与树值回传为：

\[
Q(s,a)=r(s,a)+\gamma V(s'), \qquad
V_{\mathrm{tree}}(s)=\operatorname{mean}_{a\in\mathrm{surviving\ children}} Q(s,a).
\]

终止叶的后续值设为 0，非终止状态由 Critic 补值；父节点取留下来的子节点 Q 的均值，而非最大值。[user_assistant_agent_loop.py#L374-L394](https://github.com/Quark-Medical/ATPO/blob/25af83688c06917f94245a4db06be7d0924c090c/codes/recipe/atpo/user_assistant_agent_loop.py#L374-L394)、[user_assistant_agent_loop.py#L461-L497](https://github.com/Quark-Medical/ATPO/blob/25af83688c06917f94245a4db06be7d0924c090c/codes/recipe/atpo/user_assistant_agent_loop.py#L461-L497)

launch 配置使用 M_trajectories=128、N_candidates=4、启用剪枝和 Critic，且 only_use_critic_value=True。YAML 的树搜索默认值不同，所以复现实验应以实际 run.sh 覆盖项为准。[run.sh#L116-L122](https://github.com/Quark-Medical/ATPO/blob/25af83688c06917f94245a4db06be7d0924c090c/codes/recipe/atpo/run.sh#L116-L122)、[ppo_trainer.yaml#L133-L148](https://github.com/Quark-Medical/ATPO/blob/25af83688c06917f94245a4db06be7d0924c090c/codes/recipe/atpo/config/ppo_trainer.yaml#L133-L148)

### 1.3 Advantage / Return：活动树训练使用自定义树估值，不是配置名暗示的标准 GAE

树节点存：

\[
R_t=V_{\mathrm{tree}}(s_t),\qquad
A_t=
\frac{r_t+\gamma V_{\mathrm{selected}}(s_{t+1})
             -V_{\mathrm{selected}}(s_t)}
     {\max(n_{\mathrm{explore}}(s_t),1)}.
\]

其中 only_use_critic_value=True 时，优势公式中的父、子 V 使用 Critic 值；returns 仍写入树回传的 MDP 值。探索次数对 Advantage 再做除数缩放。[user_assistant_agent_loop.py#L498-L523](https://github.com/Quark-Medical/ATPO/blob/25af83688c06917f94245a4db06be7d0924c090c/codes/recipe/atpo/user_assistant_agent_loop.py#L498-L523)

通用 trainer 另有标准 GAE：

\[
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t),\quad
\hat A_t=\delta_t+\gamma\lambda\hat A_{t+1}.
\]

[core_algos.py#L210-L257](https://github.com/Quark-Medical/ATPO/blob/25af83688c06917f94245a4db06be7d0924c090c/codes/verl/trainer/ppo/core_algos.py#L210-L257)

不过标准 agent loop 会输出预先计算好的 advantages 和 returns，trainer 发现它们后直接使用，不调用配置的 compute_advantage()。所以尽管 run.sh 写了 adv_estimator=gae、γ=1、λ=1，正常树路径采用的是上面的自定义树 Advantage，而非 GAE。这里是一个容易误读的配置/执行路径差异。[chat_model.py#L273-L336](https://github.com/Quark-Medical/ATPO/blob/25af83688c06917f94245a4db06be7d0924c090c/codes/recipe/atpo/chat_model.py#L273-L336)、[mt_trainer.py#L1293-L1332](https://github.com/Quark-Medical/ATPO/blob/25af83688c06917f94245a4db06be7d0924c090c/codes/recipe/atpo/mt_trainer.py#L1293-L1332)

chat_model.py 将一个 turn 的 Advantage 复制到该 turn 的所有生成 token；Critic 的 value_response_mask 只在每个 Assistant turn 的前 5 个 token 置位，Return 同样只写入这些位置。[chat_model.py#L273-L305](https://github.com/Quark-Medical/ATPO/blob/25af83688c06917f94245a4db06be7d0924c090c/codes/recipe/atpo/chat_model.py#L273-L305)

### 1.4 Actor、Critic 损失和参数更新

Actor 使用 clipped PPO：

\[
\rho_t(\theta)=
\exp(\log\pi_\theta-\log\pi_{\theta_{\mathrm{old}}}),
\]

\[
L_{\mathrm{PPO}}=
\operatorname{Agg}_{mask}
\max\left(
-\rho_t\hat A_t,\,
-\operatorname{clip}(\rho_t,1-\epsilon_l,1+\epsilon_h)\hat A_t
\right).
\]

当启用 KL loss 时再加 β D_KL(πθ || πref)。代码包含负优势 dual-clip 与可选截断重要性权重分支。[core_algos.py#L817-L902](https://github.com/Quark-Medical/ATPO/blob/25af83688c06917f94245a4db06be7d0924c090c/codes/verl/trainer/ppo/core_algos.py#L817-L902)、[dp_actor.py#L443-L495](https://github.com/Quark-Medical/ATPO/blob/25af83688c06917f94245a4db06be7d0924c090c/codes/verl/workers/actor/dp_actor.py#L443-L495)

Critic 使用 clipped value MSE：

\[
L_V=\frac12\operatorname{Agg}_{mask}
\max\left[
(V_\psi-R)^2,\,
(\operatorname{clip}(V_\psi,V_{\mathrm{old}}\pm\epsilon_V)-R)^2
\right].
\]

[value loss 公式](https://github.com/Quark-Medical/ATPO/blob/25af83688c06917f94245a4db06be7d0924c090c/codes/verl/trainer/ppo/core_algos.py#L1255-L1294)、[critic backward 与 optimizer step](https://github.com/Quark-Medical/ATPO/blob/25af83688c06917f94245a4db06be7d0924c090c/codes/verl/workers/critic/dp_critic.py#L223-L266)。Critic 以 Actor checkpoint 初始化，但有自己的 value head 和独立 optimizer；代码的 Critic 构造器把语言模型输出头替换成 hidden-size 到 1 的线性 value head。[model.py#L120-L139](https://github.com/Quark-Medical/ATPO/blob/25af83688c06917f94245a4db06be7d0924c090c/codes/verl/utils/model.py#L120-L139)、[run.sh#L109-L115](https://github.com/Quark-Medical/ATPO/blob/25af83688c06917f94245a4db06be7d0924c090c/codes/recipe/atpo/run.sh#L109-L115)

Critic 先于 Actor 更新；配置中 critic_warmup=4、Actor 学习率 10^-6、Critic 学习率 10^-5、KL loss 系数 0.01、use_kl_in_reward=False。YAML 中 lora_rank=0，当前 launch 没有启用 LoRA；Actor 和 Critic 按全量模型参数训练（Critic 另有 value head）。训练器分别执行 Critic/Actor backward 与 optimizer step。[run.sh#L20-L33](https://github.com/Quark-Medical/ATPO/blob/25af83688c06917f94245a4db06be7d0924c090c/codes/recipe/atpo/run.sh#L20-L33)、[run.sh#L68-L75](https://github.com/Quark-Medical/ATPO/blob/25af83688c06917f94245a4db06be7d0924c090c/codes/recipe/atpo/run.sh#L68-L75)、[run.sh#L109-L128](https://github.com/Quark-Medical/ATPO/blob/25af83688c06917f94245a4db06be7d0924c090c/codes/recipe/atpo/run.sh#L109-L128)、[ppo_trainer.yaml#L71-L79](https://github.com/Quark-Medical/ATPO/blob/25af83688c06917f94245a4db06be7d0924c090c/codes/recipe/atpo/config/ppo_trainer.yaml#L71-L79)

## 2. EA-ATPO 当前新增结构、参数、Loss 与 Reward

| 部分 | 当前代码结构与目标 | 实际训练/状态 |
|---|---|---|
| 四动作协议 | ANSWER/ASK/CONSULT/DEFER 的解析、可见状态、患者/专家调用和字段校验；没有 PyTorch 网络或可训练张量。[protocol.py、collect_counterfactuals.py#L380-L465](ea_atpo_stage/recipe/ea_atpo/collect_counterfactuals.py#L380-L465) | rollout/协议实现；不是 Actor action head |
| 手写 Utility | U=Q_out−λ_c C−λ_h H，默认四动作 cost 为 0、0.05、0.20、0.60，harm 全为 0。[utility.py#L11-L40](ea_atpo_stage/recipe/ea_atpo/utility.py#L11-L40) | 固定配置，不训练；不是 Reward Model |
| Absolute Q Router | 公共对话 TF-IDF（最多 3000 项）加 9 个结构数值特征；分别预测四动作正确率 Q 与 3 类 rollout 成本。可选可靠性特征是按病例交叉拟合的 DEFER 正确率。[router_core.py#L25-L103](experiments/ea_atpo_value_router_20261105/router_core.py#L25-L103)、[train_router.py#L141-L181](experiments/ea_atpo_value_router_20261105/train_router.py#L141-L181) | 已拟合 KRR checkpoint；100 病例、291 状态。不是神经网络或梯度优化 |
| Relative Advantage Router | 在同一 state 下构造 ASK/CONSULT/DEFER 相对 ANSWER 的配对差值，输出 3 动作 × 5 个结果/成本指标。只使用 initial public state。[relative_value_router.py#L67-L186](experiments/ea_atpo_value_router_20261105/relative_value_router.py#L67-L186) | 已拟合 KRR；按病例 70 train + 15 validation 选 α=30，再用 85 个病例拟合；15 个 Pilot internal-test 病例未用于拟合 |
| Advice reliability / utilization | 可靠性只作为 KRR 派生特征；CONSULT-Naive/Retain/Critical 是提示词与推理策略对照。没有新的可微 Advice 网络或 Loss | 推理比较/外部特征，不更新 Expert 或 Base 权重 |
| Base / Patient / Expert 模型 | Qwen 服务通过不同 system prompt 和可见信息边界承担不同角色 | 当前 Router 与 Pilot 中只做推理；没有从 Router Loss 回传到 Base/Patient/Expert |

Router 的离线拟合状态包含词表、IDF、数值归一化统计和 dual 系数；闭式求解为：

\[
K=XX^\top,\qquad \alpha=(K+\lambda I)^{-1}Y,\qquad
\hat Y=X_{\mathrm{new}}X^\top\alpha.
\]

代码用 numpy.linalg.solve 直接得到 dual 系数，不调用 optimizer/backward；这等价于线性核岭回归拟合，不是 PPO、神经网络 Q-learning 或 ATPO Critic。[router_core.py#L124-L136](experiments/ea_atpo_value_router_20261105/router_core.py#L124-L136)

Relative Router 的目标为：

\[
\Delta_{a,m}(s)=
\mathbb{E}[m(s,a)-m(s,\mathrm{ANSWER})],
\quad
a\in\{\mathrm{ASK,CONSULT,DEFER}\},
\]

其中 m 含正确率、患者轮数、专家 token、总 token 和延迟。无效 episode 在意向性分析目标中按正确率 0 计；只对 state/action/replicate index 匹配的行求差。[relative_value_router.py#L67-L128](experiments/ea_atpo_value_router_20261105/relative_value_router.py#L67-L128)

| 训练目标/结构 | 当前状态 |
|---|---|
| Absolute Q、相对 Advantage、跨拟合可靠性 KRR | 已用 rollout 标签拟合；有 .npz checkpoint 和 JSON 元数据，属于外部监督 Router |
| ATPO Actor/critic 的四动作联合训练 | 没有当前可复核的接入代码或 checkpoint |
| Four-action macro-action head、共享 Actor 生成动作与内容 | 研究计划/方法文档中提出，尚未实现 |
| 用“新事实获得”替代粗糙 effective_reward 的过程奖励 | 设计与消融方案存在于计划文档，当前 reward/loop 没有这一训练路径 |
| 新的 Huber Q 网络、pairwise ranking loss 或 expert advice utilization 模块 | 文档候选，不是当前训练代码 |
| Qwen3-32B Expert / Qwen3-8B Base 微调 | 没有；相关实验是模型推理比较 |

计划文档明确把四动作接入 ATPO tree/critic/actor 更新列为后续 Stage 3，而非已训练结果。[EA_ATPO_FEASIBLE_METHOD_V1_2026-09-29.md#L63-L76](EA_ATPO_FEASIBLE_METHOD_V1_2026-09-29.md#L63-L76)、[EA_ATPO_EXPERIMENT_PLAN_2026-09-29.md#L3-L7](EA_ATPO_EXPERIMENT_PLAN_2026-09-29.md#L3-L7)

历史交接记录称 Qwen3-4B ATPO baseline 到 step 20、E1-B 在 2 例/4 条轨迹上更新过 Actor 与 Critic；同一交接明确说明 checkpoint、日志和 rollout 没有上传。源码中能看到的是一个独立 patch 文件，不是此处官方 ATPO commit 的已应用改动。因此这些历史记录不能证明当前四动作 Router 正在联合训练。[EA_ATPO_HANDOFF_2026-10-29.md#L53-L77](EA_ATPO_HANDOFF_2026-10-29.md#L53-L77)

## 3. Q 值目标、动作成本与 continuation policy 一致性

### Q 值与成本

Absolute Router 把正确率标签和患者轮数、CONSULT token、DEFER token 分开拟合。决策时再按权重计算：

\[
U_a(s)=\hat Q_{\mathrm{correct}}(s,a)
-\lambda_{\mathrm{ASK}}\hat C_{\mathrm{patient}}(s,a)
-\lambda_{\mathrm{CONSULT}}\hat C_{\mathrm{consult}}(s,a)
-\lambda_{\mathrm{DEFER}}\hat C_{\mathrm{defer}}(s,a).
\]

Relative Router 对 accuracy 和成本分别预测动作相对 ANSWER 的差值，再以
\(\Delta U_a=\Delta Q_a-\lambda_P\Delta C_{P,a}-\lambda_E\Delta C_{E,a}\)
选择协作动作，ANSWER 的相对分数固定为 0。[train_router.py#L161-L181](experiments/ea_atpo_value_router_20261105/train_router.py#L161-L181)、[run_relative_value_experiment.py#L587-L630](experiments/ea_atpo_value_router_20261105/run_relative_value_experiment.py#L587-L630)

**在这两条 Router 路径内部，临床正确率和成本分头拟合、决策时只扣一次，数学口径是成立的。** 但成本权重和单位并非整个项目统一：旧 UtilityConfig 使用人为动作标量成本；Absolute Router 按患者轮数/专家调用 token 定价；Relative 实验使用患者轮数与专家 token 的 λ，总 token 和延迟只报告、不进入该净效用。因此这些实验的 utility 数值不可直接混用。当前也没有被测量并进入 reward 的临床 harm 项；UtilityConfig.harm 默认为 0。

KRR 训练本身是逐输出的 ridge 回归，不是直接优化净效用或保证 action ranking。Relative 标签虽按 state 和 replicate 编号配对，但随机种子上下文含 action/rollout 信息，并非对四动作使用共同随机数；所以差值标签方差仍受不同模型采样影响。

### Continuation policy

Absolute Router 的 rollout Q 标签对应“强制初始动作，然后 Base Agent 按固定策略继续”。旧的 stepwise 部署在每次公开状态更新后重新调用 Router；项目配置也明确写明它把“一步反事实 Q + Base continuation”用于“Router continuation”，存在 policy-distribution shift。[config.yaml#L42-L49](experiments/ea_atpo_value_router_20261105/config.yaml#L42-L49)、[run_stepwise_router_eval.py#L193-L201](experiments/ea_atpo_value_router_20261105/run_stepwise_router_eval.py#L193-L201)

这是当前 Absolute Q / 旧逐轮 Router 的主要数学错配：训练标签估计的是 \(Q^{\pi_{\mathrm{Base}}}(s,a)\)，逐轮部署实际需要的是 \(Q^{\pi_{\mathrm{Router}}}(s,a)\)。除非二者 continuation 相同或重新按当前策略采样，否则旧 Q 不是逐轮 Router 的无偏目标。

2026-11-06 的 Relative Value 实验改成只在初始公开状态选一次动作，随后使用固定 Base continuation；数据也只取 Pilot 的初始状态，并固定 10 个 Assistant turn、最多一次 CONSULT、最后一轮 ANSWER。这个新口径与它的 one-shot 训练目标一致，是当前更可解释的路由评估设计。[relative_value_config.yaml](experiments/ea_atpo_value_router_20261105/relative_value_config.yaml)、[run_relative_value_experiment.py#L487-L497](experiments/ea_atpo_value_router_20261105/run_relative_value_experiment.py#L487-L497)、[run_relative_value_experiment.py#L920-L932](experiments/ea_atpo_value_router_20261105/run_relative_value_experiment.py#L920-L932)

## 4. 最小、可辩护的 ATPO 方法改进

不建议再加一个与 Actor 并列的价值 Router 网络。最低限度的真正方法变化应是：

1. **一个共享 Actor 输出四类 macro-action。** 将动作类型（ANSWER/ASK/CONSULT/DEFER）作为明确的策略 token/字段纳入 Actor 的 response mask 和 log-prob。ANSWER 结束；ASK 调患者后回到同一 Actor；CONSULT 传入具体问题、接收建议后回到同一 Actor；DEFER 由专家独立作答并结束。四动作由策略采样，不再由外部 Router 替 Base 选择。
2. **保留现有 ATPO tree 和标量 Critic。** 树节点代表动作后的环境状态；按动作执行患者/专家转移。把剩余 turn、ASK/CONSULT 预算等放入决策状态或动作 mask，保证状态满足 Markov 条件。除非消融证明不足，先不加四头 Critic。
3. **定义一个净回报并贯穿树回传、Critic target、Advantage、Actor 更新：**
   \[
   r_t^{\mathrm{net}} =
   r_t^{\mathrm{clinical}}
   -\lambda_P c_t^{\mathrm{patient}}
   -\lambda_C c_t^{\mathrm{consult}}
   -\lambda_D c_t^{\mathrm{defer}}.
   \]
   每次实际发生的资源成本仅记一次；终局正确性/安全指标是临床 reward。成本权重在验证集按预注册效用/预算选择，测试集冻结。若用树搜索预计算 Advantage/Return，就用这套净回报计算树目标；不要再由 trainer 对它重复做另一遍 GAE。
4. **复用原 PPO 与 clipped-MSE Critic Loss，不另造 actor loss：**
   \[
   \hat A_t^{\mathrm{net}}
   =\operatorname{GAE}(r^{\mathrm{net}},V_\psi),\qquad
   L_{\mathrm{actor}}=L_{\mathrm{PPO}}(\hat A^{\mathrm{net}})
      +\beta\,KL(\pi_\theta\Vert\pi_{\mathrm{ref}}).
   \]
   这相当于在带成本的 MDP/期望效用目标下优化同一个 policy-gradient estimator；新贡献应落在动作空间、状态转移和成本化目标上，而不是换一个未经验证的损失名字。
5. 若终局奖励稀疏，再单独评估 potential-based shaping：
   \(r'_t=r_t^{\mathrm{net}}+\eta[\gamma\Phi(s_{t+1})-\Phi(s_t)]\)，终止状态令 \(\Phi=0\)。在 Markov 状态与一致折扣等条件下，Ng、Harada、Russell（ICML 1999）的 reward-shaping 结果保证这类势函数变换不改变最优策略；这应作为可选效率改进，不能替代真实临床终局 reward，也不要把“患者可回答”直接当作临床信息增益。

### 预计改动位置与训练参数

| 文件 | 最小改动 |
|---|---|
| codes/recipe/atpo/user_assistant_agent_loop.py | 四动作宏动作的树展开、患者/专家转移、终止条件、budget state、唯一净 reward |
| codes/recipe/atpo/chat_model.py 与 agent_loop.py | 保留动作 token 的生成、response mask、log-prob 与树级 Advantage 对齐 |
| codes/recipe/atpo/mt_reward_fn.py、mt_reward_manager.py | 定义同一净 reward 和 action cost，避免 tree/reward-manager 两套口径 |
| codes/recipe/atpo/mt_trainer.py | 选定唯一的 Advantage/Return 计算路径；确保 Critic targets 与 Actor advantages 来自同一 reward |
| codes/recipe/atpo/config/ppo_trainer.yaml、run.sh | 固定动作预算、成本单位与 λ、树预算和 continuation policy |
| codes/verl/workers/actor/dp_actor.py、core_algos.py | **首版不改 PPO 公式**；只确认采样的动作 token 已进入现有 PPO mask |

更新参数只应是原 ATPO Actor θ 与 Critic ψ；动作 token 共用 Actor 参数，Critic 继续用原 state-value head。成本权重不是可从 gold 推理的输入特征，应在验证集确定并冻结。Patient、Expert 与外部 Router 在本方法里不通过 Actor 梯度更新。

### 最小消融

所有组使用同一病例划分、Base/Patient/Expert 版本、rollout/tree 预算、成本单位和随机种子；按病例配对 bootstrap：

- A. 原 ATPO（原始动作/原始可执行 reward 路径）。
- B. 四动作共享 Actor + 仅终局临床 reward：隔离动作结构增益。
- C. 四动作共享 Actor + 净成本 reward：核心方法。
- D. C 去掉树搜索（N=1）：判断增益来自四动作目标还是 ATPO 搜索。
- E. C 去掉 CONSULT。
- F. C 去掉 DEFER。
- 可选 G. C 加/不加势函数 shaping：只有终局 reward 太稀疏时再做。

报告准确率、各动作使用率、患者/专家工作量、token/延迟、成本—准确率曲线和病例级 CI。独立 Router 应保留为对照基线。若 C 没有在独立病例和相同成本口径下优于 A/always ANSWER，就不把集成结构或 shaping 宣称为有效贡献。

## 最终判断

当前 EA-ATPO 的**新增拟合模型只有外部 KRR Router**（含词表/IDF/数值归一化统计及闭式回归系数）；尚没有新增神经模块、Actor action head、四动作 Critic、树搜索目标或梯度路径来更新 Base Agent。方法层面应表述为“ATPO 基线 + 四动作反事实 rollout + 外部价值 Router”，不能称为已完成的联合 Expert-aware ATPO 优化。

本审计只提出可复核的最小集成方案；没有改动源代码、应用 process-reward patch 或启动训练。
