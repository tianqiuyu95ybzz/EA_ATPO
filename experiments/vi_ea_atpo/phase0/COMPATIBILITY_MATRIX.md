# Phase 0 — 兼容性矩阵

| 能力 | 状态 | 当前源码 / 说明 |
|---|---|---|
| H-MDP 公开状态、动作计数和预算 | Existing | ea_atpo_stage/recipe/ea_atpo/protocol.py:18-53 |
| ANSWER 终局 Base 输出 | Existing | ea_agent_loop.py:695-706 |
| ASK→Patient hidden facts→Base continue | Existing | ea_agent_loop.py:708-754；Gold 不传给 Patient |
| CONSULT→Expert advice→Base continue | Existing | ea_agent_loop.py:756-820；独立成本和调用次数上限 |
| DEFER→Expert final→terminal | Existing | ea_agent_loop.py:822-874 |
| Expert 与患者的信息隔离 | Existing | protocol.py:91-128；当前独立测试覆盖 |
| 独立 Base / Patient / Expert API | Requires Adapter | ChatAPI 有三个 client；训练 EAATPOAgentLoop 目前 Actor/Expert 共用 server_manager |
| 将既有 Qwen3-32B-AWQ 接入训练 CONSULT/DEFER | Requires Adapter | 增加独立 Expert client/config，保持原协议与 reward；本阶段不改原算法 |
| 分开记录公开特征与离线 label | Requires New Module | Phase1 case-record exporter |
| InitialDeferralValueEstimator：helpful/harmful 概率和净收益 | Requires New Module | Phase1 CPU estimator，独立于 Critic |
| Phase2 共同初始状态下四动作配对干预 | Requires Adapter | 复用 protocol、ChatAPI 和 counterfactual helpers；限制为一次 ASK/CONSULT 后固定终止规则 |
| 信息前可观测的 Critic value | Existing | call_critic_model；真实准确性和校准仍需审计 |
| Critic value uncertainty / calibrated epistemic uncertainty | Not Available | 当前 Critic 返回单值，没有方差或预测区间接口 |
| ATPO 原树扩展及自适应剪枝 | Existing | user_assistant_agent_loop.py:236-372 |
| 以 VOI 改变展开优先级或预算 | Requires New Module | Phase2 通过后再建 VOISearchAdapter；当前不得接入 |
| 原始树 return / advantage | Existing | user_assistant_agent_loop.py:419-523 |
| PPO Actor 与 clipped Critic loss | Existing | verl/workers/actor/dp_actor.py:443-488；workers/critic/dp_critic.py:223-255 |
| 搜索采样/保留概率记录及 off-policy 修正 | Unverified | Actor response logprob 存在；树剪枝留存概率未记录，当前没有明确 importance correction |
| lambda_voi=0 行为兼容测试 | Not Available | 集成前必须新增并固定 seed 对照 |
| Phase2 干预是否可预测 | Unverified | 不可从现有静态 Router 结果推出；先导实验未运行 |
| 官方独立泛化测试 | Unverified | 400 case holdout 非官方 test 且已反复查看 |
| Phase5 大规模 PPO/ATPO | Not Available | 本轮禁止启动 |

## 需要保留的实现差异

当前 training EA 的 Expert 角色是单模型客户端上的角色提示词，而不是独立的 Qwen3-32B-AWQ 服务。counterfactual collector 和真实 pilot endpoint 配置可以拆开角色 API，但训练 AgentLoop 尚不能。任何后续接入都需要显式 expert_client 适配，不得将 CONSULT 替换为 DEFER。

原 ATPO search 是按层扩展并按剪枝规则随机/整体保留 child，不是优先级队列。VOI 首版应只改变已有搜索中的展开预算/候选顺序，且必须检查入选样本对 PPO 行为分布的影响；不能用一个想象出来的 Bellman-error+uncertainty 评分替代源码算法。
