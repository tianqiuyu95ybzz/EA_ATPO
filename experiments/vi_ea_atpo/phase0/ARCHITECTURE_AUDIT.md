# Phase 0 — 原 ATPO / EA-ATPO 架构审计

日期：2026-11-18

## 仓库与执行环境

- 仓库根目录：/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO_V1
- 分支：ea-atpo-v1
- HEAD：2aebfa8ce33db872b9b4c204377a4a11cfb09c97（Record Qwen32 Router generalization results）
- 用户提到的 2aedfa8 在当前对象库不存在：git cat-file 返回 “Not a valid object name”。
- 开始时跟踪文件无修改；工作区有 15 个既存未跟踪文件组，包括 .task_cache、Attempt01–09/original ATPO smoke 数据，以及 relative_gain_predictability_20261118。它们均保留。
- 当前终端为 Linux x86_64，仓库 .venv 为 Python 3.12 系列；研究数据脚本使用 .venv 的 pandas/pyarrow 环境。
- 当前可见 GPU 为 8 张 B20Z（每张约 183,359 MiB）。检查时 GPU 2 利用率 96%、GPU 4 利用率 100%；GPU 0/1 也有大量显存占用。模型端点 127.0.0.1:30000、30131、30132 的 /v1/models 均返回连接失败。没有启动模型服务或 GPU 任务。
- EA 真实入口：ea_atpo_stage/recipe/ea_atpo/mt_main_ea.py；自定义 AgentLoop：ea_atpo_stage/recipe/ea_atpo/ea_agent_loop.py；配置：ea_atpo_stage/recipe/ea_atpo/config/feasibility.yaml、agent.yaml；原 ATPO 入口：ATPO-official/codes/recipe/atpo/mt_main.py。

## H-MDP 状态与动作协议

自定义 EA 状态与协议定义于 ea_atpo_stage/recipe/ea_atpo/protocol.py:18-53。Action 枚举为 ANSWER、ASK、CONSULT、DEFER。PublicState 保存初始公开消息、对话、已 ASK/CONSULT 次数、剩余交互预算和已用 Actor 动作数。state_id 只哈希公开消息和动作计数，不含 Gold 或隐藏事实。

Actor 的可见动作规则和终止定义在 ea_atpo_stage/recipe/ea_atpo/ea_agent_loop.py:47-72；每个动作实际转移在同文件 609-874：

- ANSWER（695-706）：Base 直接返回 A–D 并终止。
- ASK（708-754）：只把问题交给 Patient 角色；Patient 的输入含 atomic_facts，回答变为公开 observation；然后同一 Base Actor 继续。
- CONSULT（756-820）：Base 生成具体咨询问题；Expert 只见公开病例/对话及咨询焦点，返回建议 observation；同一 Base Actor 继续。咨询有单独成本，并受 max_consults 限制。
- DEFER（822-874）：Expert 根据公开状态直接给最终选项；episode 终止。
- CONSULT 和 DEFER 共用 _expert_response/server_manager 的调用通道，但使用 protocol.py:56-84 的不同 system prompt、约束和提示正文；CONSULT 允许建议、不允许最终选项，DEFER 只能返回最终选项。当前训练 AgentLoop 没有独立 Expert model/client 字段：Actor 与 Expert 都走 configurable.server_manager（ea_agent_loop.py:398-416、1024-1066）。因此默认训练协议虽有 Expert 语义，却不等于用户既有实验中的 Qwen3-32B-AWQ 独立专家；要接入独立专家需 adapter。对照采集器 collect_counterfactuals.py 支持 base/patient/expert 三个独立 API client。
- _legal_actions 和计数限制（ea_agent_loop.py:61-72、659-664）允许最多 max_asks/max_consults 次；缺省配置为 2，动作总预算取 max_actor_actions（通常 4）。剩余动作不够继续时 ASK/CONSULT 不合法；无效动作不会调用外部角色。
- Patient 使用 hidden atomic_facts 是环境接口设计；Expert 请求由 validate_expert_view 检查，不含隐藏事实或 Gold（protocol.py:91-128）。Gold 通过 runner 配置传给离线评估，但只在生成后 _evaluate_nodes 中用于终局正确性（ea_agent_loop.py:876-907）。

## Actor、Critic、搜索和 PPO

Actor 调用以 ATPO 的 ChatModel/server manager 为后端。EAChatModel 和 EAATPOAgentLoop 分别在 ea_agent_loop.py:248-364、995-1097。仓库没有一套新建的 EA 专属 Transformer Actor。

原 ATPO 的树搜索 runner 为 ATPO-official/codes/recipe/atpo/user_assistant_agent_loop.py:224-525，节点结构为 tree_search_manager.py:63-89。搜索按当前活跃叶扩展（user_assistant_agent_loop.py:236-297）：每个 leaf 请求 N_candidates，收到全部同层候选后运行 environment，再进行剪枝。这是分层扩展，不是按节点分数排序的 best-first priority queue。

原剪枝逻辑位于 user_assistant_agent_loop.py:336-372：计算候选 q_value 方差、可选按历史统计量标准化，并计算 Critic value 与子节点平均 Q 的绝对差；0.3×标准化方差 + 0.7×绝对差低于 variance_threshold 时，以 0.9 概率随机保留一个 child，否则保留候选。这里的差值是剪枝信号，并非源码命名的 Bellman-error 模块。

EA 沿用上述 runner，只重载动作展开和评价；EAATPOSearchRunner 未实现 VOI priority。EA 的节点 Q 为动作边 reward + gamma×非终局 Critic value（ea_agent_loop.py:876-907）；树备份和 Advantage 沿用原实现（user_assistant_agent_loop.py:419-523）。树状态值是尚未剪枝子节点 net Q 的均值；advantage 按树状态值/critic 值分支计算并除以节点 explore_num。

Critic 预测入口为 user_assistant_agent_loop.py:125-149，转给 recipe/atpo/agent_loop.py 中的 CriticManagerWorker.predict；Actor 和 Critic 网络/worker 属于 ATPO/verl 的共享训练代码。Actor PPO token loss 在 ATPO-official/codes/verl/workers/actor/dp_actor.py:443-488，Critic clipped value loss 在 workers/critic/dp_critic.py:223-255。Critic loss/Actor loss 与配置由 ATPO recipe 的 actor/critic YAML 管理。

不确定性来源目前有两种不同概念：树内 q_value_variance 和 Critic-vs-child-Q discrepancy 被用于剪枝；Critic 只输出 state value，没有显式 epistemic uncertainty 接口。不能把这两者直接当作 VOI。

## 奖励、成本与折扣

当前 EA 训练奖励的实现见 ea_atpo_stage/recipe/ea_atpo/ea_agent_loop.py:543-550、876-907；EARewardManager 见 ea_reward_manager.py:33-107。奖励为终局正确性（默认 1）加上动作边成本（ASK 默认 −0.05、CONSULT −0.20、DEFER −0.60）。ANSWER 无成本。非终局 ASK/CONSULT edge 先扣成本；终局回答 edge 加正确性。gamma 取 ATPO 算法配置；Attempt07 实际使用 gamma=1。

ea_reward.py:19-55 只用于校验/报告相同 episode score；RewardManager 直接透传 rm_scores，不再重算或扣成本。没有 ASK/CONSULT 的 token/latency 货币成本写入训练 reward；这些由日志另记。训练奖励尺度与离线 Accuracy−c_D×DEFER_rate 相同的仅是正确性单位和 DEFER cost 默认值，不应把 token 成本或训练时 gamma 之外的因素混入离线 Utility。CONSULT 后再 DEFER 时，两条 action edge 各扣一次 c_C+c_D；多次允许的 ASK/CONSULT 按次数分别计费。

旧 ATPO 原奖励在 mt_reward_fn.py:76 以及 user_assistant_agent_loop.py:396-417，含 process/verifier/终局逻辑。EA runner 重载 _evaluate_nodes，因此 PPO 训练的 EA reward 以 EA 代码为准，不可用旧 reward 替换。

## 数据与已有结果定位

1,700 例配对数据在 experiments/real_case_pilot_20261103/audit/outputs/qwen32_router_generalization_20261118_attempt04/paired_rollouts/，固定 1000/300/400 split。400 holdout 来自 ATPO train split，旧报告已查看，不是官方 test，也不是新的无泄漏最终测试。新 Router 复核在 audit/relative_gain_predictability_20261118/，其中选项字段错误填零已判为不可用，旧 holdout 再用仍有研究者自由度风险。

另有原四动作可行性实验 experiments/real_case_pilot_20261103/outputs/full/，100 个 train case、最多 3 个公开状态/例、每动作 3 次 rollout。它是历史探索样本，不能当 Phase2 新的未使用样本。已有 CONSULT advice-utilization 200 状态和 CONSULT hint 20 例实验也已消耗过各自样本，须从新 pilot 的历史排除中剔除。

## Phase 0 结论

当前真实 H-MDP 与四动作实现已存在，ASK/CONSULT/DEFER 的关键语义匹配预期。未实现的是：外部独立 32B 专家接入当前训练 AgentLoop；标准化信息获取数据集；预测前 VOI；节点 VOI 排序/预算分配；VOI 置信度；独立留出验证。详见 COMPATIBILITY_MATRIX.md。Phase 0 未修改任何原始算法代码。
