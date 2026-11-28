# ATPO 树搜索代码审计与 V3 继承范围

核验了 Quark-Medical/ATPO 远端 HEAD `25af83688c06917f94245a4db06be7d0924c090c`，本地 fetch 与 amax 的 ls-remote 一致。EA-ATPO 对照提交为 `ca6e2b28a2176fde71869369fb0e49088815de7b`。本轮只审源代码和接口，使用固定源码和项目已有审计。没有启动上游服务或 RL。

| 机制 | 上游代码行为 | V3 处置 |
| --- | --- | --- |
| 根节点、前沿、父子边、终止管理 | `TreeNode` 保存轨迹、父子关系、是否终止、奖励与 Q。实际 rollout 由 `TreeSearchRunner` 驱动，`TreeSearchManager` 是通用前沿结构 | 保留节点语义、分层展开、预算、终止和轨迹记录。节点状态只含当前可见对话、有效动作、剩余预算 |
| 展开和环境反馈 | `_expand_leaves_with_assistant` 对活跃叶子批量生成子候选，达到预算时减为单候选。异步调用患者与 verifier。原实现由问题或终局回答协议分流 | 保留候选调度和适用的患者反馈接口。把候选固定在 ANSWER、ASK、DEFER 三条语义支路。ASK 调用患者，获得反馈后按固定继续策略推进。ANSWER 和 DEFER 分别以 Base 或专家最终输出终止 |
| 剪枝 | 根据父 Critic 值与子 Q 均值之差、子 Q 方差的组合决定是否缩减。在达到条件后还含概率门控及随机保留子节点 | 在估计可靠 Q 之前保留三类有效动作的最低覆盖。首轮配对关闭价值剪枝。后续可对同动作的多个候选做不确定性剪枝，记录预算、随机种子和覆盖 |
| 树值回传 | 终点后续值为 0。自底向上对可回传子节点的 `r+γV` 取均值。另有 Critic 估值和优势路径 | 保留回传框架。首轮由已观测回报和已冻结的继续策略定义 Q。不同动作及其样本覆盖分开。未知动作回报保留空值与掩码 |
| 奖励 | `mt_reward_fn.py` 计算过程分和终局正确性。标准树路径非终止 `action_reward=0.0`，终局接 outcome。标准 reward manager 返回 outcome tensor | 复用阶段奖励记录接口。质量、ASK 成本、DEFER 成本在同一账本中分开记录，每次实际动作只扣一次。ReMeDi 槽值增量只作描述，无法填入质量回报 |
| 异步与缓存 | rollout 并发生成并处理节点。`agent_loop.py` 的请求 ID 绑定服务端，支持 prefix cache 复用 | 在数据和评分协议验证后再启用性能优化。共享前缀及请求身份必须包含 Base 版本和公开状态，缓存只限当次运行 |
| 模型更新 | 上游 launch 使用 GAE/VeRL，trainer 更新 Actor 和 Critic | 本研究保持 Base 冻结。在数据够用后单独运行既有 embedding、掩码 Q 回归/蒸馏和轻量 MLP 路径。当前不启动上游 PPO |

[原始节点定义](https://github.com/Quark-Medical/ATPO/blob/25af83688c06917f94245a4db06be7d0924c090c/codes/recipe/atpo/tree_search_manager.py) · [实际 rollout、剪枝与回传](https://github.com/Quark-Medical/ATPO/blob/25af83688c06917f94245a4db06be7d0924c090c/codes/recipe/atpo/user_assistant_agent_loop.py#L225) · [会话缓存](https://github.com/Quark-Medical/ATPO/blob/25af83688c06917f94245a4db06be7d0924c090c/codes/recipe/atpo/agent_loop.py#L48) · [训练入口](https://github.com/Quark-Medical/ATPO/blob/25af83688c06917f94245a4db06be7d0924c090c/codes/recipe/atpo/run.sh)

EA-ATPO 已有 `collect_counterfactuals.py` 的独立 rollout，分离公共状态、隐含患者事实与离线 Gold，并已有终止 DEFER。它是旧四动作、多选标签协议，V3 需只保留 ASK、ANSWER、DEFER，改接当前数据的状态、患者接口和评价器。V2 的 `training/data.py`、`extract_state_embeddings.py`、`train_policy.py` 和验收代码可以复用。沿用其中的数据隔离与掩码检查。历史关于联合四动作 PPO 的建议不进入此次 V3 方案。 [已有采集器](https://github.com/tianqiuyu95ybzz/EA_ATPO/blob/ca6e2b28a2176fde71869369fb0e49088815de7b/ea_atpo_stage/recipe/ea_atpo/collect_counterfactuals.py) · [已有 V2 训练框架](https://github.com/tianqiuyu95ybzz/EA_ATPO/tree/ca6e2b28a2176fde71869369fb0e49088815de7b/experiments/ea_atpo_v2/training)

## 三动作数据生产前的验收

1. 给定同一病例与同一决策前公开状态，三个动作在独立环境副本执行。把动作不可用与解析失败分开记录。
2. ANSWER 直接终止。ASK 获取实际或经核验的模拟反馈，后续 Base 只看到允许公开的新增信息。DEFER 终止，专家输入固定为决策时刻可见状态。理想 Oracle 可以在离线评价端访问 Gold，Base-Protecting Oracle 的事后判断标明为上界。
3. 固定质量定义、继续策略、采样数、代价与缺失动作规则。忠实记录状态、动作、反馈、结果和成本。初始等覆盖实验先于剪枝验证。
4. ReMeDi 用结构化一致性，IMCS-21 用标注症状获知和受限分类，DDXPlus 用合成病种标签及 Oracle 上界。各数据集保留各自的评价单位。正式三动作 Q 和树生产均等待同数据同状态的可比结果。
