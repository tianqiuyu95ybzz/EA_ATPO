# 四动作自然探索审计

## Attempt07：自然样本覆盖不足

Attempt07 的 4 条模型 rollout 共生成 10 个 Actor 动作：

| 动作 | 采样数 | 有效数 | PPO 梯度覆盖代理 |
|---|---:|---:|---:|
| ANSWER | 2 | 2 | 2 个动作段 / 10 token |
| ASK | 6 | 4 | 6 个动作段 / 128 token |
| CONSULT | 2 | 2 | 2 个动作段 / 50 token |
| DEFER | 0 | 0 | 0 |

四条 rollout 只覆盖两个训练病例，且每个病例仅有两个采样。强制动作协议测试另行运行，没有送入 PPO。DEFER 的格式、环境转移和终止逻辑通过了独立测试，所以其缺席不是解析器不识别动作，也不是树搜索硬编码排除 DEFER；树搜索只是保留 Actor 的自然生成结果。

Attempt07 保存了整段响应联合 log-probability，但没有记录未采样动作的逐 token 条件概率或 action-level logits。因此无法从这些数据估计 `P(DEFER | state)`，也无法判断是概率低还是 4 次 rollout 偶然未抽中。初始状态提示虽列出 DEFER，但没有专门解释何时应将决策交给专家。DEFER 成本 0.60 高于 ASK 的 0.05 与 CONSULT 的 0.20；对正确性奖励 1.0 而言，这会提高选择 DEFER 所需的预期正确性，但现有样本不能把成本效应与低采样量分开。

## Attempt08：自然生成了 DEFER，但没有有效 DEFER 训练信号

Attempt08 使用状态提示中的合法动作列表，并把 `M_trajectories=N_candidates=4`。所有候选仍由当前 Actor 按随机策略自然生成，响应 token 的 log-probability 与采样使用同一提示上下文；没有用强制动作或事后标签构造 PPO 轨迹，因此无需行为策略重要性校正。合法动作列表是自然语言上下文，不是 token 级强制解码；模型仍可能违反上限，违规 ASK/CONSULT 会被记录为策略约束失败并保留 PPO 训练信号。

训练从官方训练集抽取 20 个病例，计划 10 个 optimizer step；独立验证集抽取 20 个病例。每个输入状态采样 4 个候选，训练成本为 `ASK=0.05 / CONSULT=0.20 / DEFER=0.60`。进程只完成 2 步，在第三个 batch 遇到全 DEFER 格式失败后退出；没有运行验证，也没有保存 checkpoint。

24 条原始 episode trace 的自然动作事件计数为：ASK 28、CONSULT 20、ANSWER 13、DEFER 6。24 条轨迹全部先选择 ASK；4 条在 ASK 预算用尽后再次 ASK，作为策略违规保留。20 次 CONSULT 中 19 次返回符合协议的建议，1 次输出包含最终选项而被拒绝。6 次 DEFER 全部因 Expert 未输出可解析的单一最终选项失败。因此 DEFER **有自然采样**，但有效终局和 PPO 训练覆盖仍为 0；不能说 Actor 学会了 DEFER。

Attempt08 的每步 PPO 日志显示完成 2 步，分别有 6 和 8 条 rollout 输出；`custom/num_valid` 为 6/6、8/8，正确数为 0/6 和 4/8。这个指标统计可解析终局答案，不是动作协议成功率。总 episode trace 中有 17 条策略结果可训练、7 条专家协议/环境错误不可训练，其中 4 条 ASK 上限违规轨迹属于可训练策略结果。因第三个 batch 中一个病例的所有 DEFER 专家调用失败，旧样本级异常中止了整批，另一病例的 4 条重复 ASK 轨迹没能进入 PPO。Attempt08 未通过 10 步稳定性验证，不能用两步结果声称四动作学习成功。

修复后，失败样本用全零 Actor response mask 占位，使其同批有效兄弟样本能够保留；若全批都没有可训练 Actor token，自定义 Reward Manager 会在 PPO 前停止。该修复通过 8/8 单测，真实模型的端到端确认仍待资源允许后进行。

## Expert 输出失败定位补充

Attempt08 的原始 Expert 文本没有写入 trace 或主日志，不能恢复精确字符串。可确认的内容只有 6 次 DEFER 严格格式校验失败，以及 1 次 CONSULT 命中“泄露/断言答案”正则。Attempt08 的 tokenizer template 显式设置 `enable_thinking=False`，并将 Expert generation cap 设为 384 tokens；但失败事件没有 token 数或 stop reason，无法归因到 JSON、thinking 片段或截断。

现在的解析顺序是先去除完整的 `<think>...</think>` 前缀，再移除 Qwen 传输结束符，随后校验角色格式；格式失败最多进行一次只包含格式说明的重试。CONSULT 仍拒绝最终答案泄漏；DEFER 仍只接收一项 A-D 答案。失败原文最多保存 500 字符，prompt/completion token、重试数及“达到 token cap”代理都进入事件记录。`TokenOutput` 不提供 finish reason，所以 token cap 只用于告警统计，不等同已证实截断。

Attempt08 回归输入位于 `ea_atpo_stage/recipe/ea_atpo/testdata/attempt08_expert_failures.jsonl`。其中只保留真实错误类别；示例字符串明确标记为代表性 parser 输入，不是恢复出来的模型输出。当前测试 **8/8 通过**，但 GPU4/6 的显存门槛未满足，尚无修复后的自然 DEFER 成功或 retry 失败率数据。

## 不做的推断

- 不能因为 DEFER 0 次采样就断言模型对 DEFER 概率为 0。
- 不能仅凭输出频率宣称模型已学会四动作。
- 不能把独立的强制动作测试计入自然 rollout 统计。
- 若后续加入 epsilon 强制动作或改写动作 token 分布，必须按实际混合行为策略校正 PPO log-probability；Attempt08 没有采用这种方式。
- Attempt08 虽自然采样 6 次 DEFER，但均因专家输出格式失败，不能算有效 DEFER reward/advantage/梯度覆盖。
- Attempt08 的失败原文未保留，代表性回归输入不能当作真实 Qwen 输出或用于判断错误成因。
