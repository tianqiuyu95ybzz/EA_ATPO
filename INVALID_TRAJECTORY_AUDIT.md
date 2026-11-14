# Attempt07/08 无效轨迹审计

审计对象：`experiments/ea_atpo_v1_smoke/ea_atpo_v1_attempt07/`，包括 `traces/train.jsonl`、逐步训练日志、rollout 和 manifest。Attempt07 的原始轨迹与 checkpoint 未修改。

## 分类结果

| 情况 | 观测数 | Attempt07 处理 | 是否进入 PPO |
|---|---:|---|---|
| Actor 输出无法解析的协议错误 | 0 | 未发生 | 不适用 |
| Patient/Expert 调用或服务异常 | 0 | 未发生；Patient 的 “cannot answer” 是合法模拟器响应，不是服务异常 | 不适用 |
| 重复 ASK，超过 `max_asks=1` | 2 个无效 ASK，来自 2 条 episode | `ask_budget_exhausted` 终止；无 Patient 调用，无重复 ASK 成本 | 是 |
| ASK 专属预算耗尽 | 2 条 episode | 是。触发时全局还剩 2 个决策位置，因此不是全局交互预算耗尽 | 是 |
| 全局交互预算耗尽 | 0 | 没有达到 `max_actor_actions=3` | 不适用 |
| 终局答案错误 | 2 条 episode | `ANSWER` 终止，正确性奖励为 0 | 是 |

4 条 rollout 来自 2 个病例，每例重复 2 条。两条有效终局路径为 `ASK → CONSULT → ANSWER`，最终答案均错误，episode 净回报各为 `-0.25`。另外两条为 `ASK → ASK`，第二个 ASK 触发上限，未输出最终答案，净回报各为 `-0.05`。

第二个 ASK 仍是 Actor 生成的策略动作，不是模拟器故障。它没有外部调用，边奖励为 0；前一个有效 ASK 的 `-0.05` 成本保留。两个无效 ASK 的 Advantage 分别为 `-2.5125`、`-2.6055`。这两条轨迹没有被删除，进入了 Actor/Critic 更新。

错误的终局答案同样保留训练：ANSWER 的终局正确性奖励为 0，先前 ASK/CONSULT 成本分别扣一次。两条轨迹净回报各为 `-0.25`。不应因为答案错误而把这些策略失败从 PPO 样本中剔除。

`custom/num_valid` 只表示完成文本中能否解析出一个终局答案：Step 1 为 2/2，Step 2 为 0/2。它不是 Actor 动作格式有效率，也没有用作 PPO 样本过滤条件。Step 2 的 0/2 轨迹仍生成了完整的 `AgentLoopOutput`。

## 协议/环境错误的修正

Attempt07 没有观察到协议解析或 Patient/Expert 服务异常。不过，原实现会把这类异常转换成终止叶节点并继续送入 PPO；这会将基础设施失败误当成策略回报。当前代码在 commit `2bf65f4` 中加入 `failure_category` 和 `training_eligible`：

- `invalid_actor_action` 标为 `protocol_error`；
- Actor generation、Patient、CONSULT Expert、DEFER Expert 调用异常标为 `environment_error`；
- 以上轨迹照常写入 JSONL，但不进入 PPO；
- 重复 ASK、动作预算违规和错误终局答案标为策略结果，继续进入 PPO。

这是对当前代码的修正，不会回写或重新解释 Attempt07 的原始结果。

## Attempt08：样本级异常曾中止整个 PPO batch

审计对象：`experiments/ea_atpo_v1_smoke/ea_atpo_v1_attempt08_stability/`。原始 24 条轨迹保留，训练进程在 2 次 PPO 更新后因一个样本的全部候选都发生 DEFER 专家格式错误而退出。Attempt08 没有 OOM，也没有保存 checkpoint（配置每 5 步保存，实际只完成 2 步）。

| 轨迹结果 | 数量 | 训练处理 |
|---|---:|---|
| 可训练策略结果 | 17/24 | 包括终局答错和 4 条重复 ASK 预算违规轨迹 |
| DEFER 专家格式错误 | 6/24 | `Expert Final Answer` 不满足严格解析；环境错误，不应给 Actor 负奖励 |
| CONSULT 泄漏最终选项 | 1/24 | 专家格式错误，保留轨迹审计，不应给 Actor 负奖励 |

动作级计数为 ASK 28、CONSULT 20、ANSWER 13、DEFER 6。24 条轨迹均从 ASK 开始；其中 4 条在 ASK 上限已用尽后再次 ASK，仍属于策略约束违规并保留训练资格。20 次 CONSULT 中 1 次因建议泄露最终答案而失败；6 次 DEFER 均因专家最终答案格式错误而未形成有效终局样本。

Attempt08 暴露了原过滤实现的批次级副作用：AgentLoop 为每个输入样本独立生成候选，但一个样本的所有叶子都不可训练时会抛异常。`asyncio.gather` 随即取消整个 worker batch，导致同批其他样本的有效策略轨迹不能进入 PPO。Attempt08 的第三个训练 batch 中，`case_id=6886` 的 4 条候选全部 DEFER 专家格式错误；`case_id=4256` 的 4 条重复 ASK 轨迹虽是可训练策略失败，也随 batch 中止而没有进入更新。原始 trace 仍完整保留。

### Expert 解析失败定位与证据边界

Attempt08 记录了 7 个 Expert 协议失败：6 个 DEFER，错误为 `DEFER response must contain exactly one final option`；1 个 CONSULT，错误为 `CONSULT leaked or asserted a final answer`。但当时失败事件没有保存 Expert 原始文本、completion token 数或 finish reason，Attempt08 主日志也没有这些内容。因此无法恢复“原始输出是什么”，也不能断言实际输出是 JSON、思维文本、某个具体字段缺失，或被截断。错误信息仅能证明输出不符合当时严格校验：DEFER 未完整匹配 `Expert Final Answer: <A-D>`；CONSULT 触发了 `Final Answer`、`correct answer/option` 或 `option A-D` 的泄漏检测之一。

按动作级专家请求计，26 次 Expert 调用有 7 次协议失败，Attempt08 未重试前失败率为 26.9%；按 episode 计，7/24=29.2%。

Attempt08 的 `_role_generate` 使用 chat template 的 `enable_thinking=False`，Expert 每次上限为 384 tokens。Thinking 因此不是预期格式，但由于原文未保存，不能排除模型仍生成非预期标签。旧 `TokenOutput` 只提供 token IDs 和 log-prob，没有 finish reason；失败轨迹又未记 token 数，所以也无法审计截断。原解析器没有 JSON 解析或容错；DEFER 的严格全文匹配也会拒绝带有额外文本或尾部传输标记的答案。

脱敏回归输入见 `ea_atpo_stage/recipe/ea_atpo/testdata/attempt08_expert_failures.jsonl`。文件保留真实错误类别，但明确标记原始响应不可恢复；测试文本只是代表性 parser trigger，不冒充 Qwen3-8B 的原始输出。CONSULT 测试覆盖“建议中断言正确答案”；DEFER 测试覆盖 Qwen 传输终止符归一化。

### EA 层最小修复

为避免改动 `ATPO-official`，EA AgentLoop 现在在“单个样本没有任何可训练叶子”时返回一条全零 response mask 的占位行。它只满足上游批次结构，不包含 Actor/Critic 训练 token；原始失败原因仍写在逐 episode JSONL。有效兄弟样本继续进入 PPO。`EARewardManager` 在整个 batch 的 Actor mask 全零时于 PPO 前抛出明确异常，避免 Critic/Actor 在无有效 token 时产生 NaN 更新。

Expert 侧现在每次解析失败只做一次格式修复重试，固定温度 0，不向提示词加入 gold 或 hidden facts。若重试仍失败，错误类型记为环境/协议失败并隔离 episode。失败事件现在记录最多 500 字符的原始响应、请求与重试次数、prompt/completion token 数和延迟。由于 TokenOutput 没有 finish reason，另外记录 `completion_at_token_cap` 作为触顶代理，不把它冒称为准确的截断原因。

回归测试覆盖了 Attempt08 两种解析失败类别、一次有限重试、尾部终止符归一化、占位行 mask、Reward Manager 的全失败拦截，以及四动作工程测试，8/8 通过。修复尚未在真实多样本模型 rollout 中验证；本轮检查时 GPU4/6 都只有约 56 GiB 可用且利用率为 100%，不能据此声称 Attempt08 问题已通过端到端复验。

## 证据文件

- 原始轨迹：`experiments/ea_atpo_v1_smoke/ea_atpo_v1_attempt07/traces/train.jsonl`
- 原始逐步统计：`experiments/ea_atpo_v1_smoke/ea_atpo_v1_attempt07/logs/ea_atpo_v1_smoke_20261114T034613Z.log`
- 动作级汇总：`experiments/ea_atpo_v1_smoke/ea_atpo_v1_attempt07/action_gradient_coverage.csv`
- Attempt07 汇总：`experiments/ea_atpo_v1_smoke/ea_atpo_v1_attempt07/RESULTS.md`
- Attempt08 轨迹和失败日志：`experiments/ea_atpo_v1_smoke/ea_atpo_v1_attempt08_stability/traces/train.jsonl`、`logs/ea_atpo_v1_smoke_20261114T044413Z.log`
