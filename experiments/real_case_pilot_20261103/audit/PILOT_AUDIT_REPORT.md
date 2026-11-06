# EA-ATPO Pilot 轨迹审计

## 范围与方法

本报告只读审计既有 `outputs/full/` 产物，没有改写 Pilot 轨迹或汇总文件。核对了 `REAL_PILOT_REPORT.md`、原始轨迹 JSONL、公开状态 JSONL、汇总 CSV、动作价值 CSV、运行清单、runner/协议实现和 vLLM 服务日志。逐状态重算和哈希证据见 `outputs/pilot_audit_metrics.json`；原始文件哈希写在该 JSON 中。

## 审计结论

### 初始状态与 continuation

- 文件中有 3,492 条动作轨迹、100 个病例、291 个公开状态。每个状态都有 ANSWER、ASK、CONSULT、DEFER 四动作，每动作 3 次；共 873 组完整的“状态 × 重复次数”四动作配对，没有缺失分支。
- 对每个病例/状态，3,492 条轨迹中的 `public_state` 序列化内容完全相同，`state_id` 的消息哈希全部匹配。四动作的初始比较没有混入不同患者回答或不同公开历史。
- ASK 和 CONSULT 都进入同一个 `_continue_base` continuation 函数：同一 Base 模型、温度、每轮动作语法和最多 9 个后续 Base turn；最后一轮强制终答。两者的可用动作预算有一个协议差异：ASK 后仍可 CONSULT 一次，初始 CONSULT 已用尽 CONSULT 预算，后续不能再咨询。ASK 轨迹因此允许“先问患者、之后再咨询”，CONSULT 轨迹则不允许二次咨询；结果解释应将它视作当前预注册预算下的动作定义。
- ASK 分支将 Patient 的回答加入该分支的公开对话；CONSULT 分支将 Expert 建议加入公开对话。两者之后的 Base 决策规则相同，但收到的信息不同。DEFER 是终止分支，Expert 读取初始公开状态并独立给出终答。

### CONSULT 的信息隔离与答案泄露

- 代码构造 Expert 请求时只传公开状态和具体咨询问题；Patient 的 `atomic_facts` 不传给 Expert，Gold 只在推理结束后用于评分。每次 Expert 请求前，`validate_expert_view` 会检查特权字段、未公开 atomic fact 原文和值。
- 这能证明请求构造遵守字段隔离，但不能证明模型语义上绝不泄露答案。CONSULT 输出校验只拦截规定格式、`Final Answer`、`correct answer` 和 `option A-D` 等明显答案表达，不能检测所有隐晦暗示或直接说出正确诊断。
- 原轨迹记录了 3 条 CONSULT 泄露校验失败和 1 条 CONSULT 格式失败，但由于异常在 Expert 响应进入 trace 前抛出，原始违规文本没有保存。因此只能确认校验拒绝了这些响应，不能复核具体措辞。

### 失败轨迹与有效样本偏差

- 27 条无效轨迹都保留在 JSONL，并以 0 正确率进入汇总，没有先筛掉无效样本：CONSULT 8 条、DEFER 14 条、ASK 5 条、ANSWER 0 条。全数为 Expert 输出协议校验失败，不是数据或服务中断。
- 失败集中在 7 个病例；源行 3359 占 18/27（66.7%）。该题干描述脑动脉破裂，选项却是寄生虫名称，数据题干与选项明显不匹配；Expert 多次无法按“一定输出 A-D”协议作答。其余失败病例/动作见 `pilot_audit_metrics.json`。
- 因此，准确率使用的是计划样本而非“仅有效轨迹”子集，避免了完整案例筛选偏差；但将协议失败记 0 会把模型格式失败与医学判断错误合并。报告准确率应理解为“任务协议下的端到端成功率”。

### Token、延迟与成本记录

- 有效调用的 prompt/completion token usage 和客户端请求时间会累加到每条轨迹。失败 Expert 响应在验证抛错前没有把 API usage 写入 trace；27 个被拒绝的 Expert 调用成本和延迟缺失。失败分支的成本均值是下界，不能当成完整成本。
- 原服务 `max_num_seqs=4`；runner 用 `asyncio.gather` 并行执行同一状态下四个动作。vLLM 日志中 168 个 prefix-cache 采样点命中率为 1.6%–94.2%，均值 88.6%。完整 token usage 仍按请求 prompt 计数，但缓存和并行批处理会改变本地预填充计算量及延迟。
- `total_latency_s` 是 trace 中已记录 API 请求延迟的总和，不是 episode 实际墙钟时间。四动作并发时请求有重叠，跨请求又共享 prefix cache；均值不能直接外推为串行部署延迟。Base/HTTP 重试的失败尝试也不会完整留在成功 trace 中。

### Value、成本及置信区间

- 原始 Q 是每状态 3 次 rollout 的正确率均值，协议失败已经按 0 计；成本只在动作选择公式中扣除一次。原 CSV 没有把成本先并入 Q 后再扣一次，因此没有重复扣费。
- 原始固定动作准确率和 CONSULT 配对差的 bootstrap 先将同一病例所有状态及重复 rollout 聚合成一个病例级值，再对病例重采样；这处理了同病例内相关性。原始动作最优占比和部分 value-gap 行没有置信区间；本审计重新计算了病例聚类 bootstrap 区间，包含在 `action_value_reanalysis.csv`。
- 价值重算的统一收益为正确 `1`、错误或协议失败 `0`。对每个状态，以四动作的 3 次结果均值作为经验 Q；成本敏感效用按 ASK 患者轮数、CONSULT 专家 token、DEFER 专家 token 分别扣一次。详细定义、成本组及区间见 `ACTION_VALUE_REANALYSIS.md`。
- 成本敏感结果仍受失败调用成本缺失影响，属于已观测成本下的敏感性分析；无成本动作分布不受该成本缺失影响。

## 主要限制

这是 ATPO 公开医学选择题上的 Qwen3-8B 推理与 LLM Patient/Expert 模拟，不是真实病历、患者试验或医生评估。案例 3359 的题目质量异常会集中影响失败率。现有结果适合审计协议与描述动作差异，不能作为临床安全性或可部署 Router 的证据。
