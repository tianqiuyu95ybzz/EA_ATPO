# Attempt09 配对基线评估

## 实验设置

在同一份 20 例开发集上重新恢复并评估三种策略。数据文件为 `data/test.parquet`，SHA-256 为 `419ecd2ce0738f45f7fcf67568a5202fabebf02ce2e1dea61ff65681b1752987`。逐例输出只保留匿名病例哈希、答案字母、动作和数值统计，不保留病例文本、hidden facts、gold、患者回复或专家建议。

| 项目 | 设置 |
|---|---|
| Always Answer | 未微调 Qwen3-8B，禁止交互，要求直接给答案 |
| Original ATPO | Attempt20 step 2 Actor；仅 ASK/ANSWER；训练时仅 2 次 optimizer update、2 个训练病例 |
| EA-ATPO | Attempt09 step 10 Actor；ANSWER/ASK/CONSULT/DEFER |
| Patient / Expert | 同一个 Qwen3-8B 基础模型，通过固定角色提示模拟；Expert 只看公开病例和已公开对话 |
| 解码 | `do_sample=False`、greedy、BF16、SDPA；Actor 每 episode 的累计生成回复与观察 token 预算 512（后续调用扣除已用 token），Patient 每次 256，Expert 每次 128 |
| 交互预算 | 每 episode 最多 3 个 Actor 决策、最多 1 次 ASK、最多 1 次 CONSULT；Original ATPO 不支持 CONSULT/DEFER |
| 评分 | 三种策略共用解析器；接受规范动作标记或独立选项字母；不可解析答案计错 |

各方法的系统提示按各自原生动作协议表达，因此任务指令本身不完全相同。Original ATPO 和 Always Answer 都使用 `enable_thinking=False`，避免模板开关造成额外差异。Original ATPO checkpoint 是两步工程烟测产物，不能代表充分训练的 Original ATPO 基线；这组结果只作描述性对比。

## 结果

| 方法 | 正确率 | 可解析答案 | ANSWER | ASK | CONSULT | DEFER | 平均总 tokens | 平均 Actor 决策轮数 | 平均动作成本 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Always Answer | 2/20 (10%) | 6/20 | 20 | 0 | 0 | 0 | 530.4 | 1.00 | 0.000 |
| Original ATPO Attempt20 | 7/20 (35%) | 20/20 | 20 | 20 | 0 | 0 | 1,382.8 | 2.00 | 0.050 |
| EA-ATPO Attempt09 | 4/20 (20%) | 20/20 | 20 | 3 | 3 | 0 | 1,034.2 | 1.30 | 0.0375 |

总 tokens 包括 Actor、Patient 和 Expert 请求的 prompt 与 completion tokens。CONSULT 同状态直接回答反事实调用单独记录，不计入主策略 tokens。成本使用训练配置中的 ASK=0.05、CONSULT=0.20、DEFER=0.60；正确答案奖励不计入此成本列。

## 配对结果

| 配对比较（前者正确 / 后者正确） | 两者都对 | 仅前者对 | 仅后者对 | 两者都错 | 配对准确率差与病例 bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|
| EA-ATPO vs Always Answer | 2 | 2 | 0 | 16 | +10 pp [0, +25 pp] |
| EA-ATPO vs Original ATPO | 4 | 0 | 3 | 13 | -15 pp [-30, 0 pp] |
| Original ATPO vs Always Answer | 2 | 5 | 0 | 13 | +25 pp [+10, +45 pp] |

置信区间按病例成对重采样 20,000 次，固定 seed `20261115`；这里只有 20 个病例，区间仅表示该开发集上的抽样不确定性。

EA-ATPO 在这 20 例中比 Always Answer 多答对 2 例，但 bootstrap 区间包含零；同时 Always Answer 只有 6/20 输出能按规则解析，另外 14 例生成正常终止于 EOS、但没有解析到答案。因此这项差异不能解释为已证明的临床推理提升。EA-ATPO 比当前两步 Original ATPO 少答对 3 例；由于 Original checkpoint 训练预算极小，也不能据此断言充分训练的 Original ATPO 会更好。

## CONSULT 配对反事实

EA-ATPO 在 3 例上执行了 `ASK → CONSULT → ANSWER`。每例均从咨询前的同一公开状态额外运行一次“直接回答”反事实：0 例从错变对，1 例从对变错，2 例正确性不变。纠错率为 0/3，退化率为 1/3。样本太少，不能证明 CONSULT 有独立收益。

## Checkpoint 恢复一致性

Attempt09 step10 Actor、Critic 与 tokenizer 均从恢复目录加载。恢复后的 HF greedy 推理与已保存的 SGLang 验证结果在 20/20 例上答案字母和正确性一致；动作序列 19/20 一致。Critic 能正常加载并对 probe 返回有限值。Actor、Critic、tokenizer 和开发集文件哈希见同目录 `run_manifest.json`；权重 checkpoint 保存在 PFS，不包含在 Git 提交中。

## 解释边界

这是 20 例开发集上的小样本工程验证，不是独立确认性实验。Expert 是 Qwen3-8B 模拟专家，不是真实医生。Current Original ATPO checkpoint 仅训练 2 步；Always Answer 的低可解析率也降低了准确率比较的解释力。不要据此宣称 EA-ATPO 已优于原始 ATPO或具有临床有效性。
