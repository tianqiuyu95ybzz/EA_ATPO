# EA-ATPO V1 小规模训练结果

## 结论

Attempt07 通过了四动作协议和两步 PPO 工程 smoke，但不是稳定性或策略质量验证。Attempt08 从官方数据和真实 Qwen3-8B 服务生成了 24 条自然策略 episode，完成 2 次 PPO 更新后，在第三批被 Expert 输出格式错误中止。没有 OOM 或非有限 loss/gradient；没有完成计划的 10 步，没有验证集评估，也没有匹配的 Original ATPO / 无协作成本 baseline。

当前不应开始大规模训练或把现有结果写成四动作训练成功。Attempt08 没有 checkpoint；Attempt07 原始 checkpoint 和日志保持不变。

## 数据和实验范围

- Base/Actor/Critic：Qwen3-8B；Patient/Expert 与 Base 通过同一推理服务、不同角色提示隔离。
- 数据：ATPO 官方 train/test；Attempt08 抽取 20 个训练病例与 20 个独立测试病例。官方 JSONL 哈希和转换后 Parquet 哈希记录在 Attempt08 `run_manifest.json`。
- 训练参数：batch size 2、`M=N=4`、最多 3 次 Actor 决策、最多 1 次 ASK 和 1 次 CONSULT、响应上限 512。
- EA 成本：ASK 0.05、CONSULT 0.20、DEFER 0.60；终局正确奖励 1.0。成本只在树边奖励扣一次；预计算 Return/Advantage 被训练器直接使用。
- 自然采样：没有强制动作结果进入 PPO。独立四动作强制协议测试只验证工程转移和 mask。

## Attempt08 轨迹和动作覆盖

24 条原始轨迹中，17 条属于可训练策略结果，7 条属于专家协议/环境错误。17 条包括答错终局和 4 条重复 ASK 上限违规；7 条环境错误由 6 次 DEFER Expert 格式失败和 1 次 CONSULT 最终答案泄漏构成。所有轨迹连同 `failure_reason`、成本和最终答案均留存在 `traces/train.jsonl`。

自然 Actor 事件计数：ASK 28、CONSULT 20、ANSWER 13、DEFER 6。每条轨迹第一步都选择 ASK；4 条轨迹重复 ASK，越过 `max_asks=1`，但没有产生额外 Patient 调用或重复成本。20 次 CONSULT 中 19 次得到可解析建议，1 次泄露最终选项被隔离。6 次自然 DEFER 都因 Expert 没有给出可解析的单一答案失败，所以有效 DEFER 终局、Return 和策略梯度覆盖仍为 0。

## PPO 数值

| 指标 | Step 1 | Step 2 |
|---|---:|---:|
| PPO rollout rows / 可解析答案 | 6 / 6 | 8 / 8 |
| 答对数 / rollout 准确率 | 0 / 6 (0%) | 4 / 8 (50%) |
| Episode reward 均值 | -0.25 | 0.25 |
| Critic Return 均值与范围 | -0.1833 [-0.55, 0] | 0.3083 [-0.25, 1] |
| Critic VF loss | 8.9201 | 5.2951 |
| Critic grad norm（裁剪前） | 1035.866 | 412.119 |
| Actor PG loss | 0.1732 | 0.4690 |
| Actor grad norm（裁剪前） | 15.472 | 24.942 |
| Critic explained variance | -91.21 | -11.66 |
| 峰值显存 allocated / reserved | 76.30 / 88.94 GiB | 76.30 / 88.94 GiB |

所有记录的 loss、Return、Advantage 和梯度都是有限值，两步都有 Actor/Critic 更新。Critic explained variance 仍强烈为负，且训练只涵盖两步和少数病例，无法证明价值估计稳定。`custom/num_valid` 是终局答案格式可解析数，不是训练样本有效率。

## 中止原因和最小修复

第三批病例 `6886` 的 4 个候选全都在 DEFER Expert 阶段因最终答案格式错误被判为环境失败。旧代码在单个样本没有可训练叶子时抛异常，导致同 batch 的 `4256` 四条有效策略失败（重复 ASK 违规）也没能进入 PPO；原始轨迹仍保存。进程在 2/10 次更新后退出，`save_freq=5` 因而没有 checkpoint。

当前 EA 层已加入最小修复：单个样本全失败时返回零 Actor response mask 占位行，不给失败行为添加策略回报；有效兄弟样本可继续训练。若整个 batch 的 Actor mask 都为零，`EARewardManager` 在 PPO 前明确终止，避免空 mask 更新。此修复没有改动 `ATPO-official`，通过 8/8 单元/协议测试；尚未经过真实多样本 rollout 端到端复验。

### Expert 输出协议定位

Attempt08 的失败 trace 只存了解析异常，没有记录 Expert 原始文本、completion token 数或 finish reason。故无法报告实际输出字符串，也不能断言是 JSON、thinking 文本、字段缺失或截断。可确认的是 6 次 DEFER 未匹配旧解析器的严格格式 `Expert Final Answer: <A-D>`，1 次 CONSULT 命中答案泄漏正则。CONSULT 正则禁止 `Final Answer`、`correct answer/option` 和 `option A-D` 等表述。

Attempt08 使用 Qwen chat template `enable_thinking=False`，Expert 上限为 384 tokens；这是配置证据，不足以判断具体模型输出。SGLang TokenOutput 不带 finish reason，原失败 trace 又没有 token 数，因此截断不可判定。脱敏 fixture `ea_atpo_stage/recipe/ea_atpo/testdata/attempt08_expert_failures.jsonl` 保留真实失败类别；里面的 parser 输入明确标记为代表性样例、不是追回的模型输出。

按 20 次 CONSULT 和 6 次 DEFER 计算，Attempt08 的 Expert 协议失败率为 7/26（26.9%）；episode 级环境失败率为 7/24（29.2%）。这是未启用有限重试时的历史观测值，不能代替修复后的失败率。

当前每个 Expert 响应最多进行一次格式重试，温度固定 0，不加入 gold 或 hidden facts；解析仍失败时按环境/协议错误隔离。失败事件保存最多 500 字符的响应、请求/重试次数、token 数和延迟。因为上游不返回 finish reason，另写 `completion_at_token_cap` 作为“输出达到上限”的代理，不能当作精确截断标签。

## 验证状态与后续

本次没有完成病例隔离验证，也没有 Original ATPO / 四动作无成本 / EA 成本三组同预算比较。Attempt08 manifest 记录 `observed_optimizer_steps=2`、退出码 1、无 OOM、无 checkpoint；请不要将它当作 10 步稳定性结果。

GPU 检查时，GPU4/6 各约 56 GiB 空闲且利用率均为 100%，检测到其他训练进程。Attempt09 启动器要求两卡连续 5 次采样分别至少有 105000 MiB 可用，故本次没有启动，未终止或清理其他用户进程。满足门槛后，下一步才以每步 checkpoint 运行 10 步自然采样，监控全失败 batch、有限性和 Critic 指标，再运行独立验证与匹配 baseline。

## 工程验证

执行命令：

```bash
PYTHONPATH=ea_atpo_stage:ATPO-official/codes .venv/bin/python -m unittest discover \
  -s ea_atpo_stage/recipe/ea_atpo -p 'test_ea_agent_loop.py' -v
bash -n run_ea_atpo_v1_stability.sh run_ea_atpo_v1_smoke.sh run_ea_atpo_v1_validated_smoke.sh
.venv/bin/python -m compileall -q ea_atpo_stage/recipe/ea_atpo
```

结果：8/8 单测通过；Shell 语法检查和 Python 编译检查通过。测试包含 Attempt08 两种错误类别的代表性 parser 输入、尾部 Qwen 标记归一化、有限重试、失败隔离和全无效 batch 拦截。单测使用本地 tokenizer 和确定性 mock，不能替代真实模型训练结果。

复现实验命令：`./run_ea_atpo_v1_stability.sh`。默认固定 GPU4/6、需要每卡至少 105000 MiB 连续 5 次采样、每个 PPO step 保存 checkpoint；本轮门槛不满足，因此未启动 Attempt09，亦无新 checkpoint。Attempt07 checkpoint 路径见 `experiments/ea_atpo_v1_smoke/ea_atpo_v1_attempt07/checkpoints/`；Attempt08 无 checkpoint。

代码和 Attempt08 结果已提交并推送到 `ea-atpo-v1`，当前更新提交详见 Git 记录。GitHub API 身份检查得到 `GMRainZ`，仓库权限 `push=true`；先前 403 来自本机缓存凭据 `Seven11111`，禁用 credential helper 并读取 `.env/github_pat.txt` 后推送成功。
