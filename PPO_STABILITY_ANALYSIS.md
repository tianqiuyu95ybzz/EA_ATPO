# Attempt07/08 PPO 稳定性分析

## 结果

两步更新均正常结束，Actor 和 Critic 梯度均有限且非零，checkpoint 已保存，无 OOM。两步各只有 2 条 rollout，且每个 step 的两条 rollout 都来自同一个重复采样病例；因此这些统计只能诊断工程链路，不能证明 Critic 或策略已稳定。

| 指标 | Step 1 | Step 2 |
|---|---:|---:|
| rollout 有终局答案 | 2/2 | 0/2 |
| episode reward 均值 / 范围 | -0.25 / [-0.25, -0.25] | -0.05 / [-0.05, -0.05] |
| Critic Return 均值 / 范围 | -0.150 / [-0.25, 0] | -0.025 / [-0.05, 0] |
| Return 方差（训练日志） | 0.012069 | 0.0006579 |
| Actor-token Advantage 均值 / 人口标准差 | 0.02880 / 1.0561 | -0.9970 / 1.5678 |
| Advantage 范围 | [-0.9773, 1.9625] | [-2.6055, 0.6176] |
| Critic value 均值 / 范围 | -1.790 / [-4.125, 0.0811] | 4.685 / [2.1875, 8.0625] |
| Critic VF loss | 2.1411 | 12.7497 |
| Actor PG loss | 0.01852 | 1.0968 |
| Actor grad norm | 24.016 | 34.061 |
| Critic grad norm | 624.508 | 1676.140 |
| Critic explained variance | -152.72 | -4918.44 |
| response length 均值 / 截断率 | 212.5 / 0 | 117 / 0 |
| 峰值 allocated / reserved | 76.30 / 88.20 GiB | 76.30 / 88.20 GiB |

Advantage 人口标准差由完整轨迹中每个 Actor 生成 token 继承所在动作段 Advantage 后计算；均值与日志中的 `critic/advantages/mean` 一致。Critic Return 则只在每个动作段的前 5 个 value-query token 上训练，所以其均值/方差与对所有生成 token 求值不同。

## 裁剪与参数更新

Actor/Critic 配置的 `grad_clip=1.0`。FSDP2 裁剪函数先计算全局 norm，再原地缩放梯度；训练日志记录裁剪前 norm。对应的预期裁剪比例约为：

| 模块 | Step 1 比例 | Step 2 比例 |
|---|---:|---:|
| Actor | 0.0416 | 0.0294 |
| Critic | 0.00160 | 0.000596 |

因此有限梯度裁剪后的全局 norm 应约为 1。Attempt07 没有记录裁剪后的 norm 张量，表内是依据已执行裁剪代码推算，不是单独测得值。`actor/pg_clipfrac=0` 是 PPO ratio clipping 统计；`critic/vf_clipfrac=0` 是 value clipping 统计，两者都不是 gradient clipping 比率。

对保存的 FSDP2 DTensor checkpoint 分片进行 CPU 差分，比较 Step 1 到 Step 2：

| 模块 | 参数量（两 rank 合计） | 更新 L2 | Step 1 权重 L2 | 相对更新 L2 | 最大绝对变化 |
|---|---:|---:|---:|---:|---:|
| Actor | 8,190,735,360 | 0.05807 | 2459.622 | 2.36e-5 | 1.91e-6 |
| Critic | 7,568,409,601 | 0.51167 | 2372.397 | 2.16e-4 | 1.14e-5 |

这确认两类参数都发生了更新；Actor 更新幅度很小，Critic 相对变化更大。

## 诊断

Step 2 的 Critic Return 几乎为常数，而预测 value 已从 Step 1 的负值变为约 2–8，远离 `[-0.05, 0]` 的训练目标；VF loss 和负 explained variance 因而显著恶化。大梯度没有变成非有限梯度，裁剪代码仍会执行，但 value 学习质量明显不稳定。两条重复病例样本不足以判断这是数据方差、随机初始化还是更新超调，不能据此调参。

未发现重复扣费或重复 GAE：EA edge reward 将 action cost 与终局正确性相加一次；树搜索回传一次得到 MDP Return/Advantage；日志显示 `Found pre-computed advantages in batch. Using them directly.`；EARewardManager 把 episode 净分数送入原始 reward logging 路径，不会再次修改预计算目标。关键上游实现见 `ATPO-official/codes/verl/workers/actor/dp_actor.py`、`.../critic/dp_critic.py`、`ATPO-official/codes/recipe/atpo/user_assistant_agent_loop.py` 和 `ATPO-official/codes/recipe/atpo/chat_model.py`。

因此 Attempt07 **没有通过 Critic 稳定性验证**。Attempt08 的两步数值仍有限，但样本数不足且在第三个 batch 因协议/环境错误退出，也没有完成预定的 10 步。

## Attempt08：两步真实训练，未完成稳定性验证

Attempt08 使用 20 个训练病例、batch size 2、每状态 `M=N=4`，计划 10 步，成本为 ASK 0.05、CONSULT 0.20、DEFER 0.60。训练完成 2 步后，第三批遇到某病例全部候选均为 DEFER 专家格式错误，进程退出。没有 OOM、NaN 或非有限梯度；但退出使其不能算作稳定性通过。`save_freq=5`，因此没有 checkpoint。日志与 24 条 episode trace 完整保留。

| 指标 | Step 1 | Step 2 |
|---|---:|---:|
| PPO 输出数 / 可解析终局答案数 | 6 / 6 | 8 / 8 |
| 终局正确数 / 准确率 | 0 / 6 (0%) | 4 / 8 (50%) |
| episode reward 均值 / 范围 | -0.25 / [-0.25, -0.25] | 0.25 / [-0.25, 0.75] |
| Critic Return 均值 / 范围 | -0.1833 / [-0.55, 0] | 0.3083 / [-0.25, 1] |
| Return 方差 | 0.02753 | 0.23901 |
| Actor-token Advantage 均值 / 范围 | -0.2276 / [-0.8031, 0.9223] | -0.5485 / [-2.0217, 2.9203] |
| Critic value 均值 / 范围 | 0.5317 / [-3.2969, 3.5469] | -2.3674 / [-4.5938, 0.8242] |
| Critic VF loss | 8.9201 | 5.2951 |
| Actor PG loss | 0.1732 | 0.4690 |
| Critic grad norm（裁剪前） | 1035.866 | 412.119 |
| Actor grad norm（裁剪前） | 15.472 | 24.942 |
| Critic explained variance | -91.21 | -11.66 |
| response length 均值 / 截断率 | 266 / 0 | 242.75 / 0 |
| 峰值 allocated / reserved | 76.30 / 88.94 GiB | 76.30 / 88.94 GiB |

`grad_clip=1.0`，日志记录的是裁剪前 norm；裁剪后 norm 没有单独测量。没有发现非有限 loss 或梯度。两步的 Critic explained variance 都大幅为负，说明当前两步不足以证明 Critic 拟合有效；单步 batch 也很小，不能据此可靠调参。Step 1/2 `custom/num_valid` 的 6/6、8/8 表示提交给评估器的文本能解析答案，不等于 episode 训练资格，也不等于这些答案正确。

失败原因不是显存或优化器异常，而是第三个 batch 内一个样本的 4 个候选全部遇到 DEFER Expert 格式解析错误。旧代码在单样本全失败时抛异常，连带丢弃了同 batch 另一病例的 4 条重复 ASK 策略轨迹。EA 层现已改为返回零 response mask 占位行以保留同批有效轨迹，并在全 batch 无 Actor token 时由自定义 Reward Manager 于 PPO 前停止。单元/协议回归测试 7/7 通过，端到端重跑仍未完成。

GPU 资源是明确阻塞：检查时 GPU4 有约 77 GiB 可用、GPU6 约 108 GiB 可用，两个卡各有其他用户的训练进程占用约 71 GiB；Attempt09 启动门槛要求两卡连续 5 次采样均至少有 105000 MiB 可用。训练没有启动。后续必须等资源门槛满足，再运行保存频率为每步的 10 步实验；如仍出现连续全失败 batch、非有限数值或严重 Critic 发散，应立即停止。
