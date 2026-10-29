# EA-ATPO 项目交接文档

更新时间：2026-10-29（Asia/Shanghai）

## 项目目标

本项目构建医疗 Agent 的 learn-to-defer 能力。Agent 在每个医疗问答状态下自主选择 `ANSWER`、`ASK`、`CONSULT` 或 `DEFER`。最终目标不是用规则比较四个动作，而是训练可学习的动作价值/门控模型，在相同专家预算下提升诊断准确率和成本调整效用。

方法路线：先复现官方 ATPO baseline；E1 加入事实感知过程奖励；E2 冻结 E1 Agent 后训练序贯四动作门控；E3 仅在 E1/E2 都有效后进行联合 ATPO RL。

## GitHub 仓库

仓库：<https://github.com/tianqiuyu95ybzz/EA_ATPO>

当前主分支提交：`6ad03c7946ff7e0e1259eb4d3de47d2d150ef886`

```bash
git clone https://github.com/tianqiuyu95ybzz/EA_ATPO.git
cd EA_ATPO
```

仓库保存研究增量代码、补丁、脚本和文档，不包含大模型权重、数据集、checkpoint、日志或虚拟环境。

## 原工作区和目录

原 Codex 工作区：

```text
C:\Users\华硕\.codex\.chatgpt-projects\g-p-6a6ac70b22c081918dee40e99d55627a
```

主要内容：

```text
ea_atpo_stage/                    EA-ATPO 四动作协议和反事实采集代码
remote_patch/                     运行补丁和 smoke 脚本
e0_test_atpo_reward_path.py       E0 奖励路径测试
e1_*_process_reward_switch.patch  E1 过程奖励补丁
run_e1_b_process_reward_smoke_gpu7.sh
EA_ATPO_*.md                      方法、创新点和实验计划
ATPO-official/                    本地官方 ATPO checkout（未上传 GitHub）
ATPO-official/dataset/             本地数据集副本（不上传）
```

新环境中需要单独获取官方 ATPO 源码；不要把数据集、模型和 checkpoint 上传到公开仓库。

## 已完成工作

- 官方 ATPO 代码和医疗数据曾在旧服务器上准备完成。
- Qwen3-4B SFT 已跑通：3 epochs、1500 steps，最终验证 loss 约 0.9658。
- 官方 ATPO RL baseline 曾运行到 step 20。
- 已完成 E0 奖励路径审计。
- 已确认官方代码计算过程分数，但部分中间节点的 `action_reward` 原实现被置为 0。
- 已加入默认关闭的 `enable_process_reward` 设计，并完成 E1-B 单步 smoke。
- EA-ATPO 四动作协议已实现，本地协议测试 11/11 通过。
- 已加入专家视图检查：专家不能看到 `atomic_facts`、`ground_truth`、`answer_idx` 或 gold explanation。

## 已有结果（来自旧服务器记录）

原始 checkpoint、日志和 rollout 没有上传 GitHub。

### 官方 baseline

- 记录中的测试准确率约为 `43.45% -> 46.53%`；
- 尚未完成多随机种子和病例级置信区间验证，不能作为最终论文结论；
- 旧 checkpoint 曾位于 `checkpoints/official-atpo-baseline-corrected-gpu1/global_step_20/actor_merged_hf`。

### E1-B 单步 smoke

- 2 个病例、4 条训练轨迹；
- actor 和 critic 都完成梯度更新；
- checkpoint 曾位于 `checkpoints/e1-b-process-connected-smoke-gpu7/global_step_1/`；
- 样本太小，不能证明准确率提升；暴露出患者“无法回答”时仍可能被过程奖励正向奖励的问题。

### 离线四动作 router

- 当前冻结表示 router 效用约 `0.4869`；
- always ANSWER 约 `0.4966`；
- 差值约 `-0.0097`，病例 bootstrap 置信区间跨 0；
- 尚无可信的 learned deferral 正收益证据。

## 当前关键问题

患者对部分问题返回 `The patient cannot answer this question.`，旧过程奖励可能仍给这类问题正分，Agent 可能把未知误当成阴性或阳性证据。强制 ASK 反事实分支中曾有约 72.3% 回复为无法回答，但不能直接等同于自然策略的无效提问率。

正式 C/D 组训练前必须完成：

1. 患者回复到新增 fact ID 的可靠匹配；
2. 无法回答/未知信息的非正奖励和轮次成本；
3. `unknown-as-negative` 错误率审计；
4. 至少 100 条患者/专家交互人工抽检。

## 新环境验证

获取官方 ATPO：

```bash
git clone https://github.com/Quark-Medical/ATPO.git ATPO-official
cd ATPO-official
git rev-parse HEAD
git status --short
```

必须记录实际 commit，不要假设它与旧服务器版本一致。旧 E1 补丁曾因文件路径/版本差异无法直接应用，需要先对照源码适配。

运行 EA-ATPO 协议测试：

```bash
cd EA_ATPO/ea_atpo_stage
python -m unittest recipe.ea_atpo.test_protocol -v
```

预期 11 个测试全部通过。GPU 和模型恢复前，只做协议、隐私边界、奖励路径、数据格式和 mock pilot，不把 mock 结果写成正式准确率。

## 正式实验顺序

### E0：奖励路径审计

确认过程奖励进入节点 action reward、树回传 Q、critic return、turn advantage 和 actor 更新。

### E1：四组消融

固定模型、数据、树预算、训练步数和种子：A=官方 ATPO，B=原过程分数进入树，C=新增事实信号，D=两者结合。指标包括病例级准确率、平均轮数、有效问诊率、新事实率、重复/无效提问率和训练成本。

### E2：序贯四动作门控

冻结 E1 胜出的 Agent，从真实访问状态构造动作配对 rollout，学习：

```text
U(s,a) = Q_out(s,a) - lambda_cost * Q_cost(s,a)
```

`ASK`/`CONSULT` 后允许继续门控；`ANSWER`/`DEFER` 终止。测试集只用于最终评估。

### E3：条件性联合 RL

只有 E1 和 E2 在独立测试集上通过预注册门槛，才启动联合训练；否则报告负结果。

## 服务器恢复后

旧实验服务器为 41 号机，之前通过 `jms-research-nj.agnes-ai.cn:2222` 连接。此前出现跳板机认证后连接目标服务器被拒绝。目前不能假设旧服务器上的 checkpoint、模型服务或未提交代码仍可访问。

恢复访问后第一步只读检查：主机名、GPU、进程、代码 commit、文件 hash 和 checkpoint；不要立即启动新训练。

每次正式实验必须记录：配置快照、代码 commit/hash、数据划分和 hash、启动命令、训练日志、checkpoint、指标 JSON、失败原因和下一步。

## 交接原则

- 不把旧结果表述为已证明的方法提升；
- 不把规则 router 的动作统计表述为 learned policy；
- 不把小样本 smoke 准确率当正式结果；
- 不上传医疗原始数据、模型权重和服务器密码；
- 新结论追加到实验记录，不覆盖旧结论。

