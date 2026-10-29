# Medical Agent ATPO baseline 复现状态（2026-09-08）

## 研究目标

最终目标是构建医疗 Agent，使模型能够依据当前信息状态学习何时直接 `ANSWER`、何时继续 `ASK`、何时请求专家 `CONSULT`、何时将病例 `DEFER` 给专家。当前阶段只复现官方 ATPO baseline，尚未开始 EA-ATPO 四动作方法的正式改进。本文记录的是历史 Qwen3-4B 工程复现；正式主实验应切换到官方论文使用的 Qwen3-8B Assistant policy。

## 当前服务器与资源

- 主机：`hd03-gpu2-0041`
- 项目：`/kwkj-k8s/llm_team/tqy/mywork/ATPO-official`
- GPU 0：Qwen3-8B 患者模拟器
- GPU 1–3：ATPO actor/critic/rollout 训练
- GPU 4–7：其他用户占用，未使用

## 已完成

1. 原始医疗数据及 RL 数据已准备：训练 14,256，测试 1,268。
2. Qwen3-4B 官方 SFT 已训练完成：3 epochs、1,500 steps、最终验证 loss 0.9658。
3. SFT checkpoint 已合并为 HuggingFace 模型：
   `/kwkj-k8s/llm_team/tqy/mywork/ATPO-official/checkpoints/sft-qwen3-4b-full-hf`
4. Qwen3-8B 患者模拟器已部署并通过三类检查：事实命中、缺失事实拒答、正确选项防泄漏。
5. 官方 ATPO 单卡 RL smoke 已完整通过，包含树扩展、在线 critic、reward、advantage、actor/critic 更新和 checkpoint 保存。
6. 三个独立质量 pilot 已完成，共保留 21 条轨迹：
   - 合法最终答案 9/21（42.9%）
   - 最终答对 4/21（19.0%）
   - 合法答案内准确率 4/9（44.4%）
   - 格式错误率 0
   - 有效提问较少，患者拒答占主导
7. 找到并修复三卡 critic 死锁：在线 critic 批次原先固定补齐到 8，无法被 3 个 FSDP rank 均分；现改为按 `critic_work_group.world_size` 补齐。三卡 smoke 已在 45.5 秒内端到端通过。

## 当前质量判断

工程链路已经跑通，但初始 SFT 策略的交互质量较弱。典型问题是模型询问隐藏 `atomic_facts` 中不存在的信息，患者模拟器只能拒答；部分轨迹直到最大轮数仍未给出最终答案。这不是答案泄漏或接口错误，而是当前策略尚未学会提出可回答、能改变诊断决策的问题，正是后续 ATPO RL 应优化的能力。

## 已完成的稳定性 pilot

正式训练前的三卡稳定性 pilot 已完成：

- 训练 GPU：1、2、3
- 每步病例数：6
- 每例最多树轨迹：4
- 每个节点候选数：4
- 最大对话轮数：5
- 更新步数：5
- 第 5 步保存 checkpoint
- 启动脚本：`/kwkj-k8s/llm_team/tqy/mywork/ATPO-official/run_official_atpo_pilot_4gpu.sh`
- 日志：`/kwkj-k8s/llm_team/tqy/mywork/ATPO-official/logs/official-atpo-stability-pilot.log`
- 输出：`/kwkj-k8s/llm_team/tqy/mywork/ATPO-official/outputs/official-atpo-stability-pilot/rollout`
- checkpoint：`/kwkj-k8s/llm_team/tqy/mywork/ATPO-official/checkpoints/official-atpo-stability-pilot`

结果：5 个更新 step 全部完成，总耗时 2 分 55 秒；无 NaN、OOM 或通信死锁。各步合法作答率为 91.7%–95.8%，有效提问率为 23.6%–55.1%。由于每一步抽取的病例不同，短跑准确率波动不能解释为显著提升。

## baseline Phase 1 压力测试结果

- 训练 GPU：1、2、3；GPU 0 提供患者模拟
- 从原始 SFT checkpoint 重新开始，不继承 5-step pilot 权重
- 每步病例数：24
- 每例最多树轨迹：8；每节点候选数：4
- 最大对话轮数：10；最大响应长度：2,048 tokens
- 训练步数：50；critic warmup：4 steps
- 每 10 step 保存，最多保留 2 组 actor/critic checkpoint
- 启动脚本：`/kwkj-k8s/llm_team/tqy/mywork/ATPO-official/run_official_atpo_baseline_phase1_4gpu.sh`
- 日志：`/kwkj-k8s/llm_team/tqy/mywork/ATPO-official/logs/official-atpo-baseline-phase1.log`
- 输出：`/kwkj-k8s/llm_team/tqy/mywork/ATPO-official/outputs/official-atpo-baseline-phase1/rollout`
- checkpoint：`/kwkj-k8s/llm_team/tqy/mywork/ATPO-official/checkpoints/official-atpo-baseline-phase1`

50 steps 已完成，总训练时间约 50 分 57 秒。该阶段不能作为正式 baseline：它从 smoke 配置放大，沿用了默认 `token-mean` actor loss，而官方正式脚本使用 `seq-mean-token-mean`。训练在约 step 28 后开始出现长文本退化，step 50 的合法作答率降至 0，响应长度截断率达到 90.5%，所有轨迹 reward 为 −1。其价值是验证三卡系统能持续运行，并定位出正式 loss 配置不可省略。

## 正在运行：官方损失修正版

- GPU 0：Qwen3-8B 患者模拟器
- GPU 1：Qwen3-4B actor/critic/rollout；GPU 2、3 已被其他任务占用，未抢占
- 从原始 SFT checkpoint 重新开始
- 正式加入 `actor.loss_agg_mode=seq-mean-token-mean`
- 正式加入 PPO `clip_ratio_low=0.2`、`clip_ratio_high=0.28`、`clip_ratio_c=10.0`
- 训练 20 steps，critic warmup 4 steps，每 5 step 保存
- 脚本：`/kwkj-k8s/llm_team/tqy/mywork/ATPO-official/run_official_atpo_baseline_corrected_gpu1.sh`
- 日志：`/kwkj-k8s/llm_team/tqy/mywork/ATPO-official/logs/official-atpo-baseline-corrected-gpu1.log`
- checkpoint：`/kwkj-k8s/llm_team/tqy/mywork/ATPO-official/checkpoints/official-atpo-baseline-corrected-gpu1`

## 后续顺序

1. 验收 20-step 官方损失修正版是否消除长文本崩塌。
2. GPU 2、3 空闲后，将修正版扩展回三卡并运行更长训练。
3. 根据实际吞吐确定完整 baseline 的 batch 和树宽，定期保存并在 1,268 条测试集上评估。
4. 汇报最终准确率、合法作答率、平均轮数、有效提问率、患者拒答率、重复/多问题率及训练成本。
5. baseline 稳定后，从不同信息状态分别 rollout `ANSWER / ASK / CONSULT / DEFER`，验证四动作是否存在不同优势区间。
6. 随后实现 EA-ATPO 的反事实动作价值与成本敏感 Learn-to-Defer 目标，再进入方法改进实验。
