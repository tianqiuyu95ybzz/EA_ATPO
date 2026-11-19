# Phase 0 审计后的阶段实施计划

## Phase 1 — 静态相对收益复核

复用既有 1,700 对输出与已有 feature audit，不再推理。构造公开状态文件与离线 label 文件，分离 Gold correctness 派生标签；在 train 拟合特征变换和轻量模型，validation 选模型/阈值，test 仅做已污染 holdout 的冻结复核。输出校准、Helpful/Harmful、Relative Gain、成本敏感结果。不得把 Base Error 指标称为 Relative Gain 能力。

## Phase 2 — 固定 continuation 的主动信息获取先导

在 ATPO 原始 train 数据中做全局历史排除，排除旧 1,700 队列、Attempt09 development、旧 pilot、咨询样本和已有轨迹。均匀抽取 100 train + 50 validation，不按 helpful/harmful 标签挑选。对每例创建 Direct ANSWER、Direct DEFER、ASK→continuation、CONSULT→continuation 分支；信息获取之后固定同一套 continuation policy，并记录“强制 ANSWER 的 Task Improvement”与“允许 ANSWER/DEFER 的 Decision Improvement”两种结果。当前服务未运行且 GPU 忙，本轮只运行预检/CPU 测试，不发起生成请求。

## Phase 3 — 只在 Phase2 形成可复现净收益后进行

仅使用 Phase2 train 学习经验净信息价值，validation 选模型/风险惩罚。由于 pilot 只有 150 个病例，任何拟合不确定性和 VOI 结论都标为探索性；若 CI 跨零或低成本收益不能复现，则不进入树搜索。

## Phase 4 — Phase3 通过后

新增独立 VOI search adapter，只在 ATPO 原始树搜索的展开预算/节点排序处接入；保留 reward、PPO/Critic loss、动作集合和 tree return。训练统计仅使用 train，lambda_voi 仅用 validation。必须先以 lambda_voi=0 验证原树路径、回报和 PPO targets 数值兼容，并记录搜索样本分布及可用的行为概率。

## Phase 5–6

本轮不训练 PPO/ATPO，不做最终独立测试。Phase5 至少需要 Phase2–4 通过和可比较搜索预算；Phase6 需要确认官方 test 的历史使用状态，或新抽取并冻结的独立病例集合。

## 成本映射约定

EA train reward 当前是终局正确性 − 0.05×ASK − 0.20×CONSULT − c_D×DEFER，动作 costs 每条 edge 计一次；离线 router utility 是 Accuracy − c_D×DEFER rate，属于逐病例平均回报。Phase2 采用同一 terminal-correctness 单位和固定 c_A=0.05、c_C=0.20；c_D 敏感性使用预设网格 [0, .05, .10, .20, .40, .60]。Token 与 latency 另行报告，不在同一个指标中隐式重复计费。
