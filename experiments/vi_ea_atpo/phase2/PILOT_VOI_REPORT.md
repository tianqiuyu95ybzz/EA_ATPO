# Phase 2 — ASK/CONSULT 信息获取先导实验

**当前状态：冻结病例队列与运行器就绪；最多两例真实 Smoke 已运行，正式 150 例干预 Pilot 未运行。** 本报告后续原始记录描述的是服务恢复前的准备状态；真实 Smoke 与服务状态见 `SMOKE_TEST_REPORT.md` 和 `SERVICE_RECOVERY_REPORT.md`。本报告没有把历史实验或 mock 单测写成新先导实验结果。

## 1. 固定样本

- 来源是 EA-ATPO 上游训练来源 `train.pilot.parquet`（14,256 行）；没有从官方 test 或旧 400 例 holdout 取样。
- 在仓库和同项目共享实验目录扫描了 95 个 JSONL 文件（44,264 条记录）、3 个 `case_selection.json` 和 1 个旧 sample-selection manifest；另排除 Attempt09 development prompt 身份。
- 在历史身份排除后用固定 seed `20261118` 均匀抽出 100 train 与 50 validation。病例/公开 prompt 哈希互斥。选择未读取 Gold correctness、Helpful/Harmful 或 uncertainty；没有按结果分层，因此保留自然来源分布。
- 选择文件只保存哈希化 case ID、source row index 和 prompt hash；没有 Gold、隐藏 atomic facts 或完整 prompt。完整来源校验和排除统计见 `selection_manifest.json`。

## 2. 冻结的分支协议

每个病例从同一个公开初始状态建立以下轨迹：

1. `DIRECT_ANSWER`：原 Base 终局答案。
2. `DIRECT_DEFER`：原 Expert 终局答案。
3. `ASK_TASK_ANSWER`：Base 生成一次问题，Patient 只使用环境 `atomic_facts` 回答，Base 被要求立即给出终局答案。
4. `ASK_DECISION`：使用同一 Patient 回答，Base 在 `ANSWER` 与 `DEFER` 间做一次终止决策；若 DEFER，调用原 Expert 终局接口。
5. `CONSULT_TASK_ANSWER`：Base 生成一次咨询焦点，Expert 按原 CONSULT prompt 只返回建议，然后 Base 被要求立即回答。
6. `CONSULT_DECISION`：使用同一 Expert 建议，Base 在 `ANSWER` 与 `DEFER` 间做一次终止决策；若 DEFER，调用原 Expert 的 DEFER prompt。

这两个 continuation policy 都是为本次先导实验固定的隔离条件；不是完整 H-MDP 的 unrestricted 多轮策略。它们分别度量 task improvement 和 terminal decision improvement。它们不替代原 AgentLoop，且没有把 CONSULT 当成 DEFER。

动作成本按原 EA reward 的归一化尺度：`c_A=0.05`、`c_C=0.20`；DEFER 使用冻结网格 `[0,.05,.10,.20,.40,.60]`，主成本 `.60`。轨迹 Utility 为 `terminal_correct − c_A*N_ASK − c_C*N_CONSULT − c_D*N_DEFER`。token 与 latency 单独报告，不在 Utility 内隐式重复扣费。CONSULT 后再 DEFER 将分别计一次 `c_C` 和 `c_D`。

## 3. 服务恢复前的运行状态（历史记录）

执行命令：

```bash
.venv/bin/python experiments/vi_ea_atpo/phase2/select_pilot_cases.py
.venv/bin/python experiments/vi_ea_atpo/phase2/run_information_pilot.py
```

病例哈希/源数据身份预检通过，最终队列规模为 train 100、validation 50。此历史检查中，`127.0.0.1:30131/v1/models` 与 `127.0.0.1:30132/v1/models` 均连接拒绝，故当时没有发起 smoke 或正式请求。之后已从实际模型资产建立 30481/30482 服务并运行两例 Smoke；结果见本报告开头所链接的更新报告。

`execution_manifest.json` 记录 endpoint、GPU、配置/代码/数据哈希及 `model_inference_started=false`；`failure.log` 保留端点阻塞记录。`intervention_rollouts.jsonl` 为空；两张指标 CSV 只有 schema header，**不是零分或实验测量值**。

完整 150 例分支计划 nominal API 生成次数为最多 1,800（不含服务重试和格式重试）；代码中的每次 API 请求最多 3 次传输重试，强制动作最多 4 次格式尝试。因此这是按一次成功返回计算的预算，不是严格的总请求上限。两例 Smoke 已实际执行；正式 150 例 Pilot 仍未运行。

## 4. 已有主动信息证据（仅历史背景）

以下结果已在旧样本上产生，不属于新 cohort，也不替代本报告未运行的干预比较：

- 既有 100 例四动作 Pilot 使用同一 Qwen3-8B 模拟 Base/Patient/Expert，多轮/多 rollout 设置与本次固定一次干预不同；ASK/CONSULT 的准确率点估计略高于直接 ANSWER，但未控制本次归一化动作成本，不能证明新 32B Expert 的 ASK/CONSULT 净 Utility。
- 既有 200 状态建议利用实验中，CONSULT-Critical 相对 ANSWER 的配对准确率差为 +0.5 pp，病例聚类 95% CI [−5.9,+6.8] pp；CONSULT 分支平均 token 明显较高。实验使用的是 Qwen3-8B 模拟建议，不是本次 32B Expert，也不是新选出的 cohort。
- 独立 200 例 direct ANSWER/DEFER 验证显示 32B Expert 相对 8B Base +9.5 pp，95% CI [+2.5,+16.5] pp；这是终局 DEFER 互补性证据，不等于主动 CONSULT/ASK 的信息价值。

## 5. 下一阶段门槛

Phase 2 正式 rollout 仍为 **Not Run**。恢复服务后先核对 `/v1/models` 的 checkpoint revision 与冻结配置，再对最多 2 例运行 smoke；通过动作解析、Gold/atomic-facts 隔离和日志核验后，才运行冻结的 train/validation pilot。若 validation 上看不到有实际意义的净 Utility 改善或收益排序信号，则不进入 VOI 估计训练和树搜索融合。

本阶段不能回答 ASK/CONSULT 的总体 Accuracy、Utility、负收益比例、不同成本表现或 bootstrap CI；这些结果仍待模型 rollout。
