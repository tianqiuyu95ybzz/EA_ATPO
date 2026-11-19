# VI-EA-ATPO 主报告

日期：2026-11-18  
项目：Value-of-Information Guided Expert-Aware Adaptive Tree Policy Optimization  
当前决策：**D — 现有证据尚未显示主动信息获取有足够、可预测的净收益；暂停 VOI 搜索与 RL。** 这是当前证据的 no-go，不是证明 ASK/CONSULT 在所有系统中都无价值。

## 1. 执行摘要

本轮先检查了真实仓库，再完成原 ATPO/EA-ATPO 架构审计和 1,700 例配对数据的完整重算。静态 Expert 配对准确率增益存在，但当前决策前特征无法稳定排出 Helpful 与 Harmful 的净相对收益；已反复使用的 400 例 holdout 不能再作为新泛化集。历史 ASK/CONSULT 实验没有证明稳定的成本后净增益。

为直接验证主动信息价值，本轮又准备了新的 100 train / 50 validation 病例、固定分支协议、运行器、成本评估、日志和 CPU/mock tests。但 Base/Patient 服务 `127.0.0.1:30131` 与 Expert 服务 `127.0.0.1:30132` 在预检时拒绝连接。模型 rollout **Not Run**，没有推理调用，也没有把 mock 结果当成性能结果。因此 Phase 2 尚未通过，Phase 3–6 均停止在未运行状态。

## 2. 仓库、提交与工作区

- 根目录：`/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO_V1`
- 开始时 branch：`ea-atpo-v1`
- 开始时 HEAD：`2aebfa8ce33db872b9b4c204377a4a11cfb09c97`
- 用户提及的 `2aedfa8` 不在当前 Git object database 中。
- 开始时 tracked files 干净；工作区有 `.task_cache/`、Attempt01–09/original ATPO smoke data 和旧 Router 分析等既有 untracked 文件。本轮没有覆盖或删除它们，也没有修改原 ATPO/EA-ATPO 算法源码；新增内容全部放在 `experiments/vi_ea_atpo/`。
- 运行环境：Linux x86_64，`.venv` Python 3.12、NumPy/pandas/pyarrow 可用。CPU 分析完成。
- GPU snapshot：8 张 NVIDIA B20Z；至少有 GPU 满足本轮 runner 的显存/利用率门槛，但本地两个推理 endpoint 都不可达。没有启动模型服务、GPU 推理或 RL。

## 3. Phase 0 — ATPO / EA-ATPO 结构审计（Complete）

H-MDP 状态与动作定义在 [protocol.py](/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO_V1/ea_atpo_stage/recipe/ea_atpo/protocol.py:18)。真实四动作转换在 [ea_agent_loop.py](/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO_V1/ea_atpo_stage/recipe/ea_atpo/ea_agent_loop.py:609)：ASK→Patient→Base continue、CONSULT→Expert advice→Base continue、DEFER→Expert terminal answer。CONSULT 与 DEFER 共用模型调用通道，但有不同提示词、解析协议、语义和成本。训练 AgentLoop 当前 Actor/Expert 使用同一个 server manager，因此它不能自动代表此前独立的 Qwen3-32B-AWQ Expert；Counterfactual API collector 可配置独立服务。

原树搜索在 [user_assistant_agent_loop.py](/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO_V1/ATPO-official/codes/recipe/atpo/user_assistant_agent_loop.py:224)：按层扩展活跃叶，按 q 方差及 Critic-value/子节点均值 Q 差异剪枝，不是 best-first priority queue。Critic 返回单值，没有已实现的 epistemic uncertainty。Actor PPO、Critic loss、Tree Return/Advantage 沿用 ATPO 原实现。EA reward 是终局正确性加动作 edge cost；Attempt07 的 gamma=1，ASK=.05、CONSULT=.20、DEFER=.60；每个动作 edge 扣一次，CONSULT 后 DEFER 则分别计两次成本。训练 reward 不包含隐式 token/latency 扣费。

原始源码未加入 VOI、没有改变动作集合、Reward、PPO 或 Critic。完整路径、方法差异和状态矩阵见 [架构审计](/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO_V1/experiments/vi_ea_atpo/phase0/ARCHITECTURE_AUDIT.md)、[兼容矩阵](/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO_V1/experiments/vi_ea_atpo/phase0/COMPATIBILITY_MATRIX.md) 和 [实施计划](/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO_V1/experiments/vi_ea_atpo/phase0/IMPLEMENTATION_PLAN.md)。

## 4. Phase 1 — 1,700 对数据与初始相对价值分析（Complete）

从既有 `attempt04/paired_rollouts` 重建病例记录，匹配 case selection 与源 Parquet，并从共同 Gold 独立重算 Base/Expert correctness。source row、case hash、prompt hash 和 split 一致；train/validation/test 分别 1000/300/400，case ID、prompt hash 和 source row 在 split 间唯一。integrity audit 输出位于 [paired_integrity_audit.json](/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO_V1/experiments/vi_ea_atpo/phase1/paired_integrity_audit.json)。

| Split | Base Accuracy | Expert Accuracy | Paired Gain | Helpful | Harmful | Expert parse failure |
|---|---:|---:|---:|---:|---:|---:|
| Train (1000) | 47.80% | 52.80% | +5.00 pp [1.90, 8.10] | 155 | 105 | 38 |
| Validation (300) | 44.33% | 53.33% | +9.00 pp [3.33, 14.67] | 52 | 25 | 13 |
| Test holdout (400) | 48.50% | 57.75% | +9.25 pp [4.25, 14.25] | 72 | 35 | 15 |

CI 是按病例配对、固定 seed、10,000 次 bootstrap 的本次重算。旧报告区间因 bootstrap seed/实现/有限重采样略有差异（约 [+4.3,+14.5] pp 或 [+4.5,+14.2] pp），不改变方向。测试组四格为 both wrong 134、helpful 72、harmful 35、both correct 159。Base parser failure 3/0/0；Expert 总解析失败 66 条，按真实协议失败计错，无事后修补。

新建可见特征与离线 label 分离文件。`option_count` 和 `mean_option_chars` 是旧正则解析 fallback 造成的伪零：在三个 split 中都视作 unavailable，未送入模型。其余 public length、Base confidence/entropy/margin、token 长度等只用 train 拟合预处理。Base uncertainty 被明确分成三个目标：Base Error AUROC 0.686、Helpful AUROC 0.710、Harmful AUROC 0.614。**0.710 是 Helpful 标签，不是 Base Error 或 Relative Gain AUROC。**

新初始价值模型只用 12 个可用预决策字段，train 拟合、validation 选择、test 只复核。test 上 helpful/harmful logistic score 的 AUROC 分别 0.557/0.582，Relative Gain Spearman 0.017，Brier 0.145/0.081，10-bin ECE 0.029/0.023。Ridge 的 helpful/harmful AUC 为 0.583/0.590。排序没有稳定的 Helpful-versus-Harmful 区分能力。

新初始 estimator 在 c_D=.60 选择 0% defer；所有基线与 Always Answer 相同，Test Utility=.485。Base uncertainty 在已多次分析的旧 holdout 上，c_D=.10/.20 Utility lift 分别 +.01975（CI [.0065,.03425]）和 +.013（[.003,.025]），但新 logistic 的 lift 区间跨 0，且低成本结果不能外推到主成本 .60。该 test 曾用于旧报告和多轮分析，**不是新的独立测试**。

输出包括 `observable_cases.jsonl`、`offline_labels.csv`、`observable_features.csv`、`feature_audit.csv`、`initial_value_predictions.csv`、`initial_value_analysis.csv`、`risk_calibration.csv`、`cost_policy_metrics.csv`、`initial_value_estimator.json` 与 [初始分析报告](/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO_V1/experiments/vi_ea_atpo/phase1/INITIAL_VALUE_REPORT.md)。标签与特征的重算脚本为 [run_initial_value_analysis.py](/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO_V1/experiments/vi_ea_atpo/phase1/run_initial_value_analysis.py:1)；失败归因见 [FAILURE_ATTRIBUTION.md](/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO_V1/experiments/vi_ea_atpo/phase1/FAILURE_ATTRIBUTION.md)。

## 5. Phase 2 — 主动信息获取 pilot（Prepared; Rollouts Not Run）

新病例选择从 14,256 行 ATPO train 来源数据中抽取，先扫描两个工作树的历史实验身份。95 个 JSONL、44,264 条记录和旧选择 manifest 被检查；Attempt09 dev prompts 也被排除。固定 seed `20261118` 均匀选择 100 train/50 validation；没有 test；选择不依赖 correctness、Helpful/Harmful 或 uncertainty。哈希化 cohort 与 selection manifest 已冻结。

运行脚本从原 Collector 复用 `ChatAPI`、动作解析、患者 hidden-facts 输入、CONSULT/DEFER 专家适配和安全检查。每例固定 direct ANSWER/DEFER、ASK→强制回答、ASK→ANSWER/DEFER 终止决策、CONSULT→强制回答、CONSULT→ANSWER/DEFER 终止决策六个分支。两条 continuation 分别估 task improvement 与 decision improvement。成本采用原动作尺度及冻结成本网格；token/latency 单独报告。

运行状态：两个配置服务 endpoint 连接拒绝，模型 alias/revision 无法复核，runner 因此保持无推理模式。没有运行 2 例 smoke 或 150 例正式 pilot。manifest 标记 `model_inference_started=false`；JSONL rollout 文件为空，CSV 为 header-only。150 例 nominal API generation budget 为最多 1,800（不含 HTTP/格式重试）。完整情况见 [Phase 2 pilot 报告](/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO_V1/experiments/vi_ea_atpo/phase2/PILOT_VOI_REPORT.md) 和 [execution_manifest.json](/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO_V1/experiments/vi_ea_atpo/phase2/execution_manifest.json)。

历史主动信息结果只作背景：旧 100 例四动作 Pilot 的 ASK/CONSULT 均值略高于 direct ANSWER，但多状态/重复 rollout 且模型角色都是 8B，CONSULT−ANSWER 的 paired accuracy +2.67 pp、CI [−1.78,+7.22]；不能说明新 32B Expert 的净收益。旧 200 状态 CONSULT-Critical 建议继续实验相对 ANSWER +0.5 pp、病例聚类 95% CI [−5.9,+6.8]，完整咨询链 token 较高；该建议来自 8B 模拟器。另一个旧 200-test direct DEFER cohort 中 32B Expert 相对 8B Base +9.5 pp [2.5,16.5]，证明平均终局互补性但不是 ASK/CONSULT 的因果价值。所有旧病例均从新 cohort 排除。

## 6. Phase 3 — VOI 可预测性（Not Run）

不在缺少 Phase 2 rollout target 的情况下训练 VOI estimator。Ridge/MLP/constant/uncertainty 对照、VOI MAE/RMSE/Spearman、Positive VOI AUC、top-K precision、calibration 与策略 Utility 全部 Not Run。没有用 Oracle VOI 训练或给模型输入。

## 7. Phase 4 — VOI 与 ATPO 搜索融合（Not Run）

没有添加 `VOISearchAdapter` 到原搜索源码。原剪枝并非 best-first，Critic uncertainty 与 VOI 也不是同一信号。`λ_voi=0` 搜索轨迹兼容、树预算、Actor logprob/行为分布、Tree Return/Advantage、PPO/Critic target 一致性均 Not Run；这是因为 Phase 2/3 gate 未通过，而非声称验证通过。

## 8. Phase 5–6 — 策略训练与独立泛化（Not Run）

本轮没有 PPO/ATPO/RL、完整策略训练、最终测试或新的独立 test。既有 400 test holdout 已多次用于分析，不可再宣称独立泛化。未来官方 test 使用历史审计尚未完成；Phase 2 的 150 个样本仅用于 pilot train/validation，不是最终测试。

## 9. 测试与执行命令

实际执行：

```bash
.venv/bin/python experiments/vi_ea_atpo/phase1/run_initial_value_analysis.py
.venv/bin/python experiments/vi_ea_atpo/phase1/verify_paired_integrity.py
.venv/bin/python experiments/vi_ea_atpo/phase2/select_pilot_cases.py
.venv/bin/python experiments/vi_ea_atpo/phase2/run_information_pilot.py
.venv/bin/python -m unittest discover -s experiments/vi_ea_atpo/phase2 -p 'test_*.py' -v
PYTHONPATH=ea_atpo_stage .venv/bin/python -m unittest recipe.ea_atpo.test_protocol recipe.ea_atpo.test_collect_counterfactuals -v
```

Phase 2 新增 CPU/mock adapter tests **9/9 通过**；原 protocol 与 counterfactual collector tests **19/19 通过**；Python `py_compile` 通过。完整 EA agent-loop test discovery 额外尝试失败：当前 `.venv` 缺少 `verl`（`ModuleNotFoundError`），因此那部分未执行成功。未安装依赖，避免改动项目环境。

## 10. 研究假设与局限

已有结果表明固定模型之间可存在平均配对互补性，但 helpful/harmful 选择性差、成本敏感的 Learned Router 在主成本不调用 Expert。静态 DEFER 数据不能代替 ASK/CONSULT 干预实验；另一方面，新 active pilot 因服务不可用未测量真实信息增益。因此当前不能宣称 VOI 有效，也不能宣称所有信息获取机制都无效。病例来自公开医学 MCQ 的模型模拟环境，不是真实临床病例或医生协作。

## 11. 下一步与资源

最小后续步骤：恢复并核验 8B/32B endpoint 的模型 revision；复查冻结 config、数据和 prompt hash；只对冻结 cohort 的 2 例做 smoke；检查角色隔离、解析有效率、成本和日志后，才运行固定的 150 例 train/validation pilot。nominal 最多 1,800 次模型生成（失败重试另计），先导结果完成前不启动 PPO/ATPO。GPU-hours 需用 smoke 的实际延迟和服务配置估算；当前没有足够数据给出可信时长。

## 12. 本轮交付文件

- Phase 0：`phase0/ARCHITECTURE_AUDIT.md`、`COMPATIBILITY_MATRIX.md`、`IMPLEMENTATION_PLAN.md`。
- Phase 1：`phase1/initial_value_estimator.py`、`run_initial_value_analysis.py`、`verify_paired_integrity.py` 与全部 CSV/JSON/报告。
- Phase 2：`phase2/select_pilot_cases.py`、`pilot_config.json`、`run_information_pilot.py`、`test_information_pilot.py`、`run_phase2_pilot.sh`、冻结 cohort、manifest、失败日志与 header-only 结果文件。
- 根报告与方法设计：本报告、[METHOD_DESIGN.md](/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO_V1/experiments/vi_ea_atpo/reports/METHOD_DESIGN.md)。

## 最终决策

**D — 当前主动信息获取缺乏足够的、成本后的可重复收益证据，不建议继续到 VOI 搜索或 RL。** 当前解释是“no-go，等待新 pilot 证据”，不是“已证明 ASK/CONSULT 普遍无效”。Phase 2 rollout 未运行；只有在 endpoint 恢复后按冻结协议得到正向、可预测且验证集稳定的净 Utility，才重新考虑 Phase 3。
