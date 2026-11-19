# EA-ATPO V2 50 训练病例反事实 Pilot（2026-11-20）

## 实验边界

- 固定选择种子：`20261119`；按 `case_id` 从既有训练分区确定 50 个病例，分 5 批、每批 10 例。
- 首批重跑原 10 例以比较 ASK 修复，其余 40 例为训练分区新病例。
- 50 例共 146 个冻结公开状态；每个状态独立执行 ASK / ANSWER / DEFER，ASK 按固定的“一次患者回复、最多一次 Unknown 复核、Base 强制终答”策略继续。
- Base：Qwen3-8B revision `b968826d…`；Expert：Qwen3-32B-AWQ revision `0499c3ac…`；Patient 与 Base 共用 Qwen3-8B 服务。
- 模型均关闭 thinking，temperature=0、top_p=1。所有状态使用同一公开内容；Gold 仅在三条分支完成后评分。Base/Expert 不接收 hidden facts。
- 未打开 Validation、Internal Test 或 Frozen Holdout；未运行 Q Regression、Actor 蒸馏、PPO 或 SFT。

## 分批完成情况

| 批次 | 病例 | 状态 | 轨迹/有效 | Q 标签 | ASK Unknown | DEFER 纠错/退化 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 10 | 29 | 87 / 87 | 87/87 | 21（72.4%） | 1 / 0 |
| 2 | 10 | 29 | 87 / 87 | 87/87 | 24（82.8%） | 3 / 4 |
| 3 | 10 | 30 | 90 / 90 | 90/90 | 23（76.7%） | 8 / 8 |
| 4 | 10 | 29 | 87 / 87 | 87/87 | 17（58.6%） | 7 / 2 |
| 5 | 10 | 29 | 87 / 87 | 87/87 | 13（44.8%） | 6 / 4 |
| **合计** | **50** | **146** | **438 / 438** | **438/438** | **98（67.1%）** | **25 / 18** |

每个批次均通过相同状态配对、三动作覆盖、成本一次性扣除检查。所有答案和问题格式有效，146/146 ASK 状态转移结构有效；无 API、解析或轨迹失败。全量 Q 数据为 146 个状态，每个状态含三个有效目标，成本已计入 Q。

## 汇总动作表现

| 动作 | 准确率 | 平均回报 | 平均 Token | 平均延迟 | 总动作成本 |
|---|---:|---:|---:|---:|---:|
| ANSWER | 57/146（39.0%） | 0.390 | 243 | 0.065 秒 | 0 |
| ASK | 56/146（38.4%） | 0.334 | 1,070 | 0.354 秒 | 7.30（0.05/次） |
| DEFER | 64/146（43.8%） | -0.162 | 259 | 0.115 秒 | 87.60（0.60/次） |

ASK 相对 ANSWER 的配对纠错为 10、退化为 11，平均净回报差 -0.057。DEFER 配对纠错 25、退化 18，准确率高 4.8 个百分点，但按预设成本 0.60 计费后平均净回报比 ANSWER 低 0.552。48/146 个 ASK 回复满足“非 Unknown 且文本新颖”代理标准；该指标不等于医学信息质量。

## 资源与数据完整性

模型服务通过健康检查；启动阶段 GPU 2 空闲约 182.6 GiB。模型加载后每批前后至少保留约 115.1 GiB 显存，未发生 OOM。三动作采集完成后，由本次服务管理器停止的 Base/Expert 服务已关闭，GPU 2 空闲 182,621 MiB（利用率 0%）。端口 30481/30482 已释放；独立服务 18951 未修改。

分析脚本标记了 3 个公开状态的题干/选项语义异常（共 9 条动作级轨迹）。轨迹格式仍有效，但这些状态应在后续拟合时显式报告，不能静默删除。

## 文件与复现

- 完整三动作原始轨迹：`outputs/ask_unknown_repair_50case_20261120/merged/pilot_trajectories.jsonl`
- Q Dataset：`outputs/ask_unknown_repair_50case_20261120/merged/q_dataset.jsonl`
- 全量汇总：`outputs/ask_unknown_repair_50case_20261120/merged/analysis/pilot_summary.json`
- 每批汇总 CSV：`outputs/ask_unknown_repair_50case_20261120/reports/batch_summary/batch_summary.csv`
- 最终结构/隔离/服务资源核验：`outputs/ask_unknown_repair_50case_20261120/final_validation.json`
- 逐批 manifest、GPU 快照、服务日志和权重清单均在同一 `outputs/ask_unknown_repair_50case_20261120/` 目录中。

运行脚本：`scripts/prepare_50_case_batches.py`、`scripts/run_unknown_review_smoke.py`、`scripts/run_counterfactual_pilot.py`、`scripts/build_q_dataset.py`、`scripts/analyze_counterfactual_pilot.py`、`scripts/merge_counterfactual_batches.py`、`scripts/summarize_50_case_batches.py`。每批命令及配置路径记录于本次运行 manifest 和批次目录。
