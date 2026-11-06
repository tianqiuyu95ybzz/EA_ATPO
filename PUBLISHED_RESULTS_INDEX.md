# EA-ATPO 实验结果索引

此提交整理了已完成的 Pilot、CONSULT 审计、强 Expert 对照和四动作 Router 实验。报告、逐例轨迹、CSV 汇总、模型配置、checkpoint 与必要的运行代码均已纳入版本控制。

## 真实模型 Pilot 与审计

- 最终 100 例 Pilot：`experiments/real_case_pilot_20261103/outputs/full/`
- Pilot 协议与结果审计：`experiments/real_case_pilot_20261103/audit/PILOT_AUDIT_REPORT.md`
- CONSULT 质量分析：`experiments/real_case_pilot_20261103/audit/CONSULT_QUALITY_ANALYSIS.md`
- Advice Utilization 对照：`experiments/real_case_pilot_20261103/audit/ADVICE_UTILIZATION_EXPERIMENT.md`
- 动作价值与阶段决策：`experiments/real_case_pilot_20261103/audit/ACTION_VALUE_REANALYSIS.md`、`NEXT_STAGE_DECISION.md`

## 四动作价值 Router

- 逐轮独立测试报告与原始轨迹：`experiments/ea_atpo_value_router_20261105/outputs/stepwise_test_v2/`
- 训练代码、配置、checkpoint 与强专家评估代码：`experiments/ea_atpo_value_router_20261105/`
- 强 Expert 32B 与 8B 配对评估：`experiments/ea_atpo_value_router_20261105/outputs/strong_expert_awq/`
- 汇总决策：`experiments/ea_atpo_value_router_20261105/CORE_METHOD_DECISION.md`

## 大文件排除说明

Qwen 模型权重、下载缓存、Python 环境和源数据集副本未放入 Git。Qwen3-32B-AWQ 的 Hugging Face 与 ModelScope 固定 revision、分片 SHA-256 和来源信息记录在 `experiments/real_case_pilot_20261103/assets/models/Qwen3-32B-AWQ/model_source_lock.json`；断点下载脚本在 Router 实验目录中。

当前实测没有证明 Learned Value Router 优于 Always Answer；报告保留了该负面结果及必要限制。项目没有启动 P5 Router RL 或 P8 联合 ATPO RL。
