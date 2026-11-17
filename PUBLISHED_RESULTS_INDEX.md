# EA-ATPO 实验结果索引

此索引整理已完成的 Pilot、CONSULT 审计、专家对照和四动作 Router 实验。报告与必要运行代码纳入版本控制；含病例内容的逐条轨迹、模型权重、数据集和大 checkpoint 按实验说明留在 PFS 或被 Git 忽略，公开报告只引用聚合指标和复现哈希。

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

## EA-ATPO V1 Attempt09

- 10 步 PPO 烟测与 Critic 稳定性记录：`experiments/ea_atpo_v1_smoke/ea_atpo_v1_attempt09_stability/`
- 配对开发集结果、checkpoint / tokenizer 哈希与复现 manifest：`experiments/ea_atpo_v1_smoke/ea_atpo_v1_attempt09_stability/paired_evaluation/`
- 阶段报告：`BASELINE_PAIRED_EVALUATION.md`、`CRITIC_DIAGNOSIS.md`、`ACTION_DISTRIBUTION_ANALYSIS.md`、`NEXT_EXPERIMENT_DECISION.md`
- 当前结论：恢复后的 EA-ATPO 在 20 例上答对 4 例；没有证明优于 Always Answer，且低于当前两步 Original ATPO checkpoint。CONSULT 独立收益未显示，DEFER 没有验证成功案例，Critic 10/10 步 EV 为负；不据此扩展 PPO。
- 大模型权重、恢复 checkpoint 与 Parquet 病例数据保留在 PFS，不纳入 Git；Git 中只有哈希、匿名配对指标、报告与运行脚本。

## 公开医疗 Qwen 专家筛选（2026-11-16 至 2026-11-18）

- [ReMeDi 100 例独立 L2D DEFER 确认与 32B 后续尝试](experiments/real_case_pilot_20261103/audit/L2D_REMEDI_CONFIRMATION_2026-11-17.md)
- [候选专家模型配对评估与 L2D DEFER 验证](experiments/real_case_pilot_20261103/audit/MEDICAL_EXPERT_CANDIDATE_EVALUATION_2026-11-16.md)
- [ReMeDi、PopiT、Huatuo DEFER 复测与 CONSULT 小提示 pilot](experiments/real_case_pilot_20261103/audit/L2D_EXPERT_AND_CONSULT_HINT_ANALYSIS_2026-11-17.md)
- [断点续传与模型分片哈希脚本](experiments/real_case_pilot_20261103/audit/download_medical_candidates.py)
- [200 例直接问答配对评估](experiments/real_case_pilot_20261103/audit/run_medical_candidate_eval.py)
- [20 例 DEFER 专家协议评估](experiments/real_case_pilot_20261103/audit/run_medical_candidate_defer_dev_eval.py)
- [100 例 ReMeDi 确认 runner](experiments/real_case_pilot_20261103/audit/run_l2d_remedi_confirmation.py)
- [Qwen3-32B-AWQ 新鲜病例筛选 runner 与单卡服务脚本](experiments/real_case_pilot_20261103/audit/run_l2d_expert_screen_32b.py)、[启动服务](experiments/real_case_pilot_20261103/audit/start_l2d_32b_awq_service.sh)
- ReMeDi 的 20 例开发集优势没有复现：新 100 例上 ReMeDi-DEFER 为 40%，Always Answer 为 54%，配对差 −14 pp（95% CI −23 至 −5）。当前没有通过验证的 DEFER Expert。Qwen3-32B-AWQ 的新 DEFER 确认因 GPU 资源变化未能启动完成；既有直接答题结果不能代替 DEFER 验证。未启动 Router、PPO 或 ATPO RL。
- 更新的候选筛选中，Base-DEFER/ReMeDi/PopiT 分别为 44%/41%/40%，Always Answer 为 41%；合并两个独立 ReMeDi 100 例集合后为 40.5% 对 47.5% Always Answer（配对 −7 pp，95% CI −13.5 至 −0.5）。20 例 CONSULT 小提示探索结果未显示收益，候选提示合规率为 30%–85%。没有通过验证的生产式 DEFER Expert，也没有将 Gold 作为输出或训练数据。

## 大文件排除说明

Qwen 模型权重、下载缓存、Python 环境和源数据集副本未放入 Git。Qwen3-32B-AWQ 的 Hugging Face 与 ModelScope 固定 revision、分片 SHA-256 和来源信息记录在 `experiments/real_case_pilot_20261103/assets/models/Qwen3-32B-AWQ/model_source_lock.json`；断点下载脚本在 Router 实验目录中。

当前实测没有证明 Learned Value Router 优于 Always Answer；报告保留了该负面结果及必要限制。项目没有启动 P5 Router RL 或 P8 联合 ATPO RL。
