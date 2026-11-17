# EA-ATPO 文档索引

更新日期：2026-11-17

此页标出当前有效的研究口径、实验结果和复现入口。历史计划与中间实验保留在对应目录或 Git 历史；若与较新的结果冲突，以带日期的原始实验报告和 `PUBLISHED_RESULTS_INDEX.md` 为准。

## 当前项目口径

- [项目与上游 ATPO 代码布局](PROJECT_LAYOUT.md)
- [研究综述与方法定位](EA_ATPO_RESEARCH_REVIEW_AND_NEW_METHOD_2026-11-02.md)
- [研究路线与创新点状态](EA_ATPO_ROADMAP_2026-11-02.md)
- [真实实验预注册](EA_ATPO_EXPERIMENT_PREREGISTRATION_2026-11-03.md)
- [EA-ATPO V1 实现记录](EA_ATPO_V1_IMPLEMENTATION.md)

## 已完成实验与当前结论

- [实验结果总索引](PUBLISHED_RESULTS_INDEX.md)
- [Attempt09 训练结果](experiments/ea_atpo_v1_smoke/ea_atpo_v1_attempt09_stability/ATTEMPT09_RESULTS.md)
- [Attempt09 配对评估与 Critic 诊断](experiments/ea_atpo_v1_smoke/ea_atpo_v1_attempt09_stability/paired_evaluation/)
- [Pilot 数据和协议说明](experiments/real_case_pilot_20261103/README.md)
- [公开医疗 Qwen 专家候选评估与 L2D DEFER 结果](experiments/real_case_pilot_20261103/audit/MEDICAL_EXPERT_CANDIDATE_EVALUATION_2026-11-16.md)
- [ReMeDi 100 例独立确认与 32B-AWQ 部署尝试](experiments/real_case_pilot_20261103/audit/L2D_REMEDI_CONFIRMATION_2026-11-17.md)
- [其他候选 DEFER 与 CONSULT 小提示分析](experiments/real_case_pilot_20261103/audit/L2D_EXPERT_AND_CONSULT_HINT_ANALYSIS_2026-11-17.md)
- [专家条件互补性、成本敏感性与 Router 可行性审计](EXPERT_CONDITIONED_COMPLEMENTARITY_COST_ANALYSIS.md)
- [med_qwen 模型核验与待执行配对评估计划](MED_QWEN_EXPERT_EVALUATION.md)

## 实验记录与复现脚本

- [四动作协议与运行脚本](ea_atpo_stage/)
- [Attempt09 配对评估脚本](ea_atpo_stage/scripts/run_attempt09_paired_evaluation.py)
- [实验预注册和原始结果目录](experiments/)
- [完整 smoke 运行历史](EA_ATPO_SMOKE_TEST.md)

## 文档清理

2026-11-16 将 2026-09-29 的三份早期方案草案和 2026-10-29 的旧环境交接文档移出当前文档集合；它们已被本页列出的研究综述、路线图、预注册和实现记录取代。删除只影响当前工作树，既有 Git 提交仍保留历史版本。
