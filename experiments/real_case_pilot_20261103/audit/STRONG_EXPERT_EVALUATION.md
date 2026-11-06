# Qwen3-32B 强 Expert 验证（更新版）

本报告替代 2026-11-05 下载阶段的未测状态。原始 BF16 `Qwen/Qwen3-32B` revision `9216db5781bf21249d130ec9da846c4624c16137` 仍未完成下载；随后采用官方 AWQ 量化变体 `Qwen/Qwen3-32B-AWQ`，已在 vLLM 0.33.0 上成功部署并完成冻结 200 例配对评估。AWQ 结果不冒充 BF16 结果。

## 结果

| 指标 | Qwen3-8B Base | Qwen3-32B AWQ |
|---|---:|---:|
| 总体准确率 | 44.0%（88/200；95% CI 37.0%–51.0%） | 48.0%（96/200；95% CI 41.0%–55.0%） |
| 平均 tokens/病例 | 170.32 | 170.32 |
| 平均请求延迟 | 0.065 秒 | 0.143 秒 |

配对准确率差为 +4.0 个百分点，95% 病例级 bootstrap CI 为 −2.5 至 +10.5 个百分点，未证明总体准确率提升。

Base 错误的 112 例中，AWQ 32B 答对 26 例，条件准确率/纠错率为 23.2%（95% CI 15.7%–31.4%）。Base 正确的 88 例中，有 18 例被 32B 改为错误，退化率为 20.5%（12.4%–29.3%）。配对四格表计数为：两者都正确 70、Base 对而 32B 错 18、Base 错而 32B 对 26、两者都错 86。

模型与 Base 使用相同公开题干、提示、答案解析和抽样；Gold 只用于事后评分，atomic facts 未发送给模型。两者均为 LLM，不是真实医生。条件指标已从保存的原始轨迹重算，修正了旧分析中条件分子未限定于对应分母组的问题；原始推理轨迹未修改。

## 可复现结果

- 当前完整报告：`experiments/ea_atpo_value_router_20261105/outputs/strong_expert_awq/STRONG_EXPERT_EVALUATION.md`
- 原始 32B 输出：`experiments/ea_atpo_value_router_20261105/outputs/strong_expert_awq/qwen3_32b_direct_answers.jsonl`
- 8B 基线及配对病例表：`outputs/direct_eval/` 与 `experiments/ea_atpo_value_router_20261105/outputs/strong_expert_awq/paired_correctness.csv`
- 模型源、revision 与哈希锁：`experiments/real_case_pilot_20261103/assets/models/Qwen3-32B-AWQ/model_source_lock.json`
