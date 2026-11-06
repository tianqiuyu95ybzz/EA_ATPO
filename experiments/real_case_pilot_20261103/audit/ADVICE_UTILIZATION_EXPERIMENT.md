# CONSULT 后决策机制实验

## 摘要

在同一批 200 个既有 Pilot 公开状态、同一条已保存 CONSULT 问题和逐字相同的模拟专家建议上，比较了原始 ANSWER、CONSULT-Naive、CONSULT-Retain 和 CONSULT-Critical。新运行共完成 **600 次 Qwen3-8B 调用**（后三种方法各 200 次）；200 个状态来自 97 个病例。Gold 仅用于本地事后评分，没有发给模型。

CONSULT-Critical 得到 88/200 正确（44.0%），比配对 Pilot ANSWER 的 87/200（43.5%）高 **0.5 个百分点**，病例级配对 CI 为 **−5.9 到 +6.8 个百分点**。CONSULT-Naive 和 Retain 分别是 42.0% 和 40.0%。三种方法均未显示统计上明确的准确率提升。Critical 的正确退化率最低，但错误修正率也低于 Naive。建议确实会改变答案，然而答案改变率不能证明模型理解或正确采纳了建议。

## 实验协议

- **冻结样本：** 从 Pilot 的 291 个病例/状态中，按 seed `20261105` 抽取 200 个合格状态（97 个病例）。合格状态须有有效的已保存 CONSULT 建议和同一 rollout 的有效 Pilot ANSWER；有 1 个状态没有有效 CONSULT 建议，未进入抽样框。每个状态固定一个 rollout。
- **基线 A — ANSWER：** 同一公开状态上的原始 Pilot 直接回答，不含专家建议。
- **B — CONSULT-Naive：** 新的 Qwen3-8B 继续回答；输入公开状态、原始咨询问题和逐字专家建议，不告知原 Pilot 答案。
- **C — CONSULT-Retain：** 与 B 相同，并告知原 Pilot 答案；要求声明 `KEEP` 或 `UPDATE`，允许保留原答。
- **D — CONSULT-Critical：** 与 C 相同；先要求把专家建议主张与可见病例证据逐项对照，标记无法由公开状态验证的内容，再决定保留或更新。
- B/C/D 使用 Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`，temperature 0、top-p 1、最多 128 tokens、seed `20261105`、关闭 thinking、prefix cache 关闭。三个方法共享完全相同的 `public_state_sha256` 和 `expert_advice_sha256`；答案解析器也相同。
- 实验输入消息仅由公开题干、选项、可见对话、咨询问题、专家建议，以及 C/D 所需的先前选项组成。没有把隐藏 atomic facts、Gold 或未来 Patient 回复传给模型。样本 JSONL 中的 Gold 和轨迹中的评分 Gold 均为 evaluator 字段；不属于 API `messages`。
- Pilot 没有保存先前 Base 的文字推理，只有先前选项。因此 C/D 拿到的是原答案选项，没有伪造或补写 rationale。Critical 是一次新的可见证据检查。

## 准确率与修正/退化

答案正确计 1，错误、无法按固定格式解析或 API 失败计 0，均保留在 200 的分母内。95% CI 按病例聚类 bootstrap（97 个病例、10,000 次重采样）。修正率的分母是原 Pilot ANSWER 错误的 113 个状态；退化率分母是原 ANSWER 正确的 87 个状态。

| 方法 | 最终正确 | 准确率（病例级 95% CI） | 修正原错误 | 破坏原正确答案 | 格式有效 |
|---|---:|---:|---:|---:|---:|
| ANSWER（Pilot） | 87/200 | 43.5%（33.8–53.4） | — | — | 200/200 |
| CONSULT-Naive | 84/200 | 42.0%（33.3–50.8） | 20/113 = 17.7%（10.6–25.9） | 23/87 = 26.4%（16.3–38.3） | 174/200 |
| CONSULT-Retain | 80/200 | 40.0%（31.3–49.1） | 14/113 = 12.4%（6.8–19.1） | 21/87 = 24.1%（14.1–35.4） | 195/200 |
| CONSULT-Critical | 88/200 | 44.0%（34.7–53.7） | 15/113 = 13.3%（6.8–20.9） | 14/87 = 16.1%（7.5–26.8） | 200/200 |

Naive 有 26 个回答未能按解析器要求给出单一 `Final Answer: A-D`；Retain 有 5 个。它们不是 HTTP/API 异常，仍按协议失败计 0，未重试。部分输出达到 128-token 上限；也有样本的原题干与选项不匹配，模型回答“没有合适选项”因而无法解析。该定义衡量端到端协议成功率，不能把每个无效样本都解释成单纯的医学知识错误。

## 配对差异

以下是准确率百分点差；CI 用病例为聚类单位，对相同状态配对比较，10,000 次 bootstrap。所有区间都包含 0。

| 配对差异 | 准确率差 | 95% CI |
|---|---:|---:|
| Naive − ANSWER | −1.5 pp | −8.8 至 +5.6 pp |
| Retain − ANSWER | −3.5 pp | −10.0 至 +2.9 pp |
| Critical − ANSWER | +0.5 pp | −5.9 至 +6.8 pp |
| Retain − Naive | −2.0 pp | −8.0 至 +4.4 pp |
| Critical − Naive | +2.0 pp | −5.6 至 +9.6 pp |
| Critical − Retain | +4.0 pp | −1.5 至 +9.6 pp |

Critical 相较 Naive 的点估计为 +2.0 pp，且把退化从 26.4% 降至 16.1%；但其修正率也从 17.7% 降至 13.3%，总准确率差的 CI 仍跨 0。这个结果符合“少做无依据更新”的假设，但尚不足以证明 Critical 优于其它方法。

## 建议采纳的操作指标

用最终选项是否改变原 Pilot 答案作为可观测的**答案改变率**。它不是语义层面的建议采纳标签，也不能判断改变是由建议、额外推理还是提示结构导致。

| 方法 | 有效回答中的选项改变 | 模型声明 UPDATE |
|---|---:|---:|
| CONSULT-Naive | 51/174 = 29.3%（21.7–37.6） | 不适用 |
| CONSULT-Retain | 46/195 = 23.6%（16.8–30.6） | 113/200 = 56.5%（48.6–64.6） |
| CONSULT-Critical | 36/200 = 18.0%（12.5–23.8） | 62/200 = 31.0%（24.1–38.3） |

`UPDATE` 是模型自报的决策标签，不等于最终选项不同：声明 UPDATE 后仍可给出同一字母。反之，仅看选项变化也不能知道模型是否采用了建议中的有效信息。后续评价应由独立标注员逐条判定建议是否正确/相关，以及 Base 是否恰当地接受或拒绝，而不是把“更常改答案”当成收益。

## Token 与延迟

新方法的请求是 continuation；为估计整条 CONSULT 分支，把同一 rollout 的实际 Base 咨询问句调用和 Expert 回答调用加到 continuation。每个样本恰有两个 CONSULT setup 请求，平均合计 1,273.9 tokens、0.388 秒请求延迟。先前获取公开状态的成本对所有方法相同，视为共同沉没成本而排除。

| 方法 | 完整分支平均 tokens（病例级 95% CI） | 延迟和均值秒（病例级 95% CI） |
|---|---:|---:|
| ANSWER | 550.2（545.5–555.1） | 0.067（0.066–0.068） |
| CONSULT-Naive | 1,685.2（1,669.3–1,702.4） | 1.390（1.335–1.444） |
| CONSULT-Retain | 1,702.2（1,685.7–1,719.6） | 1.002（0.967–1.037） |
| CONSULT-Critical | 1,756.2（1,739.6–1,772.8） | 1.006（0.983–1.029） |

相对 ANSWER，完整 CONSULT 分支平均多 **1,135–1,206 tokens**；病例级 paired CI 分别为 Naive +1,123 至 +1,148、Retain +1,140 至 +1,164、Critical +1,194 至 +1,218 tokens。延迟差依次为 +1.323 秒（1.269–1.378）、+0.935 秒（0.901–0.970）、+0.939 秒（0.916–0.962）。

延迟是观测量而非受控的吞吐比较：Pilot ANSWER 和 CONSULT setup 来自原始批处理服务；新的 continuation 来自 GPU 1 上 `max_num_seqs=1`、关闭 prefix cache 的单请求服务。完整 CONSULT 延迟是这两个来源的逐请求延迟相加。Token 数来自真实 API usage，包含各自的 prompt 和 completion。

## 解释与相关研究

这次提供建议的 Expert 也是同一 revision 的 Qwen3-8B 模拟器，不是 Qwen3-32B，也不是真实医生。因此实验识别的是**看到同一模拟建议后，不同 Base 决策提示的行为差异**；没有识别专家相对 Base 的独立准确率，也没有直接测出建议内容本身的因果贡献。

- [Kumaran et al., *Competing Biases underlie Overconfidence and Underconfidence in LLMs*, Nature Machine Intelligence, 2026](https://www.nature.com/articles/s42256-026-01217-9) 报告了可见初始答案带来的固着，以及模型对相反建议的过度敏感。该研究支持同时测“错误修正”和“正确退化”，而不是只优化更新率；它不证明本实验的提示机制有效。
- [Bo et al., *To Rely or Not to Rely?*, CHI 2025](https://arxiv.org/abs/2412.15584) 是人类用户实验；部分干预能减少过度依赖，但未普遍改善恰当依赖。对象与本实验不同，但提醒我们不要把较低采纳或较高采纳本身当成质量指标。
- [Ni, *Position Matters More Than Channel*, arXiv preprint, 2026](https://arxiv.org/abs/2610.34809) 在 Qwen3.5-27B 的 MMLU-Pro 设置中考察建议位置、来源归属和模型先前答案。该工作仍是单模型、特定题集的预印本；它提示后续要把“是否展示原答案”和“证据检查 prompt”分开控制。

本阶段没有真实医生或临床工作流参与。模型输出基于公开 benchmark 的模拟状态，只适合研究协议，不构成临床效用或安全性证据。

## 可复现产物

- 原始 600 条调用及完整 API messages：`outputs/advice_utilization/advice_utilization_trajectories.jsonl`
- 冻结的 200 状态：`outputs/advice_utilization/frozen_consult_inputs.jsonl`
- 指标汇总：`outputs/advice_utilization/advice_utilization_summary.csv`
- 完整链路成本汇总/逐状态明细：`outputs/advice_utilization/advice_utilization_cost_summary.csv`、`outputs/advice_utilization/advice_utilization_cost_by_state.csv`
- 配对差异及病例级 CI：`outputs/advice_utilization/advice_utilization_paired_differences.csv`
- 配置、样本和轨迹哈希：`outputs/advice_utilization/run_manifest.json`
- 实验和离线统计脚本：`run_advice_utilization_experiment.py`、`analyze_advice_utilization.py`
