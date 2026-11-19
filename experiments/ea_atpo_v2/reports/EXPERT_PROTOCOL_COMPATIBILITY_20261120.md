# V2 DEFER 与历史 200 例协议一致性审计（2026-11-20）

## 相同项与差异项

| 设置 | 历史 200 例 DEFER | 本次 V2 50 例 | 是否一致 |
|---|---|---|---|
| Base | Qwen3-8B，revision `b968826d…` | 同一 revision | 是 |
| Expert | Qwen3-32B-AWQ，revision `0499c3ac…` | 同一 revision | 是 |
| vLLM | 0.33.0 | 0.33.0 | 是 |
| 服务 GPU / 并行 | GPU 2，TP=1 | GPU 2，TP=1 | 是 |
| thinking | 关闭 | 关闭 | 是 |
| temperature / top_p | 0 / 1 | 0 / 1 | 是 |
| tokenizer | 两模型配置同 SHA；当前固定资产 chat-template SHA `a55ee1b1…` | 同一固定 revision/资产 | 版本相符；历史 manifest 未保存模板指纹 |
| Base / Expert 最大生成 Token | 64 / 128 | 24 / 32 | 否 |
| 请求 seed | 历史请求未发送 seed | 每个 state-action 发送固定 rollout seed | 否 |
| DEFER 提示 | V1 senior clinician 文本，要求 `Expert Final Answer:` | V2 independent Expert 文本，要求 `Final Answer:` | 否 |
| 解码约束 | 无结构化输出约束，生成后解析 | vLLM regex 约束 `Final Answer: [A-D]` | 否 |
| 解析规则 | Base 接受唯一 `Final Answer` 或裸字母；Expert 验证 `Expert Final Answer` | 对 `Final Answer: A-D` 做精确解析 | 否 |
| 数据 | 200 个独立历史测试病例 | 50 个训练病例、146 个公开状态 | 否 |

两组结果都按解析出的选项字母与 answer key 比较正确性，解析失败不计为正确。尽管核心模型 revision、温度和 top-p 一致，提示、token 上限、seed、约束解码、输出前缀、parser 和病例集合不同，**不能将历史 55.5% Expert / 46.0% Base 与本次 50 例准确率作为受控横向比较**。历史 manifest 没有记录 tokenizer 模板 hash；本报告核对的是相同固定 revision 下当前本地 tokenizer 配置与模板 SHA。

完整配置及提示 SHA 见 `outputs/ask_unknown_repair_50case_20261120/reports/protocol_audit/expert_protocol_compatibility.json`；无 Gold 值的逐状态选项配对记录见同目录 `expert_prediction_agreement.csv`。

## 当前 Base/Expert 预测配对

| 队列 | 状态 | 选项预测完全相同 | 纠错 | 退化 | 两者均错 |
|---|---:|---:|---:|---:|---:|
| 原 V2 10 例 Pilot | 29 | 20（69.0%） | 0 | 0 | 20 个均错状态中有 9 个预测字母不同 |
| Patient 策略重跑的首 10 例 | 29 | 19（65.5%） | 1 | 0 | 20 |
| 新 50 训练病例 | 146 | 72（49.3%） | 25 | 18 | 64 |

50 例中有 74 个状态 Base 与 Expert 选项不同：25 个状态 Expert 纠正 Base、18 个状态 Expert 使正确答案退化、31 个状态两者都错但选项不同。Base 正确 57/146（39.0%），Expert 正确 64/146（43.8%），净差 +4.8 个百分点；DEFER 成本 0.60 后平均回报为 -0.162，低于 ANSWER 的 0.390。

这些配对只描述当前 50 个训练病例和 V2 提示协议，不足以证明泛化的专家互补性，也不替代历史 200 例实验。当前数据中存在一定互补信号，但纠错与退化并存，是否可由决策前特征预测尚未测试。
