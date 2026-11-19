# Initial Relative Value Analysis

## 设计

- 复用 Attempt04 固定的 1,000/300/400 paired case；没有任何重新推理。
- 分开写入 observable_cases.jsonl 和 offline_labels.csv。Gold 不保存；correctness/relative_gain 标签不进入可见特征文件。
- 12 个决策前字段中不含选项字段；option_count 与 mean_option_chars 在旧解析器中是假零值，因此标为 unavailable 并排除。
- 插补和标准化只用 train 拟合；logistic 与 ridge 超参数、每个成本的阈值只用 validation 选择。
- 400 例 test holdout 来自 ATPO train split，已有旧报告和新 Router 报告查看过。以下 test 指标是污染 holdout 复核，不是独立泛化验证。

## 结果

| Router | Cost | Test accuracy | Test utility | Defer rate | Utility lift vs Answer (paired 95% CI) |
|---|---:|---:|---:|---:|---:|
| InitialDeferralValueEstimator | 0.60 | 0.485 | 0.485 | 0.000 | +0.000 [+0.000, +0.000] |
| Base Uncertainty | 0.60 | 0.485 | 0.485 | 0.000 | +0.000 [+0.000, +0.000] |
| Ridge Relative Gain | 0.60 | 0.485 | 0.485 | 0.000 | +0.000 [+0.000, +0.000] |

### Test 主要可预测性指标
- relative_gain_spearman: 0.0172
- relative_gain_mae: 0.2954
- relative_gain_rmse: 0.5098
- helpful_brier: 0.1451
- helpful_ece_10bin: 0.0293
- harmful_brier: 0.0807
- harmful_ece_10bin: 0.0228

## 解释边界

- InitialDeferralValueEstimator 输出 P(helpful)、P(harmful) 和两者之差。bootstrap refit SD 是训练样本扰动诊断，不是经过覆盖校准的置信界。
- Utility 的成本是归一化 DEFER 成本，不是美元或临床资源成本。
- 当前数据只涉及固定模型、一次 direct answer 与一次 Expert 最终答复；不包含 ASK/CONSULT 干预，所以不能回答主动获取信息的价值。
- 详情见 initial_value_analysis.csv、risk_calibration.csv、cost_policy_metrics.csv、initial_value_predictions.csv。

## 文件

- observable_cases.jsonl：公开 prompt、Base 输出和可见特征。
- offline_labels.csv：专家输出、正确性、相对收益和解析状态；与可见特征分开。
- feature_audit.csv：按 split 的完整特征审计。
- initial_value_estimator.json：冻结参数。
