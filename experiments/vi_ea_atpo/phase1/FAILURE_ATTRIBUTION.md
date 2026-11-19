# Phase 1 — 失败归因与证据边界

## 已复核的标签与解析

对每一行，从锁定源 Parquet 重新读取同一 Gold，并由 Base/Expert 的已解析选项独立重算 correctness。1,700 条 case ID、prompt hash、source row index 在 split 间均唯一，病例 split 没有交叉；paired 文件的 split、case hash、prompt hash 与选择 manifest 一致。`relative_gain = expert_correct - base_correct`，未把 `base_error` 当作相对收益。

test 400 的四格为 both wrong 134、helpful 72、harmful 35、both correct 159；Base 0.485、Expert 0.5775、净配对差 +0.0925。按固定 seed 和 10,000 次病例级 bootstrap 重算，配对差 CI 为 [0.0425, 0.1425]。旧报告的边界因 bootstrap 实现、seed 和有限重采样略有差异（约 [+0.043,+0.145] 或 [+0.045,+0.142]），都不改变“点估计为正”的判断。

Expert parser failure 为 train/validation/test = 38/13/15（共 66）；这些输出按部署格式约定记为错误，没有事后修补。Base parser failure 为 3/0/0。这影响端到端相对收益，但属于模型协议表现的一部分。400 例来自 ATPO train split 且已被多个旧报告反复查看；它是历史 holdout，不是新独立泛化证据。

## Learned Relative Gain 信号

新 CPU 复核只用 train 拟合预处理、logistic 和 ridge；模型配置由 validation 选择，成本阈值也逐成本只在 validation 选择。test 只应用冻结模型/阈值。该 holdout 仍有历史污染风险。

- Base entropy uncertainty 对 Base Error 的 AUROC 是 0.686；对 Helpful 是 0.710；对 Harmful 是 0.614。0.710 对应 Helpful 标签，不是 Base Error 或 Relative Gain。
- 新 helpful/harmful logistic `P(helpful)-P(harmful)` 在 test 的 Helpful AUROC 0.557、Harmful AUROC 0.582、Relative Gain Spearman 0.017；Brier 分别 0.145/0.081，10-bin ECE 分别 0.029/0.023。Brier/ECE 的数值不构成校准通过证据，尤其类别稀疏、且无外部验证。
- Ridge Relative Gain test Helpful/Harmful AUROC 为 0.583/0.590，但相对收益排序仍很弱；前一版本的 Logistic/Ridge/MLP 结果分别约 0.583/0.574/0.613 Helpful AUROC，净收益 Spearman 约 −0.052/+0.037/−0.018。模型版本间分数不完全可比，但共同点是 Helpful 识别没有转化为稳定的净收益排序。
- uncertainty 在 c_D=0.10 和 0.20 的 held-out Utility lift 为 +0.01975（95% CI [+0.0065,+0.03425]）和 +0.013（[+0.003,+0.025]）；这是已经反复查看的旧 400 例点，不能再作为新策略选择集。新 logistic 在这两个成本的 lift CI 均跨 0。c_D=0.60 时所有学习策略与 uncertainty 在 validation 冻结的阈值均为零调用，Utility 与 Always Answer 相同。

## 原因判断

1. **特征问题确实存在，但不是全部原因。** `option_count` / `mean_option_chars` 的原始零是解析 fallback，不是真实病例零；它们标为 unavailable 并从新模型剔除。可用 Base confidence、entropy、margin、prompt 和 token 特征均在决策前可见。修复零值问题没有令净收益排序变强。
2. **AUROC 标签不平衡且目标不同。** 400 例中 Helpful 占 18%、Harmful 占 8.75%；一个能找到 Base 错误或 Helpful 个案的分数仍可能同时调用 Harmful 个案。要路由的是 `P(helpful)-P(harmful)` 超过成本，而非仅提高 Helpful AUC。
3. **uncertainty 的正 AUROC 不等于成本下可路由。** c_D=0.60 需要可观测分数筛出预期净正确增益大于 0.60 的病例；当前 Helpful 概率/排序无此证据。固定 validation 阈值选择零调用是预设 Utility 下的合理保守决策，不是系统错误。
4. **不能只把 MLP 失败归因于参数量或样本量。** 旧 MLP train Helpful/Harmful AUC 高于 validation，test harmful 区分回落；这与过拟合/抽样不稳定兼容，但现有 400 例不能识别唯一原因。Ridge/Logistic 也未得到可靠净收益排序。
5. **主动信息价值仍未知。** 静态 DEFER 配对标签不能推导 ASK/CONSULT 的因果回报。既有咨询建议继续推理实验提升不显著且 token 增加；本轮新 Phase2 因 endpoint 不可用未运行，所以不将它描述为信息没有价值的最终反证。

## 决策

现有结果支持“Expert 有平均互补性”，不支持“能够在主成本下提前可靠识别收益”。Phase 2 仍待运行；在出现新 cohort 的正向、可重复净 Utility 并验证可预测性之前，不训练 VOI estimator、不改树搜索、不启动 PPO/ATPO。
