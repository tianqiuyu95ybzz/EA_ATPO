# 失败归因

## 1. 旧 Logistic AUROC 是否来自特征问题？

部分是。`option_count` 和 `mean_option_chars` 在旧提取器中“未匹配就填 0”，train/validation/test 的存储值都只有一个唯一值 0。源 prompt 审计中，原选项行正则和扩展标点/括号模式对各 split 都未识别到选项行，因此不能把 0 解释为题目确实没有选项；本轮将其标为解析不可用并排除。其他 prompt 文本特征在 1,700 例上重算完全匹配。修复前后相对收益性能应看新 test helpful/harmful AUROC 和 Spearman；旧 .573 主要是 Helpful 目标结果，不是净收益 AUROC。特征错误是混杂因素，但不足以单独归因全部失败。

## 2. MLP 接近随机是否是样本不足或过拟合？

MLP Helpful/Harmful AUROC 从 train 0.675/0.688，到 validation 0.581/0.583，再到 test 0.613/0.480；relative-gain Spearman 为 +0.148/-0.018/-0.018。Helpful 排序在 test 高于随机，但 Harmful 从 train 的 0.688 降至 test 0.480；net gain 排序在 validation/test 都接近 0。证据支持 harmful 类泛化不稳/过拟合风险，400 例不足以把它与抽样噪声完全区分。MLP 只在 1,000 条 train 上拟合，hidden=16；weight decay/class weighting 由 300 条 validation 三分类 log loss 选择。参数：`{"weight_decay": 0.01, "class_weight": "none", "best_epoch": 104, "validation_logloss": 0.7233723938553503}`。

## 3. Base uncertainty 是否同时预测 Expert 相对收益？

Base Error AUROC = 0.686 [0.633, 0.737]；Helpful AUROC = 0.710 [0.645, 0.771]。这两个目标不等价。净相对收益更应看 relative gain 的 Spearman，以及分别识别 helpful 和 harmful 的分数。

## 4. 成本 0.60 下 0% DEFER 是否合理？

各预测 Router 在 test 的调用率：`{"base_uncertainty": 0.0, "logistic_base_error": 0.0025, "logistic_relative_gain": 0.0, "ridge_relative_gain": 0.0, "mlp_relative_gain": 0.0}`；Always Answer utility = 0.485。是否 defer 的阈值只从 validation 选出。对于成本 0.60，一次调用只有在预期 `relative_gain > 0.60` 时才提升期望 Utility；若验证集没有模型能把这类病例排到成本线上方，0% 是合理的保守决策。Oracle 在 test 可看见真实标签，仅是事后上界，不能在线实现。

## 5. 是否存在可预测 Expert 优势？

test 的 Logistic Relative Gain Helpful AUROC = 0.583 [0.505, 0.660]；Harmful AUROC = 0.468 [0.366, 0.571]。其他模型见主报告目标表。预测能力需要同时考虑 helpful 捕获和 harmful 回避；只预测 Base 会错不能说明 Expert 能修复，也不能排除 Expert 会退化正确答案。

## 6. Learned Router 是否优于简单 uncertainty？

没有稳定优于。Cost .10 时 uncertainty 的 Utility lift 为 +0.0198 [+0.0068, +0.0343]；Logistic Base Error 为 +0.0288，但相对 uncertainty 的 paired 差值 +0.0090 [-0.0050, +0.0233] 跨过 0。Cost .20 时 uncertainty lift +0.0130 [+0.0025, +0.0250]，Logistic Base Error 则为 -0.0005。Cost .60 时没有 Learned Relative Gain Router 获得稳定的正收益。调用率匹配随机基线及各成本 paired 区间见 `cost_sensitivity.csv`。结论：B. 暂不进入 RL；优先改进监督 Router、校准与决策前特征，并在新病例 holdout 上复验。

## 审计边界

没有重新推理，没有改变 frozen split，没有使用 test 调参或选阈值；但相同 400 例 test 已在先前报告中看过，所以这次不是独立 holdout。Gold 仅用于离线标签，所有 Router 特征均来自可见 prompt 或 Base 完成后的状态。
