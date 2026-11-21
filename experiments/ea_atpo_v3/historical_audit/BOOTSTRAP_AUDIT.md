# Bootstrap 与病例配对审计

## 已发现的 V2 Router CI 缺陷

V2 Router 训练分支 `codex/router-training-20261120` 的 `cluster_bootstrap_delta()` 对学习策略和基线按如下方式配对：

```python
for policy_seed, baseline_seed in zip(policy_cases_by_seed, baseline_cases_by_seed):
    ...
```

学习 Router 有三个固定种子 `{17, 42, 2026}`，Always ANSWER 只有一份共享固定预测。Python `zip()` 只迭代一对，因此 seed 42、2026 两组 Router 预测没有进入原 CI。审计对旧算法使用相同 5,000 次重采样复算，旧区间与保存结果精确吻合，确认不是浮点或随机误差。

审计分支中的修订版会在基线只有一组时重复该基线，并检查策略与基线 seed 数一致；该修订**尚未合并到训练分支**。这轮没有修改其他分支。

## 原区间与修正区间

下表的点差按三 seed 等权平均、每病例等权计算。重采样单位是 `case_id`，每次抽取一个病例时保留该病例全部状态；同一病例的三个 Router seed 预测也共同保留。修正区间为 5,000 次 percentile bootstrap，NumPy PCG64，seed `20261120`。

| 方法 | 对比 Always ANSWER | 点差 | 旧 CI：只用 seed 17 | 修正 CI：三个 seed、按病例配对 |
|---|---|---:|---:|---:|
| Q-guided Distillation | Accuracy | +0.1074 | `[0.0000, 0.3222]` | `[-0.0185, 0.2556]` |
| Q-guided Distillation | Utility | +0.0906 | `[-0.0142, 0.2967]` | `[-0.0325, 0.2357]` |
| Relative Advantage MLP | Accuracy | +0.0926 | `[0.0222, 0.3222]` | `[-0.0667, 0.2593]` |
| Relative Advantage MLP | Utility | +0.0686 | `[0.0014, 0.2948]` | `[-0.0888, 0.2330]` |

校正结果显示两个方法的 Accuracy 和 Utility 区间都跨 0。相对 MLP 原先“Utility 区间严格为正”的说法必须撤回。

另外逐一比较了 Always ANSWER、Always ASK、Always DEFER 三种固定策略。附加 CSV [`results_20261121/router_fixed_baseline_ci.csv`](results_20261121/router_fixed_baseline_ci.csv) 给出 12 个三 seed paired comparisons，全部在相同 15 个 case 上做 5,000 次病例聚类重采样。Q-guided/Relative 对三种固定策略的 Accuracy 和 Utility CI 均包含 0；最佳固定策略的比较也没有显示可靠优势。

## 其他历史统计的抽样单位

- **Window B**：每个病例先聚合其所有状态，再按病例等权估计点值；10,000 次 percentile bootstrap 对病例采样，状态保持在病例簇内。Oracle 与固定动作比较使用配对的病例结果。脚本 `window_b_diagnostics/analyze_window_b.py` 的 `case_means()`、`bootstrap_ci()` 与 `summarize_cost_grid()` 实现符合这一单位。本轮用独立聚合脚本重算保存表，96 行精确一致。
- **V3 Oracle Expert**：`oracle_expert/scripts/evaluate_oracle_router.py:245-268,303-334,607-614` 以每个 `case_id` 为一个 vector 元素，固定策略和 learned policy 在相同 case 顺序形成差值向量，再对差值向量做 percentile bootstrap。Learned 多 seed 先在病例内等权平均。四格表重采样时整病例的状态同时进入分子、分母（`:662-688`）。方法正确，但只有 15 个病例，区间仍很不稳定，且是开发集。
- **病例内多个状态**：V2 Router 的 43 个状态来自 15 病例，Window B 的 146 状态来自 50 病例。不能把状态当作独立 bootstrap 单元。

## 95% CI 边界与解释

实现使用经验分布的 2.5%/97.5% percentile quantile，不对区间做 `[0,1]` 截断。Accuracy、Utility 差是有符号差，区间允许跨过 0；成本可使 Utility/差值小于 0。Action rate 本身落在 `[0,1]`。在这一问题上边界处理正确。病例数仅 15 或 50，区间覆盖率不应被视为精确的临床不确定性界限。

这些 Router bootstrap 条件于固定的 3 个已训练 checkpoint，并对病例簇重采样；它量化的是此病例样本上的配对波动，不是充分估计训练过程随机性。三 seed 的样本标准差另行报告。
