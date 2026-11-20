# EA-ATPO V2 Router 结果独立审计

日期：2026-11-20<br>
审计分支：`codex/router-results-audit-20261120`<br>
审计输入代码提交：`ba698a8d8176ae2dfe34e8f654adc12ca6208530`

## 范围与边界

本次只复算既有 50 病例 Router 实验的 Train/Dev 划分、状态重复、三个固定种子汇总、动作率、效用和病例级配对 bootstrap 区间。没有训练模型、调用 GPU、读取 Frozen Holdout、加载或修改 checkpoint，也没有覆盖历史结果。审计脚本只输出汇总统计、输入 SHA256 和方法级指标，不输出病例 ID、病例文本或 Gold。

开发集只有 15 个病例，而且已用于 early stopping 和候选方法选择。因此下列 Dev 指标是开发结果，不能解释为独立泛化证据。

## 数据与病例隔离

| 检查 | 结果 |
|---|---:|
| Train / Dev 病例 | 35 / 15 |
| Train / Dev 状态 | 103 / 43 |
| Train 与 Dev 的 case_id 交集 | 0 |
| Q 状态 / 公开状态 / embedding 行 | 146 / 146 / 146 |
| 反事实轨迹 | 438（ASK、ANSWER、DEFER 各 146） |
| 无效轨迹 / Q 标签与回报不一致 | 0 / 0 |
| 重复 Q 状态键 / 重复轨迹状态-动作键 | 0 / 0 |
| 同病例重复公开状态 / 跨 Train-Dev 相同公开状态 | 0 / 0 |
| 两种候选模型预测重复键 | 0 |

病例划分映射明确声明以 `case_id` 为单位；状态、动作轨迹、公开状态和 embedding 的键均与该划分一致。精确公开状态内容哈希也没有发现 Train/Dev 跨分区重复。这个检查能发现完全相同的状态，不能证明语义近似病例绝无重复；病例来源标识审计仍是独立于本次精确去重检查的工作。

动作成本配置为 ANSWER `0.00`、ASK `0.01`、DEFER `0.05`；Utility 按正确性扣对应动作成本计算。逐状态复算与 Q 标签的差异为 0，未发现再次扣费。

## 指标重算

主表按以下顺序计算：每例先平均该病例所有公开状态，再对病例等权平均；三个固定种子（17、42、2026）的指标再等权平均。调用率因此是“病例等权平均的病例内动作率”，不是直接把 43 个状态合并后计算的状态比例。表中同时给出状态加权统计，避免二者混称。

### 病例等权：三种子平均

| Router | Accuracy | Utility | ASK 率 | ANSWER 率 | DEFER 率 |
|---|---:|---:|---:|---:|---:|
| Always ANSWER | 0.4000 | 0.4000 | 0.0000 | 1.0000 | 0.0000 |
| Embedding Q-guided Distillation | 0.5074 | 0.4906 | 0.4741 | 0.2852 | 0.2407 |
| Embedding Relative Advantage MLP | 0.4926 | 0.4686 | 0.3481 | 0.2407 | 0.4111 |

三个 seed 的 Accuracy / Utility 样本标准差分别为：Q-guided `0.0390 / 0.0388`，Relative Advantage `0.0723 / 0.0728`。Relative Advantage 的 seed 间波动更大。

### 按状态合并：三种子状态比例的平均

| Router | Accuracy | Utility | ASK 率 | ANSWER 率 | DEFER 率 |
|---|---:|---:|---:|---:|---:|
| Embedding Q-guided Distillation | 0.5039 | 0.4875 | 0.4729 | 0.2946 | 0.2326 |
| Embedding Relative Advantage MLP | 0.4884 | 0.4647 | 0.3566 | 0.2403 | 0.4031 |

旧汇总中的 `ask_mean` / `defer_mean` 是病例等权的病例内动作率均值，不是 pooled state fraction。旧点估计和三个 seed 的汇总复算一致：最大差异为 `1.11e-16`；保存的 seed 样本标准差复算最大差异为 `7.63e-17`。因此 seed 汇总方式本身正确，需更正的是动作率的文字标签/解释。

开发集的固定动作参考：Always ANSWER Accuracy/Utility = 0.4000/0.4000；Always ASK = 0.4444/0.4344；Always DEFER = 0.4778/0.4278。Q-guided 的开发集点估计高于这些固定策略，但差异区间不支持稳定优势结论。

## Bootstrap CI 修正

原 `cluster_bootstrap_delta` 使用 `zip(policy_cases_by_seed, baseline_cases_by_seed)`。学习策略有 3 个 seed 的病例级结果，而共享的 Always ANSWER 基线只有 1 组；`zip` 静默只配对 seed 17，漏掉 seed 42 和 2026。对旧代码进行相同种子、相同 5,000 次病例重采样后，复现出的区间与原保存值完全一致，确认是 seed 截断，而不是新的随机误差。

修正后将共享基线显式重复为 3 组，并对 15 个 `case_id` 有放回抽样。每次抽样会完整保留病例内所有状态及三个固定 seed 的预测，再对病例和 seed 等权聚合。配置为 NumPy `default_rng` / PCG64、seed `20261120`、5,000 次、配对 percentile 95% CI。

| 方法 | 与 Always ANSWER 的差值 | 修正前 CI（只用 seed 17） | 修正后 CI（三个 seed，病例聚类） |
|---|---|---:|---:|
| Q-guided | ΔAccuracy = +0.1074 | [0.0000, 0.3222] | [-0.0185, 0.2556] |
| Q-guided | ΔUtility = +0.0906 | [-0.0142, 0.2967] | [-0.0325, 0.2357] |
| Relative Advantage MLP | ΔAccuracy = +0.0926 | [0.0222, 0.3222] | [-0.0667, 0.2593] |
| Relative Advantage MLP | ΔUtility = +0.0686 | [0.0014, 0.2948] | [-0.0888, 0.2330] |

四个修正后区间都包含 0。旧 Relative Advantage 区间看似严格为正的结果，是只纳入 seed 17 的产物。修正区间仍然只是经模型选择的开发集不确定性描述，不是独立测试推断。

## Checkpoint 与配置完整性

- 冻结候选文件中的 6 个 checkpoint（Q-guided 三种子、Relative Advantage 三种子）均与冻结配置记录的文件大小和 SHA256 相符。
- 候选选择规则、特征和推理配置均由 `frozen_candidate_20261120.json` 冻结；主候选为 Embedding Q-guided Distillation，模型选择依据是三 seed 的平均 Dev Utility。Relative Advantage MLP 是固定比较候选。Embedding 来自 `Qwen/Qwen3-8B` revision `b968826d9c46dd6066d109eabc6255188de91218`，公开状态输入，维度 4096；动作顺序为 ASK / ANSWER / DEFER。
- 主训练配置 SHA256：`e391a2e853ba01fbe52ee959396bc336e16ba8ca8f383680b7a724962612c81d`；冻结候选配置 SHA256：`d52466ffbe2e05a2dc571035d37c10f08883cd6d178d6540b6d171312b9a9f05`。六个 checkpoint 的逐个大小和 SHA256 比对状态保存在 `results/router_result_audit.json`。
- 既有 checkpoint reload 审计状态为 `passed`，覆盖 30 个重载检查；记录的最大预测差为 `4.17e-7`。
- 本次没有加载、写入、复制或改变任何 checkpoint。

## 结论

1. Train/Dev 按 `case_id` 隔离，且没有发现精确重复状态或跨分区状态内容泄漏。
2. Q-guided 和 Relative Advantage 的原始三种子点估计均可复算，seed 平均及样本标准差的汇总正确。
3. 原病例级 bootstrap 代码因策略 seed 数为 3、共享 baseline 行数为 1，被 `zip` 静默截断为仅 seed 17；修复后所有 95% CI 均包含零。
4. Q-guided 在本 Dev 集上的点估计高于 Relative Advantage 和固定动作基线，但 15 例 Dev 且已参与候选选择，证据不足以声称可靠优势；Relative Advantage 的旧正向 CI 不应继续引用。
5. 本报告没有打开冻结 400 例 Holdout，也没有提供新的泛化证据。

## 可复现产物

- `audit_router_results.py`：读取源 worktree 的现有结果并重新校验；不会训练或调用 GPU。
- `results/router_result_audit.json`：汇总审计结果、SHA256、数据完整性、CI 前后对比。
- `results/bootstrap_ci_before_after.csv`：病例级 CI 修正前后表。
- `results/bootstrap_ci_case_clustered.csv`：修正后的病例聚类区间。
- `results/seed_metrics_case_vs_state_weighted.csv`：三个种子的病例等权和状态加权指标。
- `REPRODUCE.md`：复现命令与环境说明。
