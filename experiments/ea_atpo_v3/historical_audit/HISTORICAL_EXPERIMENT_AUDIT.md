# EA-ATPO V2/V3 历史实验独立审计

日期：2026-11-21
审计分支：`codex/historical-audit-20261121`
本审计 worktree 初始 HEAD：`93c8b75a497b6a880953e39d0bee5ba134eeb557`

## 范围与限制

本轮只读核查已经完成的 V2/V3 实验文件、聚合预测、配置、代码和 checkpoint 清单；使用 CPU 对保存的输出进行复算。没有训练模型、调用 GPU/模型服务、读取 400 例 Frozen Holdout、修改历史实验输出或进入 C 会话的 worktree。没有复制病例原文或 Gold 到本目录。

V2 Router 主运行输入来自代码提交 `ba698a8d8176ae2dfe34e8f654adc12ca6208530`。本报告中的开发集和 Oracle 结果来自同一批已用于方法选择的病例，不能当成新的泛化证据。C 会话负责的 Gold 来源核验和新独立病例评估不在本审计范围内。

## 主要结论

1. **V2 的数据联结、成本标签和三个 seed 的点估计能够复算。** 35/15 病例划分无 case ID 交集；146 个状态各有 ASK、ANSWER、DEFER 三条有效轨迹；438 条回报均与成本分量一致。历史三 seed 均值和样本标准差与独立计算最大差低于 `1.2e-16`。
2. **原 Router CI 有真实实现错误。** `zip(3 个学习策略 seed, 1 份共享基线)` 静默只使用 seed 17。旧区间可被精确重放，但不能代表三 seed 结果。修正后的 Q-guided 和 Relative Advantage Utility CI 都包含 0。
3. **Dev 点估计偏正，但证据不足以证明 Router 优于固定策略。** 15 个 Dev 病例、43 个状态上，Q-guided Utility 为 `0.491`，Always ANSWER/ASK/DEFER 分别为 `0.400/0.434/0.428`；对三个固定策略的配对病例 bootstrap 区间均跨 0。该 Dev 同时参与 early stopping 和方法选择。
4. **没有发现直接病例级泄漏。** 输入特征边界是决策前公开状态；TF-IDF 仅用 Train 拟合；embedding 冻结并来自公开状态。审计未发现 case ID 交叉、状态键重复、精确文本状态跨 split 重复，或将 Gold、隐藏事实、未来回复、动作结果写入特征文件的路径。此结论不等于证明相似病例/来源题目没有语义重复。
5. **V2 50 例 Oracle 是训练队列上的 hindsight 诊断。** 在 ASK/DEFER 成本 `0.01/0.05` 下，事后逐状态 Oracle Accuracy/Utility 为 `0.597/0.589`，高于 Always ANSWER 的 `0.390/0.390`；它根据已观测动作结果选动作，不能作为可部署 Router 成绩。
6. **V3 Oracle 报告仍是开发集重分析。** 它复用同一 15 Dev 病例/43 状态。Oracle Conditional 100% 是根据 Gold 定义出来的上界；Learned Router + Oracle safety gate 使用事后 Base 正误，不能部署。它没有新增独立测试证据。

## 历史实验分类

| 实验 | 审计结论 | 可用于论文的表述 |
|---|---|---|
| 50 病例 V2 反事实数据（146 状态、438 轨迹） | 输入和成本标签完整、配对有效；每个 state-action 仅 K=1；全属官方 Train 分区 | 可作为数据构造/工程 pilot 与探索性分析，明确非独立测试且标签是单次实现结果 |
| Window B 固定策略与 cost grid | 指标由原始正确性和动作计数生成，CPU 复算 96 行与保存结果完全一致；Oracle 用结果选动作 | 可报告训练队列 hindsight 上界和成本敏感描述，不能代表部署策略 |
| V2 Router Train/Dev | 35/15 case 隔离成立；三 seed 点估计复算；Dev 参与 early stopping 和选择 | 可报告开发结果、seed 波动、修正后的探索性 CI；不得写成测试集优势 |
| Q-guided Distillation / Relative Advantage MLP checkpoints | 冻结清单中的 6 个 checkpoint 哈希/大小匹配；历史 30-checkpoint reload audit 通过 | 可说明推理产物可重载；无需把 checkpoint 当作疗效证据 |
| V3 Oracle Expert | 同一 15-case Dev；正确按 case 聚类 bootstrap；Oracle 条件使用答案键事后筛选 | 仅可用于解释理想上界与当前 Router 的差距，不能写成独立性能验证 |
| 确认集/端到端完整 Router | 历史审计显示剩余 20 例受到 V1 拟合污染；没有新的未触碰病例运行；没有冻结 Router 的真实多轮端到端对照 | 尚无可报告的独立泛化结果或完整序列策略效果 |

## 可以与不可以得出的结论

### 可以报告

- 在已审计的 15-case Dev 子集，三 seed Q-guided Router 的 case-macro Accuracy/Utility 点估计分别为 `0.507/0.491`；相对 Always ANSWER 的点差为 `+0.107/+0.091`。
- 采用正确的三 seed、case-cluster 配对 bootstrap 后，Accuracy 差 95% CI 为 `[-0.0185, 0.2556]`，Utility 差为 `[-0.0325, 0.2357]`。这两个区间都跨 0。
- Relative Advantage MLP 的点差为 `+0.093/+0.069`，正确区间分别为 `[-0.0667, 0.2593]` 和 `[-0.0888, 0.2330]`，同样跨 0。
- 冻结 Q-guided checkpoint 能在 CPU 上重现已保存预测，报告的最大绝对分数差小于 `7.5e-8`；历史对 30 个 checkpoint 的重载检查最大差小于 `4.2e-7`。
- 50-case 训练队列上的事后动作选择显示标签空间存在潜在协作收益，但当前训练数据和 K=1 标签不足以证明可学习、可泛化的路由规则。

### 不可以报告

- “Router 已在独立测试集上优于 Always ANSWER/ASK/DEFER”。现有 15 Dev 病例参与模型选择，且修正 CI 包含 0。
- “Oracle 实验验证了可部署专家路由”。Oracle Conditional 和 Oracle safety gate 都用了 Gold/事后正确性。
- “50-case Oracle 的 20.7 pp 增益是模型性能”。它是对真实已观测反事实结果做事后挑选的训练队列上界。
- “每个 Q 标签是可靠的期望回报”。438 条轨迹虽全部有效，但每个状态动作仅有一次 rollout。
- “完整 ASK → Router → 后续动作的序列策略已验证”。既有 learned policy 结果在配对反事实分支上选择动作；ASK 分支由固定 continuation policy 执行，不是 Router 再次决策的在线闭环。

## 论文写作建议

可以把这些结果放入方法开发/可行性小节，标题和图表需标为 **Train-cohort retrospective analysis** 或 **development-set result**。论文主结论应等待 C 的 Gold 来源核验及全新病例上的冻结策略评估。若 C 获得独立病例，可将本报告提供的 Always ANSWER、固定动作、冻结 Router 和成本公式作为预先锁定的历史参照；不要把已查看的 15 Dev 或 50-case Train Oracle 重新命名为独立测试。

## 复现和文件

- Router case split、标签、重复和三 seed 复算：[`scripts/audit_router_results.py`](scripts/audit_router_results.py)
- V2 Window B 成本和聚合重算：[`scripts/recompute_windowb_metrics.py`](scripts/recompute_windowb_metrics.py)
- Dev 与三个固定动作的配对区间：[`scripts/recompute_router_fixed_baseline_ci.py`](scripts/recompute_router_fixed_baseline_ci.py)
- 汇总表：[`PAPER_RESULT_TABLES.csv`](PAPER_RESULT_TABLES.csv)
- 机器可读输入/输出清单：[`REPRODUCIBILITY_MANIFEST.json`](REPRODUCIBILITY_MANIFEST.json)
- CPU 重算产物：[`results_20261121/`](results_20261121/)
