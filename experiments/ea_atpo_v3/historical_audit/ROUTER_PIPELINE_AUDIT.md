# Router 数据、特征、checkpoint 与复现审计

## 运行与病例划分

主数据为 V2 50 个官方 Train 病例：固定 `case_id` 划分 35 Train / 15 Dev；103 / 43 个公开状态。审计确认 Train/Dev case ID 交集为 0，同病例状态、三动作轨迹和 embedding 保持在同一 split。统计到：146 个 Q 状态、146 个公开状态、146 条 embedding、438 条有效动作轨迹（每状态 ASK/ANSWER/DEFER 各一条）。重复 Q key、状态-动作键、同病例精确重复公开状态、跨 Train/Dev 精确公开文本重复均为 0。

预测文件覆盖 43 个 Dev 状态 × 2 候选族 × 3 seeds，共 258 行；预测键重复为 0。所有候选和固定策略使用同一组 43 个 Dev 状态及同一份反事实轨迹。训练分区仅有 35 个病例，状态数 103；三 seed 不是新增独立病例。

此审计检测精确 ID 和内容哈希重复，没有根据原始病案来源建立更高层的病人身份链接；因此不能保证语义近似病例或镜像来源病例完全不存在。

## 输入特征泄漏检查

1. `prepare_costed_dataset.py:86-94` 要求 metadata 声明 `feature_boundary=public_state_before_action_only`，且 `gold_or_future_fields_in_features=false`；输出 `public_states_only.jsonl` 仅含 case/state key 和公开对话（`:144-183`）。
2. 构造训练目标的 Q 文件保留 `valid_mask` 与反事实标签。特征生成代码另取公开状态文本，标签只进入损失目标，不拼接到 Router 输入。
3. `train_evaluate.py:61-78` 中，文本 TF-IDF 的词表/统计只在 35 个 Train case 拟合；Dev 仅 transform。冻结 embedding 使用 Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`、4096 维 mean-last-hidden-state，输入是生成动作前的公开状态。没有为 Router 微调 Base。
4. 历史 manifest 和预处理代码未发现 Gold、hidden atomic facts、未来患者回复、Expert 输出、最终正确性或 rollout 结果作为决策特征的路径。它们只用作离线 Q 回归监督/评价目标。
5. 这只表明被审计的数据处理边界和代码实现没有直接标签泄漏，不证明公开状态本身不含历史系统曾公开的答案内容，也不等于临床意义上的来源去重。

## 目标和策略一致性

- 三个动作顺序固定为 ASK / ANSWER / DEFER；标签按 `final_accuracy − .01×ASK_count − .05×DEFER_count` 重建。审计确认 Q 与回报匹配，无重复成本。
- 策略指标在 `train_evaluate.py:220-253` 中先计算病例内 Accuracy、Utility 和动作率，再对病例等权平均；同一状态的配对分支按固定动作索引取结果。
- ASK 的 rollout continuation 固定为 `ASK_ONCE_THEN_FORCE_BASE_ANSWER_PATIENT_REVIEW_V2`。所以 ASK 标签对应这一固定的一次问诊过程，不代表 Router 在患者回复后再次自由选择的多轮策略。
- Q-guided Distillation 根据配对 Q 产生软标签并学习三类 action logits；Relative Advantage MLP 学习 ASK-ANSWER、DEFER-ANSWER 并把 ANSWER 设为零锚点，推理选最大优势。缺失标签以 `valid_mask` 排除，目标定义见 `train_evaluate.py:95-105`。旧 bootstrap 的错位点位于 `:256-267`；原训练分支仍有 `zip()` seed 截断。
- 保存的 Router 结果是离线反事实 action selection，不是由 Router 驱动的新病例端到端 ASK → Patient → Router 闭环评测。
- 没有可用的 Base token-level confidence/entropy，因此 uncertainty baseline 不具备同口径特征，不能从模型答对与否反向构造该特征。

## Checkpoint、选择和运行可复现性

冻结候选配置 `confirmation_20261120/configs/frozen_candidate_20261120.json` 锁定：Qwen3-8B embedding/tokenizer revision、模板 SHA、非 thinking 模式、4096 维、截断/池化、动作顺序、成本、三 seeds、16 hidden units 和模型选择规则。Q-guided 三个 checkpoint 和 Relative Advantage 三个 checkpoint 的文件大小及 SHA-256 全部与冻结配置相符。相应 hashes 与配置见该 manifest；本审计没有加载或修改 checkpoint。

历史 checkpoint reload audit 记录 30 个 checkpoint 均通过，43 Dev states 上最大预测差小于 `4.18e-7`。先前 V3 CPU replay 对三个 Q-guided checkpoint 的保存 score 重放差小于 `7.5e-8`。据此，历史结果可由已保存 inference checkpoint 和特征缓存复现；它们是 inference checkpoint，不应被假设包含 optimizer/scheduler/RNG 状态或可用于无缝续训。

模型族选择和 early stopping 都使用这 15 个 Dev case：Q-guided 按平均 Dev Utility 选为 primary；Relative Advantage 作为比较候选；各 seed 以 Dev loss early stop。此后在同一 Dev 上反复比较的点值有模型选择偏差，即使 bootstrap 修正也不能把 Dev 变成 untouched test。

## 风险判断

| 风险 | 判断 |
|---|---|
| Train/Dev 同病例泄漏 | 未发现；审计按 case ID 和状态/embedding 键重算 |
| Gold/未来信息进入 Router 特征 | 当前处理管线未发现；预处理有公开状态边界检查 |
| Q target 重复扣成本 | 未发现；raw cost 与 Q/return 核对一致 |
| bootstrap 少算两个 seed | 已确认；训练分支的旧函数仍含问题，独立审计分支的修订未合入 |
| Dev checkpoint/方法选择污染 | 确认存在；15 Dev 参与 early stopping 和主模型选择 |
| 语义重复病例 | 未充分核验；精确状态 hash 不能排除相似病例或上游重复 |
| 在线多轮策略复现 | 未完成；现有结果是反事实分支选择而非 Router 闭环 rollout |
| 独立泛化 | 未验证；冻结 Holdout 未打开，也没有新的干净病例集结果 |
