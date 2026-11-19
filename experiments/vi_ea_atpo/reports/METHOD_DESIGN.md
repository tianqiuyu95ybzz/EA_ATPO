# VI-EA-ATPO 方法设计（研究假设版）

**状态：设计冻结候选，不是已验证算法贡献。** Phase 2 干预尚未运行；VOI 可预测性、搜索增益和独立泛化均为 Not Run。

## 1. 目标与兼容边界

VI-EA-ATPO 研究在既有 EA-ATPO H-MDP 中，何时 ASK 患者或 CONSULT 专家可通过新增信息提高后续任务收益，以及这种收益能否在动作前预测。设计保留原动作语义：ANSWER 终局作答；ASK 由 Patient 用隐藏 atomic facts 回答后 Base 继续；CONSULT 由 Expert 给非终局建议后 Base 继续；DEFER 由 Expert 给终局答案。

第一版目标是“估计与搜索预算适配”，不是替换 Actor、Critic、动作集或 reward。原搜索以层为单位扩展活跃叶，使用 q 方差和 Critic value 与子节点均值 Q 的差异进行剪枝，不是当前文档中常见的 best-first priority queue。当前 Critic 是单值预测，没有 epistemic uncertainty 接口。以上实现位于 `ea_atpo_stage/recipe/ea_atpo/ea_agent_loop.py` 与 `ATPO-official/codes/recipe/atpo/user_assistant_agent_loop.py`，细节见 Phase 0 审计。

## 2. 符号与信息价值

- `s`：动作前公开 H-MDP 状态，包含公开病例、对话、动作计数和剩余预算。
- `a∈{ANSWER, ASK, CONSULT, DEFER}`：现有宏动作。
- `z`：ASK 返回的患者观察；`e`：CONSULT 返回的专家建议。
- `c_A,c_C,c_D`：按统一 reward 单位计的动作成本。
- `π_cont`：冻结的后续策略；必须在评估前定义，不能按单病例事后正确性选择。
- `U(τ)`：轨迹终局 correctness 减去按动作计数的成本。

终止动作参考值为：

```text
V_stop(s) = max(Q_answer(s), Q_defer(s))
```

若 `V` 代表在固定 continuation policy 下的期望净回报：

```text
Q_ask(s)     = -c_A + E_z[V(s_z) | s, ASK]
Q_consult(s) = -c_C + E_e[V(s_e) | s, CONSULT]

VOI_ask(s)     = Q_ask(s) - V_stop(s)
VOI_consult(s) = Q_consult(s) - V_stop(s)
```

期望必须包含未来 ANSWER/DEFER 选择以及之后发生的所有成本。仅熵下降、置信度上升或建议导致答案变化都不是 VOI 正值的充分条件。单个 rollout 的 `U(intervene) - U(stop)` 是**观测收益差**；多次配对 rollout 的均值是固定 continuation 下的经验条件收益；两者都不是精确的真实期望或最优 VOI。

Phase 2 的固定分支将分别测量：

- **Task Improvement**：直接 Base ANSWER 与 ASK/CONSULT 后强制 Base ANSWER 的配对正确性/Utility 差。
- **Decision Improvement**：信息后由 Base 在 ANSWER/DEFER 中做终止选择，并与 Direct ANSWER、Direct DEFER 和 train 上冻结的静态动作比较。

这两项是先导实验结果变量，不是动作前特征。动作前 `x(s)` 只含公开 prompt/state 特征和可取得的 Base/Actor 状态信号；当前 completion-only Pilot API 不提供 Base 选项 logprob，因此 Phase 2 把 confidence/entropy/margin 标为 unavailable，不补造值。Gold、Expert 结果、未来回答、Oracle 动作及 rollout 后标签不得进入 `x(s)`。

## 3. 估计器设计（只有 Phase 2 通过后才训练）

`VOIEstimator(x(s), budget, counts, cost)` 输出：

```text
estimated_VOI_ASK
estimated_VOI_CONSULT
estimated_VOI_uncertainty
```

候选对照按复杂度递增：常数均值、Base uncertainty heuristic、Ridge、Logistic/MLP。监督目标由 Phase 2 train 上的 paired rollout 构造：

```text
y_ask     = observed_return_ASK - train-fitted stop_value
y_consult = observed_return_CONSULT - train-fitted stop_value
```

这是给定 continuation policy 与模型版本下的 noisy empirical target，不是策略最优值标签。所有同病例 rollout 留在同一 split；标准化、校准、模型、风险惩罚与阈值只在 train/validation 拟合或选择。Validation 按 case_id 冻结；test 在本轮不会使用。

模型不确定性优先用 train-only case bootstrap 或多模型输出 spread；未经覆盖校准，不称为 confidence interval 或 LCB。可选风险评分 `VOI_hat - λ·uncertainty_hat` 只在 validation 选 λ，且需与点估计及常数策略等预算对照。

## 4. 候选树搜索融合（Phase 4 研究假设）

原搜索信号记作源码实际实现的 `S_ATPO(node)`，不把 Critic discrepancy 重新命名成 Bellman error。候选搜索调度分数为：

```text
S_VOI(node) = max(0, VOI_hat_ASK(node), VOI_hat_CONSULT(node))
S_new(node) = S_ATPO(node) + λ_voi · Normalize_train(S_VOI(node))
```

`Normalize_train` 统计仅由训练数据拟合；`λ_voi`、是否分开 ASK/CONSULT、risk penalty 仅由 validation 决定。`λ_voi=0` 必须严格回到原剪枝/展开顺序。VOI 不能删除 Actor 原本合法动作、改变动作概率或终止条件。

优先只调整每层搜索的展开预算/候选顺序，不同时修改 Reward、PPO loss、Critic loss、Actor 网络或动作集。若改变了 Actor 实际采样策略，要确认 PPO logprob 对应行为策略；当前源码保存生成 token logprob，但没有明确树剪枝 inclusion probability / importance correction。该差异必须在集成前解决或证明现有训练路径不受它影响。

## 5. 算法流程

```text
Offline pilot:
  freeze fresh case ids and one-step continuation policies
  for each case s:
    independently roll out ANSWER and DEFER from s
    roll out ASK -> patient observation -> frozen continuation
    roll out CONSULT -> expert advice -> frozen continuation
    score only after each branch terminates; save visible state, usage and costs
  select static baselines / cost thresholds on pilot train only
  evaluate paired task and decision improvement on pilot validation

If and only if information has positive, repeatable validation utility:
  fit train-only stop-value / VOI estimators from train branch outcomes
  choose model, calibration, risk penalty and decision threshold on validation
  if VOI predicts held-out pilot validation outcomes and improves utility:
    pass frozen estimator to a VOI search adapter
    adapter ranks/allocates existing search expansion budget
    Actor, Critic, reward, returns, advantages and PPO losses remain original
  otherwise stop before search integration
```

## 6. 伪代码

```python
def pilot_branch(state, intervention, continuation, gold_for_evaluator_only):
    if intervention == ASK:
        question = actor.ask(state)
        observation = patient.answer(question, hidden_atomic_facts)
        next_state = state.append_public(question, observation)
        action_cost = c_A
    elif intervention == CONSULT:
        focus = actor.consult(state)
        advice = expert.consult(public_state=state, focus=focus)
        next_state = state.append_public(focus, advice)
        action_cost = c_C

    terminal_action, answer, extra_cost = continuation(next_state)
    # Gold is first read here by evaluator, after actor/patient/expert calls.
    reward = int(answer == gold_for_evaluator_only) - action_cost - extra_cost
    return reward, terminal_action, next_state


def voi_expand(node, atpo_score, voi_estimator, train_normalizer, weight):
    if weight == 0:
        return original_atpo_expand_score(node)
    x = public_predecision_features(node.state)
    voi = voi_estimator.predict(x)
    signal = max(0.0, voi.ask, voi.consult)
    return atpo_score(node) + weight * train_normalizer(signal)
```

`hidden_atomic_facts` 仅传给 Patient；`gold_for_evaluator_only` 永不进入 actor/patient/expert request 或 `public_predecision_features`。

## 7. 复杂度与预算

对 `d` 维、`n` 个训练样本的 Ridge，直接求解约为 `O(nd²+d³)`，在线预测约 `O(d)`；Logistic 每轮 IRLS 量级相近，实际实现可用稳定的迭代 solver。对每个搜索节点评估两个动作 VOI，增加两次轻量 forward 和常数次动作比较；不应增加每个候选的新模型生成调用。Phase 2 仍需生成模拟 Patient/Expert 的新信息，因此成本主要来自 API 请求：当前 150 个病例计划最多 1,800 次 nominal generation（不含传输/格式重试）。

## 8. 失败模式与有效性威胁

- 患者模拟器只覆盖数据提供的 atomic facts，可能“无法回答”或反馈事实不准；信息获取表现依赖模拟器。
- Expert 建议可能错误、重复公开信息或诱发过度自信；信息量不代表可靠性。
- continuation policy 与真实部署策略不同会改变估计到的条件 VOI。
- 单病例单 rollout 方差高；少量成功样本无法支持精细分层、复杂 MLP 或可靠校准。
- ASK/CONSULT 成本单位与 token、延迟、GPU 资源不同；主 Utility 用归一化训练 reward，不表示实际货币或临床效用。
- 状态、提示词、服务模型 revision 改变会使训练 target shift；API alias 不是模型版本证明。
- 新病例仍来自 ATPO train 源，虽然按历史身份排除，但不等于临床人群或官方独立 test。
- 改变树剪枝留下的训练轨迹分布可能造成 PPO 行为策略不一致；树预算相同也不自动消除该偏差。

## 9. 当前验证状态

Phase 0/1 已完成；Phase 2 队列、运行器和 CPU/mock 单测已完成，但模型 rollout Not Run。因 Phase 2 净收益与 Phase 3 可预测性没有结果，本设计不接入搜索、不训练 VOI、不启动 RL。所有融合公式、风险策略和效用改善均为研究假设。
