# EA-ATPO：创新点与实验计划

版本：2026-09-29 v1.0　｜　性质：**待验证的研究方案与持续进度记录**

## 0. 摘要与边界

研究目标：在官方 ATPO 多轮医疗问诊策略上，学习何时 `ANSWER`（自主回答）、`ASK`（询问患者）、`CONSULT`（请求专家建议后继续）或 `DEFER`（将最终判断交给专家）。目标不是增加人工规则，而是从可观察状态预测每种动作的长期结果与资源成本，在相同专家预算下提高诊断表现。ACT 只作为 ASK/ANSWER 对照或动作学习参考，**不是本文的主体算法**。

本方案研究模拟医疗选择题与模拟患者/LLM 专家，不代表真实医生试验或临床部署。本文把“已完成”“拟实施”“通过条件”分开，不能把 oracle 分析、离线 router 或设计公式写成已经取得的论文结果。

主方法采用两阶段：

1. 在 ATPO 内验证并改进问诊过程的训练信号，得到基础策略 `π_base`。
2. 冻结 `π_base`，学习执行前的四动作序贯价值与专家接入策略。

四动作联合 ATPO RL 是条件性增强，不是主方法成立的前提。这样保留旧版 DP-ATPO 的稳定性，并保留真正的 CONSULT/DEFER 区别。

## 1. 与已有工作的关系：创新究竟在哪里

- 官方 ATPO：医疗对话 H-MDP、自适应树展开、树回传、turn-level advantage、PPO-style actor 与 critic MSE；原始动作主要是提问或终局回答。论文：https://arxiv.org/abs/2603.02216 。
- ACT：研究 clarification/answer 的多轮动作学习；在本文只作相关对照。论文：https://arxiv.org/abs/2406.00222 。
- SLTD：启发固定目标策略后学习序贯转交，并重视“现在转交”与“延迟转交”的长期区别；本文不声称复现其模型式 RL。论文：https://arxiv.org/abs/2109.06312 。

**候选创新点 1：事实获得驱动的 ATPO 信用分配。** 官方论文的主要训练奖励来自终局答案；当前复现代码虽计算提问格式/校验/有效性分数，却在树搜索非终止节点把 `action_reward` 置零。先动态审计真实执行路径，再比较接通已有过程分数与新增“实际获得此前未知事实”的奖励。若只靠接通现有过程分数即可获得全部收益，不把新增奖励包装成独立创新。

**候选创新点 2：执行前、序贯、成本敏感的协作价值学习。** 将问患者、问专家和完全转交定义为不同转移与工作量；从当前公开状态预测动作的长期 outcome 与后续成本，门控可在每轮重新决策，并显式考虑剩余预算。专家提示尚未出现时就要作 CONSULT 决定，因此不能用真实提示后的 `V(s⊕h)` 作推理时输入。

**候选创新点 3：严格的转交评估协议。** 在独立病例测试集、相同专家预算下比较准确率、总效用、专家负担和问诊质量；报告泄漏、专家幻觉、无效轨迹及负结果。单独的指标集合通常不足以构成算法创新，应作为方法可信度支撑。

## 2. 环境、信息边界与专家定义

公开状态 `s_t` 仅含题干/选项、已公开患者回答、已返回的专家建议、剩余轮次与预算、既往公开动作。正确选项、隐藏 `atomic_facts`、未生成的专家建议、模型 `<think>` 私有推理都不能进入策略或专家输入。隐藏事实只允许患者模拟器与训练奖励判定器使用。

| 动作 | 环境行为 | 是否终止 | 成本来源 |
|---|---|---:|---|
| ANSWER | 基础 agent 现在给诊断选项 | 是 | 模型推理 |
| ASK | agent 问患者；患者只据 hidden atomic facts 回答 | 否 | 患者负担、轮次、tokens |
| CONSULT | 独立专家看公开状态和具体问题，给有限建议；agent 继续 | 否 | 专家输入/输出、延迟、工作量 |
| DEFER | 独立专家看公开状态并给终局答案 | 是 | 完整专家工作量 |

主实验每例最多一次 CONSULT，用 `b_t∈{0,1}` 表示剩余咨询机会；DEFER 仍可在后续任何轮发生，但价格更高。若另设“总专家预算”，须单独定义状态与动作掩码，不与 `b_t` 混淆。CONSULT 不得直接给最终选项；DEFER 必须真正终止，不能让 agent 重猜。这里的专家为 Qwen3.5-35B-A3B **模拟专家**，不是人类医生。

## 3. E0：先确认官方奖励是否进入优化（当前下一步）

代码重点：`codes-repro/recipe/atpo/mt_reward_fn.py`、`mt_reward_manager.py`、`user_assistant_agent_loop.py`、`mt_trainer.py`。静态检查发现前者计算 process_reward，而 `user_assistant_agent_loop.py::_evaluate_nodes()` 对非终止节点实际使用 `node.action_reward = 0.0`，同时树备份与 advantage 使用该字段。这是**待动态验证的断点**，尚不能仅凭一处代码推断所有训练路径。

执行：固定一份真实训练配置与随机种子，构造“新增事实”“重复/无效提问”“直接作答”三条受控轨迹；逐步记录 `process_reward → node.action_reward → tree target → critic target → advantage → actor update`，与手算值对照。另保存数据划分/哈希、checkpoint、评估参数。E0 通过前不启动 E1 正式训练，不修改未动过的官方代码副本。

判定：若过程分数已通过其他路径进入优化，则修正创新点 1；若确实没有进入，进入 E1 的四组消融。

## 4. E1：Fact-aware Process-guided ATPO

维持 ATPO 的 actor、critic、树展开与 PPO-style actor loss；改变环境在提问后的即时奖励，并确认它参与节点 Q、树回传和优势。候选奖励：

`r_ask = w_new·clip(新增事实数) − w_rep·重复事实 − w_invalid·无效/无法回答 − w_turn·轮次成本`。

“新增事实”必须来自患者**实际回复**，且先前尚未公开；不能因为隐藏事实列表中存在某事实便奖励 agent。先用可审计的 fact ID 匹配，抽检语义错误；每轮奖励限幅，终局正确性仍占主导。权重只在训练/验证病例上选择。终局奖励量纲按本次复现的实际配置核对，不直接照抄论文中的 +3/0/−1。

ATPO 的数学骨架不变：

`Q_t = r_t + γ Vψ(s_{t+1})`，`A_t = r_t + γ Vψ(s_{t+1}) − Vψ(s_t)`；critic 拟合树回传值，actor 使用原 PPO-style clipped objective。**新增的是可追踪的过程信用信号，而非新的 PPO 公式。**

| 组别 | 原有过程分数入树 | 新事实信号 | 用途 |
|---|---:|---:|---|
| A 官方 ATPO | 否/按原实现 | 否 | 基线 |
| B Process-connected | 是 | 否 | 区分代码断点收益 |
| C Fact-only | 否 | 是 | 检验事实信号 |
| D Full E1 | 是 | 是 | 检验互补性 |

同一模型初始化、训练数据、步数、树预算与评估设置。先单元测试和小规模 RL smoke，再正式训练。主指标准确率与病例级置信区间；辅以平均轮数、有效问诊、新事实获取、重复/无效提问和 GPU 成本。若只是多问而准确率下降，则 E1 不通过。

## 5. E2：冻结 agent 的四动作序贯 learn-to-defer

冻结 E1 的 `π_base`。公开状态编码器首选冻结基础模型隐藏表征，后接共享轻量网络与四个动作价值头；同时估计 outcome 与整条后续轨迹的成本。区别于当前的冻结表示+Ridge 先导实验，正式模型必须在独立病例上证明收益，不能仅展示训练 loss 下降。

定义 `Q_out(s,a,b)` 为动作后遵循当前门控策略的预期终局正确性；`Q_cost(s,a,b)` 为该动作及未来动作的预期累计成本。决策仅在行动前使用可见信息：

`π_gate(s,b) = argmax_{a∈合法动作} [Q̂_out(s,a,b) − λ_c Q̂_cost(s,a,b)]`。

ANSWER/DEFER 是终止动作；ASK/CONSULT 后进入新状态，分别保留/消耗 CONSULT 机会。`Q_out(ASK)` 与 `Q_out(CONSULT)` 必须考虑**后续仍可作门控决策**；否则训练成“一次干预后始终由 base 完成”、测试却每轮重新决策，会发生训练目标错位。E1 的事实塑形奖励不混进 E2 的临床 outcome，资源成本也只扣一次。目前没有可信的临床 harm 等级标注，不把 harm 项假装成已实现损失。

训练流程：病例级 train/val/internal-test 划分；从 `π_base` 真实访问状态抽样，初始/后续状态分层；同状态对四动作做配对随机 rollout，保存 outcome、调用成本、完整 trace 和失败类型。先以 base-only 后续轨迹训练初始价值模型；随后在训练病例上用该门控继续 rollout、更新一次价值目标，验证是否改善。测试病例不参与此迭代。价值损失首选带可靠性权重的 Huber 回归：

`L_value = Σ_(s,a) w_(s,a)·Huber(Q̂_out(s,a,b), G_out(s,a,b)) + λ_costhead·Huber(Q̂_cost(s,a,b), G_cost(s,a,b))`。

辅助 pairwise ranking loss 只在验证显示净收益排序不足、且配对差异可靠时再加入；不把 3 次 rollout 的事后最佳动作当成无噪声分类标签。成本权重/阈值由验证集预注册选择，绝不在测试集调到“刚好有提升”。报告价值误差、动作排序、校准、选择率及不同成本敏感性。

旧版二元 `CONSULT vs 不咨询` gate 可作消融，但不能代表完整 DEFER。SLTD 风格的“现在 DEFER vs ASK 一轮后仍可 DEFER”作序贯性消融。若 DEFER 在公开信息下并不比 agent 强，应报告专家能力边界，不靠人为调低成本强迫 DEFER 胜出。

## 6. 可比实验、统计与停止条件

对照：官方 ATPO；E1 最优 ATPO；always ANSWER；always/随机 CONSULT；always DEFER；简单置信度规则；旧二元 gate；现有 Ridge 四头；正式 E2。相同基础模型、病例、患者/专家协议、最大轮数、专家预算与计价标准。专家抽检不少于 100 次，核查无 gold/atomic_facts/私有 think 泄漏、CONSULT 无最终答案、DEFER 独立回答、建议无明显幻觉。

主要终点：**同专家预算下的诊断准确率**与病例级成本调整效用。次要终点：CONSULT/DEFER rate、必要/可避免转交、平均轮数、有效/重复/无效问诊、新事实率、专家 tokens/延迟、无效动作和泄漏率、训练/推理成本。病例级 paired bootstrap CI，多随机种子；报告初始/后续状态和专家质量分层。不同成本点是完整曲线，不把最有利点单独当主结果。

通过条件：E1 在不降低诊断表现的前提下改善问诊，且与 B 组相比有可辨别的新增收益；E2 在预先指定的合理预算档位于独立测试病例超过 E1 base 和 always ANSWER，病例级 CI 支持正效益，专家协议审计通过。若不达标，先诊断标签方差、动作价值间隔、专家质量、状态表示与样本量，报告负结果并暂停 E3。

## 7. E3：联合 Expert-aware ATPO（条件性）

仅在 E1/E2 达标后，让 actor 在 ATPO 树中真正采样四种 macro-action，处理 ASK/CONSULT 的状态继续与 ANSWER/DEFER 的终止，四种动作的奖励进入树备份、critic 与 PPO-style actor 更新。与 E2 固定门控在同专家预算及算力下比较；若无额外收益，不宣称联合算法成功。E3 不是当前已运行实验。

## 8. 当前证据与实际状态（2026-09-29）

- 历史 Qwen3-4B SFT 与 ATPO RL baseline 已跑至 step 20；既有记录中测试准确率 43.45%→46.53%，平均轮数约 4.05→3.39，但该提升还不能被表述为已证实显著。该 4B 结果仅作为工程复现记录；正式主实验应使用官方配置的 Qwen3-8B Assistant policy。checkpoint：`checkpoints/official-atpo-baseline-corrected-gpu1/global_step_20/actor_merged_hf`。
- 修复 `<think>` 泄漏后，四动作采集得到 500 状态/250 病例/6000 条 rollout；485 状态四动作完整，有效率约 96.23%。按指定人工成本的胜出数 ANSWER 280 / ASK 94 / CONSULT 86 / DEFER 25。该胜出数不是学习策略结果。
- 冻结 Qwen3 隐状态的优势目标 router 在 98 个完整测试状态：效用 0.4869；always ANSWER 为 0.4966；差值 −0.0097，病例 bootstrap 95% CI [−0.0403, 0.0134]。TF-IDF 优势目标在当前成本下全部选 ANSWER。**尚无 learned deferral 正收益证据。**
- 数据：`outputs/ea_atpo/feasibility_thinkstrip_35b_500states_3rollouts_20260928.jsonl`；报告：`outputs/ea_atpo/router_adv_*_500states_20260929/report.json`。含泄漏的旧版 500 状态数据仅可用于管道调试。
- 当前运行架构：41 号机 GPU 4/5/6 分别承载基础模型、Qwen3-8B 患者和 Qwen3.5-35B-A3B 专家服务；GPU 7 之前空闲。该资源状态是上次检查的快照，执行前须重新核实。

## 9. 执行队列与进度记录模板

| 顺序 | 任务 | 交付物 | 进入下一步条件 | 状态 |
|---:|---|---|---|---|
| 0 | E0 奖励路径动态审计 | 手算对照、测试、配置快照 | 确认真实训练路径 | 已通过受控节点测试；配置清单待固化 |
| 1 | E1 最小补丁与 smoke | 4 组奖励日志、梯度与 checkpoint | 无泄漏、数值稳定、可加载 | B 组单步 RL smoke 通过；C/D 尚未实现 |
| 2 | E1 正式同预算消融 | 病例级指标与 CI | 过程改善且诊断不退化 | 待执行 |
| 3 | E2 标签/专家质量审计 | 方差、价值间隔、100 条抽检 | 可学习、专家协议可信 | 待执行 |
| 4 | E2 序贯价值训练与评估 | 预算曲线、校准、测试报告 | 独立测试正收益 | 待执行 |
| 5 | E3 联合 RL | 仅条件性启动 | E1/E2 均达标 | 未授权启动 |

每次进度追加：日期时间、执行者/主机、代码差异或提交、配置与随机种子、病例划分/哈希、命令、日志/结果/checkpoint 路径、成功或失败、偏差原因、下一步。不得覆写旧结论；`sources/` 中的本地镜像文件只读。

### 2026-09-29 记录

完成 ATPO/ACT/SLTD 定位复核、当前代码静态审计、清洁反事实数据和负结果报告复核；形成两阶段可实现候选方案。下一步执行 E0，暂不启动新 GPU 训练。

### 2026-09-29 E0/E1-B 追加记录

- 在 41 号机 `codes-repro/recipe/ea_atpo/test_atpo_reward_path.py` 新增受控节点测试。按实际 baseline 的 `correctness_score=3`：有效问诊计算到 `process_reward=1.0`，重复问诊为 `0.1`，但原实现的两个非终止节点 `action_reward=0.0`；正确终局答案 `action_reward=3.0`。树 Q、备份和 turn advantage 的手算与运行值一致。
- 仅在工作副本 `codes-repro/recipe/atpo/user_assistant_agent_loop.py` 加入默认关闭的 `enable_process_reward` 开关。打开后有效问诊的节点奖励由 0 变 1.0，示例 Q 由 0.4 变 1.4、advantage 由 0.2 变 1.2；默认关闭时保留 baseline 行为。原文件备份由 patch 工具保留为 `.orig`。
- `python -m py_compile` 通过；奖励路径与协议测试合计 17/17 通过。此时**还没有完成 RL smoke 或正式训练**。患者端口 30002 可用，GPU 7 为空闲快照。
- 下一步：用 GPU 7 做 E1-B 单步 RL smoke，核对真实训练 batch 中的奖励/优势、actor/critic 更新与 checkpoint；如失败先修工程问题。不得把受控单元测试当成端到端训练结论。
- 补全 `codes-repro/recipe/atpo/agent_loop.py` 的配置透传，使 `tree_search.enable_process_reward` 从训练配置传至树运行器；两处开关默认关闭，官方 baseline 行为不变。奖励路径与协议测试重跑 17/17 通过。
- 首次单步 smoke 在 Ray 初始化时因 `/tmp` 临时目录路径过长触发 AF_UNIX 107 字节限制，尚未进入模型训练；保留 `logs/e1-b-process-connected-smoke-gpu7-20260929.log`。缩短运行专用临时目录后，于 41 号机 GPU 7 重新启动 PID 117489；日志 `logs/e1-b-process-connected-smoke-gpu7-20260929-v2.log`，运行脚本 `run_e1_b_process_reward_smoke_gpu7.sh`。截至本次记录已进入数据加载，**尚未判定 smoke 通过**。

### 2026-09-29 E1-B 单步 RL smoke 完成

- 重跑进程 PID 117489 正常退出；一个训练 step 耗时约 4 分 45 秒。2 个病例生成 4 条训练轨迹，格式有效率 100%。这 4 条恰好均答对，样本太小，**不能解释为准确率提升**。
- 训练器日志确认 `Found pre-computed advantages in batch`；`actor/pg_loss=-1.4928`、`critic/vf_loss=6.9475`，actor/critic 均有梯度并完成更新。`critic/returns/max=4.0` 高于终局正确奖励 3.0，支持过程奖励进入树回传；日志中的 `critic/rewards/mean=3.0` 仍是 outcome-only 字段，不能据此误判中间奖励未入树。
- actor 和 critic 模型、优化器及额外状态均保存于 `checkpoints/e1-b-process-connected-smoke-gpu7/global_step_1/`；4 条 rollout 位于 `outputs/ea_atpo/e1-b-process-connected-smoke-gpu7/rollout/1.jsonl`。运行后 GPU 7 不再有本次训练进程。
- 该小样本 `effective_rate=0.25`、`human_reject_rate=0.75`，提示“接通原有有效性分数”可能不足以鼓励真正有信息价值的提问；下一步先检查 atomic_facts 与患者回复如何可靠对齐，再实现 C/D 的新事实奖励。正式比较需同预算、多种子，不能使用这次 smoke 作方法效果结论。

### 2026-09-29 B 组风险复核与后续约束

- 直接抽查 smoke 轨迹：患者对发热/脱水和体检问题回答 `The patient cannot answer this question.`，agent 却在后续推理中把“无法回答”当作“无发热”及“可能严重脱水”的证据。这是具体的**未知信息被误当作阴性/阳性证据**的失败模式；终局选项恰好正确不能掩盖它。
- 新增受控测试确认：格式正确、verifier 判为 `<Normal>`、但患者无法回答的提问仍获 `process_reward=0.5`（格式 0.1 + verifier 0.4）。B 组将其接入树后也给节点 `action_reward=0.5`。奖励路径测试 5/5 通过。故 **B 组只作为机制消融，不直接扩大为正式主方法训练**。
- C/D 设计必须先建立可靠的“实际患者回复→新 fact ID”判定，并对无法回答设非正奖励/轮次成本。训练时可用 hidden atomic_facts 作环境评判，但 actor 输入和专家输入仍不许见到；任何语义匹配器先人工抽检。增加 `unknown-as-negative` 错误率作为问诊安全性分析指标。
- 对**清洁 500 状态数据的强制 ASK 分支**做只读复核：1500 条首次患者回复中 1084 条为 `cannot answer`（72.3%）；初始状态 486/750（64.8%），较晚状态 598/750（79.7%），另有 1 条缺失回复。该比例是强制 ASK 反事实分支，**不能直接当作自然 ATPO 策略的无效提问率**，但解释了当前 ASK 标签噪声与“直接回答”塌缩风险。进入 C/D 前需按病例/信息状态抽检：是模型提问不对、atomic_facts 覆盖不足，还是患者模拟器过度拒答。
