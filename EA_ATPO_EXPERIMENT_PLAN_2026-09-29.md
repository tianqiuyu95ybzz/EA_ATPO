# EA-ATPO 实验计划与进度（2026-09-29）

## 方法定位与当前结论

目标是改进 **ATPO** 的多轮医疗问诊与适时专家转交；ACT 只作为 ASK/ANSWER 对照或动作预热参考，不作为本文的优化算法。当前 `codes-repro/recipe/ea_atpo/` 是独立的四动作反事实采集与离线 router 包，**尚未把 CONSULT/DEFER 接入 ATPO 树搜索、critic 和 actor 更新**。现阶段不能称为已完成 Expert-aware ATPO 算法。

已完成的官方 baseline：Qwen3-4B SFT 与 ATPO RL 至 step 20；既有记录显示测试准确率 SFT 43.45%、ATPO 46.53%，平均轮数约 4.05→3.39。此差值仍需按病例做置信区间，不能先写成显著提升。训练 checkpoint：`checkpoints/official-atpo-baseline-corrected-gpu1/global_step_20/actor_merged_hf`。

已完成的四动作可行性采集：修复公开状态中的 `<think>` 泄漏后，重新得到 500 状态、250 病例、6000 条轨迹、485 个四动作均有效状态，有效率 96.23%。结果：`outputs/ea_atpo/feasibility_thinkstrip_35b_500states_3rollouts_20260928.jsonl`。按当前人为成本的胜出数 ANSWER 280 / ASK 94 / CONSULT 86 / DEFER 25；这些是带采样噪声的 *oracle-style* 探索统计，不是学得策略的测试性能。

已完成的离线 router：按病例划分 150/50/50 训练/验证/测试；冻结 Qwen3 隐状态的优势目标模型在 98 个完整测试状态上，当前成本档位的效用 0.4869，始终 ANSWER 为 0.4966，差值 -0.0097，病例 bootstrap 95% CI [-0.0403, 0.0134]；TF-IDF 优势目标全部选择 ANSWER，效用 0.4966。绝对价值目标也未超过 always ANSWER。**目前没有可宣称的 learned deferral 收益。** 报告位于 `outputs/ea_atpo/router_adv_*_500states_20260929/report.json`。

## 新发现：官方 ATPO 过程奖励的真实接入状态

`codes-repro/recipe/atpo/mt_reward_fn.py` 已计算提问格式、verifier 和有效回复分数，因此“原 ATPO 完全没有过程奖励”不准确。但是 `codes-repro/recipe/atpo/user_assistant_agent_loop.py` 的 `_evaluate_nodes()` 对非终止节点保留了注释 `# node.action_reward = reward_info["process_reward"]`，实际设置为 `node.action_reward = 0.0`；其后树备份与 advantage 使用 `node.action_reward`。`mt_reward_manager.py` 也计算过程奖励，但对训练返回的 `reward_tensor` 是 outcome tensor。**需先用最小测试确认上述执行路径及实际训练配置**，不能仅凭静态代码推断全部训练行为。

## E0：基线与奖励路径审计（先做；不占 GPU）

1. 锁定 step-20 baseline 的启动脚本、配置、数据哈希、评估集与采样参数，保存可复现清单；确认 actor、critic、树搜索采用的实际路径。
2. 构造三条受控轨迹：获得新 fact、重复/无效提问、直接正确回答。记录 `process_reward`、`node.action_reward`、树备份值、advantage 与 actor 更新输入；明确各分数是否进入优化，以及是否发生重复计分。
3. 验证树的 `gamma`、终止奖励与现有校验奖励量纲；检查问诊模拟器是否只读 `atomic_facts`，专家是否只读公开状态。检查无效轨迹的处理、`<think>` 清洗、不同病例拆分和推理时无 gold。
4. 输出 E0 审计表与单元测试。若过程分数已经通过其他路径生效，改写 Stage 1 假设，不进行重复添加。

## E1：真正的 Process-guided ATPO（ATPO 主线，GPU 训练）

只在 E0 确认的树节点奖励接口实施**增量事实获得奖励**；不直接照搬既有 `effective_reward`，因为“患者能回答”不等于“取得对诊断有用的新信息”。维护每病例已获知 fact 集合，在患者回复后计算新增、重复、无效三类信号；原始隐藏 fact 只供训练环境判定，不进入 actor 可见状态。把新增过程信号接入 `node.action_reward`，使树备份、critic target、turn-level advantage 和 actor 梯度一致更新；测试时不需要隐藏 fact 判定器。

预注册四组：A 官方 ATPO；B 仅接通原有 process_reward；C 仅加入新增 fact 信号；D 两者结合。先用小规模 smoke 检查每组优势值/梯度有限、checkpoint 可加载、同一轨迹奖励与手算相符，再同数据、同模型、同训练步数/rollout 预算完成正式比较。主要指标：病例级准确率、平均轮数、有效提问率、新 fact 获得率、重复/无效提问率；同时报告训练成本。至少三个随机种子或等计算量重复，病例级 bootstrap CI。若 D 仅提高过程指标却降低诊断准确率，不能宣称主方法成功。

## E2：固定 ATPO actor 的专家门控（两阶段 learn-to-defer）

先冻结 E1 胜出的 base actor；从其真实访问的状态采样，病例级 train/val/test 分离，初始与后续状态分层。对四动作做配对 rollout，保存每动作 outcome、推理调用数、token、延迟、失败类型和成本；测试集只用于最终评估。CONSULT 是低工作量、针对公开状态与具体问题的建议，之后由 base actor 继续；DEFER 是高工作量独立终局判断。两者都不得读取 gold、隐藏 fact 或 base 私有思考。

学习推理前可预测的增益 `Q_out(s,a)-Q_out(s,ANSWER)`，成本只在决策时扣一次。避免使用 rollout 后的专家提示、患者答复或正确答案作特征。与 always ANSWER、always ASK/CONSULT/DEFER、随机/规则门控、原绝对 Q 头比较；主指标为**同专家调用预算下的准确率**及成本调整效用，辅以 consult/defer rate、必要/可避免转交、专家工作量。当前 3 次 rollout、150 训练病例噪声大；先量化每动作 return 方差、动作价值间隔和标签翻转率，再决定扩大 rollout 或病例数，不能盲目加模型规模。

进入下一阶段的门槛：至少一个预注册合理成本/预算档位在独立病例测试集上超过 always ANSWER，病例级 95% CI 不跨 0；效果在不同种子、初始/后续状态分层和专家质量审计下保持。若达不到，报告负结果并回到状态表示、专家协议、数据难度和监督噪声诊断，不进入 E3。

## E3：Expert-aware ATPO 联合优化（条件性增强）

只有 E1、E2 达标后，才把 CONSULT/DEFER 作为环境动作纳入 ATPO 的分支扩展、终止条件、树备份、critic 与 actor 更新。对照 E2 固定门控，在**相同专家调用预算、训练算力与评估病例**下比较。逐项测试四动作合法性、成本只扣一次、终止/继续语义、数据泄漏及信用分配。E3 不是当前正在运行的实验，也不是论文成立的先决条件。

## 立即执行顺序与记录规则

下一步先完成 E0；在完成审计前不启动新 GPU 训练，也不改动官方代码副本。所有实验应保存：不可变配置快照、代码差异、随机种子、数据划分/哈希、启动命令、日志、checkpoint、指标 JSON、失败原因与结论。每次推进在本文件追加日期、完成项、证据路径、偏差和下一步；不要覆盖早先结论。旧版含 `<think>` 泄漏的 500 状态数据只可用于管道调试，不得进入正式方法比较。

### 2026-09-29 进度记录

- 完成 ATPO/ACT 方法定位复核；确认当前四动作包是后接离线策略学习，而非 ATPO 联合优化。
- 完成 500 状态清洁数据、绝对价值与优势目标 router 报告复核；两种目标均未展示优于 always ANSWER 的测试收益。
- 静态代码发现官方过程分数已计算，但树节点中间奖励设为零；列为 E0 第一优先验证点。
- GPU 状态：41 号机 GPU 4/5/6 仍承载 base/patient/expert 服务，GPU 7 空闲；前四卡为其他任务。当前没有本项目 router 训练进程。
- 下一个可执行任务：E0 受控轨迹奖励路径测试与可复现清单；随后依据结果决定 E1 最小代码改动。
