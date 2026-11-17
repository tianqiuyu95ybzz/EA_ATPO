# L2D 专家与 CONSULT 小提示验证

日期：2026-11-17

> **后续审计口径说明（2026-11-17）：** 本历史报告曾把两个独立 100 例 cohort 作描述性汇总。新的[专家互补性与成本审计](../../../EXPERT_CONDITIONED_COMPLEMENTARITY_COST_ANALYSIS.md)将各 cohort 分开计算病例配对指标；当前结论以分 cohort 报告为准，不将不同病例集拼成单一配对估计。

## 结论

- **Qwen3-32B-AWQ 不是已经选定的 Expert。** 它曾作为“大模型候选”被部署筛选：此前 200 例直接答题为 48% 对 Qwen3-8B 的 44%，但配对差 +4 pp 的 95% CI 为 −2.5 至 +10.5 pp，不能证明它更好。AWQ 是为降低 32B 权重显存而采用的量化格式；不是医疗微调，也没有证据证明 AWQ 版本更适合 DEFER。
- **DEFER 本身没有显示稳定的准确率损失；换成较弱或不匹配的 Expert 才是主要问题。** 同一 Qwen3-8B 改用 DEFER 提示，在两批各 100 例上分别为 53% 对 54%、44% 对 41%；合并后 Base-DEFER 48.5%，Always Answer 47.5%，配对差 +1 pp（95% CI −2 至 +4）。
- **ReMeDi 没有表现出足够的条件互补性。** 两批独立 100 例上合计 81/200（40.5%），低于 Always Answer 的 95/200（47.5%）；配对差 −7 pp（95% CI −13.5 至 −0.5）。它在 Base 错误时纠正 16/105，在 Base 正确时破坏 30/95。医学对话微调不能代替目标 benchmark 上的条件可靠性验证。
- **另外两个候选也没有显示 DEFER 收益。** PopiT 在新 100 例为 40% 对 Base 的 41%；Huatuo 在当前严格 DEFER 格式下 100/100 均解析失败，因此其 0% 是协议失败，不是可解释的医疗准确率。
- **CONSULT 小提示的 20 例探索性 pilot 未显示收益，并暴露出提示合规问题。** Base、ReMeDi、PopiT 的合规提示率分别为 85%、30%、50%；端到端准确率分别为 45%、15%、20%，Always Answer 为 50%。此 20 例 pilot 不能证明一般性退化，但不足以支持继续训练 Router 或 RL。
- **当前 DEFER 成本设定使“完美 Expert”也无法胜过 Always Answer 的净值。** 若 `U_DEFER = p_correct − 0.60`，完美 Expert 的最大净值是 0.40，低于合并直接回答准确率 0.475。Gold 驱动的 Oracle 只能作为理论上界，不能冒充真实模型或用于常规训练样本。

本阶段没有真实医生参与。所有“Expert”均为语言模型。没有启动 Router、PPO 或 ATPO RL。

## 为什么尝试 Qwen3-32B-AWQ

尝试它的理由仅是：在此前固定 200 例直接回答筛选中，32B-AWQ 的点估计比 8B 高 4 个百分点，且本地已有完整权重；AWQ 通过量化降低权重显存，理论上比 BF16 32B 更容易部署。它原本是候选筛选，不是已验证的临床专家，也不是论文方案所必需的组件。

该直接答题差异的区间跨 0；此前 32B-DEFER 测试因共享 GPU 上没有足够 KV cache 未产生病例结果。**部署失败没有性能含义，直接答题结果也不能替代 DEFER 条件测试。** 因此不建议仅凭参数量继续占用资源启动 32B。若以后重测，应先证明它在冻结公开病例上能提高 Base 错误条件准确率且不破坏过多 Base 正确答案，再决定是否承担部署成本。

完整历史结果见 [32B-AWQ 直接答题报告](../../ea_atpo_value_router_20261105/outputs/strong_expert_awq/STRONG_EXPERT_EVALUATION.md)。

## DEFER 配对评估

### 新候选筛选集

新抽取 100 个唯一公开状态，seed `20261118`。它们与旧固定 200 例、ReMeDi 独立确认的 100 例和 Attempt09 20 例开发集不重合。各模型看到相同公开病例，Gold 和 hidden atomic facts 仅用于事后评分。Qwen3 模型使用 BF16、greedy、关闭 thinking、最多 128 tokens。

| 方法 | 正确率（病例 bootstrap 95% CI） | 格式有效率 | 相对 ANSWER 配对差（95% CI） | Base 错时答对 | Base 对时退化 |
|---|---:|---:|---:|---:|---:|
| Qwen3-8B Always Answer | 41/100，41%（31–51） | 100% | — | — | — |
| Qwen3-8B Base-DEFER | 44/100，44%（35–54） | 99% | +3 pp（−1 至 +8） | 4/59 | 1/41 |
| Qwen3-4B-ReMeDi-DEFER | 41/100，41%（31–51） | 100% | 0 pp（−9 至 +9） | 11/59 | 11/41 |
| PopiT-Qwen3-4B-Medical-SFT-DEFER | 40/100，40%（30–49） | 99% | −1 pp（−11 至 +9） | 13/59 | 14/41 |
| HuatuoGPT-o1-7B-DEFER | 0/100，0% | 0% | −41 pp（−51 至 −31） | 0/59 | 41/41 |

Huatuo 的 100 条输出全部未通过 `Expert Final Answer: A-D` 解析；99 条到达 token 上限。它不能被视为 0% 医疗知识准确率，应记为当前协议下不可调用。PopiT/ReMeDi 的纠错数与退化数都按同一病例逐项配对。

### ReMeDi 的两批 100 例合并

两批为不重合的公开状态，分别在 2026-11-17 和 2026-11-18 运行。合并病例级 bootstrap 95% CI，配对差按同一病例计算。

| 方法 | 正确率 | 与 Always Answer 的配对差 | 两者都对 | Answer 对 / Expert 错 | Answer 错 / Expert 对 | 两者都错 |
|---|---:|---:|---:|---:|---:|---:|
| Always Answer | 95/200，47.5%（40.5–54.5） | — | — | — | — | — |
| Qwen3-8B Base-DEFER | 97/200，48.5%（41.5–55.5） | +1 pp（−2 至 +4） | 91 | 4 | 6 | 99 |
| Qwen3-4B-ReMeDi-DEFER | 81/200，40.5%（34.0–47.5） | −7 pp（−13.5 至 −0.5） | 65 | 30 | 16 | 89 |

ReMeDi 在 Base 错误状态的正确率是 16/105（15.2%）；在 Base 正确状态的退化率是 30/95（31.6%）。这解释了为什么“一个模型本身能答对一些题”不等于“交接给它可以改善 L2D”：交接是否有价值取决于它能否在 Base 错误状态补位，并避免破坏 Base 已经答对的状态。

### 成本上界

Attempt09 使用 `λ_DEFER = 0.60` 的归一化成本，收益定义为准确率减动作成本。按合并直接回答准确率 0.475 计算：

```text
U_DEFER = p_correct − λ_DEFER
Oracle upper bound = 1.000 − 0.600 = 0.400 < 0.475 = U_ANSWER
```

在当前成本下，即便完美正确的 Oracle 也不能成为净值最优动作；要让完美 Oracle 净值超过直接回答，`λ_DEFER` 必须小于 0.525。按 λ=0.60，要超过 0.475 的直接回答净值，需要 `p_correct > 1.075`，数学上不可能。这里的 0.60 是研究中的归一化参数，不是医生收费；报告应并列呈现原始准确率和成本敏感性，不能把人工参数解释为真实经济成本。

若使用真实 Gold 为 Oracle 直接生成正确选项，这只能作为**明确标注的算法上界**，不是真实模型性能，也不能把 Gold 发给 Agent/Expert 或作为正常 Actor 轨迹。当前只报告上界计算，没有生成 Oracle 轨迹或训练数据。

## CONSULT 小提示探索性 pilot

20 个配对公开状态取自 seed `20261118` 的候选筛选集 offset 5–24；这批病例此前已经进入候选 DEFER 评估，所以本结果是探索性机制 pilot，不是独立测试。对所有模型固定同一问题：“Which one concrete, visible clue in this case should the Base Agent notice? Give no interpretation or answer.” 候选模型只需指出一个可见线索，不得给诊断、治疗、选项或最终答案；Base 收到提示后继续回答。Gold/hidden facts 未进入任何提示。

| 方法 | 准确率（95% CI） | 配对差 vs ANSWER（95% CI） | 提示格式有效 | 修正 ANSWER 错误 | 破坏 ANSWER 正确 | 观测 tokens/分支* |
|---|---:|---:|---:|---:|---:|---:|
| Always Answer Qwen3-8B | 10/20，50%（30–70） | — | — | — | — | 167.8 |
| Base Qwen3-8B hint + continuation | 9/20，45%（25–65） | −5 pp（−25 至 +15） | 17/20，85% | 2/10 | 3/10 | 454.9 |
| ReMeDi hint + continuation | 3/20，15%（0–30） | −35 pp（−55 至 −15） | 6/20，30% | 0/10 | 7/10 |
| PopiT hint + continuation | 4/20，20%（5–40） | −30 pp（−55 至 −5） | 10/20，50% | 1/10 | 7/10 |

准确率将无效提示/最终答案失败记为错误，并保留在 20 的分母内。错误修正分母是 ANSWER 错误的 10 例；正确退化分母是 ANSWER 正确的 10 例。严格提示仍未让医疗候选稳定产出小提示：ReMeDi 多次超出 15 词或直接解释诊断；PopiT 有时没有 `Expert Hint:` 格式，或给出解释性判断。没有将无效输出筛出后再报告成功率。

\* 分支 token 数是实际运行到答案或协议失败时的总 prompt+completion tokens。协议失败的分支没有继续调用 Base，因此这些 token 均值是“截至成功/失败”的观测成本，不能与完整成功分支成本等同。咨询问题使用固定文本、未由 Base 生成；该实验不覆盖完整 L2D 动作生成开销。延迟来自 CPU 12 线程，不应外推为 GPU 服务延迟。提示准确率的区间是 20 个配对状态的病例 bootstrap；该样本很小且已用于候选筛选。

由于 20 例 pilot 的准确率没有提升、专家提示合规率低，**没有继续把同一提示扩成 100 例，也没有将其当作 CONSULT 质量结论**。初始 5 例协议 smoke 中发现 Base 自动生成的问题会带入诊断猜测，故在任何 continuation 之前停止；那 5 例不进入上述准确率表。

## 后续决策

1. 暂停 32B-AWQ 常驻部署；它是一个尚未完成 DEFER 验证的筛选条件，不是默认 Expert。仅当明确需要检验更大模型，并且有安全显存余量时，才重启配对评估。
2. 不将 Gold 做成真实 DEFER 答案。可保留 Oracle 100% 作为报告中的理论上界；不要把它加入正常 PPO/Router 训练或宣传为真实性能。
3. 当前没有通过验证的 DEFER Expert，也没有通过提示合规的小提示来源。停止扩大 Router/PPO/ATPO RL。
4. 若后续继续 CONSULT，优先使用临床人员或协议合规率达到预注册门槛的模型；Base 保持最终决策权。先单独评估建议质量与 Base 的恰当利用，再评估动作策略。
5. 任何“高概率正确”结论都要在独立病例上校准模型置信度，并报告 risk–coverage、条件正确率和配对退化；模型自报信心不能替代条件可靠性。

## 复现文件

- 候选 DEFER runner：`run_l2d_prior_candidates_defer.py`；100 例输出及运行清单留在忽略目录 `outputs/l2d_prior_candidates_defer_20261118_gpu5/`。数据哈希 `3783ebd70d2282ab31c19f5030bf122301d34465516dee222438835e57c9c051`；样本状态哈希 `5255522035eae49ce84b08d33ec9422eef2604615dcf4c364adbbc07f6c42274`；轨迹 SHA-256 `c3d81a2b5f2bef31f07fb05a2bd41eeeef7a861ebad4690a44d21787aaef19d6`。
- CONSULT hint runner：`run_l2d_consult_hint_experiment.py`；用 CPU 12 线程复现该 20 例子集时，指定一个新的输出目录，例如 `../EA_ATPO/.venv-real-pilot/bin/python experiments/real_case_pilot_20261103/audit/run_l2d_consult_hint_experiment.py --device cpu --threads 12 --limit-cases 20 --offset 5 --output-dir experiments/real_case_pilot_20261103/audit/outputs/l2d_consult_hint_reproduction_202611/`。原输出和清单留在忽略目录 `outputs/l2d_consult_hint_pilot20_20261118_cpu/`；轨迹 SHA-256 `d0541f66d6096d31ece611cef24fcc8e9c17d4d58f8725189d96b02738a6067b`，配对 CSV SHA-256 `a1a1c9a4e1a518f24a4defdcc2882b6d29782573edfc814c8b833b82be5889db`。
- Candidate run manifest records two stages: initial partial execution on GPU5 and completion after resume on GPU3; GPU1/GPU7 were not used. Initial and resumed runner SHA-256 values are `50c8ef39d2cca2e009699361fa7dba4f392a376b461d34e2b2a0625e890de606` and `4727e53fcc06ceb331de2dac0e25d081aa9a7f4b14a6850bae72412195a16054`. The current runner adds a GPU1/GPU7 safety guard after the experiment; it does not change the recorded trajectories.
- CONSULT inference manifest runner SHA-256 is `0843c5e40e186b7b1ca88e09dc7e967ffa067f36cbe4b5a737bbd75a3667eb6b`. The working runner now includes post-run summary-statistic corrections; the raw trajectory and paired-result hashes above match the manifest. A reproduction should use a fresh output directory rather than resume these completed artifacts.
- 轨迹只含病例哈希、模型生成的问题/提示、预测字母、正确性、token、延迟及协议字段；不含病例原文、Gold、hidden facts 或逐字最终答案解释。`run_manifest.json` 保存模型 revision/config 哈希、配置、样本哈希和运行环境。
