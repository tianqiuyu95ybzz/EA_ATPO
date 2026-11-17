# Qwen3-32B-AWQ DEFER 配对验证结果

**执行状态：** 六阶段验证已完成；本次不启动 PPO/ATPO RL。
**实验目录：** `outputs/qwen32_defer_validation_20261119_attempt01/`（共享 PFS 本地保存，未纳入 Git）
**结果分析版本：** `analysis_v4/`；早期 `analysis/`、`analysis_v2/`、`analysis_v3/` 均保留。
**当前结论：** Qwen3-32B-AWQ 在一个独立 200 例测试队列上比 Qwen3-8B Base 准确率高，但决策前特征尚不能可靠预测专家优势；Learned Router 没有可靠胜过简单基线，因此不满足重启 RL 的门槛。

## 1. 实验设置与数据完整性

固定模型为 Qwen3-8B Base（revision `b968826d9c46dd6066d109eabc6255188de91218`）和 Qwen3-32B-AWQ Expert（revision `0499c3ac83fdef8810b907a23894ba91e95eddd8`）。权重分片、配置、tokenizer 的 SHA-256 保存在 `model_fingerprint.json`；32B-AWQ 分片与锁定下载清单校验一致。

实验按 `case_id` 严格分成 200 个训练病例、100 个验证病例和 200 个独立测试病例。训练、验证、测试的病例 ID、病例哈希和公开 prompt 哈希互斥。200 例测试队列在模型调用前冻结；其结果未用于拟合系数、挑选超参数、阈值或成本。病例抽样、排除历史暴露病例的审计和源数据哈希见 `case_selection.json` 与 `paired_rollouts/run_manifest.json`。

另有 20 例协议 smoke，Base/Expert 输出均 20/20 可解析，Base 不确定性均可提取，所有原始回答均保存。随后完成 500 个配对模型请求（训练 200、验证 100、测试 200）；Base 输出均有效，Expert 有 16 条格式无效输出（训练集 10、验证集 6、测试集 0）。无效输出按错误计入且没有从数据中剔除。每条轨迹保存 raw completion、prompt/completion token 数、finish reason、解析状态、延迟与响应哈希；提示正文、Gold 和隐藏 atomic facts 没有写入 JSONL。

Base 与 Expert 收到相同病例题干和选项，系统指令按 ANSWER/DEFER 角色区分；采样参数为 temperature 0、top-p 1、thinking 关闭，Base 最多生成 64 tokens、Expert 128 tokens。Expert 是 Qwen3-32B-AWQ 模拟模型，不是真实医生。数据是公开医学选择题，不是真实临床病历。

GPU2 运行期间显存使用峰值 111,172 MiB、最低空闲 71,453 MiB；服务已停止，实验端口 30156/30157 已释放。本次没有结束或修改其他进程。

## 2. 配对准确率与专家互补性

测试集病例级 bootstrap 10,000 次，随机种子 `20261121`。配对正确性四格表如下：

| Base | Expert | 病例数 |
|---|---|---:|
| 对 | 对 | 76 |
| 对 | 错 | 16 |
| 错 | 对 | 35 |
| 错 | 错 | 73 |

| 指标 | 结果 | 95% 配对/病例级区间 |
|---|---:|---:|
| Base 准确率 | 46.0% | 39.0%–53.0% |
| Expert DEFER 准确率 | 55.5% | 48.5%–62.5% |
| Expert − Base 准确率差 | +9.5 pp | +2.5–+16.5 pp |
| Base 错误时，Expert 修正率 | 35/108 = 32.4% | 23.8%–41.5% |
| Base 正确时，Expert 退化率 | 16/92 = 17.4% | 9.9%–25.3% |

在这个测试 cohort 上，Expert 有净纠错优势，但也会破坏一部分 Base 正确答案。正的总体配对差来自 35 次纠错减去 16 次退化，不能据此假定任意病例都适合 DEFER。尚未有第二个独立队列复现，所以“可重复互补性”仍未建立。

## 3. 预决策特征与 Learned Router

Router 只读取决策时可见的信息：公开病例长度/词数/选项等统计、Base 已生成的选项答案，以及选项 token logprob 熵和概率间隔。它回归 `Expert correctness − Base correctness`，不读取 Gold、Expert 输出或事后正确性作为输入。

系数仅用 200 个训练病例拟合；Ridge alpha 在 100 个验证病例上选择（alpha=100）；六档成本阈值也只在验证集锁定。`policy_lock.json` 明确记录 `test_split_opened_by_fitter=false`、`oracle_used_for_fitting=false`。

| 预测诊断 | 结果 |
|---|---:|
| Train MSE/相关性 | MSE 0.271，Pearson r=0.211 |
| Validation MSE/相关性 | 0.265，Pearson r=0.111 |
| 独立 Test MSE/相关性 | 0.248，Pearson r=0.054，95% CI −0.084–0.194 |
| Test 上识别 Expert 修正 Base 错误的 AUC | 0.596，95% CI 0.489–0.697 |

测试集相关性区间包含 0，AUC 区间包含 0.5；当前特征没有显示可复现的专家优势预测能力。训练/验证表现略有信号，但独立测试没有确认。该模型仅是诊断性第一版 Learned Router，不作为可部署路由器。

## 4. 六档成本与策略比较

固定 `Utility = Accuracy − λ × DEFER 调用率`，`平均成本 = λ × DEFER 调用率`。成本单位是预注册的归一化单位，不是美元或实际 GPU 费用。所有阈值均由验证集选择，测试集不挑成本。表内每格顺序为 **Accuracy / Utility / 调用率 / 平均成本**。Random DEFER 同时列出固定种子的实际抽样结果和精确随机化期望；抽样只依赖病例 ID 与种子 `20261122`。

| λ | Always Answer | Always DEFER | Random 实际 | Random 期望 | Uncertainty DEFER | Learned Router | Oracle 上界 |
|---:|---|---|---|---|---|---|---|
| 0.00 | .460/.460/0%/0 | .555/.555/100%/0 | .510/.510/51%/0 | .508/.508/51%/0 | .500/.500/32%/0 | .530/.530/62%/0 | .635/.635/17.5%/0 |
| 0.05 | .460/.460/0%/0 | .555/.505/100%/.050 | .510/.485/51%/.026 | .508/.483/51%/.025 | .500/.484/32%/.016 | .530/.499/62%/.031 | .635/.626/17.5%/.009 |
| 0.10 | .460/.460/0%/0 | .555/.455/100%/.100 | .510/.459/51%/.051 | .508/.457/51%/.051 | .455/.450/5%/.005 | .530/.468/62%/.062 | .635/.618/17.5%/.017 |
| 0.20 | .460/.460/0%/0 | .555/.355/100%/.200 | .455/.453/1%/.002 | .461/.459/1%/.002 | .455/.445/5%/.010 | .455/.451/2%/.004 | .635/.600/17.5%/.035 |
| 0.40 | .460/.460/0%/0 | .555/.155/100%/.400 | .455/.451/1%/.004 | .461/.457/1%/.004 | .455/.441/3.5%/.014 | .455/.447/2%/.008 | .635/.565/17.5%/.070 |
| 0.60 | .460/.460/0%/0 | .555/−.045/100%/.600 | .455/.449/1%/.006 | .461/.455/1%/.006 | .460/.460/0%/0 | .455/.443/2%/.012 | .635/.530/17.5%/.105 |

病例级配对 Utility 差异显示：Learned Router 在 λ=0 时优于 Always Answer（+0.070，95% CI +0.015–+0.125），但未可靠胜过 Uncertainty（+0.030，−0.020–+0.080）或 Random 期望（+0.022，−0.014–+0.056）。λ=0.05、0.10 时与简单策略的差异区间都跨 0；λ=0.20、0.40 时 Learned 低于 Always Answer；λ=0.60 时低于 Always Answer 和 Uncertainty（均 −0.017，95% CI −0.039–−0.003）。因此，在预注册成本网格内，没有成本点证明 Learned Router 能可靠超过简单基线。

Always DEFER 的测试点估计相对 Always Answer 净增益为 +0.095；所以按点估计，其全量接管的盈亏平衡成本是 λ=0.095。该增益的区间为 +0.025–+0.165，说明 λ=0.05 的净效用差仍不确定；在 λ=0.60，Always DEFER 的 Utility 为 −0.045。

Oracle 仅在事后 `Expert correctness − Base correctness > λ` 时 DEFER，完全依赖 Gold/结果标签，只是不可部署上界。λ=0.60 时它以 17.5% 调用率取得 0.635 Accuracy、0.530 Utility，相对 Always Answer 的配对 Utility 差为 +0.070（95% CI +0.050–+0.092）。这显示理论上有选择专家的空间，但不代表任何已学习策略做到了这一点。

完整准确率、Utility、调用率、平均成本和各项置信区间在 `outputs/.../analysis_v4/cost_sensitivity.csv` 与 `paired_policy_comparisons.csv`。

## 5. 风险—覆盖与正确答案退化

覆盖率指没有转给 Expert、仍由 Base 作答的比例。λ=0 时，验证集锁定阈值在测试集实际调用 Expert 124/200（62%），保留 Base 的覆盖率为 38%；调用病例中包含 67/108（62%）个 Base 错误，修正 24 个错误，同时破坏 10 个 Base 正确答案。未转交病例的 Base 风险是 53.9%，接近全体 Base 风险 54.0%，因此该路由没有明显把高风险病例集中挑出来；净准确率提升主要来自大量调用，而不是有力的风险识别。

在主成本 λ=0.60 时，验证阈值对应 1% 调用率，测试集实际调用 4/200。覆盖率为 98%（病例 bootstrap 95% CI 96.0%–99.5%），覆盖病例的 Base 错误率 54.1%，几乎等于全体 Base 风险。4 次接管中没有修正 Base 错误，反而破坏 1 个 Base 正确答案；系统 Accuracy 从 46.0% 降至 45.5%，最终风险从 54.0% 升至 54.5%。错误捕获率仅 1.9%（95% CI 0%–4.8%）。所以主成本下没有证据表明选择性 DEFER 能减少高风险错误，并确实出现正确答案退化。

完整阈值曲线及风险、覆盖率、纠错和退化区间见 `outputs/.../analysis_v4/risk_coverage.csv`。所有风险阈值只从验证分数生成。

## 6. 调用成本、异常与复现

主成本 λ=0.60 下的平均推理量：Always Answer 169.3 tokens / 0.0435 s；Always DEFER 243.4 / 0.0948 s；Uncertainty 169.3 / 0.0435 s；Learned Router 176.6 / 0.0457 s；Oracle 上界 212.5 / 0.0603 s。时间是本次 GPU 服务请求延迟，不是线上负载或临床服务 SLA。

重要产物位于 PFS 本地运行目录：

- 原始 Base/Expert 输出与 token/延迟/解析遥测：`paired_rollouts/base_answer_trajectories.jsonl`、`qwen32_defer_trajectories.jsonl`。
- 配对训练/验证/测试病例及评分：`paired_rollouts/paired_{train,validation,test}.jsonl`；报告只发布汇总，不列病例 ID。
- 模型版本、源数据和结果哈希：`paired_rollouts/run_manifest.json`、`case_selection.json`、`model_fingerprint.json`。
- 路由器系数与冻结阈值：`router/policy_lock.json`、`router/validation_policy_selection.csv`。
- 当前分析产物：`analysis_v4/summary.json`、`base_expert_2x2.csv`、`cost_sensitivity.csv`、`paired_policy_comparisons.csv`、`risk_coverage.csv`、`expert_advantage_predictability.json`、`analysis_manifest.json`。
- 固定种子选中的随机路由病例：`analysis_v4/random_policy_selections.json`。

复现入口：`bash experiments/real_case_pilot_20261103/audit/run_qwen32_defer_validation.sh`；脚本先锁模型指纹和病例选择，再启动服务，GPU/服务启动和依赖环境见同目录两个 `start_qwen*_defer_validation_service.sh`。脚本会检查已有服务、smoke、rollout 与遥测文件并拒绝覆盖；重跑时先将本次完整输出目录归档到另一个 PFS 名称，再在空目录运行。只重做测试集分析可运行固定环境下的 `analyze_qwen32_defer_validation.py`；分析器也会拒绝覆盖已有 `analysis_v4/`，如需另一个分析版本，先递增代码中的分析目录版本。

## 7. 六阶段结果与结论

| 阶段 | 状态 | 结论 |
|---|---|---|
| 1. 环境、模型与数据锁定 | 完成 | 固定模型 revision、权重/tokenizer 哈希、源数据哈希和 200/100/200 病例切分。 |
| 2. 20 例协议 smoke | 通过 | 20/20 成对答案有效，20/20 Base logprob 不确定性有效，raw 输出完整。 |
| 3. 200 例配对 DEFER | 完成 | Expert 准确率 55.5% 对 Base 46.0%；纠错和退化均非零。 |
| 4. 五类基线与 Oracle 上界 | 完成 | Oracle 上界有明显空间；它没有进入训练或测试时路由。Random 同时报单次抽样和期望。 |
| 5. 预决策预测、成本、风险—覆盖 | 完成，但路由器未通过 | 测试优势预测接近随机；主成本下没有风险筛选收益，Learned 未可靠胜过简单基线。 |
| 6. 阶段决策 | No-Go：不启动 RL | 尚无第二 cohort 重复结果，且 Learned Router 未可靠胜过简单基线。 |

**med_qwen** 仍保留为候选；权重、基座和配置通过核验并用独立病例运行前，不报告任何准确率或专家收益。
