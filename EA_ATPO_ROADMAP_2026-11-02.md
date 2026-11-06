# EA-ATPO 后续计划与创新点

更新日期：2026-11-02

## 一、研究创新点

| 编号 | 内容 | 在项目中的定位 | 当前证据与边界 |
|---|---|---|---|
| I0 | 医疗对话中四种动作：ANSWER、ASK、CONSULT、DEFER；CONSULT 后继续由 agent 推理，DEFER 由专家终局接管 | 研究问题与系统设定 | 保留为项目主线，但单独提出四动作/渐进转交不足以构成算法创新。序贯转交、信息获取和指导式转交均已有相关工作 |
| I1（主候选） | **按协作通道可靠性学习序贯路由价值**：分别估计患者回答可用率、专家提示对后续推理的帮助价值、专家独立接管的正确率；每轮按公开状态和剩余资源重新选择动作 | 新方法候选 | 合成实验中，“难度+专家可靠性”在固定专家预算下为 63.11%，比无可靠性特征高 2.60 个百分点（配对 95% CI [1.83, 3.37]）。尚未在真实医疗数据或 LLM 上验证 |
| I1a | 将 CONSULT 提示效用与 DEFER 终局准确率分开估计 | I1 的关键机制 | 不能假定同一个专家在“给建议”和“独立作答”两种任务上能力相同；当前合成实验尚未把这两个可靠性参数分开，需在真实实验中补测 |
| I1b | 根据患者实际拒答更新“本病例后续提问可获得信息”的估计；UNKNOWN 保持为未知，不变成阴性事实 | I1 的状态更新与安全机制 | 项目交接记录指出 forced-ASK 分支拒答较多；这是明确的工程风险。UNKNOWN 处理和信息增益已有相关研究，暂不作为独立新颖性主张 |
| I2 | 在相同专家工作预算和患者负担下评价准确率、校准及渠道质量变化 | 实验协议贡献 | 必须使用病例级独立测试、paired bootstrap/置信区间，并报告 ASK 拒答、专家 tokens/延迟、无效提示和 unknown-as-negative；预算评估是验证设计，不单独包装成算法创新 |

**文献定位原则：** ATPO、ProMed、MedCIA 已涉及医疗多轮信息获取或信息增益；MIND 涉及证据状态与可靠性元数据；Sequential Learning-to-Defer、Guided Deferral、varying-expert 和 expert-conditioned-advice 工作覆盖序贯/专家转交的邻近问题。因此论文应把重点放在“医疗对话中患者回答通道、非终局专家提示、终局接管三种不同转移的可靠性估计与成本敏感序贯决策”上，并在投稿前继续做逐篇方法对照。见[调研记录](EA_ATPO_RESEARCH_REVIEW_AND_NEW_METHOD_2026-11-02.md)。

## 二、CONSULT 数据从哪里来、如何构建

当前项目没有可直接使用的真实医生 CONSULT 问答数据集。上游 ATPO 数据来自公开医学选择题基准的多轮改写；旧服务器交接记录为 train 14,256、test 1,268，恢复数据后仍需重新核实切分与哈希。[ATPO 上游 README](https://github.com/Quark-Medical/ATPO)

| 数据部分 | 构建来源 | 模型可见范围 | 用途 |
|---|---|---|---|
| 病例和选项 | 上游公开医学 MCQ 改写样本 | agent、patient、expert 只看题干、选项和已公开对话 | 构造初始状态 |
| 患者事实 | 每条样本的 `atomic_facts` | 只给 patient simulator；agent/expert 不可见 | 患者回答 ASK 或返回 UNKNOWN |
| CONSULT 问句 | 由 base agent 根据当前公开状态和它自己的不确定点生成，例如 `Consult: 当前最需要鉴别 A 与 B 的哪项证据？` | 仅当前公开对话、预算和模型自身状态 | 控制咨询是否有明确目标；不能从 gold explanation 派生 |
| CONSULT 回复 | 由独立专家 LLM 在线生成；若做真人子集，则由医生按同一公开状态和问句给提示 | 专家只看公开状态和 CONSULT 问句，不看 atomic_facts/gold/future response | 生成非终局建议，格式限制为 `Expert Advice:` |
| 训练/评价结果 | agent 在提示后继续推理产生最终答案；训练评估器在 rollout 结束后比对 gold | gold 只给 outcome evaluator，不能进任何 actor/expert 输入 | 形成 `Q_out(s, CONSULT)`、正确率、token/延迟等成本 |

建议的单条 CONSULT 轨迹：

```text
公开病例与对话
    ↓
base agent 输出 CONSULT: <基于公开状态的具体问题>
    ↓
expert LLM 看到公开病例 + 该问题，输出 Expert Advice: <短提示>
    ↓
base agent 继续动作，可再次 ASK/CONSULT，也可 ANSWER/DEFER
    ↓
终局 outcome evaluator 记录正确性；另记专家 token、延迟和患者负担
```

`answer_info`、标准答案解释、正确选项片段都不能用来生成咨询问句或提示，否则咨询轨迹已经带有目标答案。可以从**公开病例**提炼某一项待鉴别证据，但应由 agent 从公开状态生成，并保留原状态以供专家判断；不能从隐藏答案里“切一小条”再让专家确认。

**实现状态（2026-11-03）：** 上述代码缺口已在本地采集器修复：base agent 根据当前公开状态生成定向 CONSULT 问题，再传给 expert；续接循环支持 ASK/CONSULT/ANSWER/DEFER，默认每例最多一次 CONSULT；DEFER 终止并由 expert 接管。初始状态生成、公开对话和 JSONL trace 都剔除 `<think>` 私有推理；expert 只看公开病例、对话和定向问句，patient 的 UNKNOWN 保持 UNKNOWN；无终局答案的 rollout 记录明确失败原因。协议与采集器 mocked 测试 **17/17** 通过，Python 语法检查和 `git diff --check` 通过。尚未接真实 endpoint/病例运行，CONSULT 文本质量仍未验证。

## 三、执行路线

| 阶段 | 后续工作 | 关键交付物 | 进入/通过标准 | 状态与依赖 |
|---|---|---|---|---|
| P0 资源恢复与复现清单 | 固定官方 ATPO commit、VeRL commit、配置、数据划分与哈希；找到 Qwen3-8B SFT checkpoint、患者/专家服务地址和可用 GPU；复核访问权限与数据来源 | `repro_manifest.json`、启动配置、环境版本、服务健康检查记录 | 文件、模型、服务和 GPU 可用；不得以历史日志替代当前可复现资产 | **部分完成，pilot 阻塞。** `experiments/real_case_pilot_20261103/outputs/run_manifest.json` 固定了项目 commit、配置/脚本哈希、环境、GPU 与 endpoint 检查。工作区及历史挂载均未找到原始 train/test Parquet 或两个历史 checkpoint；公开源 JSONL 下载尝试因吞吐低于 1 MiB/30–60s 而停止。系统 Python 3.12 缺依赖，但现有 Python 3.11 共享环境具备 `aiohttp/pandas/pyarrow/datasets`。本次快照 GPU3 空闲（约 4 MiB 显存使用），其余卡有不同程度占用；30001–30003 均拒绝连接。上游 README 说明仓库提供 RL 代码，SFT 需外部准备 |
| P1 方法预注册 | 固定主假设、主要预算档、病例级切分、随机种子、成本口径、统计方法与停止条件 | 一页实验预注册表 | 测试集不能参与校准、阈值选择或模型选择；一次锁定主终点与比较组 | **草案已写**于 `EA_ATPO_EXPERIMENT_PREREGISTRATION_2026-11-03.md`；live rollout 前按恢复数据哈希锁定 |
| P2 交互标签与泄漏审计 | 抽检至少 100 条 ASK/CONSULT/DEFER 轨迹；标注 patient response 为 answerable / unanswerable / ambiguous；独立评 CONSULT 提示是否有证据依据、是否帮助后续 agent；统计 DEFER 正确率 | 标注指南、双人/仲裁记录、标签一致性、泄漏报告、unknown-as-negative 率 | 专家看不到 gold、hidden atomic facts、未来回复或 agent 私有 think；CONSULT 不输出终局答案；DEFER 必须终止 | **标注指南已写**于预注册文件；无 rollout，尚未开始人工标注 |
| P3 复现基础策略 | 运行官方 ATPO baseline 或验证现有可用 checkpoint；固定 Assistant policy 为 Qwen3-8B，与官方口径对齐；在验证集和独立测试集报告基础指标 | baseline checkpoint/commit、训练日志、病例级评测 JSON、复现差异说明 | 格式、准确率、轮数、拒答和成本均稳定；checkpoint 可加载；不得把历史 4B smoke 当正式结果 | 依赖 P0；上游代码不含 SFT 训练代码 |
| P4 收集配对反事实 | 固定随机种子抽取 100 例；在相同公开状态比较 ANSWER/ASK/CONSULT/DEFER，评估 token/延迟/成本与可靠性分层 | 无泄漏 JSONL、配对结果 CSV、完整运行报告 | 病例级 train/val/test 隔离；问句来自公开状态；CONSULT 可继续、DEFER 可终止；失败样本有原因码 | **runner 和恢复脚本已就绪，live pilot 阻塞。** 最新预检 `status=blocked`、0 条病例轨迹；原因是无 Parquet 和三个服务均不可达。成本敏感性网格、纠错/退化与动作优势分析均尚无真实结果。运行入口见 `experiments/real_case_pilot_20261103/README.md` |
| P5 实现可靠性校准路由 | 训练公开状态上的患者回答可用率、CONSULT 后续收益、DEFER 独立正确率估计；另拟合 Q_out 与 cost；使用验证集校准风险下界；预算耗尽时合法动作掩码生效 | 可加载 gate checkpoint、校准图、动作价值误差与不确定性分析 | 推理输入不包含 gold/atomic_facts/未来交互；CONSULT 与 DEFER 价值头分开；成本只扣一次 | 依赖 P4；先做离线 gate，不启动联合 RL |
| P6 主实验与消融 | 比较 ATPO base、当前 outcome-value gate、难度特征、患者可靠性、专家可靠性、三者组合、always ANSWER、规则/随机对照；在相同专家预算曲线上测试 | 主要结果表、paired case bootstrap CI、多种子结果 | I1 只有在预注册预算上超过 base/always ANSWER、CI 支持正收益、校准合格且 unknown-as-negative 不升高时才通过 | 依赖 P5；至少多随机种子；不得按测试集挑预算 |
| P7 渠道变化压力测试 | 改变患者拒答率、专家可靠度、难度比例；检查路由校准与性能是否失效；报告 CONSULT 和 DEFER 各自变化 | reliability-shift 曲线、分层误差、失效案例 | 增益在目标 stress slices 不靠隐藏状态；发现漂移时能检测或退回保守策略 | 依赖 P6；以模拟器 stress test 为主，不称临床验证 |
| P8 条件性联合 ATPO | 仅当 P6/P7 通过后，把 CONSULT/DEFER 真正纳入 ATPO 树搜索、critic 和 actor 更新；与冻结 gate 同预算对比 | 联合训练代码、checkpoint、预算匹配结果 | 联合 RL 在独立测试上优于冻结 gate，且成本、泄漏、终止语义均通过审核；否则报告负结果并停止扩展 | **未启动。** 不作为论文成立的先决条件 |
| P9 论文与可复现发布 | 整理相关工作、算法、局限、数据/模型许可、匿名化日志和运行说明；只公开可发布材料 | 论文草稿、复现包、结果摘要 | 不包含原始隐私数据、密钥、模型权重或服务器凭据；清楚标识模拟专家而非真人医生 | 依赖真实结果；不得先于 P6 宣称方法有效 |

## 四、当前完成与下一步

- 已完成：阅读 `tqy/创新点.txt`、EA-ATPO 仓库交接和四动作实现；完成聚焦式文献调研；协议与采集器检查 17/17 通过。
- 已完成：P1 预注册草案和 P2 标注定义；病例样本与双人标注尚待真实 rollout 产生后执行。
- 已完成：CPU 合成路由实验，30 个随机种子；固定专家工作预算 0.20 units/case。2026-11-03 复跑结果不变：expert reliability 为 63.11%，比 no-reliability 高 2.60 个百分点（paired 95% CI [1.83, 3.37]）；患者+专家特征组合为 62.33%，低于 expert-only。
- 不能据此声称真实模型提升：合成实验使用手写 outcome 公式和完美可靠性信号，没有真实 CONSULT 文本或医疗病例。
- **建议当前推进顺序：恢复 P0 数据与三路推理端点 → 完成 P4 真实病例 pilot → 审阅 CONSULT/DEFER 泄漏及价值结果 → 再决定是否进入 P5。** 本轮没有启动 P5/P8，也未将合成数据当真实结果。当前只有 GPU3 空闲，但工作区缺少 Parquet/模型 checkpoint，30001–30003 无服务；因此 P4 不能执行，四动作价值学习暂不进入。

合成实验与复现命令见[实验目录](experiments/reliability_conditioned_router_20261102/README.md)。
