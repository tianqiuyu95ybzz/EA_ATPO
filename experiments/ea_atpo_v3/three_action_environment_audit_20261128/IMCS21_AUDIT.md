# IMCS-21 官方资源与 ASK 接口审计

审计依据为作者仓库提交 `836048f491704011dba229a5bd04db5f7f4d2bbe`、论文 v3 第 3–5 节和作者维护的 CBLUE 发布说明。本轮只读源码及元数据。出版社附录链接未能获取，附录全文独立复核为 **UNKNOWN**。以下接口结论来自已固定的实际源码。
[论文第 5.5 节](https://arxiv.org/html/2204.08997#S5.SS5) · [期刊论文与附录入口](https://doi.org/10.1093/bioinformatics/btac817) · [固定仓库](https://github.com/lemuria-wchen/imcs21/tree/836048f491704011dba229a5bd04db5f7f4d2bbe) · [CBLUE 作者说明](https://github.com/lemuria-wchen/imcs21-cblue#数据格式)

## 数据与任务

- 源于中文在线儿科多轮咨询。论文统一按对话划分 Train 2,472、Dev 833、Test 811，共 4,116 条，覆盖 10 类病种。任务为命名实体识别 NER、对话动作分类 DAC、症状标签推断 SLI、医疗报告生成 MRG、诊断导向对话策略 DDP。症状词表需按所选版本固定。
  [论文第 4–5 节](https://arxiv.org/html/2204.08997#S4) · [官方 Train/Dev/Test 元数据](https://api.github.com/repos/lemuria-wchen/imcs21/git/trees/836048f491704011dba229a5bd04db5f7f4d2bbe?recursive=1)
- 发布结构以记录 ID 为键，包含 `self-report`、`dialogue`、`diagnosis`、`report`、`explicit_info.Symptom`、`implicit_info.Symptom`。句级标注含说话人、文本、BIO 实体和 `dialogue_act`，新版说明另含归一化症状、症状状态和 `local_implicit_info`。询问意图由动作类别表达。DDP slot 是归一化症状。记录 ID 可作为复现键，患者实体级身份及跨划分隔离为 **UNKNOWN**。
  [说明与发布字段](https://github.com/lemuria-wchen/imcs21-cblue#数据格式) · [SLI 评价器](https://github.com/lemuria-wchen/imcs21/blob/836048f491704011dba229a5bd04db5f7f4d2bbe/task/SLI/eval_sli.py#L84-L108)
- 症状标注含 `1` POS、`0` NEG、`2` NS。未记录的症状由 DDP 返回 UNK，来源与 NS 不同。官方 `diagnosis` 可作十类 benchmark 评价标签，取得原病种的临床确认链为 **UNKNOWN**。报告中的诊断由标注者根据完整对话推断，需与官方类别分开，整份报告也不进入初态。
  [论文第 4.1、4.3、5.5 节](https://arxiv.org/html/2204.08997#S4) · [DQN 状态编码](https://github.com/lemuria-wchen/imcs21/blob/836048f491704011dba229a5bd04db5f7f4d2bbe/task/DDP/Flat_DQN/agent/utils.py#L89-L112)

## Simulator 类型与调用边界

| 类别 | 对 IMCS-21 的判断 |
| --- | --- |
| A 作者原生独立患者服务 | 未找到 IMCS-21 自研的独立服务。DDP README 明确标注其基线代码来源 |
| B 作者提供的接口或代码 | 已确认。作者仓库随 DDP 发布从 DISCOpen-MedBox 复用的规则患者及状态跟踪器 |
| C 历史回放 | 可以按发布对话复放历史问答，另有 B 能力 |
| D 只有标签需从零实现 | 无此需要。已有 B 的查询机制，发布格式需薄适配 |

[DDP README](https://github.com/lemuria-wchen/imcs21/blob/836048f491704011dba229a5bd04db5f7f4d2bbe/task/DDP/README.md) · [原始 DISCOpen-MedBox 项目](https://github.com/Guardianzc/DISCOpen-MedBox-DialoDiagnosis)

1. `User.initialize(dataset, goal_index=...)` 可钉住病例，向 agent 初始公布全部显式症状，疾病请求字段使用未知占位。省略 `goal_index` 会随机选病例。连续多轮可用 `User.next(agent_action, turn)`，直到轮次或终止动作结束。每个分支重新初始化独立实例并使用同一个 split 和 goal index，源码上可以复现不同查询序列。运行验证在 amax 为 **UNKNOWN**。
   [初始化与变更状态](https://github.com/lemuria-wchen/imcs21/blob/836048f491704011dba229a5bd04db5f7f4d2bbe/task/DDP/Flat_DQN/user_simulator/user.py#L62-L192)
2. agent 动作为结构化 `request`，携带 `request_slots`。命中隐含槽返回该槽的已标注状态，未记录槽返回未知，已有槽走重复/约束失败路径。源码会遍历请求字典，调用方需限制每次恰好一个症状；采用复制后的当步对应槽反馈，避免将可变模拟器状态、未问症状和 reward 交给 Base。完整中文自由文本的问答映射及真实未来反馈均未提供。
   [规则反馈](https://github.com/lemuria-wchen/imcs21/blob/836048f491704011dba229a5bd04db5f7f4d2bbe/task/DDP/Flat_DQN/user_simulator/user.py#L200-L231) · [论文结构化任务](https://arxiv.org/html/2204.08997#S5.SS5)
3. `preprocess.py` 读取 `label/exp_sxs/imp_sxs` 列表式记录，作者现行数据说明是按记录 ID 的 `diagnosis/explicit_info/implicit_info` 字典。已有 DDP 预处理不能原样套用。若许可明确，仅需把现有标签映射到 `disease_tag` 与显式/隐含的 goal schema。另有关键泄漏点，原 `DialogueManager.next` 把 `user.goal["disease_tag"]` 传给旧 agent。V3 只能复用患者端或严格审计的公开状态入口，禁止沿用整个 manager 调用链接 Base。
   [预处理字段](https://github.com/lemuria-wchen/imcs21/blob/836048f491704011dba229a5bd04db5f7f4d2bbe/task/DDP/preprocess.py#L14-L46) · [manager 的疾病标签参数](https://github.com/lemuria-wchen/imcs21/blob/836048f491704011dba229a5bd04db5f7f4d2bbe/task/DDP/Flat_DQN/dialogue_manager/dialogue_manager.py#L42-L64)

**准入结论。** 结构化 ASK 专项具备可复用候选接口。数据权利与格式桥接尚待确认。真实咨询中由新问题引发的自然语言反馈仍无观察依据。官方十类标签有助于受限分类，未找到同决策状态的专家回答或原生 DEFER。许可与大小见 [DATA_LICENSES.md](DATA_LICENSES.md)。
