# DDXPlus Gold 与 Oracle 审计

[官方仓库](https://github.com/mila-iqia/ddxplus) 和 [English figshare](https://figshare.com/articles/dataset/DDXPlus_Dataset_English_/22687585) 定义 PATHOLOGY 为合成患者的 ground-truth 病种；它是生成过程标签，不是真实患者临床专家核验结果。DIFFERENTIAL_DIAGNOSIS 是生成系统给出的病种、概率对列表；所抽 50 例概率和约为 1，但这不证明实际临床分布，也不证明包含所有医学鉴别诊断。EVIDENCES 是隐含状态；INITIAL_EVIDENCE 是其中启动用的一项。数据集原生提供 train/validate/test；本轮仅取官方 validate，未读取其 test。

官方英文元数据包含 223 个 evidence 和 49 个条件；实际文件中 49/49 有 ICD-10 字段且 49 个值互异。原型使用官方 condition_name **精确匹配**做标准病种标签评价，未对自由文本、同义词或 ICD 映射作未经审计的推断。ICD 字段可供进一步标准化，但代码范围、临床含义及自由文本标准化仍需单独验证。这个任务只能衡量数据集标签分类，不能评价诊疗回答完整性。

`ddx_cpu_smoke.py` 使用官方 CSV（Python literal 安全解析）及原始 ontology，不依赖第三方文本转换。每例初始只公开 INITIAL_EVIDENCE；ASK 显式 evidence code 后才暴露其数据值；未知代码返回 UNKNOWN，未记录 evidence 保守返回 UNKNOWN，不把缺席误称否定。重复查询收费为零，轨迹保留全部动作和费用。Router 只获取 view，PATHOLOGY 和差分诊断不在 view。此为原型：未实现完整的官方 default_value、code_question 层级/依赖、自然语言问题映射或真实患者回复；这些必须在正式实验前按官方代码核对和冻结。view 隔离是接口约束，不是进程安全沙箱。严格的 offline evaluator 负责标签，Router 不能接收患者对象内的隐藏字段。

Oracle A 返回已知病种；Oracle B 以事后病种相等作为 Base 正确性，正确时保留原标签，否则返回病种。A 的 ANSWER 和 B 的 ANSWER 原样保留；二者仅作为离线理论评估，并非真实 Expert。若 Base 已正确，两 Oracle 都输出相同病种；不能据此声称已经在实际模型上证明保护收益。
