# IMCS-21 原始病种与准入审计

[作者论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC9848052/)形式化 MCR 为自述、对话与病种类别 y；论文明确报告撰写者未获真实病种，而依据对话推断报告中的 diagnosis。故病种 y 与人工 Medical Report 中 diagnosis 是两个不同对象。实际检查 [官方仓库](https://github.com/lemuria-wchen/imcs21) commit `836048f491704011dba229a5bd04db5f7f4d2bbe`：原始 `train.json`、`dev.json` 顶层按记录 ID 索引，y 对应每条的 `diagnosis`；`report` 才是人工报告。train 2,472/2,472、dev 833/833 非空，均有 10 个病种；这只证明文件含标签。论文将病种视作既有 MCR 的类别，并在报告分析中称之为真实病种，但未在已核对资料中说明最初病种如何确定、是否有独立临床确认。故标作**官方基准标签、来源链未完全确认**，不得称为独立临床认证 Gold。

官方代码 `task/SLI/eval_sli.py` 对隐含症状 `implicit_info.Symptom` 使用 1=POS、0=NEG、2=不确定 NS；未记录是 UNK，不等于 NEG。`explicit_info.Symptom` 是患者自述已知症状列表。该结构可支撑仅使用显式状态初始化、逐次查询隐藏症状并返回标注状态的**离线患者模拟**，并非真实患者对自由提问的反事实答复。已查看作者 DDP 预处理：它引用 `label/exp_sxs/imp_sxs` 列表结构，当前仓库 train/dev 文件是以 `diagnosis/explicit_info/implicit_info` 索引的字典，不能直接宣称该脚本开箱即用；需独立审计适配。完整对话和 report 含未来信息，不得作为初始 Router 输入。

仓库有 train/dev/test 文件，所查 train/dev ID 交集为零；无法由此证明患者无跨记录重复，或其他实验/预训练未见过。官方 README 同时称 test 在 CBLEU 托管，版本间应谨慎区分。若病种来源与适用许可核实，精确官方病种标签可评估受限的疾病分类准确率，而非完整诊疗质量。仓库未见明确许可证，论文可访问或代码公开不等于确认学术数据使用权；正式使用须明确权利条款。故本轮不把 IMCS 标为通过准入，也未使用它运行 Oracle 样本验证。
