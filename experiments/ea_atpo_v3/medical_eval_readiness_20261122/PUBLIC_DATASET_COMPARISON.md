# 公开评价资源比较：候选而非已引入

所有规模和标签语义若未注明本地实查，均为发布方/论文声明；本轮未下载、未检视样本或生成任何 Gold。没有证据将下列资源的单病例标签直接转移到现有 20 例 ReMeDi 问诊。

| 资源 | Gold / 实际标注 | ASK / 诊断及病例划分 | 许可 / 准入 | 与本项目的关系 |
|---|---|---|---|---|
| [HealthBench](https://openai.com/index/healthbench/) | 5,000 条模拟或对抗生成的健康对话，逐对话医生编写的评分条目；不是现有 ReMeDi 个体诊断 Gold | 部分条目评估获取上下文；公开整体/Hard/Consensus 有重叠子集，未证实患者级 train/dev/test 独立 | [simple-evals 代码 MIT](https://github.com/openai/simple-evals/blob/main/LICENSE)；数据与具体使用/再分发条件需单独核实，作者请求不要公开样本原文 | 首选**外部医生评分框架候选**，只用于其自身病例或获得医生批准的新映射；不能直接给旧 Pilot 打分或提供自动 Oracle 回答 |
| [MedVAL-Bench 1.0.2](https://physionet.org/content/medval-bench/1.0.2/) | 发布方称 12 名医生对 840 个生成医学文本评错误及风险；是评价者标注，不是本项目病例 Gold | 含病人问题、药物问答及对话到记录任务；不直接提供交互患者，病例/患者划分需获授权后检查 | PhysioNet credentialed 1.5、培训和 DUA；上游数据各有条款，未获取授权 | 候选用于独立验证事实/安全评价器，不能直接评 ReMeDi 回答或作患者诊断 |
| [MedDialogRubrics](https://huggingface.co/datasets/AQ-MedAI/MedDialogRubrics) | 发布方称合成病例、LLM 生成后经领域专家筛修的 must-ask rubric；非真实患者及非逐诊断临床确证 | 旨在多轮问询和患者代理；当前卡未明确 train/dev/test 或患者独立性，未实读数据 | 页面无元数据许可证，指向 LICENSE；2026-11-22 仓库文件清单未见 LICENSE；先核实文件、版本和权利 | 适合作为未来 ASK 评价候选，**暂不引入**；不能直接映射 20 例 |
| [DDXPlus](https://github.com/mila-iqia/ddxplus) | 官方合成病种 PATHOLOGY 和差分列表，为合成标签而非真实临床专家诊断；本地前 50 validation 映射已查 | 结构化 evidence 可模拟 ASK，但依赖和默认值尚待实现/验证；官方 train/validate/test 无患者 ID，本地发现跨 train/validate 精确重复 | [英文发布 CC BY 4.0](https://figshare.com/articles/dataset/DDXPlus_Dataset_English_/22687585)；隔离、代码和规则需先冻结 | **项目已选主候选**，仅做合成诊断分类的 Gold-backed 上界，不评完整医疗回答 |
| [IMCS-21](https://github.com/lemuria-wchen/imcs21) | 原始类别 diagnosis 与人工推断的报告 diagnosis 不同；类别的上游形成和临床确认链未完全核实 | 中文多轮与 POS/NEG/NS 标注；官方 train/dev/test；患者级独立性未知，模拟器需适配 | 仓库数据许可未定位清晰声明，需核权 | 已选真实中文对话辅助候选；先核权和标签链，不直接充当确定 Gold |
| [Med-HALT](https://medhalt.github.io/) | 考试题与记忆/虚构问题的基准答案或构造标签，不是患者诊断 | 非患者多轮 ASK；不同来源 split 与污染需各自核对 | 页面说数据限研究及非商业，代码 Apache-2.0；上游题目权利需核实 | 仅作为医学事实/幻觉能力的额外对照，不能将知识题答案赋给 ReMeDi 病例 |
| ReMeDi / MedDG | 医生历史发言与语义/实体标注；当前无逐时点独立验证的临床 Gold | 原始回放提供患者后续话语；不能当作对任意自主 ASK 的反事实反馈，患者级隔离未知 | ReMeDi README MIT，MedDG 仓库 GPL-3.0；上游内容权利及共享须核查 | 实际中文咨询与人工审查，不把普通医生句子当 Oracle |

推荐顺序：先保持 DDXPlus 主线的准入修复，取得真实医学评价资源可用许可后，以 HealthBench 自身案例/医生标准评安全与信息获取，并将 IMCS-21 用于受权中文回放。MedVAL-Bench 需凭证和 DUA；MedDialogRubrics 需先核清许可证与完整数据；Med-HALT 仅独立知识对照。以上均不是对现有 20 例的现成人工评分，也没有新数据导入。
