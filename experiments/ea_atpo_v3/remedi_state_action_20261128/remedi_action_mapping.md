# ReMeDi 动作映射与适配

构建基线为项目提交 `47f8eff35b3dea3e44ce21083f2a01a99fc99cee`、官方数据提交 `f94bb9abfba48f467669d2454771bbe263cc4ae5`。原始 JSON 使用 `dialogue` 和 `information`，轮次使用 `turn`、`role`、`sentence`、`message`、`actions`。标签多见 `intent`、`slot`、`value1`、`value2`、`text`、`range`。缺失或变体字段保持原样。按官方处理规则合并相邻同角色话语，各条样本保留原对话 ID、原轮次、所属文件和映射后的组位置。原始字段可据此回查。

| 官方文件标签 | 医生动作含义 | 本轮映射 |
| --- | --- | --- |
| `Inquire` | 对症状、时长等槽进行问询 | ASK |
| `Diagnosis` | 发表诊断判断 | ANSWER |
| `QuestionAnswering` | 针对患者问题作答 | ANSWER |
| `Recommend` | 建议检查、用药、治疗或注意事项 | OTHER。该标签也覆盖需追加检查的建议，缺少区分终止回答的标志 |
| `Inform` | 告知事实或解释 | OTHER |
| `Chitchat` | 社交性话语 | OTHER |
| `Other` | 其余动作 | OTHER |

同组若同时出现 `Inquire` 与直接回答标签，单独记 MIXED，从单动作模仿样本中排除。历史回放分析保留该组的问询成分。缺失标签归入 OTHER，并记下缺失标记。患者侧只有已出现的话语中带非空有效值的 `Inform` 标注进入当前槽状态。通用 `other`、缺名槽和未知值占位不计为已知信息，明确的否定值照常保留。所有动作标签只进入监督或审计记录。

ANSWER CSV 分列动作、槽和实体参考的可用率。字面实体对照使用医疗槽的 `value1` 与原话语做规范化子串匹配，另检查 `range` 指向的标注文本，允许发布文件中闭合与半开结束位置。Action Consistency、Slot Consistency 和 Medical Entity Coverage 的独立预测指标需待后续生成并对齐，现阶段留空。源码回复的实体字面匹配率只反映标签与原文的对齐，医学正确性需要独立医学评价。

[论文第 3、5 节](https://irlab.science.uva.nl/wp-content/papercite-data/pdf/yan-2022-remedi.pdf) · [官方标注指南](https://github.com/yanguojun123/Medical-Dialogue/blob/f94bb9abfba48f467669d2454771bbe263cc4ae5/annotation_guide/MDS_annotation_guide.pdf)
