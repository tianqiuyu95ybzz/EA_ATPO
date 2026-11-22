# 人工评审准备

`BLINDED_REVIEW_INSTRUCTIONS.md`、`LICENSE_AND_REVIEW_RIGHTS.md` 和 `PRIVACY_RISK_STATS.json` 记录评审与风险状态；`summarize_human_reviews.py` 使用现有 private 两张评分表、病例映射及原 blind key。没有真实人类评分、权利和隐私确认时脚本会拒绝统计与解盲；它不创建或推断任何评分。评审材料仍不得对外发送。`privacy_stats.py` 只用本地源做精确匹配和简单模式统计，不上传或打印原文。

汇总示例在获得真实表单、受授权的私有 attestation 和输出目录后由数据管理员运行。参数为 `--scores`、`--pairs`、`--case-map`、`--blind-key`、`--attestation`、`--out`；结果只能写入新的 private 路径。attestation 必须明确 `human_review_completed`、`reviewers_qualified`、`privacy_and_rights_cleared` 和 `rubric_unchanged` 为 true。脚本核验的是声明的存在，不能独立核验人员资质。可选 `--utility-config` 需明确权重和严重错误惩罚，并声明解盲前已冻结；否则只给逐维比较，Utility 为 N/A。1–5 分平均及按病例 percentile bootstrap 为探索性统计，并非确证临床显著性。空值须由评审者明确标记“无法判断”，不能静默变成分数。

真实评分、任意原始对话、病例映射、医生身份和生成文本禁止提交。原数据及先前成果不覆盖；DEFER 没有合格完整配对，保持 N/A。
