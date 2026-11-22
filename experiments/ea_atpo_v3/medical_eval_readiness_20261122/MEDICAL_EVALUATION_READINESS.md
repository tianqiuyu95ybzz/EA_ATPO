# 医学评价与权限准备（2026-11-22）

## 已验证状态

本次先 fetch 当前 C 独立分支，远端与开始 HEAD `8fcbd488987da4b7d5a2db9b9a063a11b382102f` 一致，未覆盖其他工作区。用户给出的 `quality_audit_20261121/HUMAN_EVALUATION_SUMMARY.md` 和 `human_review_readiness_20261122/human_review_readiness.py` 不存在；对应现有文件分别是 `review_privacy_20261122/HUMAN_EVALUATION_SUMMARY.md` 与 `human_review_readiness_20261122/summarize_human_reviews.py`，已读取。最新 REPORT、盲评说明、指标和受限本地包均核对。

20 例 / 40 答复逐条计算 SHA-256，40/40 与旧评审材料逐字相等；单条 digest 和匿名选项对应保留在本地 private，不发布病例链接。排序多重集摘要 `1095c93b2c829d0055aa7c0de1a0d18371925f89c47df5ad2254ae0db35d4b63`；本地受限包与评分表分别与上一轮清单 SHA 一致，整个 50 例输出文件哈希一致，所有评分及配对评分字段为 0 个已填。DEFER 在已有 50 例无完整同状态经核验参考，N/A；3 条可能含部分医生建议的候选不构成 Gold 或完整对照。

## 原始来源及使用限制

[ReMeDi 固定版本 README](https://raw.githubusercontent.com/yanguojun123/Medical-Dialogue/f94bb9abfba48f467669d2454771bbe263cc4ae5/readme.md) 声明所有资源为 MIT，[许可证](https://raw.githubusercontent.com/yanguojun123/Medical-Dialogue/f94bb9abfba48f467669d2454771bbe263cc4ae5/MIT-license.txt)列明版权及许可声明须随复制或实质部分保留；不能忽略这一明示许可。[作者论文](https://staff.fnwi.uva.nl/m.derijke/wp-content/papercite-data/pdf/yan-2022-remedi.pdf)称在线咨询来自春雨医生、部分个人和机构信息经匿名化并经人工标注；它没有证明本次具体对话的所有上游权利及后续评审访问已获许可。

[春雨医生现行公开服务协议](https://www.chunyuyisheng.com/agreement)第 8 条提出平台内容权利和使用限制。网页现行协议不自动证明其与历史样本收集时间、ReMeDi 研究者的授权或本项目评审者之间的适用关系，也不能由此断言某一具体使用违法或必然获许。作者 README 的 MIT 声明与平台条款、上游原始内容和个人信息处理依据的关系须由有权的项目数据治理/法律负责人核实。没有找到可直接证明这 20 例可以向本项目外部医学评审者发送或在本地进行二次评审的单独授权。

## 可行的受控本地评审方案：设计，不是已获准许

在不重新构造样本的前提下，可准备只在受控项目环境内的评审方案：先核实数据权利、人员身份和处理依据，完成独立隐私审查并记录授权；对 20 例逐条最小化个人信息，同时记录临床语义损失，控制评审者与密钥的权限；评审时关闭外部上传/联网、限制复制导出并记录访问；评分结果、解盲映射和原文分离，结束后按批准的保留计划处置。若远程屏幕或第三方托管可暴露原文，也属于对外访问，必须纳入核查；不是“无文件传输就一定合规”。这些是可实施的控制建议，不是对适用法律或当前系统的合规认证；目前材料留在本地，无外发和人工评分。

已有受限版本中 20/20 例至少一段不少于 12 字的上下文与本地公开源逐句一致；儿童月龄、病史、治疗和时间组合可能关联到人。原始模型输出不可改，评审材料额外变换会影响可比性，必须逐项记录，且不自动判定匿名。参照 [HHS 风险方法说明](https://www.hhs.gov/hipaa/for-professionals/special-topics/de-identification/index.html)，不据此断言美国法律适用或已完成认证。

本次不读取 Frozen Holdout，不重跑 50 例、不训练、不发送病历。评审等待真实有资质人员和数据治理结论；当前无医学评分、无 Utility 和无 Router 改善证据。
