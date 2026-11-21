# ATPO 答案来源逐条审计

## 实际核查

2026-11-21 从作者项目指向的 Google Drive 下载 [MedQA](https://github.com/jind11/MedQA) 和 [MedMCQA](https://github.com/medmcqa/medmcqa) 原始压缩包，分别 SHA-256 `1c2ca8130b3d86d9a99a432ab9bef14f3bb9807bef20facd9ac86ba36960f629` 和 `16c1fbc6f47d548d2af7837b18e893aa45f45c0be9bda0a9adfff3c625bf9262`。作者仓库当时 commit 分别为 `27b02f66aac217933c9648a06f82e9f720377925` 和 `c59ef14ca1990266c4107c7864b45a20fd93e5e0`；仓库 commit 固定链接声明，Drive 对象本身未声明不可变 revision，故以实际压缩包 hash 标识。源 ATPO 文件 SHA-256 `434a6734dce931262f2ed9e3201a3751e57db3609b1612ab9666835ac8832fdc`。

129 条历史动作结果对应 43 状态、15 病例、15 个不同 ATPO 源行。对每个病例在作者的 train/dev/test 存档中用四个规范化选项全文匹配，再按官方正确答案**文本**映射 ATPO 选项字母，避免直接比较可能重排的字母。按病例：13 例只有一个候选，2 例各有两个候选；所有 17 个候选在各自映射后均与 ATPO 答案键一致。按动作记录：111 条唯一选项集合候选、18 条存在歧义。所有候选位于对应压缩包的 train 文件。13 个唯一候选中，11 例的 `Problem` 片段也是原题精确子串；但初始公开 prompt 经提取/改写，原题完整病史没有逐字保留。2 例同选项多候选虽答案一致，也不能确定唯一原题。不得将候选一律称为逐条官方 Gold 确认。

私有逐动作 JSONL `private/private_129_gold_provenance_v3.jsonl` 含 case/state/action、ATPO 行、当前答案键、候选官方 ID 或文件行定位、原始答案、答案一致性、候选数、出处、split、压缩包版本和仓库许可说明。MedQA 作者 JSONL 没有稳定 question ID，此处的 `train.jsonl:行号` 是存档内定位符，**不是官方 ID**；MedMCQA 提供 UUID。逐条文件 SHA-256 `722d251adc696dfb09441cf4d8cbda695564d5be7010d20a8c7b3e9aa7974864`。它含答案与可能的私人信息，留在 PFS，不上传 GitHub。病例内多状态并非重复独立病例；15 个 ATPO 源行各不相同，但当前检查无法排除跨题同义或病例级重复。不存在足够证据认定为医生审核 Gold。

既有 Oracle runner 的检查记录 129/129 prompt-ATPO 源记录一致、129/129 历史动作正确性与该源键一致，这证明内部一致，不证明原题语义完整保留。项目模拟患者与历史生成标签不是 Gold；此前 169 条不一致仍为 PENDING，未自动判错，也未对 291 条进行全量人工审核。

## 权利、来源及 split 的限制

作者仓库各带 MIT 许可证，但仓库许可并不能单独证明第三方医学考试题的所有版权、隐私及商业训练/再分发权利。文件只存 PFS。MedQA 原存档 US train/dev/test 是五选项，ATPO 使用四选项，因此按选项文本映射，不能假设字母位置未变。MedMCQA 选项 `cop` 在原存档为 1–4，作者 [数据读取代码](https://github.com/medmcqa/medmcqa/blob/main/dataset.py) 将其减一。

注意作者 MedMCQA README 表述的 dev/test 规模与实际下载归档文件名不一致：归档 `dev.json` 为 4,183 条且含答案，`test.json` 为 6,150 条且不含答案；README 的表格则将 dev/test 数量反向写成 6,150/4,183。本文以归档**实际文件名**标注候选 split，不擅自消解文档冲突。作者仓库、[MedQA README](https://github.com/jind11/MedQA) 及 [ATPO README](https://github.com/Quark-Medical/ATPO) 均需结合原文件理解，不将代码许可推定为患者数据权利。
