# DDXPlus 官方字段、模拟环境与 Oracle 审计

固定作者源码为 `69e8cf50a15ebb8979a57bb1f30ba5bb7f587f66`，英文数据目标为 Figshare v2。原论文实验使用法文发布，英文发布为同格式版本。本次未获取患者数据文件，仅使用作者文档的合成示例、版本元数据与已有项目摘要。
[NeurIPS 论文](https://papers.neurips.cc/paper_files/paper/2022/hash/cae73a974390c0edd95ae7aeae09139c-Abstract-Datasets_and_Benchmarks.html) · [字段文档](https://github.com/mila-iqia/ddxplus/blob/69e8cf50a15ebb8979a57bb1f30ba5bb7f587f66/README.md#patient-description) · [Figshare 英文数据](https://figshare.com/articles/dataset/DDXPlus_Dataset_English_/22687585)

| 项目 | 发布语义与官方文档示例 |
| --- | --- |
| 病例定位 | 英文患者记录为六列，无原生患者 ID。使用固定 archive SHA-256、split 与原始行号定位，独立性和患者身份无法由此证明 |
| 初始可见证据 | `INITIAL_EVIDENCE` 是已固定的起始证据，包含于完整 `EVIDENCES`。文档示例为 `E_91`，人口学还有 `AGE=18` 与 `SEX=M`。人口学是否公开须在三臂统一，官方 BASD 初始化包含年龄与性别编码 |
| 隐藏证据 | 完整 `EVIDENCES` 仅留在患者环境。二元症状以证据码表示，类别或多选采用 `证据码_@_值码`。示例同一 `E_54` 有 `V_161` 和 `V_183` 两个值。这是一个证据的两个合法值 |
| 阳性、阴性及未知 | BASD 中未问状态为 `NONE_VAL`。问二元项后，列于病例证据列表为 `PRES_VAL`，未列出的项为 `ABS_VAL`。未列的类别或多选走元数据默认值。该规则属于合成环境。例 `E_130` 的默认是 `V_11`，并与 `E_129` 相关 |
| 病史或风险因素 | 证据元数据 `is_antecedent` 标出 antecedent，和症状一并通过证据码查询。证据类型 B/C/M，共 223 个证据 |
| 疾病标签 | `PATHOLOGY` 是合成生成时指定的目标病种，匹配 `condition_name`。官方文档示例目标为 `URTI`，整体有 49 类 |
| 疾病概率 | `DIFFERENTIAL_DIAGNOSIS` 是病例的候选疾病—概率对列表。示例 Bronchitis 约 0.1917，排序最高仍可能与指定 `PATHOLOGY` 不同。概率属于数据集生成过程，非临床患病概率或 Base 置信度 |
| Split | Figshare 提供 Train、Validate、Test。作者约报 130 万合成记录。项目历史英文包核验得到 Train 1,025,602、Validate 132,448，Test 的本地行数 **UNKNOWN**，本次只核验发布包元数据 |

[证据和病种元数据](https://github.com/mila-iqia/ddxplus/blob/69e8cf50a15ebb8979a57bb1f30ba5bb7f587f66/README.md#evidence-description) · [BASD 目标状态编码](https://github.com/mila-iqia/ddxplus/blob/69e8cf50a15ebb8979a57bb1f30ba5bb7f587f66/code/basd/env.py#L253-L302) · [项目既有去重审计](../ddx_phase3_20261121/OVERLAP_AUDIT.json)

## 官方模拟代码与复用路径

作者仓库发布 BASD 和 AARLC 的可执行规则环境。BASD 更贴合同病例重复分支。其 `initialize_state(indices=[i])` 支持固定病例索引、重置轮次和查询状态。应保持数据自带初始证据启用。若初始证据缺失或禁用，源码会从可用二元症状随机取值，不能用于约定固定 s0 的配对。初始化返回五元组，同时带标签、差分索引、概率和严重度，只有第一个返回值 `init_state` 可进入后续经白名单投影的公开观测。完整 `target_state` 和评估字段只留在环境。
[BASD 初始化](https://github.com/mila-iqia/ddxplus/blob/69e8cf50a15ebb8979a57bb1f30ba5bb7f587f66/code/basd/env.py#L305-L391) · [AARLC 对照](https://github.com/mila-iqia/ddxplus/blob/69e8cf50a15ebb8979a57bb1f30ba5bb7f587f66/code/aarlc/ddxplus_code/env.py#L301-L420)

BASD `step(s,a_p,done)` 用目标字段切片和 action mask 更新当前指定证据，再返回状态、奖励和 done。每个证据对应独立切片，因此单次选一个证据时只会改变该证据及可选的轮次计数。M 类型可合法更新该证据的多个值。代码构造了 `code_question` 相关索引，但 `step` 没有据此强制拒绝子证据。若实验要求根证据前置，需单独固定可用动作约束并核验。Base 只能读公开状态、当前问题的投影答复和固定标签词表。不要传 `reward_s`、全部已知证据、`target_state`、差分、病种或未选择字段。
[BASD 单证据掩码](https://github.com/mila-iqia/ddxplus/blob/69e8cf50a15ebb8979a57bb1f30ba5bb7f587f66/code/basd/env.py#L159-L174) · [step 与 reward](https://github.com/mila-iqia/ddxplus/blob/69e8cf50a15ebb8979a57bb1f30ba5bb7f587f66/code/basd/env.py#L394-L420) · [关联索引](https://github.com/mila-iqia/ddxplus/blob/69e8cf50a15ebb8979a57bb1f30ba5bb7f587f66/code/basd/env.py#L639-L687)

## 同状态三动作

- **ANSWER**。将同一 s0 的约定人口学和唯一起始证据公开给冻结 Qwen3-8B，按固定 49 类词表输出单病种，与该病例 `PATHOLOGY` 精确匹配。候选词表对所有病例相同，仅评价器持有目标。
- **ASK**。把同一 s0 复制到独立分支，选择一个未揭示且可询问的证据，以 BASD 的公开状态差分得到一个证据的反馈。更新后用同一 Base 和诊断协议终止。无效、重复、不满足依赖及默认值必须分开统计。全量隐藏证据只可作为另定义的信息上界。
- **DEFER**。Standard Oracle 直接返回同一病例的 `PATHOLOGY`，输出只在分支终局或评估端出现。Base-Protecting Oracle 用事后 Gold 判断 Base 是否精确命中，命中则保留，失败则返回 Gold。在标准标签口径下两种 Oracle 通常等价。病例级 Gold 一致与同初态对照成立，但 Oracle 本身是拥有额外真值的理想端点。仅依赖 s0 的独立专家、临床真实性及开放式答案质量仍缺证据。
- **可做程度**。官方环境提供查询和状态转移，三宏动作封装、可见状态投影、病例索引重置、成本与 Oracle 评价由项目定义。旧项目 `ddx.py` 有同病例 ASK 一次、Standard/Protecting Oracle 的历史验证原型。旧结果不能替代 BASD 接口验收。历史记录另发现 Train/Validate 交集有 1,625 种完全相同记录，正式 split 仍需处理。正式 Q 范围限定相同标签尺度下的合成诊断，和 ReMeDi 医疗回复质量分开。

[历史 50 例报告](../ddx_phase3_20261121/PHASE3_RESULT_REPORT.md) · [旧原型](../ddx_phase3_20261121/ddx.py) · [工程边界](../ddx_phase3_20261121/ENGINEERING_AND_ADMISSION.md)
