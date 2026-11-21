# EA-ATPO V3 ReMeDi Dialogue Pilot 数据审计

日期：2026-11-21
用途：支持 C 运行真实模型推理；本轮仅做 CPU 数据准备与评测工具，不运行模型、不训练、不调用 GPU。

## 数据来源与文件完整性

使用服务器上已存在的 ReMeDi 文件，没有重新下载。原始文件位于数据准备 worktree 的私有目录：

`/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO_V1_public_medical_data_20261121/experiments/ea_atpo_v3/public_medical_20261121/private/remedi_base.json`

文件大小 `16,596,996` bytes，SHA-256 `b27b6b1222255cc87cbc6cc0fb6ce7de98e0b6341f5ea61892bb92f247383458`。JSON 能完整读取，共 1,557 条对话，原始 `dialogue` ID 1,557 个且唯一；本地数据准备分支的 checksum 验证对 ReMeDi 返回 OK。MedDG 文件也已在服务器存在，但 ReMeDi 可用，所以没有使用 MedDG fallback；没有读取或修改 C 的 Gold 数据目录。

ReMeDi 角色来自原文件 `information[].role`，正文来自 `information[].sentence`。原始轮次顺序保留，不合并连续同角色消息，不把不同 `dialogue` ID 混在一起。

## 候选检查与抽样

结构审计只统计字段、角色、顺序和精确重复，不判断临床质量：

| 项 | 数量 |
|---|---:|
| 原始对话 | 1,557 |
| 首轮为患者且之后有医生发言 | 1,555 |
| 空消息/结构异常而排除 | 2 |
| 规范化后完全重复对话而排除 | 20 |
| 去重后的结构有效对话 | 1,535 |
| 有至少一组医生疑问后紧接患者回复的唯一对话 | 1,292 |
| 满足问答回放条件的相邻问答对（去重后） | 3,593 |
| 最终 Pilot | **100** |

从 1,292 条符合回放条件的唯一对话中，按 `SHA256("EA_ATPO-ReMeDi-pilot:20261121:" + 原始 dialogue ID)` 升序确定性选取 100 条；没有按模型表现、Gold 或回复内容挑选。最终 100 条包含 2,299 个原始轮次（患者 1,162、医生 1,137），每例 5–50 轮，平均 22.99 轮；100 个原始 ID 均唯一。每例至少有一组医生疑问与患者后续反馈，合计 **281** 组 ASK 回放关联。

医生疑问检测是显式问号/中文疑问表达的规则启发式；关联要求源序列中医生消息后**紧接**非空患者消息。没有跨轮猜测，也不会把其他后续患者发言当作该问题的回答。规则会有误检可能，因此这是待 C 检查的回放候选，不是人工验证后的问答标签。

## 私有文件与字段

私有输出目录：

`/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO_V1_v3_pilot_prep_20261121/experiments/ea_atpo_v3/dialogue_pilot_20261121/private/`

- `pilot_samples.jsonl`：100 条完整源对话，保留 `source_dialogue_id`、`source_order_index`、原始角色/正文/轮次索引、首条患者问题、每条历史医生发言及 281 组相邻医生问题—患者回复配对。
- `pilot_manifest.json`：源文件哈希、确定性筛选规则、筛选数量、排除数量、选中原始 ID、样本文件哈希。仅保存在私有目录。
- `simple_rule_routes.jsonl`：无模型置信度时的文本规则回退；当前 100 条因初始信息字段不完整而全部选 ASK，等同 Always ASK，不作为有区分度的 EA-ATPO 结果。

权限为目录 `0700`、文件 `0600`。原始对话、样本 JSONL、manifest、路由行和模型输出均不得提交 GitHub。

样本每行顶层字段包括：`sample_id`、`dataset`、`source_dialogue_id`、`initial_patient_question`、`messages`、`historical_clinician_response_references`、`ask_replay_pairs`、`data_semantics`。医生回复明确标记为 `verified_expert_gold=false`。

## ANSWER / ASK / DEFER / EA-ATPO 评测接口

- **ANSWER**：Base 只看到首轮患者输入并直接生成回答。
- **ASK**：测试集回放采用指定的历史医生疑问及其紧邻患者反馈，随后让 Base 继续回答。这是“真实历史信息回放”基线，不等同于 Base 自己生成问题。若测试的是 Base 自己生成的问题，只能在预先冻结的精确匹配规则成功时提供相应患者回复；不匹配则记录 `replay_mismatch`，不能借用无关消息。
- **DEFER**：C 使用真实 Expert 服务生成回答。历史医生回复只作为带 turn index 的 `historical_clinician_response_reference`，绝不是 Gold，也不替代真实模型输出。不同轮次医生回复对应不同已见信息，比较时要记录参考回复的信息状态。
- **EA-ATPO-simple-rule**：提供无模型训练的透明规则实现。若 Base 置信度可用，固定阈值为 `<0.35 → DEFER`、`[0.35,0.65) → ASK`、其余 ANSWER（阈值未校准）；同时有显式高风险词规则。没有置信度时回退到初始病史字段规则。本数据集上该回退恰好 100/100 ASK，因此应与 Always ASK 区分报告但不能宣称有增益。

规则只读取首轮患者公开文字和可用动作回放，不读取 Gold、医生后续答案或最终结果。它不是临床分诊规则。

## 统一评价

本轮提供了固定盲评 prompt `blind_evaluation_prompt.md` 和 CPU 聚合器 `scripts/evaluate_pilot.py`。回答质量评分 1–5 归一化为 `(score−1)/4`；Utility 定义为：

```text
Utility proxy = normalized blind answer-quality − 0.01 × ASK_count − 0.05 × DEFER_count
```

成本沿用 EA-ATPO V2 主配置，且每个动作计费一次。该分数是**盲评质量代理**，不是准确率或临床正确性。ASK 有效信息获取率只统计盲评对“患者回复是否回答医生问题并带来相关新信息”的判断；未评分样本不当作无效。失败样本及原因写入私有输出。

本轮没有调用模型评估服务，也没有启动或使用 GPU；没有盲评结果。后续若 C 的既有评测服务可用，可按统一 prompt 对随机打乱且隐藏策略名的回答评分。医生回复的文本相似度不得作为正确性指标。

## 复现

```bash
python scripts/build_pilot_samples.py \
  --input /path/to/existing/private/remedi_base.json \
  --output-dir /path/to/private/dialogue_pilot_20261121/private \
  --max-samples 100 --seed 20261121

python scripts/evaluate_pilot.py \
  --predictions /path/to/private/predictions.jsonl \
  --ratings /path/to/private/blind_ratings.jsonl \
  --output-dir /path/to/private/evaluation_summary \
  --ask-cost 0.01 --defer-cost 0.05
```

生成器拒绝覆盖非空目录。其他复现说明见 `RESULT_SCHEMA.md` 和 `C_HANDOFF.md`。本报告只记录结构与工程结果，不代表来源权利已获得法律确认，也不证明历史临床内容正确。
