# EA-ATPO V3 下一阶段实验准入

| 准入项 | 结论 |
| --- | --- |
| A ReMeDi ASK/ANSWER 行为模仿 | 支持，4,986 个单动作标签。Train 用于拟合，Dev 用于选择，Test 留待协议冻结 |
| B IMCS-21 可比较 ASK 结果 | 任务定义支持。已有 DDP 协议和标签描述，发布文件适配、许可与本机可访问数据仍待确认 |
| C DDXPlus 同状态 Oracle DEFER | 可定义合成诊断的离线理想上界。正式配对需复用验证后的初始状态和隐藏证据隔离，并解决数据可访问性及独立划分 |
| D 同病例同状态三动作净收益 | 当前 0 个可比样本 |
| 正式 Q-guided Distillation | 暂不准入 |

最小下一阶段是 ReMeDi 独立 ASK/ANSWER 验证。先只用 Train/Dev，固定官方 `Inquire`、`Diagnosis`、`QuestionAnswering` 的复用口径，比较实际医生动作、规则和 DPL imitation，分开报告动作、槽和实体，不用实体覆盖代替医学正确性。在引入 Q 前，恢复有权限的现有 IMCS-21 和 DDXPlus 私有数据包，复测 IMCS 结构化 ASK 环境及 DDXPlus 证据依赖。任选具备适配标签的专项任务，在同病例、同可见初态、同冻结 Base 下采集 ANSWER、ASK 后回答和离线 Oracle DEFER 的配对结果。专项诊断指标与中文咨询结构指标分别报告。

冻结原始 `Qwen/Qwen3-8B` revision `b968826d9c46dd6066d109eabc6255188de91218`。只有可靠、可比较的动作结果才能建成 Q。后续用已有冻结 Base embedding 和轻量 MLP 做 Q-guided Distillation，输出 ANSWER、ASK、DEFER 概率。质量定义、ASK/DEFER 成本、缺失动作掩码和病例级隔离须先冻结。Standard Oracle 和 Base-Protecting Oracle 继续报告为离线理想对照，后者依赖事后 Gold。全程在 63 执行。

复现时先恢复 `47f8eff35b3dea3e44ce21083f2a01a99fc99cee` 记录的受控准备目录，保留历史 Pilot 文件及 `prepared-with-spans/SHA256SUMS`。原始 ReMeDi-base SHA-256 为 `b27b6b1222255cc87cbc6cc0fb6ce7de98e0b6341f5ea61892bb92f247383458`。从本目录复制构建器到受控工作区，并指定新输出路径：

```bash
umask 077
CUDA_VISIBLE_DEVICES='' python3 build_remedi_state_action.py \
    --source "$RECOVERED_PRIVATE_SOURCE" --output "$NEW_RUN_DIR"
(cd "$NEW_RUN_DIR/private" && sha256sum -c SHA256SUMS)
```

构建步骤无随机切分、未运行模型、未生成 Gold，也未训练。公共分支仅包含源代码、方法、配置和匿名汇总。
