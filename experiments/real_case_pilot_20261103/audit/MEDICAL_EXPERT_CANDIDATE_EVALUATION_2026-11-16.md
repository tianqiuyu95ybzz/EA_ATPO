# 公开医疗 Qwen 专家候选评估

日期：2026-11-16；2026-11-17 更新结论。

用途：筛选 EA-ATPO 的 L2D 专家候选。所有候选都是语言模型，不是真实医生。新一轮 100 例确认已撤销 ReMeDi 的临时推荐；详细结果见[独立确认报告](L2D_REMEDI_CONFIRMATION_2026-11-17.md)。

## 当前结论（2026-11-17）

**当前没有通过 L2D DEFER 验证的专家模型。** ReMeDi 在先前 20 例开发集上的 +1 例优势没有复现；新 100 例上，ReMeDi-DEFER 为 40/100，Always Answer 为 54/100，Base-DEFER 为 53/100。ReMeDi 相对 Always Answer 的配对差为 −14 pp（95% CI −23 至 −5）；ReMeDi 在 Base 错误时改对 5/46，在 Base 正确时改错 19/54。按 DEFER 成本 0.60，净值为 −0.20，低于 Always Answer 的 0.54。

因此撤销此前“暂选 ReMeDi”的判断，当前不应训练四动作 Router、PPO 或 ATPO RL。Qwen3-32B-AWQ 在先前 200 例直接问答中为 48% 对 8B 的 44%，配对差 +4 pp（95% CI −2.5 至 +10.5），但它尚未完成新的 DEFER 条件验证。新 32B-DEFER 实验因 GPU 资源变化导致 vLLM 无可用 KV cache 而未进入推理；详见独立确认报告，不能将部署失败误写成性能结果。

## 2026-11-16 阶段筛选结论（历史记录）

在 20 个固定开发病例上，Qwen3-4B-ReMeDi-DEFER 答对 6/20（30%），同提示词下 Qwen3-8B Base-DEFER 答对 5/20（25%），PopiT 答对 3/20（15%）；ReMeDi 的答案格式 20/20 有效。ReMeDi 相对 Base-DEFER 只多答对 1 例，配对准确率差为 +5 pp，病例级 bootstrap 95% CI 为 [0, 15] pp。此结果仅作为候选筛选信号，后来被 100 例新样本更新。

200 例直接问答筛选中，Qwen3-8B 为 87/200（43.5%），ReMeDi 为 77/200（38.5%），PopiT 为 78/200（39.0%），HuatuoGPT-o1 为 23/200（11.5%）。三个医疗候选均未超过同跑 Base。候选排序使用了固定筛选样本，因此该部分只用于筛选。

## 模型来源与冻结版本

- [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)：Attempt09 Base Agent 的底座。
- [FreedomIntelligence/HuatuoGPT-o1-7B](https://huggingface.co/FreedomIntelligence/HuatuoGPT-o1-7B)：模型卡描述为医疗推理候选，基于 Qwen2.5-7B。
- [VioletAmethyst/Qwen3-4B-ReMeDi](https://huggingface.co/VioletAmethyst/Qwen3-4B-ReMeDi)：模型卡面向多轮医疗对话生成，基于 Qwen3-4B。
- [Hoshino-Yumetsuki/PopiT-Qwen3-4B-Medical-SFT-1128](https://huggingface.co/Hoshino-Yumetsuki/PopiT-Qwen3-4B-Medical-SFT-1128)：模型卡描述为中文医疗 SFT，基于 Qwen3-4B-Instruct。

固定 revision：

| 角色 | Hugging Face 仓库 | Revision |
|---|---|---|
| Base | `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` |
| HuatuoGPT-o1-7B | `FreedomIntelligence/HuatuoGPT-o1-7B` | `6487c8071a3f87702a6816ba7a09f38921b685ed` |
| Qwen3-4B-ReMeDi | `VioletAmethyst/Qwen3-4B-ReMeDi` | `dfa418ca1c3c0f8879afd6d356b88083f01ca032` |
| PopiT-Qwen3-4B-Medical-SFT | `Hoshino-Yumetsuki/PopiT-Qwen3-4B-Medical-SFT-1128` | `f7f95e22f5fd2c8c015abfba9fe249d6b80b6f97` |

## 200 例直接问答筛选

### 评测设置

- 数据：ATPO `test.pilot.parquet`，1,268 行，SHA-256 `3783ebd70d2282ab31c19f5030bf122301d34465516dee222438835e57c9c051`。
- 固定抽样：200 例，seed `20261104`；样本 ID 哈希 `ca53dce93c31f962ccee1e32406e48333248faae4c42b55af04a32a59071186f`。Base 与三个候选使用完全相同的病例和公开提示。
- 生成：Transformers CPU，BF16，32 线程，greedy，关闭 Qwen3 thinking，`max_new_tokens=128`。Qwen2 Huatuo 使用其原生 chat template；所有模型使用相同的任务提示和 A–D 严格解析器。解析失败计错并保留在 200 例分母。
- 运行环境：Python 3.12，PyTorch 2.14.0+cu130，Transformers 5.20.2，Intel Xeon Platinum 8558P；本实验未调用 GPU。
- Gold 仅在生成后用于评分；提示只包含病例公开信息。模型加载时没有传入 gold 或 hidden atomic facts。
- 区间：按病例重采样的 10,000 次 percentile bootstrap 95% CI。延迟是 CPU 上单请求的模板处理及生成时间，不含模型加载时间，也不代表 GPU 服务延迟。

### 总体与配对结果

| 模型 | 正确数 / 200（95% CI） | 格式有效率 | 相对 Base 准确率差（95% CI） | Base 错误时模型正确率 | Base 正确时退化率 | 平均生成 tokens | 平均 CPU 请求秒 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-8B Base | 87/200，43.5% [37.0, 50.5] | 100% | — | — | — | 5.00 | 0.96 |
| HuatuoGPT-o1-7B | 23/200，11.5% [7.5, 16.0] | 20% | -32.0 pp [-39.5, -24.0] | 10/113，8.8% [4.4, 14.2] | 74/87，85.1% [77.0, 92.0] | 11.81 | 1.82 |
| Qwen3-4B-ReMeDi | 77/200，38.5% [32.0, 45.5] | 100% | -5.0 pp [-11.5, 1.5] | 16/113，14.2% [8.0, 21.2] | 26/87，29.9% [20.7, 40.2] | 5.00 | 0.74 |
| PopiT-Qwen3-4B-Medical-SFT | 78/200，39.0% [32.5, 46.0] | 100% | -4.5 pp [-11.5, 2.5] | 20/113，17.7% [10.6, 24.8] | 29/87，33.3% [24.1, 43.7] | 5.00 | 0.70 |

配对四格表：

| 候选 | Base 对 / 候选对 | Base 对 / 候选错 | Base 错 / 候选对 | Base 错 / 候选错 |
|---|---:|---:|---:|---:|
| HuatuoGPT-o1-7B | 13 | 74 | 10 | 103 |
| Qwen3-4B-ReMeDi | 61 | 26 | 16 | 97 |
| PopiT-Qwen3-4B-Medical-SFT | 58 | 29 | 20 | 93 |

Huatuo 的 160 条无效回答中，只有 11 条到达 128-token 上限；主要问题是已正常结束的输出未包含解析器要求的单个 A–D 答案，不是单纯的截断问题。它在当前 Agent 协议下不适合作为可直接调用的专家。ReMeDi 和 PopiT 输出格式稳定，但它们在 Base 错误病例上的纠错比例仍低，且会把部分 Base 正确答案改错。

## L2D DEFER 专家协议对照

### 设置与解析器修复记录

- 数据：Attempt09 固定开发集 20 例，Parquet SHA-256 `419ecd2ce0738f45f7fcf67568a5202fabebf02ce2e1dea61ff65681b1752987`；公开状态哈希 `db800a9c33921621ed1aac31b9d3673a0c7e2ef267de5206e4572f902f56c72a`。
- 五个独立 arm：Qwen3-8B Always Answer、Qwen3-8B DEFER、Huatuo DEFER、ReMeDi DEFER、PopiT DEFER。所有 DEFER 模型使用 Attempt09 的同一专家提示和严格 `Expert Final Answer: A-D` 协议。gold 与 hidden atomic facts 没有进入任何消息。
- DEFER 成本使用 Attempt09 固定值 `0.60`，只扣一次；净值定义为正确率减去 0.60。该成本是实验中的归一化动作成本，不是美元费用。未把 Token 费用加入净值。
- 首轮试运行曾错误地用 DEFER parser 解析 Always Answer，造成 0/20 可解析。发现后修正为 direct-answer parser，并在相同病例上完整重跑五个 arm。下表仅使用修复后的结果；首轮统计不纳入结论。
- 区间为 20 个病例上的病例级 10,000 次 percentile bootstrap 95% CI，样本很小，区间离散且较宽。

### 结果

| 策略 | 正确数 / 20（95% CI） | 格式有效率 | 平均总 tokens | DEFER 成本 | 净值 |
|---|---:|---:|---:|---:|---:|
| Qwen3-8B Always Answer | 5/20，25% [5, 45] | 100% | 168.1 | 0 | 0.25 |
| Qwen3-8B Base-DEFER | 5/20，25% [10, 45] | 100% | 242.1 | 0.60 | -0.35 |
| HuatuoGPT-o1-7B DEFER | 0/20，0% [0, 0] | 0% | 360.1 | 0.60 | -0.60 |
| **Qwen3-4B-ReMeDi DEFER** | **6/20，30% [10, 50]** | **100%** | 242.1 | 0.60 | **-0.30** |
| PopiT-Qwen3-4B-Medical-SFT DEFER | 3/20，15% [0, 30] | 100% | 242.1 | 0.60 | -0.45 |

ReMeDi 相对 Always Answer 的配对准确率差为 +5 pp，95% CI [-10, 20]；在 15 个 Always Answer 错误病例中改对 2 例，在 5 个 Always Answer 正确病例中退化 1 例。ReMeDi 相对 Base-DEFER 多答对 1 例，配对差 +5 pp，95% CI [0, 15]；该差异只来自一例 discordant case。PopiT 相对 Always Answer 改对 1 例、退化 3 例。Huatuo 20/20 均未满足 DEFER 答案格式。

以当前成本计算，ReMeDi DEFER 净值 -0.30，低于 Always Answer 的 0.25，动作层面仍不划算。小样本只支持把 ReMeDi 作为下一轮 L2D 实验的候选专家，不支持“专家更可靠”或“DEFER 已有正收益”的结论。此处只评估 DEFER 最终答案；不能把结果外推到 CONSULT 建议质量。

## 模型与运行哈希

数据、模型 revision、配置哈希、模型分片 SHA-256、随机种子及评测代码 SHA 均记录在被 Git 忽略的本地 `outputs/` 运行 manifest 中；下表补充完整权重分片 SHA-256。Base 权重哈希在推理完成后逐分片计算并回填 manifest。

| 模型 | 配置 SHA-256 |
|---|---|
| Qwen3-8B | `f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30` |
| HuatuoGPT-o1-7B | `76f08452a03b3ac80784789ff08ce46fc9dff070dcd67b864a0c4d3c624e466d` |
| Qwen3-4B-ReMeDi | `9f62b27f0bcbdcc16c0fe03044db909602c0637d1e38b71a1f03f8b7ac14717e` |
| PopiT-Qwen3-4B-Medical-SFT | `3795f164618a29784d4451e9f34ae52a8d2f973abd8a649eeb2bf798037e40bd` |

SHA-256（完整模型权重分片）：

| 模型 / 文件 | SHA-256 |
|---|---|
| Qwen3-8B `model-00001-of-00005.safetensors` | `31d6a825ae35f11fb85b195b4c42c146c051e446433125a215336abdf95cbf5f` |
| Qwen3-8B `model-00002-of-00005.safetensors` | `5991236cea6fe21f3d43cab0f0e84448734fbbe0789816202989f2ddc9d18282` |
| Qwen3-8B `model-00003-of-00005.safetensors` | `c5185c4794be2d8a9784d5753c9922db38df478ce11f9ed0b415b7304d896836` |
| Qwen3-8B `model-00004-of-00005.safetensors` | `b5ee7de71fbf17db3d5704e0c8f2bc7d005ca9e1d7ca2aeb19827b0cfcaa917a` |
| Qwen3-8B `model-00005-of-00005.safetensors` | `20c2d6366ab85c90786ccdd829cd2b9e7d30ef3b2ebbb998280e7e4014b542ff` |
| Huatuo `model-00001-of-00004.safetensors` | `786675266ec7a214f7e58fa5645873dfd0ef211c9c8428d30c881fd27cdfd6b5` |
| Huatuo `model-00002-of-00004.safetensors` | `8f7909b83e4a51e40907bb9363328ae6992461abb8896a17cafd9e522ab50f6a` |
| Huatuo `model-00003-of-00004.safetensors` | `50e17982a71539c21f025d49ea426561c02d4f1a7860d93c5e3c9e822725cc38` |
| Huatuo `model-00004-of-00004.safetensors` | `c6c20b8b86e82797154e3f9bbda66ca448b10422cf1e074ebc2dd45f41e0354e` |
| ReMeDi `model-00001-of-00005.safetensors` | `956627d89fe162cfb566b5caec2f5a9acb18bdfe4b1632f012324e47e829180c` |
| ReMeDi `model-00002-of-00005.safetensors` | `922e06e1fd32230bc450e648efa158664292dcb2739f7bfbfa4ec7e781e6a182` |
| ReMeDi `model-00003-of-00005.safetensors` | `d4e8d43976bf5e1a82642f4ee1e2fa4a7ec9ce33e8445cff58b256ddc8cb6b76` |
| ReMeDi `model-00004-of-00005.safetensors` | `34fc73cfc97bcb7638b26e774dd41f8f94e0c3c91dd417f76ad3c7e69566bebb` |
| ReMeDi `model-00005-of-00005.safetensors` | `d3afcc1f995d9efad2a84836b10292d7377857721d24e6c03dfa02dde3176e52` |
| PopiT `model-00001-of-00002.safetensors` | `881ee2025de5697e57f3727f79055be7931ecc1633687089390389f7308fc082` |
| PopiT `model-00002-of-00002.safetensors` | `8d490003c6563790ecbe8b98ff96251f98b311181c6290c2a23b1bf80a0319f4` |

## 文件与复现

- [断点续传与模型哈希清单脚本](download_medical_candidates.py)
- [200 例候选模型配对评测脚本](run_medical_candidate_eval.py)
- [20 例 L2D DEFER 专家协议评测脚本](run_medical_candidate_defer_dev_eval.py)
- 完整私有逐条输出、直接问答中含 gold 的评分轨迹及 manifests 仅保留在本地 `outputs/medical_expert_candidates_20261116/` 与 `outputs/medical_candidate_defer_dev_20261116/`；未纳入 Git。
- 运行时 Git HEAD：`8044d665e28adac4803a48772093b5ae2d246753`；200 例 runner SHA-256：`522f5423a98541aa8eeb01be38e063076738ead2575a1188ad1e3ba4b76b64a0`；修复后的 DEFER runner SHA-256：`836800d14a4f1c28f7a12da82aba346ef539d843e76d79583373471c901503f1`。

复现实验使用 `EA_ATPO/.venv-real-pilot` 环境，先执行模型下载/哈希脚本，再执行：

```bash
python experiments/real_case_pilot_20261103/audit/run_medical_candidate_eval.py --limit-cases 10 --threads 32
python experiments/real_case_pilot_20261103/audit/run_medical_candidate_eval.py --resume --limit-cases 200 --threads 32
python experiments/real_case_pilot_20261103/audit/run_medical_candidate_defer_dev_eval.py --threads 32
```

Smoke 与完整评测共用同一个本地输出目录；复现时若从头运行，应先移走旧 `outputs/medical_expert_candidates_20261116/` 或改用全新目录。评测使用公开模型推理，没有启动 Router、PPO 或 ATPO RL 训练；没有真实医生参与，也没有临床有效性结论。
