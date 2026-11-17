# L2D 专家候选独立确认：ReMeDi 未通过

日期：2026-11-17

## 当前结论

此前在 20 例开发集上暂选的 Qwen3-4B-ReMeDi，在新抽取的 100 例上没有复现优势。Always Answer 正确 54/100；Qwen3-8B 按 DEFER 专家协议作答正确 53/100；ReMeDi-DEFER 正确 40/100。ReMeDi 相对 Always Answer 低 **14 个百分点**（病例配对 bootstrap 95% CI：**−23 至 −5 pp**），而且在 Base 已答对的 54 例中有 19 例退化。按 Attempt09 的 DEFER 成本 0.60 计算，ReMeDi 净值为 −0.20，低于 Always Answer 的 0.54。

因此撤销“暂选 ReMeDi”的阶段性结论。当前没有通过这组实验验证的 DEFER Expert，**不启动 Router、PPO 或 ATPO RL**。Qwen3-32B-AWQ 的既有直接答题结果略高于 8B，但配对区间包含 0；本轮新鲜 100 例上的 32B-DEFER 确认因 GPU 资源变化未完成，不能据此把 32B 宣称为合格 Expert。

## 实验设置

- 数据：ATPO `test.pilot.parquet` 1,268 行，SHA-256 `3783ebd70d2282ab31c19f5030bf122301d34465516dee222438835e57c9c051`。
- 样本：seed `20261117`，100 个唯一公开状态；排除先前固定 200 例的源行，并按公开状态哈希检查 Attempt09 20 例开发状态。与前一组 200 例源行无交集；Attempt09 状态重叠为 0。
- 配对臂：Qwen3-8B Always Answer；同一 Qwen3-8B 使用 DEFER 专家协议；Qwen3-4B-ReMeDi 使用同一 DEFER 专家协议。所有 DEFER 调用均为每例一次。
- 消息：各臂使用相同病例公开信息。DEFER 两臂使用完全相同的专家任务协议。Gold 与 hidden atomic facts 只在生成后评分，且不写入输出；推理前运行了可见信息泄漏检查。
- 生成：greedy，BF16，最多 128 token，Qwen3 thinking 关闭；Transformers 4.54.1、PyTorch 2.7.1+cu128、单卡 GPU6（NVIDIA B20Z）。三臂各 100/100 完成，格式有效率均为 100%，300 条请求均正常结束。
- 统计：报告准确率的病例级 bootstrap 95% CI；配对差用病例正确性差值 bootstrap。无效格式保留在分母。本试验的 DEFER 成本是归一化效用单位，不是货币费用。

## 结果

| 策略 | 正确数 | 准确率（95% CI） | 平均总 tokens | 平均生成延迟 | DEFER 净值（成本 0.60） |
|---|---:|---:|---:|---:|---:|
| Qwen3-8B Always Answer | 54/100 | 54% [44%, 64%] | 169.48 | 0.167 秒 | 0.54 |
| Qwen3-8B Base-DEFER | 53/100 | 53% [43%, 62%] | 243.60 | 0.201 秒 | −0.07 |
| Qwen3-4B-ReMeDi DEFER | 40/100 | 40% [31%, 50%] | 243.60 | 0.213 秒 | −0.20 |

配对结果：

| 对比（行=Always Answer，列=候选） | 两者正确 | Base 对、候选错 | Base 错、候选对 | 两者错误 | 配对准确率差（95% CI） |
|---|---:|---:|---:|---:|---:|
| Qwen3-8B Base-DEFER | 51 | 3 | 2 | 44 | −1 pp [−5, 3] |
| ReMeDi-DEFER | 35 | 19 | 5 | 41 | −14 pp [−23, −5] |

ReMeDi 与 Base-DEFER 直接配对时，两者都正确 36 例、Base-DEFER 对而 ReMeDi 错 17 例、Base-DEFER 错而 ReMeDi 对 4 例、两者都错 43 例；配对差 **−13 pp**（95% CI：**−22 至 −5 pp**）。

在 Always Answer 错误的 46 例中，ReMeDi 只改对 5 例（10.9%，95% CI：2.2%–21.7%）；在 Always Answer 正确的 54 例中，ReMeDi 改错 19 例（35.2%，95% CI：22.2%–48.1%）。修正/退化比为 5/19。Qwen3-8B Base-DEFER 只修正 2 例并退化 3 例，准确率差与零无法区分；扣除动作成本后仍低于 Always Answer。

这表明本次低 DEFER 收益不是格式失败造成的：所有输出都被严格解析。ReMeDi 在这组病例中既没有更高的总体准确率，也没有足够的条件纠错率抵消其错误退化和动作成本。

## Qwen3-32B-AWQ 后续尝试

Qwen3-32B-AWQ 官方量化权重在本机完整，revision 为 `0499c3ac83fdef8810b907a23894ba91e95eddd8`，原始来源和分片校验见 `assets/models/Qwen3-32B-AWQ/model_source_lock.json`。已有 200 例直接答题评估为 48% 对 8B 的 44%，配对差 +4 pp（95% CI：−2.5 至 +10.5 pp）；该结果不是 DEFER 条件可靠性证据。

为进一步筛选，在新建的 seed `20261118` 样本上准备了另一组 100 个与先前 200 例及 ReMeDi 100 例不重叠的公开状态；其中有 1 个 Attempt09 状态哈希与测试集重合，已排除。最终 100 个状态唯一，样本哈希为 `5255522035eae49ce84b08d33ec9422eef2604615dcf4c364adbbc07f6c42274`。启动 vLLM 时，另一项 GPU6 作业随后占用约 112 GiB；vLLM 加载了 18.24 GiB 权重后报告 `No available memory for the cache blocks`，没有进入 API ready，也没有生成任何 32B 病例结果。该服务已退出，没有保留后台推理进程；候选筛选脚本和启动日志保留在本机输出目录。日志 SHA-256：`d88147a0667bdf3f1fdbd99299652edd06e3b917fbf608caaf1f4444e601c24d`。不能把此次部署失败记作模型性能失败。

本轮启动时 GPU6 曾有约 178 GiB 空闲，三臂 100 例实验的峰值 PyTorch allocated memory 约 15.4 GiB。32B 服务尝试期间资源状况改变；当前不在剩余约 70 GiB 空闲的 GPU6 上叠加新模型。

## 复现与文件

- 完整运行命令：

  ```bash
  CUDA_VISIBLE_DEVICES=6 .venv/bin/python -u \
    experiments/real_case_pilot_20261103/audit/run_l2d_remedi_confirmation.py \
    --gpu-index 6 --limit-cases 100 \
    --output-dir experiments/real_case_pilot_20261103/audit/outputs/l2d_remedi_confirmation_20261117_gpu6_full
  ```

- 运行时 Git commit：`c52db83dfb9428e1404733b8fe6bb27fd5550315`；runner SHA-256：`a527fbfea99c3d67a35f233174dcf7b94cd30885de0cc5dcae7bbca9a3dc6c56`。
- 选中公开状态哈希：`872b407665ce5b2ec772c2ad8b749f600805ddaf6ab9d9e55e5a9c4c72781a68`；源行索引哈希：`e4735d67f225f233b58933da58658a8da50eeeed07a12e7fa2c709ab4171817a`。
- Qwen3-8B config SHA-256：`f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30`。
- ReMeDi config SHA-256：`9f62b27f0bcbdcc16c0fe03044db909602c0637d1e38b71a1f03f8b7ac14717e`；固定 revision：`dfa418ca1c3c0f8879afd6d356b88083f01ca032`。
- 原始逐条轨迹、CSV 与 run manifest 留在 PFS 的忽略目录 `outputs/l2d_remedi_confirmation_20261117_gpu6_full/`；轨迹只含病例哈希、预测字母、正确性、token、延迟和解析状态，不含 prompt、Gold、hidden facts 或原始模型回复。哈希：轨迹 `99844a25ae957697471e7e2d48ad2b16ff4b0261b783c6d91ddcd89b012bdfd9`；汇总 `11458cde9a811cec26d113deb3395a23eba983aad20bd7f829baccafff21c6ad`；配对表 `175887082bd25bff57f3b0f81aab18d2dc4f50937a47e1ff372302efbdbe1945`。
- 32B 独立样本筛选 runner：`run_l2d_expert_screen_32b.py`；本轮单卡服务启动脚本：`start_l2d_32b_awq_service.sh`。32B 服务日志仅记录部署信息与启动失败，没有逐病例结果。

本实验是公开选择题上的 LLM 模拟 Expert 评估，不是真实病历验证，也没有真实医生参与。
