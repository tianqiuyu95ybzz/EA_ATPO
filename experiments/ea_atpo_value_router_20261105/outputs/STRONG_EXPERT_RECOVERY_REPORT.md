# Qwen3-32B 强 Expert 恢复与条件可靠性评估

**状态：** AWQ 权重分片已逐片校验并通过 vLLM 部署；固定 200 例配对评估完成。

## 固定候选与环境

| 项目 | 值 |
|---|---|
| 模型候选 | [`Qwen/Qwen3-32B-AWQ`](https://huggingface.co/Qwen/Qwen3-32B-AWQ) |
| 固定 revision | `0499c3ac83fdef8810b907a23894ba91e95eddd8` |
| 权重来源 | `modelscope`；source revision `c14568581d436715d61d40f471dee3cac433d1d9` |
| 模型格式 | 官方 AWQ 量化，未将它描述为 BF16 结果 |
| 推理框架 | vLLM 0.33.0，服务配置使用 `--quantization awq` |
| 推理 GPU 检查 | GPU 2: NVIDIA B20Z, 183359 MiB, 138203 MiB（服务启动前空闲 138203 MiB）（显存空闲量随其他任务变化） |
| 文件恢复状态 | verified 4/4 shards (19,325,481,744/19,325,481,744 bytes) |
| 权重目录 | `experiments/real_case_pilot_20261103/assets/models/Qwen3-32B-AWQ/`（项目 PFS） |

Hugging Face TLS 端点在当前服务器不可达。Qwen 官方 GitHub 指向其 ModelScope 组织；ModelScope 仓库 `Qwen/Qwen3-32B-AWQ` 的固定 revision 已核对，四个分片 size/SHA-256 与 Hugging Face 固定 revision 的 LFS 清单完全一致，配置、index、tokenizer 和词表文件也逐字节匹配已落盘 HF 文件。下载器使用 ModelScope Range 接口续传，同时仍以 HF LFS SHA-256 作为权重校验基准。官方 [AWQ model card](https://huggingface.co/Qwen/Qwen3-32B-AWQ) 提供 vLLM 推理配置；BF16 下载不完整，官方 [Qwen3-32B-FP8](https://huggingface.co/Qwen/Qwen3-32B-FP8) 未下载。

Tokenizer JSON 与 Qwen3-8B 官方固定 revision 中的文件具有相同 SHA-256 `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`，按字节哈希验证后在 PFS 内硬链接复用，避免重复下载。模型 index、tokenizer config、vocab 和其它元数据来自 AWQ 固定 revision。

## 冻结病例上的条件可靠性

评估沿用此前固定的 200 个独立 test 病例（seed `20261104`），逐病例检查 public prompt SHA 和 Gold 与保存的 Qwen3-8B baseline 一致。两模型使用同一公开题干、提示、温度 0、选项解析；Gold 只供事后评分，hidden atomic facts 不进入请求。无效输出按错误保留。Expert 是 LLM 模拟器，不是真实医生。

| 指标 | Qwen3-8B Base | Qwen3-32B AWQ |
|---|---:|---:|
| 总体准确率（病例级 95% CI） | 0.440（95% CI 0.370–0.510） | 0.480（95% CI 0.410–0.550） |
| 有效输出率 | 1.000 | 1.000 |
| 平均 total tokens | 170.320 | 170.320 |
| 平均 latency 秒 | 0.065 | 0.143 |

配对准确率差（32B−8B）：0.040（95% CI -0.025–0.105）。
32B 在 8B 错误病例上的条件准确率：0.232（95% CI 0.157–0.314）；修正比例：0.232（95% CI 0.157–0.314）。
32B 破坏 8B 正确答案的比例：0.205（95% CI 0.124–0.293）。

完整配对四格表见 `outputs/strong_expert_awq/paired_correctness.csv`。

## 文件和当前限制

- `outputs/strong_expert_awq/qwen3_32b_direct_answers.jsonl`：候选模型逐病例轨迹（评估完成后）。
- `outputs/strong_expert_awq/strong_expert_summary.csv`、`paired_correctness.csv`：配对结果与病例级 bootstrap。
- `outputs/strong_expert_awq/run_manifest.json`、`model_service_manifest.json`：模型、数据哈希、revision、GPU 与推理参数。
- `experiments/real_case_pilot_20261103/assets/models/Qwen3-32B-AWQ/model_source_lock.json`：ModelScope 固定 revision 与 HF LFS SHA-256 对照。
- `recompute_strong_expert_statistics.py`：从原始配对轨迹离线重算条件指标和 bootstrap。
- 本机共享 GPU 显存和网络吞吐会影响完成时间；所有分片和缓存位于项目工作区。模型是 LLM Expert simulator，不是临床医生。

