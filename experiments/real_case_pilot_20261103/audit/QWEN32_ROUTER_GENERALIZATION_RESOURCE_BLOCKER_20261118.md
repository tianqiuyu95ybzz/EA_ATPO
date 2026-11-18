# Qwen3-32B Router 泛化扩展：资源阻塞历史与实际资源记录

**当前 Attempt:** `qwen32_router_generalization_20261118_attempt04`

**检查时间：** 2026-11-18（Asia/Shanghai）
**状态：** 早期 GPU 阻塞已解除。用户授权 GPU1/2/5 后，Attempt04 使用 GPU1 部署 Qwen3-32B-AWQ、GPU5 部署 Qwen3-8B，完成 smoke 和 1,700 例配对采集；GPU2 未使用。完整结果见 [Attempt04 报告](QWEN32_ROUTER_GENERALIZATION_20261118_RESULTS.md)。

## Attempt04 实际资源使用

- 启动前：GPU1 空闲 108,823 MiB、利用率 0%；GPU5 空闲 182,625 MiB、利用率 0%。
- 50 次 GPU telemetry 采样中的峰值：GPU1 已用 118,862 MiB（最低空闲 63,763 MiB），GPU5 已用 27,238 MiB（最低空闲 155,387 MiB）。
- 32B 服务为 AWQ、`gpu_memory_utilization=0.25`；8B 为 BF16、`gpu_memory_utilization=0.12`。两项服务已由本次运行器关闭；未停止或改动其他进程。
- Attempt01 曾因 ZMQ IPC socket 路径过长而在推理前失败；Attempt02 在 `TMPDIR` 使用短路径修复后正常部署。此次实验无 OOM。

以下内容保留当时资源尚未满足门槛的历史记录，不表示当前仍阻塞。

## 已完成

- 扫描 EA_ATPO 与 EA_ATPO_V1 的 84 个历史 JSONL、35,684 条记录及历史病例选择清单。
- 排除历史病例 ID、prompt/case hashes、source row index 和 Attempt09 开发集病例。
- Attempt02 在首轮历史清单中有 34 个官方 test-case 重叠；Attempt03 完整纳入旧 `sample_selection_manifest.json` 后，官方 test split 仅剩 8 个未暴露病例。两者均未运行推理。
- Attempt04 冻结 1,000 train /300 validation /400 test-source holdout，全部来自官方 train split，按 `case_id` 和 hash 切分。官方 test parquet 不能再提供预注册所需的 400 例。
- 清单在共享 PFS：`outputs/qwen32_router_generalization_20261118_attempt04/case_selection.json`；SHA256 `11d95c40d8c34bcbeae4dd8c18698fcee726f22db77b7f718d457d9527bc0cfa`。历史排除 cache 也仅保存在 PFS，不推送病例行号或哈希。
- 新模型服务没有启动；没有读取本次测试集标签，也没有新增 Base/Expert 输出或 Router 性能结果。

## GPU 安全门槛与最近快照

2026-11-18 03:21 UTC 的一次性启动门槛检查中，GPU3 仅有 56,783 MiB（55.5 GiB）空闲且利用率 100%，低于 81,920 MiB / 25% 的 32B Expert 门槛。启动脚本立即退出，明确记录 `No service started`；GPU2 没有在这次尝试中被查询，最近先前快照为 111.8 GiB 空闲、100% 利用率。没有重试其他 GPU。

上一轮单次资源检查发现：

| GPU | 可用显存 | 利用率 | 判定 |
|---:|---:|---:|---|
| 2 | 111.8 GiB | 100% | 不启动 |
| 3 | 56.9 GiB | 100% | 不启动 |
| 6 | 56.4 GiB | 100% | 不启动 |
| 7 | 56.3 GiB | 100% | 不启动 |

GPU 0/4/5 当时也被高占用任务使用；GPU1 遵循先前“不使用 GPU1/7”的约束（之后用户曾允许 GPU7，但本次快照中 GPU7 仍满载）。没有停止或干扰他人进程，也没有启动模型服务。

上次同机双服务实验的实测峰值约 111 GiB。为保留约 60 GiB 余量，不在同一张卡上同时加载两个模型，除非空闲显存至少约 130 GiB。优先分别部署时，后续脚本应保留：

- Qwen3-32B-AWQ 卡：启动前至少 80 GiB 空闲显存，GPU 利用率不高于 25%。
- Qwen3-8B Base 卡：启动前至少 45 GiB 空闲显存，GPU 利用率不高于 25%。
- 继续固定模型 revision、vLLM 版本、模型量化、提示词、解析器和生成参数。不要用减小模型、CPU offload 或改量化来规避门槛。

以上是按已锁定的 AWQ/同版本模型服务显存配额设置的保守启动门槛，并不代表当前有符合条件的 GPU。每次实际启动前仍需单次检查显存、利用率和服务端口；不循环轮询，也不抢占已有任务。

## 影响

当前无法生成新队列的配对 Base/Expert 结果，所以不能诚实报告训练集与测试集的特征分布、专家优势标签分布、模型预测误差、AUROC、成本曲线、配对置信区间或病例级策略结果。也不能据此判断新 Router 是否学到条件专家优势。历史 200 例和旧 Router 的结果只作为历史背景，不可替代本次冻结队列。Attempt04 的测试集是 fresh train-source holdout；未来报告需要明确这个限制。

符合门槛的 GPU 出现后，应从冻结清单恢复配对推理，先完成 20 例协议 smoke；通过后运行其余训练、验证、测试病例。测试结果只能在模型与验证阈值全部锁定后读取一次。
