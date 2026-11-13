# Qwen3-32B-AWQ 双卡启动验证

## 结果

2026-11-14 在物理 GPU 4、6 上以 vLLM 0.33.0、Tensor Parallel=2 启动 Qwen3-32B-AWQ。两张卡各 183,359 MiB；首次预检各约有 112 GiB 空闲。模型 API 最终完成请求，返回 READY（prompt 19 tokens、completion 2 tokens），随后服务已关闭。

这说明当前 109 GiB 级别的单卡空闲显存足以承载这份 AWQ 推理服务，也说明模型可以分配到两张 GPU。服务设置 gpu_memory_utilization=0.20；加载日志显示模型权重约占每卡 9.15 GiB。采样到的整卡峰值包括同卡既有任务占用，约为 GPU4 108,466 MiB、GPU6 108,774 MiB；停服后空闲显存恢复到约 112 GiB。没有停止或修改其他进程。GPU 1、7 未被用于本次启动。

## 启动问题和修复

1. 第一次在模型加载前因 TMPDIR 太长，导致 ZMQ IPC socket 超过 107 字符限制。改为 /tmp/eaatpo32b。
2. 第二次权重加载后，FlashInfer 编译 trtllm_mnnvl_comm 失败，原因是 CUDA 编译器与其 CUDA 头文件不兼容。
3. 仅关闭普通 FlashInfer all-reduce 后，vLLM 的 fuse_allreduce_rms 仍尝试编译同一个模块。
4. 显式设置 pass_config.fuse_allreduce_rms=false，并让 TP 使用 PYNCCL，最终服务启动并完成生成。

完整日志保留了以上所有尝试。复现启动和停止命令分别见 start_tp2_service.sh 与 stop_tp2_service.sh。

## 边界

这是 32B AWQ 模型的推理部署验证，不是 BF16 部署结果，也不是 EA-ATPO 的 Actor-Critic 训练结果。多卡训练还需同时容纳 Actor、Critic、优化器状态、rollout 服务和 KV cache；不能仅凭这次推理启动断定完整训练所需显存。此前 Qwen3-8B 原始 ATPO 双卡训练 smoke 已另行通过，EA 四动作 Actor/树搜索尚未由这次服务测试验证。
