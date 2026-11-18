# Attempt04 复现实验说明

本说明对应 `qwen32_router_generalization_20261118_attempt04`。Attempt02/03 在模型调用前因历史病例污染/官方 test pool 耗尽而作废。本轮冻结 1,000/300/400 个未暴露病例，均来自 ATPO train split，测试组是新 case_id holdout，不是官方 test split。目前受 GPU 安全门槛限制，真实模型推理、Router 拟合和测试集分析尚未运行。不要把代码或锁定的 cohort 描述当作性能结果。

## 固定设置

- Expert：Qwen3-32B-AWQ，HF revision `0499c3ac83fdef8810b907a23894ba91e95eddd8`，AWQ。
- Base：Qwen3-8B，HF revision `b968826d9c46dd6066d109eabc6255188de91218`。
- 原有 Attempt01 的 prompt 构造、DEFER parser 和请求格式；temperature 0、top-p 1、禁用 thinking；Base 64 tokens、Expert 128 tokens；Base 选项 logprobs top-20。
- 服务使用 vLLM OpenAI-compatible API，每模型独占一张不同 GPU；Expert GPU 启动前至少 81,920 MiB 空闲，Base 至少 46,080 MiB 空闲，利用率不高于 25%。脚本只检查一次并立即退出，不等待资源，也不抢占其他进程。
- 容许的 GPU 参数为此前确认的 `{0,2,3,4}`。同一张 GPU 不得同时放两个模型。

## 执行顺序

1. 确认固定队列存在且尚无 Attempt04 推理输出：`outputs/qwen32_router_generalization_20261118_attempt04/case_selection.json`。
2. 选两张符合启动门槛的卡，例如 3 和 2。仅当两个设备都通过门槛时启动：

   ```bash
   bash experiments/real_case_pilot_20261103/audit/run_qwen32_router_generalization.sh \
     --expert-gpu 3 --base-gpu 2
   ```

   脚本先启动服务并跑独立的 20 例协议 smoke；只有 smoke 完整且 Expert 解析失败为 0 才开始 1,000/300/400 配对采集。结束后关闭本次创建的服务，再运行拟合器和一次性测试分析。

3. 若推理中断，先审计 partial JSONL 和 service logs，然后在新空闲 GPU 上使用 `--resume`。脚本会保留已完成的 service attempt、为恢复创建新的服务目录，并从未完成 response 继续：

   ```bash
   bash experiments/real_case_pilot_20261103/audit/run_qwen32_router_generalization.sh \
     --expert-gpu 3 --base-gpu 2 --resume
   ```

   重启模型服务前仍须单次通过显存和利用率门槛。`--resume` 仅复用未完成病例；不重跑已完成 response。完成模型采集后 shell 会自动执行拟合与一次性分析。

4. 配对 rollout 全部完成后，锁定策略：

   ```bash
   /pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/.venv-real-pilot/bin/python \
     experiments/real_case_pilot_20261103/audit/fit_qwen32_router_generalization.py
   ```

   该脚本只读取 train/validation JSONL。它创建 `router_generalization/policy_lock.json`，测试集不会被打开。

5. 确认 policy lock 和数据 hash 后，仅运行一次独立测试分析：

   ```bash
   /pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/.venv-real-pilot/bin/python \
     experiments/real_case_pilot_20261103/audit/analyze_qwen32_router_generalization.py
   ```

   分析脚本拒绝覆盖已有最终结果。它输出策略成本表、配对病例表、bootstrap 区间、类别/特征分布、模型预测误差和最终 Markdown 报告。

## 输出与保护

所有大文件、原始 response、服务日志、病例 ID 和病例级 CSV 留在共享 PFS 的 Attempt04 `outputs/` 目录，不推送到 GitHub。仓库提交仅包含代码、预注册和不含病例 ID/hash 的状态报告。原始 prompt、Gold、hidden atomic facts 不写入模型请求日志或轨迹；配对标签只由本地评估器在两路模型回答后计算。

若安全门槛失败，脚本在启动任一服务前退出；不改变模型、量化、提示词、parser、生成长度、GPU 显存配额或测试阈值来绕过失败。
