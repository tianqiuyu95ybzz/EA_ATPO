# Attempt04 复现实验说明

本说明对应 `qwen32_router_generalization_20261118_attempt04`。Attempt02/03 在模型调用前因历史病例重叠/官方 test pool 耗尽而作废。Attempt04 已完成 20 例 smoke、1,700 例 Base/Expert 配对推理、Router 拟合和一次性 400 例测试分析。三组来自 ATPO train split 的新 case_id holdout，test 不是官方 ATPO test split。结果见 [最终报告](QWEN32_ROUTER_GENERALIZATION_20261118_RESULTS.md)。

## 固定设置

- Expert：Qwen3-32B-AWQ，HF revision `0499c3ac83fdef8810b907a23894ba91e95eddd8`，AWQ。
- Base：Qwen3-8B，HF revision `b968826d9c46dd6066d109eabc6255188de91218`。
- 原有 Attempt01 的 prompt 构造、DEFER parser 和请求格式；temperature 0、top-p 1、禁用 thinking；Base 64 tokens、Expert 128 tokens；Base 选项 logprobs top-20。
- 服务使用 vLLM OpenAI-compatible API，每模型独占一张不同 GPU；Expert GPU 启动前至少 81,920 MiB 空闲，Base 至少 46,080 MiB 空闲，利用率不高于 25%。脚本只检查一次并立即退出，不等待资源，也不抢占其他进程。
- 按用户授权可用 GPU `{1,2,5}`；实际 Expert 使用 GPU1、Base 使用 GPU5，GPU2 未使用。同一张 GPU 不放两个模型。

## 本次执行记录

1. Attempt04 固定队列 `outputs/qwen32_router_generalization_20261118_attempt04/case_selection.json`，SHA256 `11d95c40d8c34bcbeae4dd8c18698fcee726f22db77b7f718d457d9527bc0cfa`。清单仅留在 PFS。
2. 实际使用的命令：

   ```bash
   bash experiments/real_case_pilot_20261103/audit/run_qwen32_router_generalization.sh \
     --expert-gpu 1 --base-gpu 5 --resume
   ```

   第一次服务启动尝试因 ZMQ IPC socket 路径过长退出，没有模型推理；已将服务 `TMPDIR` 改为短符号链接，再以 services/attempt02 重启。GPU 启动门槛通过后，脚本先完成 20 例协议 smoke（两模型解析失败均为 0），再完成 1,700 例采集；服务已由脚本关闭。

3. 训练与验证 Router 阈值锁定后，fit 脚本只读 train/validation。随后 test analysis 读取 test 一次并写出 CSV/manifest。首次分析在 Markdown 组装阶段因 `auc_by_model` 初始化顺序错误退出，但指标 CSV 和 `analysis_manifest.json` 均已写好。已修复源码；报告 finalizer 读取已生成汇总 CSV，并读取 raw trajectories 中的 `valid`、`parse_error`、`finish_reason`、`raw_response` 字段来归类格式失败；它不读取 Gold/正确性标签来重新计算 test 指标，也不重算 bootstrap：

   报告生成命令：

   ```bash
   /pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/.venv-real-pilot/bin/python \
     experiments/real_case_pilot_20261103/audit/finalize_qwen32_router_generalization_report.py
   ```

   finalizer 不重新拟合、不调阈值、不重算准确率或测试区间。若在全新副本从头复现实验，仍按运行器的 smoke→rollout→fit→一次性 analyze 流程执行；当前 Attempt04 outputs 已存在，禁止在原目录重跑测试或覆盖结果。

## 输出与保护

所有大文件、原始 response、服务日志、病例 ID 和病例级 CSV 留在共享 PFS 的 Attempt04 `outputs/` 目录，不推送到 GitHub。仓库提交仅包含代码、预注册和不含病例 ID/hash 的状态报告。原始 prompt、Gold、hidden atomic facts 不写入模型请求日志或轨迹；配对标签只由本地评估器在两路模型回答后计算。

## 已完成的输出

- PFS Attempt04：`outputs/qwen32_router_generalization_20261118_attempt04/`，包括原始 responses、病例级配对 CSV、policy lock、模型与 GPU telemetry。
- `paired_rollouts/run_manifest.json`：1,700 行、train/validation/test 各自数据哈希、模型 revision、runner hash 和 invalid 统计。
- `analysis/analysis_manifest.json`：policy/test SHA256、bootstrap 次数与 seed，记录 test 在 policy lock 后一次性分析。
- `analysis/parser_failure_diagnostics.csv`：仅记录解析失败类别、split、finish_reason、计数，不含模型原文。
- 仓库中的最终报告和复现脚本不含病例 ID、病例文本、Gold 或 hidden facts。详细 source hashes 见 Attempt04 PFS 的 execution provenance 记录。

若安全门槛失败，脚本在启动任一服务前退出；不改变模型、量化、提示词、parser、生成长度、GPU 显存配额或测试阈值来绕过失败。
