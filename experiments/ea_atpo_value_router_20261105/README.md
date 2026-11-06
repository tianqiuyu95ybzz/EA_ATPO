# EA-ATPO 四动作价值 Router（2026-11-05）

本目录实现了基于 Pilot 反事实结果的轻量动作价值模型和独立病例实际交互评估。它不启动 P5 Router RL 或 P8 联合 ATPO RL。

## 方法边界

- 训练源是 `real_case_pilot_20261103/outputs/full/pilot_trajectories.jsonl`：100 个病例、291 个公开状态、每状态/动作 3 条 rollout。
- `case_id` 做 70/15/15 train/validation/internal-test 切分。TF-IDF 词表和归一化统计只在当前训练病例拟合。专家可靠性特征是基于 DEFER 正确率的 5 折病例交叉拟合值。
- Router 的可用输入只有决策前 `public_state` 中 user/assistant 可见文本及其长度、轮数等结构统计。Gold、atomic facts、状态构造轨迹、Pilot rollout 正确性、未来患者回复和专家建议均不进入 Router。
- 每个动作分别用线性核岭回归预测 `Q_out`、完整分支患者模拟器轮次、CONSULT 专家 token、DEFER 专家 token。路由效用为 `Q_out − λASK·branch_patient_turns − λCONSULT·branch_consult_tokens/1000 − λDEFER·branch_defer_tokens/1000`，每个资源分量只扣一次。
- 无效 rollout 按错误结果 0 保留。成本缺失的调用在成本监督中用当前训练切分内同动作有效调用均值补齐；训练和运行 manifest 会记录缺失数量。
- 独立 test 来自 ATPO 官方 test split，避开之前固定强专家评估的 200 个 source row。逐轮主评估在每次公开状态更新后重新路由，最多 10 次决策、每 episode 最多 1 次 CONSULT；Base/Expert temperature 0.2，Patient Simulator temperature 0.0。每条请求种子由可见请求内容和温度的 SHA-256 派生。
- 当前部署的 Base、Patient Simulator、Expert Simulator 均为 Qwen3-8B。Qwen3-32B 量化模型直接答案评估独立进行；没有强专家结果时，不把 8B 模拟器称作 32B 或医生。

## 运行顺序

```bash
experiments/ea_atpo_value_router_20261105/run_training.sh
experiments/ea_atpo_value_router_20261105/run_full_router_eval.sh
experiments/ea_atpo_value_router_20261105/run_stepwise_router_eval.sh
experiments/ea_atpo_value_router_20261105/continue_awq_expert_after_download.sh
```

如需手动恢复基础模型服务：

```bash
EA_ATPO_ROUTER_GPU=2 experiments/ea_atpo_value_router_20261105/start_router_model_service.sh
```

正式交互脚本以 `--resume` 支持从已落盘病例/动作继续。服务器启动时需要 Qwen3-8B 权重和一个至少有 35 GiB 空闲显存的 GPU；默认使用物理 GPU 2 和端口 30131，可用 `EA_ATPO_ROUTER_GPU`、`EA_ATPO_ROUTER_PORT` 覆盖。

## 强 Expert 量化候选

恢复官方 AWQ（固定 Hugging Face revision 与分片哈希）：

```bash
experiments/ea_atpo_value_router_20261105/download_official_qwen3_32b_awq.sh
```

Hugging Face TLS 端点在本服务器不可达，因此下载器从 Qwen 官方 ModelScope 组织恢复。ModelScope 仓库 `Qwen/Qwen3-32B-AWQ` 固定到 revision `c14568581d436715d61d40f471dee3cac433d1d9`；四个权重分片的 SHA-256 与 HF revision `0499c3ac83fdef8810b907a23894ba91e95eddd8` 的 LFS 哈希完全一致，metadata 也与已经从 HF 获取的文件逐字节匹配。`model_source_lock.json` 保存来源和哈希锁，Range 下载可续传。当前已完成 vLLM 0.33.0 部署及冻结 200 病例配对评估：32B AWQ accuracy 48.0%，8B Base 44.0%，配对差 +4.0pp（95% CI −2.5 至 +10.5pp）；32B 在 Base 错误病例上的条件准确率为 23.2%，在 Base 正确病例上的退化率为 20.5%。条件指标已根据原始病例配对重算，报告列出完整病例级 bootstrap 区间。

需要重跑时使用 `EA_ATPO_EXPERT_VARIANT=AWQ experiments/ea_atpo_value_router_20261105/run_quantized_expert_eval.sh`。若改用 FP8，需先下载官方 `Qwen/Qwen3-32B-FP8` 固定 revision `aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df`，再将环境变量设为 `FP8`。候选回答与保存的 8B 结果使用同一组 200 个病例、同一提示、同一解析规则，结果写入精度独立的目录。

## 主要文件

- `train_router.py`、`router_core.py`：按病例切分、特征拟合、交叉拟合可靠性、价值模型和 checkpoint。
- `run_router_eval.py`：独立 test 上真实动作调用；JSONL 支持断点续跑。
- `summarize_router_eval.py`：策略汇总、病例级 paired bootstrap、Accuracy–Cost 图表和报告。
- `run_stepwise_router_eval.py`：每次 ASK/CONSULT 更新公开状态后重新运行 Router，记录逐轮动作；这是 Router 交互价值的主评估。
- `summarize_stepwise_router_eval.py`：逐轮评估汇总、与多个基线的配对 bootstrap 和 Accuracy–Cost 图。

`outputs/ROUTER_EXPERIMENT_REPORT.md` 是首动作诊断：初始 action 由策略选择，后续 continuation 仍由 Base Agent 选择。不要将它解释为逐轮 Router 评估。逐轮主结果见 `outputs/stepwise_test_v2/STEPWISE_ROUTER_EXPERIMENT_REPORT.md`。`outputs/stepwise_test/` 保留了被协议错误中止的部分调试记录，不参与汇总。
- `run_quantized_expert_eval.py`：32B FP8/AWQ 候选与冻结 8B 基线配对比较。
- `recompute_strong_expert_statistics.py`：从已保存 200 对原始轨迹重算配对条件指标、病例级 bootstrap 和报告，不调用模型。
- `continue_awq_expert_after_download.sh`：等待逐轮 Router 评估与 AWQ 权重校验完成后，自动启动 200 例强 Expert 对照，并在完成后释放其服务。
- `write_strong_expert_recovery_report.py`：按分片校验/评估状态更新恢复报告。
- `reproducibility_manifest.json`：Git commit、数据/checkpoint/配置/轨迹的 SHA-256 与模型 revision；强 Expert 完成后自动刷新。
- `config.yaml`、`model_service_config.yaml`：实验和服务配置。
- `artifacts/checkpoint/`：正式 Learned Value Router；`artifacts/checkpoint_no_reliability/` 是可靠性特征消融模型。
- `outputs/`：交互轨迹、策略汇总、置信区间、测试样本 manifest、运行日志和报告。

评估集准确率差异以病例为单位 bootstrap。强专家是模型模拟器；公开 ATPO 题目不代表真实临床病历；项目当前没有医生参与。
