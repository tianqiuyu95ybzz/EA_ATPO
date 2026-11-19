# Phase 2 服务诊断

**检查时间：** 2026-11-18 22:18 Asia/Shanghai  
**仓库：** `EA_ATPO_V1`，branch `ea-atpo-v1`，HEAD `2aebfa8ce33d`  
**结论：** Base 与 Expert 服务均未运行；本轮没有启动服务，也没有发送推理请求。Phase 2 Pilot 暂不具备启动条件。

> 历史快照：本诊断记录的是 22:18 的启动前状态，之后已由 `SERVICE_RECOVERY_REPORT.md` 更新。当前服务在 GPU 2 运行，真实服务 Smoke 结果和尚未通过的 Patient simulator 事实覆盖问题见 `SMOKE_TEST_REPORT.md`。

## 配置端点与连接状态

冻结的 `pilot_config.json` 配置如下：

| 角色 | 配置地址 | served model alias | 当前状态 |
|---|---|---|---|
| Base | `127.0.0.1:30131/v1` | `qwen3-8b-audit` | `GET /v1/models` 连接被拒绝 |
| Patient simulator | `127.0.0.1:30131/v1` | `qwen3-8b-audit` | `GET /v1/models` 连接被拒绝 |
| Expert | `127.0.0.1:30132/v1` | `qwen3-32b-candidate` | `GET /v1/models` 连接被拒绝 |

`ss` 在 30000、30131、30132、30146、30147、30156、30157、30256、30257 等配置及历史端口上均未发现监听。对这些地址的只读 `/v1/models` 探测全部返回 `Connection refused`。进程检查没有找到 vLLM API server 或 SGLang inference server。最近一次 Router 泛化实验的服务 manifest 记录了 30256（Qwen3-32B-AWQ）和 30257（Qwen3-8B），但这两个端口当前也不可达；历史 manifest 不能证明服务仍在线。

运行器的最新 preflight manifest 记录：`services_ready=false`、`model_inference_started=false`。没有请求 `/v1/chat/completions`。

## GPU 与启动门槛

`nvidia-smi` 在 preflight 时记录的本机快照（显存单位 MiB）：

| GPU | 型号 | 总显存 | 已用 | 空闲 | 利用率 |
|---:|---|---:|---:|---:|---:|
| 0 | NVIDIA B20Z | 183359 | 138258 | 45101 | 100% |
| 1 | NVIDIA B20Z | 183359 | 74386 | 108973 | 0% |
| 2 | NVIDIA B20Z | 183359 | 0 | 183359 | 0% |
| 3 | NVIDIA B20Z | 183359 | 37194 | 146165 | 100% |
| 4 | NVIDIA B20Z | 183359 | 114000 | 69359 | 100% |
| 5 | NVIDIA B20Z | 183359 | 142212 | 41147 | 100% |
| 6 | NVIDIA B20Z | 183359 | 158320 | 25039 | 100% |
| 7 | NVIDIA B20Z | 183359 | 148444 | 34915 | 100% |

Phase 2 runner 的通用 GPU gate 通过，是因为 GPU 1 满足“空闲至少 24 GiB 且利用率不高于 50%”的通用条件。该 gate 不代表目标模型所需 GPU 已预留；它也没有检查启动脚本指定的 GPU 3、6 或 7。当前不应据此启动服务。进程快照中 GPU1 的约 74 GiB 占用来自其他 FastGen 训练进程；GPU3–7 同时存在其他训练任务。

## 模型资产、版本和推理参数

### 版本不一致

Phase 2 endpoint alias 源自旧的 direct reliability evaluation 配置。对应的 `start_audit_model_services.sh` manifest 声明：

- Base：Qwen3-8B，revision `b968826d9c46dd6066d109eabc6255188de91218`，BF16，alias `qwen3-8b-audit`，端口 30131。
- Expert：Qwen3-32B **BF16**，revision `9216db5781bf21249d130ec9da846c4624c16137`，alias `qwen3-32b-candidate`，端口 30132。

项目主实验和最近的 Router 泛化 holdout 使用的是 Qwen3-32B-**AWQ**，revision `0499c3ac83fdef8810b907a23894ba91e95eddd8`。attempt04 的历史服务 manifest 记录了该 AWQ Expert 和 Qwen3-8B Base（revision `b968826d9c46dd6066d109eabc6255188de91218`），两者均为 vLLM 0.33.0、temperature 0、top_p 1、max_model_len 4096、max_num_seqs 1、关闭 thinking；但其历史端口是 30256/30257，当前不可达。

因此，现有 Phase 2 配置并未证明它会复用主实验的 AWQ Expert。当前模型 alias 也不携带 checkpoint revision，在线 `/v1/models` 元数据不可用，不能确认实际模型版本。

### 本地资产和运行环境

- `EA_ATPO_V1/experiments/real_case_pilot_20261103/assets/models/Qwen3-8B` 不存在。
- 同一 V1 目录下的 `Qwen3-32B-AWQ` 只有约 3.8 KiB 元数据，没有 `config.json` 或 safetensors 权重。
- 旧项目目录 `EA_ATPO/experiments/real_case_pilot_20261103/assets/models/` 下有完整约 16 GiB 的 Qwen3-8B 和约 19 GiB 的 Qwen3-32B-AWQ 资产，并存在 `.venv-real-pilot`。旧目录的 Qwen3-32B BF16 目录约 100 MiB，不能视为完整权重。
- V1 的 `start_audit_model_services.sh` 指向 V1 内不存在的 `.venv-real-pilot` 和模型路径，不能直接启动当前配置的 BF16 服务。

该 BF16 启动脚本还有 GPU provenance 不一致：启动命令使用默认 `EXPERT_GPU=6`，显存门槛也检查 GPU6；但它写入的服务 manifest 将 Expert 的 `gpu_physical_index` 固定记录为2。若将来使用该脚本，必须先修正 manifest 记录，否则日志会误报 Expert 所在 GPU。该脚本本轮没有执行。

### 请求参数

Phase 2 `ChatAPI.generate` 发送 `temperature=0`、动作对应的 `max_tokens`，并按角色发送 `enable_thinking`；没有显式发送 `top_p` 或随机 seed。Direct Answer、动作生成和 Expert 输出的 token 上限分别由 adapter 设置（16、256、384），Patient response 上限为 256。旧 direct reliability evaluator 使用 `temperature=0`、`top_p=1`、`max_tokens=64`、关闭 thinking。历史 Router 泛化服务 manifest 也记录 `top_p=1`、关闭 thinking，而当前 Phase 2 Base 请求配置 `enable_thinking=true`。

这些差异可能是 Phase 2 动作协议的有意设置，但在 Pilot 执行前须明确冻结。当前运行器的模型版本字段只是指向 execution manifest 的 alias 元数据，并不保证记录服务端 checkpoint revision 或 vLLM 版本。

## 不启动服务时的已有命令与显存估算

按项目主实验的 **Qwen3-8B + Qwen3-32B-AWQ** 模型对，已有启动脚本为：

```bash
bash experiments/real_case_pilot_20261103/audit/start_qwen8_base_complementarity_service.sh
bash experiments/real_case_pilot_20261103/audit/start_qwen32_awq_complementarity_service.sh
```

脚本会分别使用 GPU3/端口30147/alias `qwen3-8b-complementarity-base` 和 GPU7/端口30146/alias `qwen3-32b-awq-complementarity`。这些端口和 alias 与当前 `pilot_config.json` 不同，需先让运行配置与选定的 checkpoint、端口和 alias 一致；不应仅改 endpoint 而不记录版本。脚本还要求 GPU3 空闲至少 31,000 MiB 且利用率不高于35%，GPU7 空闲至少70,000 MiB且利用率不高于30%。本次 GPU3 利用率为100%，GPU7 空闲约34,915 MiB且利用率100%，两者均未通过启动门槛。

| 模型/脚本参数 | 配置的 vLLM 显存预算上限估算 | 检查时权重资产大小 | 当前启动判断 |
|---|---:|---:|---|
| Qwen3-8B BF16，`gpu_memory_utilization=0.14` | 约25,670 MiB（总显存的14%） | 约16 GiB | GPU3 空闲量足够，但利用率100%，脚本会拒绝启动 |
| Qwen3-32B-AWQ，`gpu_memory_utilization=0.25` | 约45,840 MiB（总显存的25%） | 约19 GiB | GPU7 仅空闲约34,915 MiB且利用率100%，脚本会拒绝启动 |
| 当前 alias 配置所指 BF16 pair：8B `0.20` / 32B `0.45` | 约36,672 / 82,512 MiB | V1 缺 8B 权重，32B BF16 权重目录缺失；V1 运行环境缺失 | 不可按当前 V1 launcher 启动；其 Expert 也与主实验 AWQ 不一致 |

这些数值是脚本按 `gpu_memory_utilization × 总显存` 设定的 vLLM 显存预算上限估算，不是实际启动后的实测占用。模型服务没有启动，因此本轮没有实测推理显存。

若保留当前 30131/30132 alias 配置，历史 BF16 启动入口是：

```bash
bash experiments/real_case_pilot_20261103/audit/start_audit_model_services.sh
```

当前 V1 仓库缺少该入口所需的 `.venv-real-pilot`、8B 权重及 BF16 32B 权重；其 Expert 默认显存门槛还要求 GPU6 空闲至少80,000 MiB，而本轮 GPU6 仅空闲约25,039 MiB。即使修复权重/显存问题，也要先纠正该脚本写 manifest 时将 GPU6 记录为 GPU2 的问题。该命令仅供定位，不应在当前状态下运行。

## 本轮执行记录

执行了只读端口/进程/GPU/资产检查，并运行：

```bash
.venv/bin/python experiments/vi_ea_atpo/phase2/run_information_pilot.py
```

该命令为默认 preflight，不带 `--smoke-cases` 或 `--run-pilot`，刷新 execution manifest/failure log，并将成本和 VOI 指标文件保持为仅含表头的空结果；`intervention_rollouts.jsonl` 仍为0字节。结果为 `preflight_only_service_or_gpu_blocked`、`services_ready=false`、`model_inference_started=false`。冻结队列校验通过：100 train、50 validation、case_id 唯一，prompt/data 哈希匹配。队列未重抽样。

**Pilot Ready：否。** 前置条件是确定并冻结 32B Expert 版本（主实验 AWQ 或明确选择 BF16），令 endpoint/alias/模型 manifest 一致；然后在服务对应 GPU 通过启动门槛时启动已有服务并重跑 preflight。服务恢复前不执行 Smoke Test 或 150 例 Pilot。
