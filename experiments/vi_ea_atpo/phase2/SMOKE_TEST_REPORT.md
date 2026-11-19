# Phase 2 真实 Smoke Test 报告

**日期：** 2026-11-18  
**Smoke Test 的服务状态：** Base、共享 Base 的 Patient role、32B-AWQ Expert 均已真实推理  
**分支执行状态：** 通过  
**Patient 事实覆盖质量：** 未通过；同一冻结队列两例复核后，仍有一个已提供事实未被反馈  
**Phase 2 Pilot Ready：** **否**

## 执行范围

按要求只取冻结队列确定性前缀：1 个 train 和 1 个 validation 病例，没有重新抽样、没有 test 病例、没有启动全部 150 例。为保留首次运行，本报告保留了两次输出：

- 初次 smoke（原 Patient prompt v1）：`phase2/intervention_rollouts.jsonl`
- 同两例、工程性 prompt 修订后的复核（v2）：`phase2/smoke_recovery_v2/intervention_rollouts.jsonl`

两次都得到 12 行分支结果：Direct Answer、Direct Defer、ASK task answer、ASK decision、CONSULT task answer、CONSULT decision 各一行/病例。每次运行的 trace 含 21 个唯一成功 API 完成事件；同一 ASK/CONSULT acquisition trace 同时写在 task 与 decision 两个结果中，统计调用数时已去重。没有扩大独立病例数。v2 因观察到 validation 的 Patient 错误拒答而修改提示，因此这次复核是工程调试记录，不是未触碰过的验证结果。

唯一事件遥测：v1 为 Base 14 次、Patient 2 次、Expert 5 次，共 21 次；累计输入 11,148 tokens、输出 205 tokens，trace latency 合计约 2.89 秒。v2 为 Base 14 次、Patient 2 次、Expert 5 次，共 21 次；累计输入 11,276 tokens、输出 205 tokens，trace latency 合计约 1.73 秒。Latency 是请求逐项相加，不是整次运行 wall time；以上 tokens/latency 不计入 Utility。

## 真实执行结果

两次运行中，所有 Direct/ASK/CONSULT/DEFER 分支都返回 A–D 终局选项，解析通过，无分支异常。Direct Defer 调用了 Expert 并终止；CONSULT 调用 Expert 得到 `Expert Advice:`，然后由 Base 继续选择终局答案或 DEFER，没有把 CONSULT 替换为 DEFER。基线和干预从同一冻结 `PublicState` 建立；不同分支的状态对象互相隔离。

Patient 的运行结果揭示一个质量问题：train 样本问及源 facts 中没有的实验室数值，返回 UNKNOWN 合理；validation 样本问及既往脊柱创伤，源 `atomic_facts` 明确写有跌落导致的颈椎爆裂骨折，但 Patient 两次均返回 UNKNOWN。提示 v2 加入了语义匹配、同义改述和广义问题回答规则，仍未使它召回这条事实。因此 Patient endpoint 可用，但 ASK 获得信息的仿真 fidelity 未通过验收。不会基于这两个病例继续调 prompt 来制造通过结果。

## 小样本观察（只用于 Smoke 验收，不作为收益证据）

两个病例在 Direct Answer 都答对、Direct Defer 都答错。ASK 后强制 Base 回答为 2/2 正确；CONSULT 后强制 Base 回答也为 2/2 正确。但两个 Direct Answer 本来已正确，所以该 smoke 没有观察到 task improvement，也不足以说明 ASK/CONSULT 的 VOI。

预设主成本 `c_D=0.60` 下的每例平均 utility：

| 分支 | Train utility | Validation utility | 说明 |
|---|---:|---:|---|
| Direct Answer | 1.00 | 1.00 | 两例都正确 |
| Direct Defer | -0.60 | -0.60 | 两例 Expert 终局答案均错误，另扣 `c_D` |
| ASK → forced answer | 0.95 | 0.95 | 正确，但另扣 `c_A=0.05` |
| ASK → answer/defer decision | 0.35 | 0.95 | Train 选择 DEFER 后 Expert 错误；Validation 直接回答 |
| CONSULT → forced answer | 0.80 | 0.80 | 正确，但另扣 `c_C=0.20` |
| CONSULT → answer/defer decision | 0.80 | 0.80 | 两例都由 Base 选择 ANSWER |

这些数值每 split 仅 `n=1`，不得当作效果评估或推广。不同成本的计算表保存在 `paired_intervention_metrics.csv` 和 recovery 目录对应文件中。

## 成本、日志与信息隔离检查

- 每行均满足 `total_utility = terminal_correct - 0.05*N_ASK - 0.20*N_CONSULT - 0.60*N_DEFER`，`action_cost` 也逐项吻合；CONSULT 后若再 DEFER，会分别计算咨询成本与接管成本。
- traces 记录动作、获取到的内容、后续状态、终局动作、答案、正确性、token/延迟、提示版本以及 Base/Expert checkpoint revision。
- patient `atomic_facts` 只进入 Patient 请求；Expert 的 CONSULT/DEFER 请求由 `build_expert_messages` 构造，并通过 `validate_expert_view` 拒绝未揭示事实。Gold 在生成结束后只用于评分。结果 JSONL 中未写入 `ground_truth` 或完整 hidden facts。
- 成本表中的 token 和 latency 是遥测字段，不重复计入主 utility。
- 运行配置与调用显式使用 temperature 0、top-p 1、禁用 Qwen3 thinking；服务模型别名/revision 与本地 manifest 一致。
- 两次 Patient fact-recall miss 已追加到 `phase2/smoke_quality_failure.log`；完整响应和上下文保留在对应 rollout JSONL 中。

## 服务健康结果

`services/health_check.json` 显示：Base 最小问答返回有效选项，Patient role 通过共享 Base 返回文本，AWQ Expert 的 DEFER 返回有效选项；`/v1/models` 身份与 manifest revision 均匹配。服务和启动问题详见 `SERVICE_RECOVERY_REPORT.md` 与 `DEPLOYMENT_PLAN.md`。

## 实际运行命令与验证

```bash
bash experiments/vi_ea_atpo/phase2/start_vi_services.sh
bash experiments/vi_ea_atpo/phase2/check_vi_services.sh
/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/.venv-real-pilot/bin/python experiments/vi_ea_atpo/phase2/run_information_pilot.py
/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/.venv-real-pilot/bin/python experiments/vi_ea_atpo/phase2/run_information_pilot.py --smoke-cases 2
/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/.venv-real-pilot/bin/python experiments/vi_ea_atpo/phase2/run_information_pilot.py --smoke-cases 2 --output-dir experiments/vi_ea_atpo/phase2/smoke_recovery_v2
/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/.venv-real-pilot/bin/python experiments/vi_ea_atpo/phase2/test_information_pilot.py
/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/.venv-real-pilot/bin/python -m py_compile experiments/vi_ea_atpo/phase2/manage_vi_services.py experiments/vi_ea_atpo/phase2/run_information_pilot.py ea_atpo_stage/recipe/ea_atpo/collect_counterfactuals.py
```

CPU adapter tests: **9/9 passed**. Both real smoke manifests report `completed`, `services_ready=true`, `gpu_gate_passed=true`, `selected_run_cases=2`; no test cases were used.

## Decision

The model-service deployment and action adapters pass the smoke checks. The Patient simulator does not reliably return known facts even after one bounded prompt revision, so **do not start the formal 150-case Phase 2 Pilot yet**. Keep both smoke outputs, prompt-v2 failure, and service logs. The next fix should make Patient fact retrieval verifiable without exposing Gold or changing the four action meanings; then revalidate on a newly frozen development cohort before using the current validation cases for efficacy claims. No PPO/ATPO RL was started.
