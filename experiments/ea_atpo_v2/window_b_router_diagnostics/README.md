# Window B Router Diagnostics

本目录包含 EA-ATPO V2 Router 泛化诊断、相对价值选择层和脱敏汇总结果。它是独立于窗口 A 的分析；不修改窗口 A 文件，不使用其新病例确认集，也不读取 Frozen Holdout。

## Contents

- `selection.py` — 相对优势选择、保守 margin、动作掩码、ASK budget 和无新增信息处理。
- `run_diagnostic.py` — 从已存在的 50 例训练运行重载冻结 embedding 小头，复现开发分数，比较 Distillation、Relative Advantage 和固定 margin 网格，输出病例聚类 bootstrap 汇总。
- `margin_grid.json` — 预先列出的成本和 margin 配置。
- `REPORT.md` — 数据审计、泛化与误差分析、对照结果及候选决策。
- `results_20261120_final/` — 汇总 CSV 与审计 JSON；不包含病例 ID、状态文本、逐状态预测、原始病例或模型 checkpoint。

## Run

需要原有训练运行目录中的 costed Q Dataset、固定 case split、公开状态 embedding 缓存及 manifest、训练/开发汇总、保存的开发分数和小头 checkpoint。它们不复制到本目录。

```bash
export ROUTER_RUN_ROOT=/path/to/EA_ATPO_V1_router_training/experiments/ea_atpo_v2/router_training/outputs/real_q_router_20261120
export ROUTER_DIAGNOSTIC_OUTPUT=experiments/ea_atpo_v2/window_b_router_diagnostics/results_reproduction
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python -m experiments.ea_atpo_v2.window_b_router_diagnostics.run_diagnostic \
  --run-root "$ROUTER_RUN_ROOT" \
  --output-dir "$ROUTER_DIAGNOSTIC_OUTPUT"
```

脚本通过 `case_id` 校验 35/15 划分，审计 K=1、成本只计算一次、公开状态特征边界和 checkpoint 配置。统计指标是旧 Train/Dev 的离线诊断，不是新病例确认结果。
