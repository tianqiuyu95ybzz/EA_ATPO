#!/usr/bin/env python3
"""Refresh the strong-Expert recovery and fixed-sample evaluation report."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
MODEL = PROJECT / "experiments/real_case_pilot_20261103/assets/models/Qwen3-32B-AWQ"
BASE_MODEL = PROJECT / "experiments/real_case_pilot_20261103/assets/models/Qwen3-8B"
OUT = HERE / "outputs"
AWQ_OUT = OUT / "strong_expert_awq"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def get_metric(rows, metric, group):
    return next((x for x in rows if x.get("metric") == metric and x.get("group") == group), None)


def main():
    status_path = MODEL / "range_download_status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {"status": "missing", "expected_shards": []}
    expected = status.get("expected_shards", [])
    verified_bytes = 0
    shard_state = []
    for entry in expected:
        path = MODEL / entry["name"]
        verified = False
        if path.exists() and path.stat().st_size == int(entry["size"]):
            verified = sha(path) == entry["sha256"]
        shard_state.append((entry["name"], int(entry["size"]), verified))
        if verified:
            verified_bytes += int(entry["size"])
    part_files = list((MODEL / ".range_parts").rglob("*.part")) if (MODEL / ".range_parts").exists() else []
    tmp_files = list((MODEL / ".range_parts").rglob("*.tmp.*")) if (MODEL / ".range_parts").exists() else []
    partial_bytes = sum(p.stat().st_size for p in part_files) + sum(p.stat().st_size for p in tmp_files)
    total_bytes = sum(int(x["size"]) for x in expected)
    report_path = OUT / "STRONG_EXPERT_RECOVERY_REPORT.md"
    OUT.mkdir(parents=True, exist_ok=True)

    eval_manifest_path = AWQ_OUT / "run_manifest.json"
    eval_manifest = json.loads(eval_manifest_path.read_text()) if eval_manifest_path.exists() else {}
    evaluation_done = eval_manifest.get("status") == "completed"
    failure_path = AWQ_OUT / "deployment_failure.json"
    deployment_failure = json.loads(failure_path.read_text()) if failure_path.exists() else None
    model_manifest_path = AWQ_OUT / "model_service_manifest.json"
    model_manifest = json.loads(model_manifest_path.read_text()) if model_manifest_path.exists() else {}
    summary = []
    summary_path = AWQ_OUT / "strong_expert_summary.csv"
    if summary_path.exists():
        with summary_path.open(newline="", encoding="utf-8") as stream:
            summary = list(csv.DictReader(stream))

    def fmt(metric, group, ci=False):
        row = get_metric(summary, metric, group)
        if row is None or row.get("value") in (None, ""):
            return "未测"
        value = float(row["value"])
        text = f"{value:.3f}"
        if ci and row.get("ci95_low") not in (None, "") and row.get("ci95_high") not in (None, ""):
            text += f"（95% CI {float(row['ci95_low']):.3f}–{float(row['ci95_high']):.3f}）"
        return text

    gpu_line = "未记录"
    try:
        if model_manifest.get("gpu_physical_index") is not None:
            gpu_index = int(model_manifest["gpu_physical_index"])
            result = subprocess.run(
                ["nvidia-smi", f"--id={gpu_index}", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"],
                check=True, capture_output=True, text=True, timeout=10,
            )
            gpu_line = f"GPU {gpu_index}: {result.stdout.strip()}（服务启动前空闲 {model_manifest.get('gpu_free_mib_before_start', '未知')} MiB）"
        else:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free", "--format=csv,noheader,nounits"],
                check=True, capture_output=True, text=True, timeout=10,
            )
            candidates = []
            for line in result.stdout.splitlines():
                parts = [x.strip() for x in line.split(",")]
                if len(parts) == 4:
                    candidates.append((int(parts[3]), int(parts[0]), f"GPU {parts[0]}: {parts[1]}, {parts[2]} MiB, free {parts[3]} MiB"))
            if candidates:
                gpu_line = "部署时动态选择当前空闲显存最多的 GPU；当前最多：" + max(candidates)[2]
    except Exception:
        pass

    if evaluation_done:
        recovery_state = "AWQ 权重分片已逐片校验并通过 vLLM 部署；固定 200 例配对评估完成。"
        download_summary = f"verified {len([x for x in shard_state if x[2]])}/{len(expected)} shards ({verified_bytes:,}/{total_bytes:,} bytes)"
    elif status.get("status") == "all_shards_verified":
        if deployment_failure:
            recovery_state = f"四个 AWQ 权重分片已校验，但 vLLM 部署或固定 200 例评估失败（exit code {deployment_failure.get('exit_code')}）。"
        else:
            recovery_state = "四个 AWQ 权重分片均已校验；vLLM 部署和固定 200 例条件可靠性评估待运行。"
        download_summary = f"verified {len([x for x in shard_state if x[2]])}/{len(expected)} shards ({verified_bytes:,}/{total_bytes:,} bytes)"
    elif status.get("status") == "interrupted_or_failed_resumable":
        recovery_state = f"{status.get('source', '官方模型源')} 下载中断，已保留断点；verified {len([x for x in shard_state if x[2]])}/{len(expected)} shards。vLLM 和 32B 条件可靠性评估未运行。"
        download_summary = f"verified {len([x for x in shard_state if x[2]])}/{len(expected)} shards ({verified_bytes:,}/{total_bytes:,} bytes); resumable range parts {partial_bytes:,}/{total_bytes:,} bytes"
    else:
        recovery_state = "AWQ 权重仍在断点续传；尚未完整校验和部署，也没有 32B 准确率结果。"
        download_summary = f"verified {len([x for x in shard_state if x[2]])}/{len(expected)} shards ({verified_bytes:,}/{total_bytes:,} bytes); resumable range parts {partial_bytes:,}/{total_bytes:,} bytes"

    lines = [
        "# Qwen3-32B 强 Expert 恢复与条件可靠性评估", "",
        f"**状态：** {recovery_state}", "",
        "## 固定候选与环境", "",
        "| 项目 | 值 |",
        "|---|---|",
        "| 模型候选 | [`Qwen/Qwen3-32B-AWQ`](https://huggingface.co/Qwen/Qwen3-32B-AWQ) |",
        "| 固定 revision | `0499c3ac83fdef8810b907a23894ba91e95eddd8` |",
        f"| 权重来源 | `{status.get('source', 'Hugging Face')}`；source revision `{status.get('source_revision', status.get('revision', '未记录'))}` |",
        "| 模型格式 | 官方 AWQ 量化，未将它描述为 BF16 结果 |",
        "| 推理框架 | vLLM 0.33.0，服务配置使用 `--quantization awq` |",
        f"| 推理 GPU 检查 | {gpu_line}（显存空闲量随其他任务变化） |",
        f"| 文件恢复状态 | {download_summary} |",
        "| 权重目录 | `experiments/real_case_pilot_20261103/assets/models/Qwen3-32B-AWQ/`（项目 PFS） |",
        "",
        "Hugging Face TLS 端点在当前服务器不可达。Qwen 官方 GitHub 指向其 ModelScope 组织；ModelScope 仓库 `Qwen/Qwen3-32B-AWQ` 的固定 revision 已核对，四个分片 size/SHA-256 与 Hugging Face 固定 revision 的 LFS 清单完全一致，配置、index、tokenizer 和词表文件也逐字节匹配已落盘 HF 文件。下载器使用 ModelScope Range 接口续传，同时仍以 HF LFS SHA-256 作为权重校验基准。官方 [AWQ model card](https://huggingface.co/Qwen/Qwen3-32B-AWQ) 提供 vLLM 推理配置；BF16 下载不完整，官方 [Qwen3-32B-FP8](https://huggingface.co/Qwen/Qwen3-32B-FP8) 未下载。", "",
        "Tokenizer JSON 与 Qwen3-8B 官方固定 revision 中的文件具有相同 SHA-256 `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`，按字节哈希验证后在 PFS 内硬链接复用，避免重复下载。模型 index、tokenizer config、vocab 和其它元数据来自 AWQ 固定 revision。", "",
        "## 冻结病例上的条件可靠性", "",
        "评估沿用此前固定的 200 个独立 test 病例（seed `20261104`），逐病例检查 public prompt SHA 和 Gold 与保存的 Qwen3-8B baseline 一致。两模型使用同一公开题干、提示、温度 0、选项解析；Gold 只供事后评分，hidden atomic facts 不进入请求。无效输出按错误保留。Expert 是 LLM 模拟器，不是真实医生。", "",
    ]
    if evaluation_done:
        lines += [
            "| 指标 | Qwen3-8B Base | Qwen3-32B AWQ |",
            "|---|---:|---:|",
            f"| 总体准确率（病例级 95% CI） | {fmt('accuracy','base_8b',True)} | {fmt('accuracy','candidate_expert_32b',True)} |",
            f"| 有效输出率 | {fmt('valid_rate','base_8b')} | {fmt('valid_rate','candidate_expert_32b')} |",
            f"| 平均 total tokens | {fmt('mean_total_tokens','base_8b')} | {fmt('mean_total_tokens','candidate_expert_32b')} |",
            f"| 平均 latency 秒 | {fmt('mean_latency_s','base_8b')} | {fmt('mean_latency_s','candidate_expert_32b')} |",
            "",
            f"配对准确率差（32B−8B）：{fmt('paired_accuracy_delta_candidate_minus_base','candidate_expert_32b-base_8b',True)}。",
            f"32B 在 8B 错误病例上的条件准确率：{fmt('candidate_accuracy_given_base_error','paired_conditional',True)}；修正比例：{fmt('correction_rate_given_base_error','paired_conditional',True)}。",
            f"32B 破坏 8B 正确答案的比例：{fmt('degradation_rate_given_base_correct','paired_conditional',True)}。", "",
            "完整配对四格表见 `outputs/strong_expert_awq/paired_correctness.csv`。", "",
        ]
    elif deployment_failure:
        lines += [
            f"部署/评估失败阶段：{deployment_failure.get('stage')}；exit code `{deployment_failure.get('exit_code')}`；尝试 GPU `{deployment_failure.get('gpu')}`。服务日志见 `outputs/strong_expert_awq/model_service.log`，评估轨迹和 manifest 保留在同目录。32B 性能指标仍为未测。", "",
        ]
    else:
        candidate_status = "未测：权重仍在断点下载/校验" if status.get("status") in ("running", "interrupted_or_failed_resumable") else "未测：权重尚未完整并通过校验"
        lines += [
            "| 指标 | Qwen3-8B Base | Qwen3-32B AWQ |",
            "|---|---:|---:|",
            f"| 总体准确率 | 44.0%（88/200；95% CI 37.0%–51.0%） | {candidate_status} |",
            "| 32B−8B 配对准确率差 | — | 未测 |",
            "| 32B 在 Base 错误病例的条件准确率 | — | 未测 |",
            "| Expert 修正/退化比例 | — | 未测 |", "",
            f"固定评估入口为 `run_quantized_expert_eval.sh`。当前 32B 尚未部署；下载状态 `{status.get('status')}`，最近错误：`{status.get('error', '无')}`。可通过 `download_official_qwen3_32b_awq.sh` 从保存的 Range 分片续传；未使用参数规模代替可靠性结果。", "",
        ]
    lines += [
        "## 文件和当前限制", "",
        "- `outputs/strong_expert_awq/qwen3_32b_direct_answers.jsonl`：候选模型逐病例轨迹（评估完成后）。",
        "- `outputs/strong_expert_awq/strong_expert_summary.csv`、`paired_correctness.csv`：配对结果与病例级 bootstrap。",
        "- `outputs/strong_expert_awq/run_manifest.json`、`model_service_manifest.json`：模型、数据哈希、revision、GPU 与推理参数。",
        "- `experiments/real_case_pilot_20261103/assets/models/Qwen3-32B-AWQ/model_source_lock.json`：ModelScope 固定 revision 与 HF LFS SHA-256 对照。",
        "- `recompute_strong_expert_statistics.py`：从原始配对轨迹离线重算条件指标和 bootstrap。",
        "- 本机共享 GPU 显存和网络吞吐会影响完成时间；所有分片和缓存位于项目工作区。模型是 LLM Expert simulator，不是临床医生。", "",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"updated {report_path}: {recovery_state}")


if __name__ == "__main__":
    main()
