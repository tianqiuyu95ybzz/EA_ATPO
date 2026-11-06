#!/usr/bin/env python3
"""Hash the final method, data, checkpoint, and actual-evaluation artifacts."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    files = [
        "experiments/real_case_pilot_20261103/outputs/full/pilot_trajectories.jsonl",
        "experiments/real_case_pilot_20261103/assets/test.pilot.parquet",
        "experiments/real_case_pilot_20261103/runtime-requirements.lock.txt",
        "experiments/ea_atpo_value_router_20261105/README.md",
        "experiments/ea_atpo_value_router_20261105/config.yaml",
        "experiments/ea_atpo_value_router_20261105/model_service_config.yaml",
        "experiments/ea_atpo_value_router_20261105/run_training.sh",
        "experiments/ea_atpo_value_router_20261105/train_router.py",
        "experiments/ea_atpo_value_router_20261105/router_core.py",
        "experiments/ea_atpo_value_router_20261105/run_stepwise_router_eval.sh",
        "experiments/ea_atpo_value_router_20261105/run_stepwise_router_eval.py",
        "experiments/ea_atpo_value_router_20261105/summarize_stepwise_router_eval.py",
        "experiments/ea_atpo_value_router_20261105/start_quantized_expert_service.sh",
        "experiments/ea_atpo_value_router_20261105/run_quantized_expert_eval.sh",
        "experiments/ea_atpo_value_router_20261105/run_quantized_expert_eval.py",
        "experiments/ea_atpo_value_router_20261105/recompute_strong_expert_statistics.py",
        "experiments/ea_atpo_value_router_20261105/download_official_qwen3_32b_awq.sh",
        "experiments/ea_atpo_value_router_20261105/download_official_qwen3_32b_awq_ranged.py",
        "experiments/ea_atpo_value_router_20261105/continue_awq_expert_after_download.sh",
        "experiments/ea_atpo_value_router_20261105/write_strong_expert_recovery_report.py",
        "experiments/ea_atpo_value_router_20261105/refresh_reproducibility_manifest.py",
        "experiments/ea_atpo_value_router_20261105/CORE_METHOD_DECISION.md",
        "experiments/ea_atpo_value_router_20261105/artifacts/case_split.json",
        "experiments/ea_atpo_value_router_20261105/artifacts/training_summary.json",
        "experiments/ea_atpo_value_router_20261105/artifacts/checkpoint/router_checkpoint.npz",
        "experiments/ea_atpo_value_router_20261105/artifacts/checkpoint_no_reliability/router_checkpoint.npz",
        "experiments/ea_atpo_value_router_20261105/outputs/stepwise_test_v2/run_manifest.json",
        "experiments/ea_atpo_value_router_20261105/outputs/stepwise_test_v2/summary_manifest.json",
        "experiments/ea_atpo_value_router_20261105/outputs/stepwise_test_v2/router_policy_summary.csv",
        "experiments/ea_atpo_value_router_20261105/outputs/stepwise_test_v2/paired_baseline_comparisons.csv",
        "experiments/ea_atpo_value_router_20261105/outputs/stepwise_test_v2/accuracy_cost_curve.csv",
        "experiments/ea_atpo_value_router_20261105/outputs/stepwise_test_v2/policy_episodes.jsonl",
        "experiments/ea_atpo_value_router_20261105/CORE_METHOD_DECISION.md",
        "experiments/ea_atpo_value_router_20261105/outputs/STRONG_EXPERT_RECOVERY_REPORT.md",
        "experiments/ea_atpo_value_router_20261105/outputs/strong_expert_awq/qwen3_32b_direct_answers.jsonl",
        "experiments/ea_atpo_value_router_20261105/outputs/strong_expert_awq/paired_correctness.csv",
        "experiments/ea_atpo_value_router_20261105/outputs/strong_expert_awq/STRONG_EXPERT_EVALUATION.md",
        "experiments/ea_atpo_value_router_20261105/outputs/strong_expert_awq/model_service_manifest.json",
        "experiments/real_case_pilot_20261103/assets/models/Qwen3-32B-AWQ/model_source_lock.json",
        "experiments/real_case_pilot_20261103/audit/run_strong_expert_comparison.py",
    ]
    strong_manifest = HERE / "outputs/strong_expert_awq/run_manifest.json"
    strong_summary = HERE / "outputs/strong_expert_awq/strong_expert_summary.csv"
    for optional in (strong_manifest, strong_summary):
        if optional.exists():
            files.append(str(optional.relative_to(PROJECT)))
    model_status_path = PROJECT / "experiments/real_case_pilot_20261103/assets/models/Qwen3-32B-AWQ/range_download_status.json"
    model_status = json.loads(model_status_path.read_text()) if model_status_path.exists() else {"status": "missing"}
    repo_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()
    experiment_relative = Path("experiments/ea_atpo_value_router_20261105")
    def label(item):
        path = Path(item)
        try:
            return str(path.relative_to(experiment_relative))
        except ValueError:
            return str(path)
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": repo_commit,
        "pilot_training_data_sha256": sha(PROJECT / files[0]),
        "official_test_data_sha256": sha(PROJECT / files[1]),
        "router_training_seed": 20261105,
        "router_test_seed": 20261106,
        "router_model": {
            "class": "linear-kernel ridge action value model",
            "base_model": "Qwen/Qwen3-8B",
            "base_revision": "b968826d9c46dd6066d109eabc6255188de91218",
            "input_boundary": "current public_state user/assistant messages and derived structure only",
        },
        "cost_utility": {
            "ASK_per_patient_turn": 0.01,
            "CONSULT_per_1000_expert_tokens": 0.02,
            "DEFER_per_1000_expert_tokens": 0.02,
            "sensitivity_sets": ["no_cost", "reference", "higher_cost"],
        },
        "strong_expert": {
            "repo_id": "Qwen/Qwen3-32B-AWQ",
            "revision": "0499c3ac83fdef8810b907a23894ba91e95eddd8",
            "download_status": model_status.get("status"),
            "download_source": model_status.get("source"),
            "source_revision": model_status.get("source_revision"),
            "modelscope_weight_upload_commit": model_status.get("modelscope_weight_upload_commit"),
            "download_method": model_status.get("method"),
            "download_chunk_mib": model_status.get("chunk_mib"),
            "download_max_workers": model_status.get("max_workers"),
            "download_read_timeout_s": model_status.get("read_timeout_s"),
            "download_range_retries": model_status.get("range_retries"),
            "completed_shards": model_status.get("completed_shards", []),
            "direct_eval_manifest_status": json.loads(strong_manifest.read_text()).get("status") if strong_manifest.exists() else "not_started",
        },
        "files_sha256": {label(item): sha(PROJECT / item) for item in files},
        "rl_training_started": False,
    }
    output = HERE / "reproducibility_manifest.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
