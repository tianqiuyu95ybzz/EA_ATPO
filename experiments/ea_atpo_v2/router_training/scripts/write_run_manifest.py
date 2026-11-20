#!/usr/bin/env python3
"""Write a local, hash-bound manifest for the completed Router experiment."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path


def digest(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def package_version(name):
    try: return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: return None


def git_value(repo, *args):
    try:
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--run-root", type=Path, required=True)
    args = ap.parse_args()
    repo, root = args.repo.resolve(), args.run_root.resolve()
    manifest_path = root / "run_manifest.json"
    if manifest_path.exists(): raise FileExistsError(manifest_path)
    fixed_paths = {
        "source_q": repo / "experiments/ea_atpo_v2/recovery_bundle_20261120/q_dataset.jsonl",
        "source_trajectories": repo / "experiments/ea_atpo_v2/recovery_bundle_20261120/pilot_trajectories.jsonl",
        "processed_q": root / "q_dataset_cost_ask001_defer005.jsonl",
        "case_split": root / "case_split_35_train_15_dev.json",
        "embedding_cache": root / "qwen3_8b_frozen_public_embeddings.jsonl",
        "embedding_manifest": root / "qwen3_8b_frozen_public_embeddings.jsonl.manifest.json",
        "primary_config": repo / "experiments/ea_atpo_v2/router_training/configs/primary_20261120.json",
        "sensitivity_config": repo / "experiments/ea_atpo_v2/router_training/configs/regularization_sensitivity_20261120.json",
        "primary_results": root / "training_final/training_results.json",
        "sensitivity_results": root / "regularization_sensitivity/training_results.json",
        "reload_audit": root / "training_final/checkpoint_reload_audit.json",
        "report": repo / "experiments/ea_atpo_v2/router_training/reports/ROUTER_TRAINING_REPORT_20261120.md",
        "run_commands": repo / "experiments/ea_atpo_v2/router_training/reports/RUN_COMMANDS_20261120.md",
    }
    missing = [name for name, path in fixed_paths.items() if not path.is_file()]
    if missing: raise FileNotFoundError(f"required run artifact missing: {missing}")
    embedding = json.loads(fixed_paths["embedding_manifest"].read_text(encoding="utf-8"))
    primary = json.loads(fixed_paths["primary_results"].read_text(encoding="utf-8"))
    sensitivity = json.loads(fixed_paths["sensitivity_results"].read_text(encoding="utf-8"))
    reload_audit = json.loads(fixed_paths["reload_audit"].read_text(encoding="utf-8"))
    checkpoint_paths = sorted((root / "training_final/checkpoints").rglob("*.pt"))
    sensitivity_checkpoints = sorted((root / "regularization_sensitivity/checkpoints").rglob("*.pt"))
    manifest = {
        "status": "completed_train_dev_experiment",
        "created_utc": subprocess.check_output(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], text=True).strip(),
        "git": {"branch": git_value(repo, "branch", "--show-current"),
                "implementation_commit": git_value(repo, "rev-parse", "HEAD"),
                "working_tree_status": git_value(repo, "status", "--short")},
        "data": {name: {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}
                 for name, path in fixed_paths.items() if name in ("source_q", "source_trajectories", "processed_q", "case_split", "embedding_cache", "primary_config", "sensitivity_config")},
        "models": {"embedding_model_id": embedding["model_id"], "embedding_revision": embedding["model_revision"],
                   "embedding_config_hash": embedding["embedding_config_hash"],
                   "embedding_device": embedding["device"], "embedding_dtype": embedding["dtype"],
                   "embedding_device_name": embedding.get("device_name"), "embedding_seed": embedding["seed"],
                   "embedding_state_count": embedding["state_count"], "embedding_dimension": embedding["embedding_dimension"],
                   "router_training_device": "CPU (CUDA_VISIBLE_DEVICES empty)"},
        "gpu_resource_check": {"embedding_gpu_index": 2, "pre_launch_free_memory_mib": 171865,
                               "pre_launch_utilization_percent": 1,
                               "other_process_present": True,
                               "other_process_action": "left running; did not stop, modify, or signal it",
                               "embedding_completed_without_oom": True},
        "runtime": {"python": sys.version, "torch": package_version("torch"),
                    "transformers": package_version("transformers"), "numpy": package_version("numpy"),
                    "cpu_threads": 1, "deterministic_algorithms": True},
        "experiment": {"action_order": ["ASK", "ANSWER", "DEFER"],
                       "costs": {"ANSWER": 0.0, "ASK": 0.01, "DEFER": 0.05},
                       "case_split": {"unit": "case_id", "train_cases": 35, "dev_cases": 15,
                                      "seed": 20261120, "optimization_seeds": [17, 42, 2026]},
                       "primary_result_status": primary["status"],
                       "regularization_sensitivity_status": sensitivity["status"],
                       "checkpoint_reload_audit_status": reload_audit["status"],
                       "primary_checkpoint_count": len(checkpoint_paths),
                       "sensitivity_checkpoint_count": len(sensitivity_checkpoints),
                       "frozen_holdout_opened": False, "official_validation_or_internal_test_used": False},
        "script_hashes": {},
        "checkpoints": [],
        "non_claims": ["Dev results are not an independent test.", "K=1 action/state labels are one realized rollout.",
                       "No Router deployment, PPO, GRPO, or Actor LoRA was run."],
    }
    for rel in ("experiments/ea_atpo_v2/router_training/scripts/prepare_costed_dataset.py",
                "experiments/ea_atpo_v2/router_training/scripts/train_evaluate.py",
                "experiments/ea_atpo_v2/router_training/scripts/verify_checkpoints.py",
                "experiments/ea_atpo_v2/training/extract_state_embeddings.py"):
        path = repo / rel
        manifest["script_hashes"][rel] = digest(path)
    for group, paths in (("primary", checkpoint_paths), ("regularization_sensitivity", sensitivity_checkpoints)):
        for path in paths:
            manifest["checkpoints"].append({"run": group, "path": str(path), "sha256": digest(path), "bytes": path.stat().st_size})
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "manifest": str(manifest_path),
                      "primary_checkpoints": len(checkpoint_paths), "sensitivity_checkpoints": len(sensitivity_checkpoints),
                      "git_commit": manifest["git"]["implementation_commit"]}, indent=2))


if __name__ == "__main__": main()
