#!/usr/bin/env python3
"""Run Q/Actor pipeline smoke on the ten-case Pilot without claiming generalization."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from . import ACTION_NAMES
from .audit_q_dataset import audit
from .data import read_jsonl
from .evaluate_policies import available_policies, case_level_metrics, q_diagnostics
from .metrics import aggregate_rollouts
from .predict import load_unique_states
from .evaluate_policies import read_predictions


TRAINING_DIR = Path(__file__).resolve().parent
V2_DIR = TRAINING_DIR.parent
PROJECT_ROOT = V2_DIR.parents[1]
DEFAULT_CASE_SPLIT = PROJECT_ROOT / "experiments/ea_atpo_value_router_20261105/artifacts/case_split.json"
DEFAULT_COST_GRID = [0.00, 0.05, 0.10, 0.20, 0.40, 0.60]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run_command(command: list[str], *, cwd: Path, env: dict[str, str], log) -> None:
    log.write("$ " + shlex.join(command) + "\n")
    log.flush()
    result = subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    log.write(f"\n[exit_code={result.returncode}]\n\n")
    log.flush()
    if result.returncode != 0:
        raise RuntimeError(f"command failed with exit code {result.returncode}: {shlex.join(command)}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-dataset", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--case-split", type=Path, default=DEFAULT_CASE_SPLIT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--max-features", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261120)
    args = parser.parse_args()
    q_path, trajectories_path, split_path = args.q_dataset.resolve(), args.trajectories.resolve(), args.case_split.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite smoke output: {output}")
    output.mkdir(parents=True)
    (output / "audit").mkdir()
    command_file = output / "commands.sh"
    invocation_file = output / "invocation.sh"
    log_path = output / "training_smoke.log"
    py = sys.executable
    commands = [
        [py, "-m", "experiments.ea_atpo_v2.training.audit_q_dataset",
         "--q-dataset", str(q_path), "--trajectories", str(trajectories_path),
         "--run-manifest", str(trajectories_path.parent / "run_manifest.json"),
         "--case-split", str(split_path), "--report", str(output / "audit" / "dataset_audit.json"),
         "--expected-cases", "10", "--expected-states", "29", "--expected-valid-labels", "87"],
        [py, "-m", "experiments.ea_atpo_v2.training.train_q",
         "--q-dataset", str(q_path), "--case-split", str(split_path),
         "--real-data-confirmed", "--smoke-only", "--output-dir", str(output / "q_fit"),
         "--epochs", str(args.epochs), "--patience", str(args.patience),
         "--hidden-dim", str(args.hidden_dim), "--max-features", str(args.max_features),
         "--batch-size", str(args.batch_size), "--seed", str(args.seed)],
        [py, "-m", "experiments.ea_atpo_v2.training.train_policy",
         "--q-dataset", str(q_path), "--case-split", str(split_path),
         "--real-data-confirmed", "--smoke-only", "--output-dir", str(output / "policy_fit"),
         "--epochs", str(args.epochs), "--patience", str(args.patience),
         "--hidden-dim", str(args.hidden_dim), "--max-features", str(args.max_features),
         "--batch-size", str(args.batch_size), "--seed", str(args.seed), "--temperature", "1.0"],
        [py, "-m", "experiments.ea_atpo_v2.training.predict",
         "--checkpoint", str(output / "q_fit" / "q_regressor.pt"), "--states", str(q_path),
         "--output", str(output / "q_predictions.jsonl")],
        [py, "-m", "experiments.ea_atpo_v2.training.predict",
         "--checkpoint", str(output / "policy_fit" / "policy_head.pt"), "--states", str(q_path),
         "--output", str(output / "policy_predictions.jsonl")],
        [py, "-m", "experiments.ea_atpo_v2.training.merge_predictions",
         "--inputs", str(output / "q_predictions.jsonl"), str(output / "policy_predictions.jsonl"),
         "--output", str(output / "all_predictions.jsonl")],
    ]
    invocation = [
        py, "-m", "experiments.ea_atpo_v2.training.run_training_smoke",
        "--q-dataset", str(q_path), "--trajectories", str(trajectories_path),
        "--case-split", str(split_path), "--output-dir", str(output),
        "--epochs", str(args.epochs), "--patience", str(args.patience),
        "--hidden-dim", str(args.hidden_dim), "--max-features", str(args.max_features),
        "--batch-size", str(args.batch_size), "--seed", str(args.seed),
    ]
    invocation_file.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        + f"cd {shlex.quote(str(PROJECT_ROOT))}\n"
        + "export CUDA_VISIBLE_DEVICES=''\nexport OMP_NUM_THREADS=1\nexport MKL_NUM_THREADS=1\n"
        + shlex.join(invocation) + "\n",
        encoding="utf-8",
    )
    command_file.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\nexport CUDA_VISIBLE_DEVICES=''\nexport OMP_NUM_THREADS=1\nexport MKL_NUM_THREADS=1\n"
        + "\n".join(shlex.join(c) for c in commands) + "\n",
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"

    started = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(timespec="seconds")
    status = "failed"
    with log_path.open("w", encoding="utf-8") as log:
        try:
            for command in commands:
                run_command(command, cwd=PROJECT_ROOT, env=env, log=log)

            audit_report = json.loads((output / "audit" / "dataset_audit.json").read_text(encoding="utf-8"))
            q_manifest = json.loads((output / "q_fit" / "training_manifest.json").read_text(encoding="utf-8"))
            policy_manifest = json.loads((output / "policy_fit" / "training_manifest.json").read_text(encoding="utf-8"))
            train_cases = set(q_manifest["train_case_ids"])
            val_cases = set(q_manifest["validation_case_ids"])
            if train_cases & val_cases:
                raise ValueError("internal smoke split has case_id overlap")
            if val_cases != set(policy_manifest["validation_case_ids"]) or train_cases != set(policy_manifest["train_case_ids"]):
                raise ValueError("Q and policy smoke used different case partitions")

            all_rollout_rows = read_jsonl(trajectories_path)
            all_states = aggregate_rollouts(all_rollout_rows)
            smoke_states = {key: value for key, value in all_states.items() if key[0] in val_cases}
            all_predictions = read_predictions(output / "all_predictions.jsonl")
            smoke_predictions = {key: value for key, value in all_predictions.items() if key[0] in val_cases}
            if not smoke_states or set(smoke_states) != set(smoke_predictions):
                raise ValueError("smoke validation states and predictions are not aligned")
            policies, unavailable = available_policies(smoke_states, smoke_predictions, uncertainty_threshold=None)
            summaries, paired, curve, normalized = case_level_metrics(
                smoke_states, smoke_predictions, policies, DEFAULT_COST_GRID, None, 0.05, args.seed
            )
            q_metrics = q_diagnostics(smoke_states, smoke_predictions, [0.05, 0.0, 0.60])
            if "q_guided_distillation" not in policies or "static_l2d" not in policies:
                raise ValueError("Q/Actor predictions did not reach smoke metric code")
            (output / "normalized_smoke_metrics.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in normalized), encoding="utf-8"
            )
            write_json(output / "smoke_metrics.json", {
                "status": "completed", "synthetic_data": False,
                "purpose": "training_pipeline_smoke_only",
                "not_generalization_evidence": True,
                "evaluation_cases_are_the_internal_early_stop_and_calibration_cases": True,
                "case_ids": sorted(val_cases), "n_cases": len(val_cases), "n_states": len(smoke_states),
                "cost_grid_lambda_defer": DEFAULT_COST_GRID, "ask_cost": 0.05,
                "policies_evaluated": policies, "policies_unavailable": unavailable,
                "policy_metrics": summaries, "paired_bootstrap_vs_always_answer": paired,
                "accuracy_cost_curve": curve, "q_diagnostics": q_metrics,
                "stored_net_return_reused_for_utility": False,
                "note": "These metrics only demonstrate code-path execution on a two-case partition of the ten-case engineering Pilot. They are not independent evaluation or a generalization estimate.",
            })
            for file in (output / "q_fit" / "q_regressor.pt", output / "policy_fit" / "policy_head.pt"):
                if not file.is_file() or file.stat().st_size == 0:
                    raise ValueError(f"checkpoint missing/empty: {file}")
            status = "completed"
        except Exception as exc:
            log.write(f"\n[smoke_error] {type(exc).__name__}: {exc}\n")
            log.flush()
            raise
        finally:
            audit_report_path = output / "audit" / "dataset_audit.json"
            audit_summary = json.loads(audit_report_path.read_text(encoding="utf-8")) if audit_report_path.is_file() else {}
            manifest = {
                "status": status,
                "purpose": "real_pilot10_training_pipeline_smoke_only",
                "not_generalization_evidence": True,
                "formal_training_started": False,
                "large_model_finetuning_started": False,
                "frozen_holdout_opened": False,
                "gpu_used": False,
                "cuda_visible_devices": "",
                "started_at_utc": started,
                "finished_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(timespec="seconds"),
                "inputs": {
                    "q_dataset": str(q_path), "q_dataset_sha256": sha256(q_path),
                    "raw_trajectories": str(trajectories_path), "raw_trajectories_sha256": sha256(trajectories_path),
                    "case_split": str(split_path), "case_split_sha256": sha256(split_path),
                },
                "outputs": {
                    "audit": str(output / "audit" / "dataset_audit.json"),
                    "q_checkpoint": str(output / "q_fit" / "q_regressor.pt"),
                    "policy_checkpoint": str(output / "policy_fit" / "policy_head.pt"),
                    "q_predictions": str(output / "q_predictions.jsonl"),
                    "policy_predictions": str(output / "policy_predictions.jsonl"),
                    "smoke_metrics": str(output / "smoke_metrics.json"),
                    "training_log": str(log_path), "commands": str(command_file),
                    "invocation": str(invocation_file),
                },
                "case_count": audit_summary.get("case_count"),
                "state_count": audit_summary.get("state_count"),
                "valid_action_target_count": audit_summary.get("valid_action_target_count"),
                "action_order": list(ACTION_NAMES),
                "training_parameters": {"epochs": args.epochs, "patience": args.patience,
                                        "hidden_dim": args.hidden_dim, "max_features": args.max_features,
                                        "batch_size": args.batch_size, "seed": args.seed},
            }
            write_json(output / "smoke_manifest.json", manifest)
    print(json.dumps({"status": status, "output_dir": str(output),
                      "training_log": str(log_path), "smoke_manifest": str(output / "smoke_manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
