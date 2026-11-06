#!/usr/bin/env python3
"""Run a paired, real-case four-action feasibility pilot; never trains a model."""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import random
import statistics
import subprocess
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


PILOT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PILOT_DIR.parents[1]
RECIPE_DIR = PROJECT_DIR / "ea_atpo_stage"
sys.path.insert(0, str(RECIPE_DIR))
ACTION_ORDER = ("ANSWER", "ASK", "CONSULT", "DEFER")
SUMMARY_FIELDS = ("section", "group", "metric", "n_cases", "n_states", "value", "ci95_low", "ci95_high", "notes")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def git_value(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=PROJECT_DIR, check=True,
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def read_config(path: Path) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return dict(value)


def endpoint_status(settings: Mapping[str, Any]) -> dict[str, Any]:
    url = str(settings["base_url"]).rstrip("/") + "/models"
    headers = {}
    key_env = settings.get("api_key_env")
    api_key = os.environ.get(str(key_env), "") if key_env else ""
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=3) as response:
            body = json.loads(response.read().decode("utf-8"))
            model_rows = body.get("data", [])
            model_ids = [str(item.get("id", "")) for item in model_rows]
            model_metadata = [
                {key: item[key] for key in ("id", "created", "owned_by", "model_revision") if key in item}
                for item in model_rows
            ]
            expected = str(settings.get("model", ""))
            return {
                "url": url,
                "http_status": response.status,
                "reachable": True,
                "configured_model": expected,
                "model_ids": model_ids,
                "model_metadata": model_metadata,
                "configured_model_listed": expected in model_ids if model_ids else None,
                "api_key_env": key_env,
                "api_key_present": bool(api_key),
            }
    except urllib.error.HTTPError as exc:
        return {
            "url": url, "http_status": exc.code, "reachable": True,
            "configured_model": str(settings.get("model", "")),
            "model_ids": [], "configured_model_listed": False,
            "model_metadata": [],
            "api_key_env": key_env, "api_key_present": bool(api_key),
            "error": "HTTP error; response body omitted",
        }
    except Exception as exc:
        return {
            "url": url, "reachable": False,
            "configured_model": str(settings.get("model", "")),
            "model_ids": [], "configured_model_listed": False,
            "model_metadata": [],
            "api_key_env": key_env, "api_key_present": bool(api_key),
            "error": f"{type(exc).__name__}: {exc}",
        }


def gpu_inventory() -> list[dict[str, Any]]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    devices = []
    for line in result.stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) == 5:
            devices.append({
                "index": fields[0], "name": fields[1],
                "memory_total_mib": int(fields[2]), "memory_used_mib": int(fields[3]),
                "utilization_pct": int(fields[4]),
            })
    return devices


def local_tokenizer_inventory() -> dict[str, Any]:
    cache = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface"))) / "hub"
    cached_qwen_dirs = []
    if cache.is_dir():
        cached_qwen_dirs = sorted(
            entry.name for entry in cache.iterdir()
            if entry.is_dir() and any(tag in entry.name.lower() for tag in ("qwen3-8b", "qwen3.5", "qwen3-35b"))
        )
    versions = {}
    for package in ("torch", "transformers", "tokenizers", "vllm", "aiohttp", "pandas", "pyarrow", "datasets", "PyYAML"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "transformers_installed": importlib.util.find_spec("transformers") is not None,
        "tokenizers_installed": importlib.util.find_spec("tokenizers") is not None,
        "qwen_model_cache_entries": cached_qwen_dirs,
        "package_versions": versions,
    }


def historical_checkpoint_inventory(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    inventory = []
    for item in config.get("historical_checkpoint_paths", []):
        path = Path(str(item.get("path", "")))
        inventory.append({
            "role": item.get("role"),
            "path": str(path),
            "exists": path.exists(),
        })
    return inventory


def build_manifest(data_path: Path, config_path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    dependencies = {name: importlib.util.find_spec(name) is not None for name in (
        "aiohttp", "pandas", "pyarrow", "yaml", "datasets", "torch", "transformers",
        "huggingface_hub", "vllm", "accelerate",
    )}
    endpoints = {
        name: endpoint_status(settings)
        for name, settings in config.get("endpoints", {}).items()
    }
    try:
        import platform
        python_version = platform.python_version()
    except Exception:
        python_version = sys.version
    model_manifest_path = PILOT_DIR / "assets/models/model_manifest.json"
    service_manifest_path = PILOT_DIR / "outputs/model_service_manifest.json"
    source_manifest_path = PILOT_DIR / "assets/source_manifest.json"
    adapter_manifest_path = PILOT_DIR / "assets/adapter_manifest.json"
    service_config_path = PILOT_DIR / "model_service_config.yaml"
    reproduction_files = (
        "PILOT_RUNTIME_REQUIREMENTS.in", "runtime-requirements.lock.txt",
        "restore_public_atpo_data.sh", "adapt_atpo_parquet.py", "download_model.sh",
        "start_model_service.sh", "run_service_role_smoke.py", "run_smoke_test.sh",
        "run_full_pilot.sh",
    )

    def read_optional_json(path: Path) -> dict[str, Any] | None:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    manifest: dict[str, Any] = {
        "created_at_utc": utc_now(),
        "status": "preflight",
        "project_git_commit": git_value("rev-parse", "HEAD"),
        "project_git_status": git_value("status", "--short"),
        "pilot_script_sha256": sha256_file(Path(__file__)),
        "collector_sha256": sha256_file(RECIPE_DIR / "recipe/ea_atpo/collect_counterfactuals.py"),
        "protocol_sha256": sha256_file(RECIPE_DIR / "recipe/ea_atpo/protocol.py"),
        "python_version": python_version,
        "python_executable": sys.executable,
        "platform": sys.platform,
        "dependencies": dependencies,
        "local_tokenizer_inventory": local_tokenizer_inventory(),
        "data_path": str(data_path),
        "data_exists": data_path.is_file(),
        "data_sha256": sha256_file(data_path) if data_path.is_file() else None,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path) if config_path.is_file() else None,
        "source": config.get("source", {}),
        "source_data_manifest": read_optional_json(source_manifest_path),
        "data_adapter_manifest": read_optional_json(adapter_manifest_path),
        "model": config.get("model", {}),
        "model_manifest": read_optional_json(model_manifest_path),
        "model_service": read_optional_json(service_manifest_path),
        "model_service_config_path": str(service_config_path),
        "model_service_config_sha256": sha256_file(service_config_path) if service_config_path.is_file() else None,
        "reproduction_artifact_sha256": {
            name: sha256_file(PILOT_DIR / name)
            for name in reproduction_files if (PILOT_DIR / name).is_file()
        },
        "models": {name: settings.get("model") for name, settings in config.get("endpoints", {}).items()},
        "checkpoint_paths": {name: settings.get("checkpoint_path") for name, settings in config.get("endpoints", {}).items()},
        "historical_checkpoint_inventory": historical_checkpoint_inventory(config),
        "endpoints": endpoints,
        "inference": {
            "temperature": config.get("temperature"),
            "state_temperature": config.get("state_temperature"),
            "num_cases": config.get("num_cases", 100),
            "max_states_per_case": config.get("max_states_per_case", 3),
            "max_assistant_turns": config.get("max_assistant_turns"),
            "max_consults_per_rollout": config.get("max_consults_per_rollout", 2),
            "rollouts_per_action": config.get("rollouts_per_action"),
            "seed": config.get("seed"),
            "thinking": {name: settings.get("enable_thinking") for name, settings in config.get("endpoints", {}).items()},
            "protocol_controls": config.get("protocol", {}),
        },
        "cost_sensitivity": config.get("cost_sensitivity", {}),
        "gpus": gpu_inventory(),
        "trajectory_count": 0,
    }
    blockers = []
    if not data_path.is_file():
        blockers.append("train Parquet not found; original local train/test Parquet could not be located")
    for name, present in dependencies.items():
        if not present:
            blockers.append(f"missing runtime dependency: {name}")
    if not endpoints:
        blockers.append("no Base/Patient/Expert endpoints configured")
    for name, status in endpoints.items():
        if not status.get("reachable"):
            blockers.append(f"{name} endpoint is unreachable")
        elif status.get("http_status") != 200:
            blockers.append(f"{name} endpoint did not return HTTP 200 from /models")
        elif status.get("configured_model_listed") is False:
            blockers.append(f"{name} configured model alias was not listed by /models")
    manifest["blockers"] = blockers
    manifest["status"] = "blocked" if blockers else "ready"
    return manifest


def write_blocked_artifacts(output_dir: Path, manifest: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest["trajectory_count"] = 0
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    status_record = {
        "record_type": "run_status",
        "status": manifest.get("status", "blocked"),
        "created_at_utc": manifest["created_at_utc"],
        "trajectory_count": 0,
        "blockers": manifest["blockers"],
        "note": "This is a status record, not a generated case trajectory.",
    }
    (output_dir / "pilot_trajectories.jsonl").write_text(json.dumps(status_record, ensure_ascii=False) + "\n", encoding="utf-8")
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerow({
            "section": "run_status", "group": "pilot", "metric": "status",
            "n_cases": 0, "n_states": 0, "value": "blocked",
            "notes": "; ".join(manifest["blockers"]),
        })
    write_report(output_dir / "FEASIBILITY_PILOT_REPORT.md", manifest, [])
    write_report(output_dir / "REAL_PILOT_REPORT.md", manifest, [])
    with (output_dir / "action_value_analysis.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerow({
            "section": "run_status", "group": "pilot", "metric": "status",
            "n_cases": 0, "n_states": 0, "value": "blocked",
            "notes": "; ".join(manifest["blockers"]),
        })


def _bootstrap_ci(values: Sequence[float], seed: int, repeats: int = 10000) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        value = float(values[0])
        return value, value
    rng = random.Random(seed)
    means = [statistics.mean(rng.choices(values, k=len(values))) for _ in range(repeats)]
    means.sort()
    return means[int(0.025 * repeats)], means[min(repeats - 1, int(0.975 * repeats))]


def summarize(records: Sequence[Mapping[str, Any]], config: Mapping[str, Any], seed: int) -> list[dict[str, Any]]:
    trajectories = [r for r in records if r.get("record_type") == "trajectory"]
    if not trajectories:
        return []
    rows: list[dict[str, Any]] = []

    def add(section: str, group: str, metric: str, values: Sequence[float], notes: str = "") -> None:
        low, high = _bootstrap_ci(values, seed)
        rows.append({
            "section": section, "group": group, "metric": metric,
            "n_cases": len(values), "n_states": "", "value": statistics.mean(values) if values else "",
            "ci95_low": low if low is not None else "", "ci95_high": high if high is not None else "", "notes": notes,
        })

    by_case_action: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    paired: dict[tuple[str, str, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    by_state: dict[tuple[str, str], dict[str, list[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for item in trajectories:
        case_id = str(item["case_id"])
        action = str(item["action"])
        by_case_action[(case_id, action)].append(item)
        pair_key = (case_id, str(item["state_id"]), int(item["rollout_index"]))
        paired[pair_key][action] = item
        by_state[(case_id, str(item["state_id"]))][action].append(item)

    for action in ACTION_ORDER:
        outcome_case_means = []
        token_case_means = []
        latency_case_means = []
        patient_call_case_means = []
        patient_token_case_means = []
        base_token_case_means = []
        total_token_case_means = []
        total_latency_case_means = []
        valid_case_means = []
        for (case_id, chosen_action), items in by_case_action.items():
            if chosen_action != action:
                continue
            outcome_case_means.append(statistics.mean(float(item.get("outcome", 0.0)) for item in items))
            token_case_means.append(statistics.mean(float(item.get("expert_prompt_tokens", 0)) + float(item.get("expert_completion_tokens", 0)) for item in items))
            latency_case_means.append(statistics.mean(float(item.get("expert_latency_s", 0.0)) for item in items))
            patient_call_case_means.append(statistics.mean(sum(t.get("actor") == "patient" for t in item.get("trace", [])) for item in items))
            patient_token_case_means.append(statistics.mean(float(item.get("patient_prompt_tokens", 0)) + float(item.get("patient_completion_tokens", 0)) for item in items))
            base_token_case_means.append(statistics.mean(float(item.get("base_prompt_tokens", 0)) + float(item.get("base_completion_tokens", 0)) for item in items))
            total_token_case_means.append(statistics.mean(float(item.get("total_prompt_tokens", 0)) + float(item.get("total_completion_tokens", 0)) for item in items))
            total_latency_case_means.append(statistics.mean(float(item.get("total_latency_s", 0.0)) for item in items))
            valid_case_means.append(statistics.mean(float(bool(item.get("valid"))) for item in items))
        add("action", action, "accuracy", outcome_case_means, "Case-clustered mean over generated states and replicates; invalid rollout scores 0.")
        add("action", action, "valid_rate", valid_case_means)
        add("cost", action, "expert_tokens_per_rollout", token_case_means)
        add("cost", action, "expert_latency_seconds_per_rollout", latency_case_means)
        add("cost", action, "patient_turns_per_rollout", patient_call_case_means)
        add("cost", action, "patient_tokens_per_rollout", patient_token_case_means)
        add("cost", action, "base_tokens_per_rollout", base_token_case_means)
        add("cost", action, "total_tokens_per_rollout", total_token_case_means)
        add("cost", action, "total_latency_seconds_per_rollout", total_latency_case_means)

    comparison_diffs: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    correction_count = answer_wrong_count = degradation_count = answer_correct_count = 0
    for branches in paired.values():
        if not all(action in branches for action in ACTION_ORDER):
            continue
        ans = float(branches["ANSWER"].get("outcome", 0.0))
        con = float(branches["CONSULT"].get("outcome", 0.0))
        defer = float(branches["DEFER"].get("outcome", 0.0))
        case_id = str(branches["ANSWER"]["case_id"])
        comparison_diffs[case_id]["consult_minus_answer"].append(con - ans)
        comparison_diffs[case_id]["consult_minus_defer"].append(con - defer)
        if ans == 0:
            answer_wrong_count += 1
            correction_count += int(con == 1)
        else:
            answer_correct_count += 1
            degradation_count += int(con == 0)
    add("comparison", "CONSULT-ANSWER", "paired_accuracy_delta", [statistics.mean(x["consult_minus_answer"]) for x in comparison_diffs.values() if x["consult_minus_answer"]])
    add("comparison", "CONSULT-DEFER", "paired_accuracy_delta", [statistics.mean(x["consult_minus_defer"]) for x in comparison_diffs.values() if x["consult_minus_defer"]])
    rows.extend([
        {"section": "revision", "group": "CONSULT", "metric": "correction_count", "n_cases": len(comparison_diffs), "n_states": len(paired), "value": correction_count, "ci95_low": "", "ci95_high": "", "notes": f"Denominator: {answer_wrong_count} paired ANSWER errors."},
        {"section": "revision", "group": "CONSULT", "metric": "correction_rate_given_answer_wrong", "n_cases": len(comparison_diffs), "n_states": len(paired), "value": correction_count / answer_wrong_count if answer_wrong_count else "", "ci95_low": "", "ci95_high": "", "notes": "CONSULT branch correct where ANSWER branch was wrong."},
        {"section": "revision", "group": "CONSULT", "metric": "degradation_count", "n_cases": len(comparison_diffs), "n_states": len(paired), "value": degradation_count, "ci95_low": "", "ci95_high": "", "notes": f"Denominator: {answer_correct_count} paired ANSWER successes."},
        {"section": "revision", "group": "CONSULT", "metric": "degradation_rate_given_answer_correct", "n_cases": len(comparison_diffs), "n_states": len(paired), "value": degradation_count / answer_correct_count if answer_correct_count else "", "ci95_low": "", "ci95_high": "", "notes": "CONSULT branch wrong where ANSWER branch was correct."},
    ])

    reliability = defaultdict(list)
    case_delta = {case: statistics.mean(values["consult_minus_answer"]) for case, values in comparison_diffs.items() if values["consult_minus_answer"]}
    for case_id in case_delta:
        defer_records = by_case_action.get((case_id, "DEFER"), [])
        if defer_records:
            reliability[case_id] = statistics.mean(float(item.get("outcome", 0.0)) for item in defer_records)
    ranked = sorted(reliability, key=lambda case_id: (reliability[case_id], case_id))
    thirds = max(1, len(ranked) // 3)
    reliability_groups = {
        "low_observed_DEFER_accuracy": ranked[:thirds],
        "mid_observed_DEFER_accuracy": ranked[thirds:2 * thirds],
        "high_observed_DEFER_accuracy": ranked[2 * thirds:],
    }
    for group, cases in reliability_groups.items():
        add("expert_reliability_slice", group, "CONSULT_minus_ANSWER_accuracy", [case_delta[case] for case in cases if case in case_delta], "Post-hoc descriptive strata by observed DEFER accuracy; never an input to agent or router.")

    lambdas = config.get("cost_sensitivity", {})
    expert_lambdas = lambdas.get("lambda_expert_per_1k_tokens", [0.0, 0.02, 0.05, 0.1])
    patient_lambdas = lambdas.get("lambda_patient_per_turn", [0.0, 0.01, 0.03])
    priority = {"ANSWER": 3, "ASK": 2, "CONSULT": 1, "DEFER": 0}
    state_decisions = []
    for (case_id, state_id), actions in by_state.items():
        values = {}
        for action, items in actions.items():
            values[action] = {
                "q": statistics.mean(float(item.get("outcome", 0.0)) for item in items),
                "tokens": statistics.mean(float(item.get("expert_prompt_tokens", 0)) + float(item.get("expert_completion_tokens", 0)) for item in items),
                "patient_turns": statistics.mean(sum(t.get("actor") == "patient" for t in item.get("trace", [])) for item in items),
                "total_tokens": statistics.mean(float(item.get("total_prompt_tokens", 0)) + float(item.get("total_completion_tokens", 0)) for item in items),
                "latency_s": statistics.mean(float(item.get("total_latency_s", 0.0)) for item in items),
            }
        if all(action in values for action in ACTION_ORDER):
            state_decisions.append((case_id, values))
    n_decisions = len(state_decisions)
    for opponent in ("ANSWER", "DEFER"):
        consult_better = sum(values["CONSULT"]["q"] > values[opponent]["q"] for _, values in state_decisions)
        consult_equal = sum(values["CONSULT"]["q"] == values[opponent]["q"] for _, values in state_decisions)
        consult_worse = sum(values["CONSULT"]["q"] < values[opponent]["q"] for _, values in state_decisions)
        rows.extend([
            {"section": "dominance", "group": f"CONSULT_vs_{opponent}", "metric": "consult_better_state_share", "n_cases": len({case for case, _ in state_decisions}), "n_states": n_decisions, "value": consult_better / n_decisions if n_decisions else "", "ci95_low": "", "ci95_high": "", "notes": "Uncosted empirical outcome value Q_out."},
            {"section": "dominance", "group": f"CONSULT_vs_{opponent}", "metric": "consult_equal_state_share", "n_cases": len({case for case, _ in state_decisions}), "n_states": n_decisions, "value": consult_equal / n_decisions if n_decisions else "", "ci95_low": "", "ci95_high": "", "notes": "Uncosted empirical outcome value Q_out."},
            {"section": "dominance", "group": f"CONSULT_vs_{opponent}", "metric": "consult_worse_state_share", "n_cases": len({case for case, _ in state_decisions}), "n_states": n_decisions, "value": consult_worse / n_decisions if n_decisions else "", "ci95_low": "", "ci95_high": "", "notes": "Uncosted empirical outcome value Q_out."},
        ])
    for lambda_e in expert_lambdas:
        for lambda_p in patient_lambdas:
            wins: dict[str, int] = defaultdict(int)
            gaps: dict[str, list[float]] = defaultdict(list)
            policy_by_case: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
            for case_id, values in state_decisions:
                utilities = {
                    action: values[action]["q"] - float(lambda_e) * values[action]["tokens"] / 1000.0 - float(lambda_p) * values[action]["patient_turns"]
                    for action in ACTION_ORDER
                }
                ordered = sorted(
                    utilities.items(),
                    key=lambda kv: (kv[1], -values[kv[0]]["tokens"], -values[kv[0]]["patient_turns"], priority[kv[0]]),
                    reverse=True,
                )
                wins[ordered[0][0]] += 1
                gaps[case_id].append(ordered[0][1] - ordered[1][1])
                selected, selected_utility = ordered[0]
                policy_by_case[case_id]["selected_accuracy"].append(values[selected]["q"])
                policy_by_case[case_id]["selected_defer_rate"].append(float(selected == "DEFER"))
                policy_by_case[case_id]["selected_expert_tokens"].append(values[selected]["tokens"])
                policy_by_case[case_id]["selected_patient_turns"].append(values[selected]["patient_turns"])
                policy_by_case[case_id]["selected_total_tokens"].append(values[selected]["total_tokens"])
                policy_by_case[case_id]["selected_latency_seconds"].append(values[selected]["latency_s"])
                policy_by_case[case_id]["selected_utility"].append(selected_utility)
                policy_by_case[case_id]["best_vs_second_gap"].append(ordered[0][1] - ordered[1][1])
            label = f"lambdaE={lambda_e:g};lambdaP={lambda_p:g}"
            total_states = max(1, len(state_decisions))
            for action in ACTION_ORDER:
                rows.append({
                    "section": "cost_sensitivity", "group": label,
                    "metric": f"optimal_action_share:{action}",
                    "n_cases": len({case for case, _ in state_decisions}),
                    "n_states": len(state_decisions), "value": wins[action] / total_states,
                    "ci95_low": "", "ci95_high": "",
                    "notes": "No-cost baseline is lambdaE=0; lambdaE penalizes expert k-tokens, lambdaP patient turns.",
                })
            gap_by_case = {case: statistics.mean(gs) for case, gs in gaps.items() if gs}
            rows.append({
                "section": "cost_sensitivity", "group": label,
                "metric": "mean_best_vs_second_value_gap",
                "n_cases": len(gap_by_case), "n_states": len(state_decisions),
                "value": statistics.mean(gap_by_case.values()) if gap_by_case else "",
                "ci95_low": "", "ci95_high": "",
                "notes": "Utilities use one common cost formula across all actions.",
            })
            for metric in (
                "selected_accuracy", "selected_defer_rate", "selected_expert_tokens",
                "selected_patient_turns", "selected_total_tokens", "selected_latency_seconds",
                "selected_utility", "best_vs_second_gap",
            ):
                per_case = [statistics.mean(metrics[metric]) for metrics in policy_by_case.values() if metrics[metric]]
                add("policy_tradeoff", label, metric, per_case, "Case-clustered post-hoc oracle over the four measured action outcomes; not a learned or deployable router. Utility costs are sensitivity units, not dollars.")

    for case_id, metrics in comparison_diffs.items():
        deltas = metrics.get("consult_minus_answer", [])
        if deltas:
            rows.append({
                "section": "consult_gain_distribution", "group": "CONSULT-ANSWER",
                "metric": "case_mean_paired_accuracy_delta", "n_cases": 1,
                "n_states": len(deltas), "value": statistics.mean(deltas),
                "ci95_low": "", "ci95_high": "", "notes": f"case_id={case_id}",
            })
    failure_counts: dict[tuple[str, str], int] = defaultdict(int)
    action_counts: dict[str, int] = defaultdict(int)
    for item in trajectories:
        action = str(item.get("action", "unknown"))
        action_counts[action] += 1
        if item.get("error"):
            failure_counts[(action, str(item["error"]).split(":", 1)[0])] += 1
    for action in ACTION_ORDER:
        rows.append({
            "section": "failure_analysis", "group": action, "metric": "failure_rate",
            "n_cases": len({str(x.get("case_id")) for x in trajectories if x.get("action") == action}),
            "n_states": action_counts[action],
            "value": sum(count for (a, _), count in failure_counts.items() if a == action) / action_counts[action] if action_counts[action] else "",
            "ci95_low": "", "ci95_high": "", "notes": "Invalid or nonterminal rollout; see error-type rows below.",
        })
    for (action, error_type), count in sorted(failure_counts.items()):
        rows.append({
            "section": "failure_analysis", "group": action, "metric": f"error_count:{error_type}",
            "n_cases": len({str(x.get("case_id")) for x in trajectories if x.get("action") == action and str(x.get("error", "")).startswith(error_type)}),
            "n_states": action_counts[action], "value": count,
            "ci95_low": "", "ci95_high": "", "notes": "Errors are retained in the raw JSONL with their full failure reason.",
        })
    return rows


def write_report(path: Path, manifest: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# FEASIBILITY PILOT REPORT",
        "",
        f"Status: **{str(manifest.get('status', 'unknown')).upper()}**  ",
        f"Created: {manifest.get('created_at_utc', 'unknown')}  ",
        f"Project commit: `{manifest.get('project_git_commit')}`  ",
        f"Pilot seed: `{manifest.get('inference', {}).get('seed', 'not run')}`",
        "",
        "## Scope and protocol",
        "",
        f"This inference-only pilot does not train a router or run joint ATPO. It samples {manifest.get('inference', {}).get('num_cases', 100)} training cases with seed {manifest.get('inference', {}).get('seed', 'unknown')}; each case can contribute up to {manifest.get('inference', {}).get('max_states_per_case', 3)} public states. Each state is branched into ANSWER, ASK, CONSULT, and DEFER with identical visible history, with {manifest.get('inference', {}).get('rollouts_per_action', 3)} rollouts per action (at most {int(manifest.get('inference', {}).get('num_cases', 100)) * int(manifest.get('inference', {}).get('max_states_per_case', 3)) * 4 * int(manifest.get('inference', {}).get('rollouts_per_action', 3))} action trajectories). Public states are built with a temperature-0 base/patient trajectory; outcome branches use the configured rollout temperature.",
        "ASK sends the generated question to the patient simulator, whose prompt contains the patient facts; the patient's answer becomes public dialogue. CONSULT sends an agent-generated focused question plus only the public case/dialogue to the expert simulator; the advice is returned to the base agent. DEFER sends that same public state to the expert for a terminal answer. Expert prompts are checked for hidden atomic facts and privileged key names. Gold is not passed to prompts or pre-call validators; it is read only after a branch returns for post-hoc scoring.",
        "",
        "The expert-reliability slices are post-hoc descriptions using observed DEFER correctness. They are not model inputs, do not represent a controlled expert-quality intervention, and do not establish a causal change in reliability. The configured cost grid penalizes expert tokens and patient turns in utility units; it is a sensitivity analysis, not a measured monetary price. Actual token use and latency are recorded only after successful calls.",
        "",
        "## Environment recovery",
        "",
        f"- Train Parquet: `{manifest.get('data_path')}` (exists: `{manifest.get('data_exists')}`; SHA-256: `{manifest.get('data_sha256')}`)",
        f"- Official upstream source snapshot: `{manifest.get('source', {}).get('repository')}` at `{manifest.get('source', {}).get('commit')}`.",
        f"- Source recovery note: {manifest.get('source', {}).get('local_recovery_status', 'No local source copy recorded.')}",
        f"- Data/schema audit: `{json.dumps(manifest.get('data_adapter_manifest', {}), ensure_ascii=False)}`",
        "- The [official ATPO README](https://github.com/Quark-Medical/ATPO) describes adapted public-benchmark JSONL inputs and conversion to Parquet; the [upstream dataset directory](https://github.com/Quark-Medical/ATPO/tree/main/dataset) currently lists `merged_train_dataset.jsonl` and `medqa_test_dataset.jsonl`. The README names `rl_train_dataset.jsonl` for training while its conversion command uses the extensionless `rl_train_dataset` prefix; the pinned restore script follows the actual filename in the dataset tree. This environment did not contain the historical Parquet files. Any regenerated Parquet is labeled reconstructed and written to a separate assets directory.",
        f"- Configured model aliases: `{json.dumps(manifest.get('models', {}), ensure_ascii=False)}`; exact checkpoint revisions are unverified unless returned by a live model service.",
        f"- Historical checkpoint paths checked: `{json.dumps(manifest.get('historical_checkpoint_inventory', []), ensure_ascii=False)}`",
        f"- GPU snapshot: `{json.dumps(manifest.get('gpus', []), ensure_ascii=False)}`",
        f"- Runtime modules: `{json.dumps(manifest.get('dependencies', {}), ensure_ascii=False)}`",
        f"- Interpreter: `{manifest.get('python_executable')}` (Python `{manifest.get('python_version')}`)",
        f"- Tokenizer/model cache: `{json.dumps(manifest.get('local_tokenizer_inventory', {}), ensure_ascii=False)}`",
        f"- Model files: `{json.dumps(manifest.get('model_manifest', {}), ensure_ascii=False)}`",
        f"- Model service: `{json.dumps(manifest.get('model_service', {}), ensure_ascii=False)}`",
        f"- Endpoint check: `{json.dumps(manifest.get('endpoints', {}), ensure_ascii=False)}`",
        "",
        "## Results",
        "",
    ]
    if not rows:
        lines.append("No model trajectories were collected. No accuracy, correction, degradation, value-gap, reliability-slice, latency, token-use, or cost-sensitivity result is available. The configured cost grid is only a planned analysis and must not be read as an experimental result.")
        lines.append("")
    else:
        lines.append("| Section | Group | Metric | N cases | N states | Value | 95% CI | Notes |")
        lines.append("|---|---|---|---:|---:|---:|---|---|")
        for row in rows:
            ci = f"[{row.get('ci95_low', '')}, {row.get('ci95_high', '')}]" if row.get("ci95_low") != "" else ""
            lines.append("| " + " | ".join(str(row.get(key, "")).replace("|", "\\|") for key in ("section", "group", "metric", "n_cases", "n_states", "value")) + f" | {ci} | {row.get('notes', '')} |")
        lines.append("")
    if manifest.get("blockers"):
        lines.extend(["## Blockers", ""])
        lines.extend(f"- {item}" for item in manifest["blockers"])
        lines.append("")
    if manifest.get("failure"):
        lines.extend(["## Run failure", "", str(manifest["failure"]), ""])
    lines.extend([
        "## Deliverables",
        "",
        "- `pilot_trajectories.jsonl`: trajectory records after a successful run. While blocked, it contains only an explicit `run_status` record and zero trajectories.",
        "- `summary.csv`: action outcomes, paired CONSULT comparisons, observed expert-reliability slices, and cost sensitivity after a successful run; otherwise a blocked-status row.",
        "- `run_manifest.json`: code/data/config hashes, model aliases, environment, GPU snapshot, endpoint checks, and blockers.",
        "- `model_service_config.yaml`: pinned HF model revision, local path, serving runtime, and GPU limits.",
        "- `run_pilot.py`: deterministic case sampler and paired inference runner.",
        "",
        "## Interpretation",
        "",
        "Do not treat synthetic routing results from other experiment folders as real model performance. The Base, Patient, and Expert roles all use the same Qwen3-8B model with different prompts and information access; Patient and Expert are simulated agents, not real clinicians. A completed pilot supports only feasibility and descriptive comparisons on this benchmark; it is not clinical validation.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_completed_artifacts(output_dir: Path, manifest: dict[str, Any], records: Sequence[Mapping[str, Any]], config: Mapping[str, Any], seed: int) -> None:
    manifest["status"] = "completed"
    manifest["trajectory_count"] = sum(r.get("record_type") == "trajectory" for r in records)
    trajectories = [r for r in records if r.get("record_type") == "trajectory"]
    manifest["valid_trajectory_count"] = sum(bool(r.get("valid")) for r in trajectories)
    manifest["failed_trajectory_count"] = len(trajectories) - manifest["valid_trajectory_count"]
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = summarize(records, config, seed)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    write_report(output_dir / "FEASIBILITY_PILOT_REPORT.md", manifest, rows)
    write_report(output_dir / "REAL_PILOT_REPORT.md", manifest, rows)
    analysis_sections = {
        "action", "comparison", "revision", "cost", "dominance", "cost_sensitivity",
        "policy_tradeoff", "expert_reliability_slice", "consult_gain_distribution", "failure_analysis",
    }
    with (output_dir / "action_value_analysis.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(row for row in rows if row.get("section") in analysis_sections)


async def _build_states(row: Mapping[str, Any], source_index: int, base: Any, patient: Any, max_states: int, max_turns: int, state_temperature: float):
    from recipe.ea_atpo.protocol import PublicState, parse_action, visible_response

    extra = row["extra_info"]
    initial = extra["prompt"]
    if hasattr(initial, "tolist"):
        initial = initial.tolist()
    initial_messages = tuple({"role": str(item["role"]), "content": str(item["content"])} for item in initial)
    raw_facts = extra.get("atomic_facts", [])
    if hasattr(raw_facts, "tolist"):
        raw_facts = raw_facts.tolist()
    facts = [str(item) for item in raw_facts] if isinstance(raw_facts, (list, tuple)) else [str(raw_facts)]
    ground_truth = row.get("reward_model", {}).get("ground_truth", extra.get("ground_truth"))
    case_id = f"row-{source_index}:item-{extra.get('index', row.get('index', source_index))}"
    current = PublicState(case_id, initial_messages)
    construction_trace: list[dict[str, Any]] = []
    states = []
    for state_index in range(min(max_states, max_turns)):
        states.append((current, facts, ground_truth, list(construction_trace), case_id, state_index))
        if state_index + 1 >= min(max_states, max_turns):
            break
        call = await base.generate(current.messages, temperature=state_temperature)
        try:
            action, content = parse_action(call.content)
        except ValueError:
            construction_trace.append({"actor": "base", "error": "invalid_format", "prompt_tokens": call.prompt_tokens, "completion_tokens": call.completion_tokens, "latency_s": call.latency_s})
            break
        action_text = f"Question: {content}" if action.value == "ASK" else (f"Final Answer: {content}" if action.value == "ANSWER" else f"{action.value}: {content}")
        construction_trace.append({"actor": "base", "action": action.value, "response": action_text, "prompt_tokens": call.prompt_tokens, "completion_tokens": call.completion_tokens, "latency_s": call.latency_s})
        if action.value != "ASK":
            break
        patient_call = await patient.generate([
            {"role": "system", "content": "Answer only from the supplied patient facts. If no supplied fact answers the question, reply exactly: The patient cannot answer this question. Give only a concise patient response."},
            {"role": "user", "content": json.dumps({"patient_facts": facts, "question": content}, ensure_ascii=False)},
        ], temperature=0.0, max_tokens=256)
        response = visible_response(patient_call.content)
        construction_trace.append({"actor": "patient", "response": response, "prompt_tokens": patient_call.prompt_tokens, "completion_tokens": patient_call.completion_tokens, "latency_s": patient_call.latency_s})
        current = PublicState(
            current.sample_id,
            current.initial_messages,
            (*current.dialogue, {"role": "assistant", "content": action_text}, {"role": "user", "content": response}),
        )
    return states


async def execute_pilot(args: argparse.Namespace, config: dict[str, Any], manifest: dict[str, Any], output_dir: Path) -> None:
    import pandas as pd
    from recipe.ea_atpo.collect_counterfactuals import ChatAPI, rollout_action
    from recipe.ea_atpo.protocol import Action, PublicState

    data_path = Path(args.data)
    frame = pd.read_parquet(data_path)
    if len(frame) < args.cases:
        raise ValueError(f"train split has {len(frame)} rows; requested {args.cases} cases")
    selected_indices = random.Random(args.seed).sample(range(len(frame)), args.cases)
    trajectory_path = output_dir / "pilot_trajectories.jsonl"
    states_path = output_dir / "pilot_public_states.jsonl"
    manifest_path = output_dir / "run_manifest.json"
    if args.resume and manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        immutable_keys = (
            "project_git_commit", "pilot_script_sha256", "collector_sha256", "protocol_sha256",
            "data_sha256", "config_sha256", "python_version", "python_executable",
        )
        mismatches = [key for key in immutable_keys if previous.get(key) != manifest.get(key)]
        if mismatches:
            raise ValueError("cannot resume with changed run inputs: " + ", ".join(mismatches))
        if previous.get("selected_cases") not in (None, args.cases) or previous.get("sampled_source_rows") not in (None, selected_indices):
            raise ValueError("cannot resume with a different case sample")
    elif not args.resume and (trajectory_path.exists() or states_path.exists()):
        raise FileExistsError("output already contains pilot state/trajectory JSONL; use --resume or a new output directory")

    manifest.update({
        "data_sha256": sha256_file(data_path),
        "train_rows": len(frame),
        "sample_seed": args.seed,
        "sampled_source_rows": selected_indices,
        "selected_cases": args.cases,
        "max_states_per_case": args.max_states_per_case,
        "rollouts_per_action": args.rollouts_per_action,
        "max_consults_per_rollout": int(config.get("max_consults_per_rollout", 2)),
    })
    manifest["inference"].update({
        "num_cases": args.cases,
        "seed": args.seed,
        "max_states_per_case": args.max_states_per_case,
        "rollouts_per_action": args.rollouts_per_action,
        "temperature": args.temperature,
    })
    manifest["status"] = "running"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    def read_records(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        loaded = []
        valid_bytes = 0
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.endswith("\n"):
                    tail_path = path.with_suffix(path.suffix + ".truncated_tail")
                    tail_path.write_text(line, encoding="utf-8")
                    with path.open("r+b") as repair:
                        repair.truncate(valid_bytes)
                    break
                try:
                    loaded.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
                valid_bytes += len(line.encode("utf-8"))
        return loaded

    previous_trajectories = [r for r in read_records(trajectory_path) if r.get("record_type") == "trajectory"]
    existing_keys = {
        (str(r["case_id"]), str(r["state_id"]), str(r["action"]), int(r["rollout_index"]))
        for r in previous_trajectories
    }
    previous_states = [r for r in read_records(states_path) if r.get("record_type") == "public_state"]
    states_by_source: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in previous_states:
        states_by_source[int(record["source_row_index"])].append(record)
    for records in states_by_source.values():
        records.sort(key=lambda r: int(r["state_index"]))

    clients = {}
    for name in ("base", "patient", "expert"):
        settings = config["endpoints"][name]
        clients[name] = ChatAPI(
            settings["base_url"], settings["model"],
            timeout_s=float(config.get("timeout_s", 180)),
            enable_thinking=settings.get("enable_thinking"),
        )
        key_env = settings.get("api_key_env")
        api_key = os.environ.get(str(key_env), "") if key_env else ""
        if api_key:
            clients[name].client.headers.update({"Authorization": f"Bearer {api_key}"})

    try:
        with trajectory_path.open("a", encoding="utf-8") as trajectory_handle, states_path.open("a", encoding="utf-8") as state_handle:
            trajectory_handle.write(json.dumps({"record_type": "run_status", "status": "running", "seed": args.seed, "resumed": bool(args.resume)}, ensure_ascii=False) + "\n")
            trajectory_handle.flush()
            for source_index in selected_indices:
                row = frame.iloc[source_index].to_dict()
                if source_index not in states_by_source:
                    case_states = await _build_states(
                        row, source_index, clients["base"], clients["patient"],
                        args.max_states_per_case, int(config.get("max_assistant_turns", 10)),
                        float(config.get("state_temperature", 0.0)),
                    )
                    for state, _facts, _gold, state_trace, case_id, state_index in case_states:
                        saved = {
                            "record_type": "public_state",
                            "source_row_index": source_index,
                            "state_index": state_index,
                            "case_id": case_id,
                            "state_id": state.state_id,
                            "sample_id": state.sample_id,
                            "initial_messages": list(state.initial_messages),
                            "dialogue": list(state.dialogue),
                            "state_construction_trace": state_trace,
                        }
                        state_handle.write(json.dumps(saved, ensure_ascii=False) + "\n")
                        state_handle.flush()
                        states_by_source[source_index].append(saved)

                extra = row["extra_info"]
                raw_facts = extra.get("atomic_facts", [])
                if hasattr(raw_facts, "tolist"):
                    raw_facts = raw_facts.tolist()
                facts = [str(item) for item in raw_facts] if isinstance(raw_facts, (list, tuple)) else [str(raw_facts)]
                ground_truth = row.get("reward_model", {}).get("ground_truth", extra.get("ground_truth"))
                case_states = []
                for saved in states_by_source[source_index]:
                    state = PublicState(
                        str(saved["sample_id"]),
                        tuple(dict(x) for x in saved["initial_messages"]),
                        tuple(dict(x) for x in saved["dialogue"]),
                    )
                    if state.state_id != saved["state_id"]:
                        raise ValueError(f"cached public state hash mismatch for {saved['state_id']}")
                    case_states.append((
                        state, facts, ground_truth, saved.get("state_construction_trace", []),
                        str(saved["case_id"]), int(saved["state_index"]),
                    ))

                for state, facts, ground_truth, state_trace, case_id, state_index in case_states:
                    for rollout_index in range(args.rollouts_per_action):
                        pending_actions = [
                            action for action in Action
                            if (case_id, state.state_id, action.value, rollout_index) not in existing_keys
                        ]
                        if not pending_actions:
                            continue
                        branches = await asyncio.gather(*[
                            rollout_action(
                                action, state, facts, ground_truth, clients["base"], clients["patient"], clients["expert"],
                                temperature=args.temperature,
                                max_turns=int(config.get("max_assistant_turns", 10)),
                                max_consults=int(config.get("max_consults_per_rollout", 2)),
                            ) for action in pending_actions
                        ])
                        for action, result in zip(pending_actions, branches):
                            branch_trace = result.get("trace", [])
                            consult_questions = [x.get("consult_question") for x in branch_trace if x.get("consult_question")]
                            expert_advice = [x.get("response") for x in branch_trace if x.get("actor") == "expert" and x.get("action") == "CONSULT"]
                            patient_responses = [x.get("response") for x in branch_trace if x.get("actor") == "patient"]
                            expert_tokens = int(result.get("expert_prompt_tokens", 0)) + int(result.get("expert_completion_tokens", 0))
                            total_tokens = int(result.get("total_prompt_tokens", 0)) + int(result.get("total_completion_tokens", 0))
                            action_cost_grid = {}
                            for lambda_e in config.get("cost_sensitivity", {}).get("lambda_expert_per_1k_tokens", [0.0]):
                                for lambda_p in config.get("cost_sensitivity", {}).get("lambda_patient_per_turn", [0.0]):
                                    label = f"lambdaE={float(lambda_e):g};lambdaP={float(lambda_p):g}"
                                    action_cost_grid[label] = float(lambda_e) * expert_tokens / 1000.0 + float(lambda_p) * int(result.get("patient_turns", 0))
                            record = {
                                "record_type": "trajectory",
                                "case_id": case_id,
                                "source_row_index": source_index,
                                "state_index": state_index,
                                "state_id": state.state_id,
                                "public_state": state.messages,
                                "state_construction_trace": state_trace,
                                "rollout_index": rollout_index,
                                "consult_question": consult_questions,
                                "expert_advice": expert_advice,
                                "patient_response": patient_responses,
                                "final_answer": result.get("answer"),
                                "correctness": result.get("outcome"),
                                "failure_reason": result.get("error"),
                                "expert_cost_tokens": expert_tokens,
                                "action_cost": {
                                    "total_tokens": total_tokens,
                                    "latency_seconds": result.get("total_latency_s", 0.0),
                                    "patient_turns": result.get("patient_turns", 0),
                                    "expert_tokens": expert_tokens,
                                    "monetary_cost_usd": None,
                                    "sensitivity_penalty_units": action_cost_grid,
                                },
                                **result,
                            }
                            trajectory_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                            trajectory_handle.flush()
                            existing_keys.add((case_id, state.state_id, action.value, rollout_index))
        all_records = read_records(trajectory_path)
        records = [r for r in all_records if r.get("record_type") == "trajectory"]
        with trajectory_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"record_type": "run_status", "status": "completed", "trajectory_count": len(records)}, ensure_ascii=False) + "\n")
    finally:
        await asyncio.gather(*(client.close() for client in clients.values()), return_exceptions=True)
    write_completed_artifacts(output_dir, manifest, records, config, args.seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(PILOT_DIR / "assets/train.pilot.parquet"))
    parser.add_argument("--config", default=str(PILOT_DIR / "pilot_config.yaml"))
    parser.add_argument("--output-dir", default=str(PILOT_DIR / "outputs"))
    parser.add_argument("--cases", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-states-per-case", type=int, default=None)
    parser.add_argument("--rollouts-per-action", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume from the saved public states and completed case/state/action rollouts.")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = read_config(config_path)
    args.seed = int(config.get("seed", 20261103) if args.seed is None else args.seed)
    args.max_states_per_case = int(config.get("max_states_per_case", 3) if args.max_states_per_case is None else args.max_states_per_case)
    args.rollouts_per_action = int(config.get("rollouts_per_action", 3) if args.rollouts_per_action is None else args.rollouts_per_action)
    args.temperature = float(config.get("temperature", 0.2) if args.temperature is None else args.temperature)
    output_dir = Path(args.output_dir)
    manifest = build_manifest(Path(args.data), config_path, config)
    if manifest["blockers"] or args.preflight_only:
        if args.preflight_only and not manifest["blockers"]:
            manifest["status"] = "preflight_only"
        existing_trace = output_dir / "pilot_trajectories.jsonl"
        if args.resume and manifest["blockers"] and existing_trace.is_file():
            manifest["status"] = "blocked_resume"
            try:
                old_lines = existing_trace.read_text(encoding="utf-8").splitlines()
                old_records = [json.loads(line) for line in old_lines if line.strip()]
                trajectory_count = sum(record.get("record_type") == "trajectory" for record in old_records)
            except (OSError, json.JSONDecodeError):
                trajectory_count = -1
            manifest["trajectory_count"] = trajectory_count
            with existing_trace.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "record_type": "run_status", "status": "blocked_resume",
                    "created_at_utc": manifest["created_at_utc"],
                    "blockers": manifest["blockers"],
                }, ensure_ascii=False) + "\n")
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            write_report(output_dir / "FEASIBILITY_PILOT_REPORT.md", manifest, [])
            write_report(output_dir / "REAL_PILOT_REPORT.md", manifest, [])
            print(json.dumps({"status": manifest["status"], "trajectory_count": trajectory_count, "blockers": manifest["blockers"], "output_dir": str(output_dir)}, ensure_ascii=False, indent=2))
            return
        write_blocked_artifacts(output_dir, manifest)
        print(json.dumps({"status": manifest["status"], "blockers": manifest["blockers"], "output_dir": str(output_dir)}, ensure_ascii=False, indent=2))
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        asyncio.run(execute_pilot(args, config, manifest, output_dir))
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failure"] = f"{type(exc).__name__}: {exc}"
        trajectory_path = output_dir / "pilot_trajectories.jsonl"
        if trajectory_path.exists():
            with trajectory_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"record_type": "run_status", "status": "failed", "reason": manifest["failure"]}, ensure_ascii=False) + "\n")
            manifest["trajectory_count"] = sum(
                1 for line in trajectory_path.read_text(encoding="utf-8").splitlines()
                if line and json.loads(line).get("record_type") == "trajectory"
            )
        else:
            manifest["trajectory_count"] = 0
        (output_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        write_report(output_dir / "FEASIBILITY_PILOT_REPORT.md", manifest, [])
        write_report(output_dir / "REAL_PILOT_REPORT.md", manifest, [])
        raise
    print(json.dumps({"status": "completed", "output_dir": str(output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
