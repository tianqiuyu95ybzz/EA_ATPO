#!/usr/bin/env python3
"""Safe lifecycle and health checks for the VI-EA-ATPO Phase 2 vLLM services."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[3]
PHASE2 = Path(__file__).resolve().parent
STAGE = ROOT / "ea_atpo_stage"
sys.path.insert(0, str(STAGE))

from recipe.ea_atpo.collect_counterfactuals import (  # noqa: E402
    ChatAPI,
    _base_action,
    _defer_to_expert,
    _patient_messages,
    _call_metrics,
)
from recipe.ea_atpo.protocol import Action, PublicState, visible_response  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError(f"Unsupported service configuration: {path}")
    return config


def service_url(config: Mapping[str, Any], service: Mapping[str, Any]) -> str:
    return f"http://{config['network']['host']}:{int(service['port'])}/v1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run_text(command: list[str], timeout: int = 20) -> str:
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=timeout)
    return result.stdout.strip()


def gpu_snapshot(config: Mapping[str, Any]) -> dict[str, Any]:
    index = int(config["gpu"]["physical_index"])
    line = run_text([
        "nvidia-smi", f"--id={index}",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ])
    columns = [part.strip() for part in line.split(",")]
    if len(columns) != 6:
        raise RuntimeError(f"Could not parse GPU status: {line}")
    result = {
        "physical_index": int(columns[0]), "name": columns[1],
        "total_mib": int(columns[2]), "used_mib": int(columns[3]),
        "free_mib": int(columns[4]), "utilization_pct": int(columns[5]),
    }
    if result["physical_index"] != index or result["name"] != config["gpu"]["expected_name"]:
        raise RuntimeError(f"GPU identity mismatch: expected index/name {index}/{config['gpu']['expected_name']}, got {result}")
    return result


def gpu_compute_pids(config: Mapping[str, Any]) -> list[int]:
    index = int(config["gpu"]["physical_index"])
    try:
        output = run_text([
            "nvidia-smi", f"--id={index}", "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ])
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Could not query compute processes on GPU {index}: {exc.stderr}") from exc
    pids: list[int] = []
    for item in output.splitlines():
        item = item.strip()
        if item and item.upper() != "N/A":
            pids.append(int(item))
    return sorted(set(pids))


def process_group(pid: int) -> int | None:
    try:
        return os.getpgid(pid)
    except (ProcessLookupError, PermissionError):
        return None


def process_cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return ""


def port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, int(port))) == 0


def get_models(url: str, timeout: float = 3.0) -> tuple[bool, list[str], str | None]:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/models", timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        return True, [str(item.get("id", "")) for item in body.get("data", [])], None
    except Exception as exc:  # report the actual local service status
        return False, [], f"{type(exc).__name__}: {exc}"


def verify_model_assets(role: str, service: Mapping[str, Any]) -> dict[str, Any]:
    model_path = Path(str(service["model_path"]))
    config_path = model_path / "config.json"
    index_path = model_path / "model.safetensors.index.json"
    tokenizer_path = model_path / "tokenizer_config.json"
    for required in (config_path, index_path, tokenizer_path):
        if not required.is_file():
            raise FileNotFoundError(f"{role}: required model file missing: {required}")
    model_cfg = json.loads(config_path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    tokenizer_cfg = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    shards = sorted(set(index.get("weight_map", {}).values()))
    if not shards:
        raise ValueError(f"{role}: safetensors index has no weight shards")
    shard_rows = []
    for name in shards:
        shard = model_path / name
        if not shard.is_file() or shard.stat().st_size <= 0:
            raise FileNotFoundError(f"{role}: model shard missing or empty: {shard}")
        shard_rows.append({"name": name, "size_bytes": shard.stat().st_size})
    if model_cfg.get("model_type") != "qwen3" or not tokenizer_cfg.get("chat_template"):
        raise ValueError(f"{role}: expected Qwen3 model and model-native tokenizer chat template")

    integrity: dict[str, Any]
    if role == "base":
        revision_tree = model_path / ".cache/huggingface/trees" / f"{service['model_revision']}.json"
        if not revision_tree.is_file():
            raise FileNotFoundError(f"Base revision snapshot proof missing: {revision_tree}")
        tree = json.loads(revision_tree.read_text(encoding="utf-8"))
        tracked = tree.get("files", {})
        for row in shard_rows:
            expected_size = (tracked.get(row["name"]) or {}).get("lfs_size")
            if expected_size is None:
                expected_size = (tracked.get(row["name"]) or {}).get("size")
            if expected_size is None or int(expected_size) != row["size_bytes"]:
                raise ValueError(f"Base shard differs from cached revision tree: {row['name']}")
        integrity = {"source": "Hugging Face cached revision tree", "revision": service["model_revision"],
                     "all_indexed_shards_present_and_size_matched": True}
        if model_cfg.get("quantization_config") is not None or model_cfg.get("torch_dtype") != "bfloat16":
            raise ValueError("Base checkpoint is not the expected BF16 Qwen3-8B")
    else:
        lock_path = model_path / "model_source_lock.json"
        status_path = model_path / "range_download_status.json"
        if not lock_path.is_file() or not status_path.is_file():
            raise FileNotFoundError("AWQ source lock or range download verification record missing")
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        status = json.loads(status_path.read_text(encoding="utf-8"))
        expected = str(service["model_revision"])
        if lock.get("huggingface_revision") != expected or status.get("revision") != expected:
            raise ValueError("AWQ model revision does not match the pinned service revision")
        if status.get("status") != "all_shards_verified":
            raise ValueError("AWQ model shards are not marked all_shards_verified")
        completed = set(status.get("completed_shards", []))
        if not set(shards).issubset(completed):
            raise ValueError("AWQ verification manifest does not contain every indexed shard")
        lock_sizes = {item["name"]: int(item["size"]) for item in lock.get("weight_files", [])}
        if any(lock_sizes.get(row["name"]) != row["size_bytes"] for row in shard_rows):
            raise ValueError("AWQ shard sizes do not match the pinned model source lock")
        quant = model_cfg.get("quantization_config") or {}
        expected_quant = service["quantization_config"]
        for key in ("quant_method", "bits", "group_size", "version", "zero_point"):
            if quant.get(key) != expected_quant.get(key):
                raise ValueError(f"AWQ quantization config mismatch for {key}: {quant.get(key)!r}")
        integrity = {"source": "model_source_lock.json + range_download_status.json",
                     "revision": expected, "status": status.get("status"),
                     "all_indexed_shards_present_and_size_matched": True,
                     "shard_sha256_recomputed_this_run": False}

    return {
        "role": role,
        "model_id": service["model_id"],
        "model_revision": service["model_revision"],
        "model_path": str(model_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "index_sha256": sha256_file(index_path),
        "shard_count": len(shard_rows),
        "total_weight_bytes": sum(item["size_bytes"] for item in shard_rows),
        "shards": shard_rows,
        "integrity_evidence": integrity,
        "model_config_quantization": model_cfg.get("quantization_config"),
        "tokenizer_has_chat_template": True,
    }


def validate_runtime(config: Mapping[str, Any]) -> dict[str, Any]:
    runtime = config["runtime"]
    expected_python = Path(runtime["python"]).resolve()
    if Path(sys.executable).resolve() != expected_python:
        raise RuntimeError(f"Run this command with configured Python {expected_python}, got {sys.executable}")
    installed = {name: importlib.metadata.version(name) for name in ("vllm", "torch", "transformers", "aiohttp", "PyYAML")}
    if installed["vllm"] != str(runtime["expected_vllm_version"]):
        raise RuntimeError(f"vLLM version mismatch: expected {runtime['expected_vllm_version']}, got {installed['vllm']}")
    if installed["torch"].split("+")[0] != str(runtime["expected_torch_version"]):
        raise RuntimeError(f"Torch version mismatch: expected {runtime['expected_torch_version']}, got {installed['torch']}")
    if not Path(runtime["vllm"]).is_file() or not os.access(runtime["vllm"], os.X_OK):
        raise FileNotFoundError(f"vLLM executable unavailable: {runtime['vllm']}")
    return {"python": sys.version, "python_executable": str(expected_python), "packages": installed}


def verify_torch_gpu(config: Mapping[str, Any]) -> dict[str, Any]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpu"]["cuda_visible_devices"])
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Configured CUDA_VISIBLE_DEVICES must expose exactly one available GPU")
    name = torch.cuda.get_device_name(0)
    if name != config["gpu"]["expected_name"]:
        raise RuntimeError(f"CUDA logical device 0 maps to {name}, expected {config['gpu']['expected_name']}")
    return {"cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "logical_device": 0,
            "physical_gpu_index": int(config["gpu"]["physical_index"]), "device_name": name,
            "compute_capability": list(torch.cuda.get_device_capability(0)), "cuda_available": True}


def manifest_path(config: Mapping[str, Any], role: str) -> Path:
    return Path(config["artifacts"]["service_dir"]) / role / "service_manifest.json"


def service_pid_path(config: Mapping[str, Any], role: str) -> Path:
    return Path(config["artifacts"]["service_dir"]) / "pids" / f"{role}.pid"


def write_state(config: Mapping[str, Any], state: Mapping[str, Any]) -> None:
    atomic_json(Path(config["artifacts"]["state_file"]), state)


def read_state(config: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(config["artifacts"]["state_file"])
    if not path.is_file():
        return {"status": "not_started", "services": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def check_service_identity(config: Mapping[str, Any], role: str) -> tuple[bool, dict[str, Any]]:
    service = config["services"][role]
    url = service_url(config, service)
    reachable, ids, error = get_models(url)
    expected_id = service["served_model_name"]
    if not reachable:
        return False, {"role": role, "url": url + "/models", "reachable": False, "error": error}
    path = manifest_path(config, role)
    manifest = None
    if path.is_file():
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = None
    matched = bool(
        expected_id in ids and manifest
        and manifest.get("served_model_name") == expected_id
        and manifest.get("model_revision") == service["model_revision"]
        and manifest.get("model_path") == str(Path(service["model_path"]).resolve())
        and manifest.get("framework_version") == str(config["runtime"]["expected_vllm_version"])
    )
    return matched, {
        "role": role, "url": url + "/models", "reachable": True,
        "model_ids": ids, "expected_alias": expected_id,
        "expected_revision": service["model_revision"],
        "service_manifest_path": str(path), "service_manifest_found": manifest is not None,
        "service_manifest_matches": matched, "error": None if matched else "alias/revision/path/framework did not all match",
    }


def build_environment(config: Mapping[str, Any], role: str) -> dict[str, str]:
    env = os.environ.copy()
    service = config["services"][role]
    runtime = config["runtime"]
    gpu = config["gpu"]
    cache = Path(runtime["cache_root"]) / role
    for name in ("hf_home", "xdg", "torch", "triton", "cuda", "vllm", "tmp", "flashinfer"):
        (cache / name).mkdir(parents=True, exist_ok=True)
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu["cuda_visible_devices"]),
        "CUDA_HOME": str(runtime["cuda_home"]),
        "HF_HOME": str(cache / "hf_home"),
        "HF_HUB_CACHE": str(cache / "hf_home" / "hub"),
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "XDG_CACHE_HOME": str(cache / "xdg"),
        "TORCH_HOME": str(cache / "torch"),
        "TRITON_CACHE_DIR": str(cache / "triton"),
        "CUDA_CACHE_PATH": str(cache / "cuda"),
        "VLLM_CACHE_ROOT": str(cache / "vllm"),
        "TMPDIR": str(Path(f"{runtime['tmp_root']}-{os.getuid()}") / role),
        "FLASHINFER_WORKSPACE_BASE": str(cache / "flashinfer"),
        "VLLM_USE_FLASHINFER_SAMPLER": "0" if runtime.get("disable_flashinfer_sampler") else "1",
        "PATH": str(Path(runtime["python"]).parent) + os.pathsep + str(Path(runtime["cuda_home"]) / "bin") + os.pathsep + env.get("PATH", ""),
    })
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    return env


def build_command(config: Mapping[str, Any], service: Mapping[str, Any]) -> list[str]:
    runtime = config["runtime"]
    gpu = config["gpu"]
    cmd = [
        str(runtime["vllm"]), "serve", str(service["model_path"]),
        "--host", str(config["network"]["host"]), "--port", str(service["port"]),
        "--served-model-name", str(service["served_model_name"]),
        "--dtype", str(service["dtype"]),
        "--max-model-len", str(service["max_model_len"]),
        "--max-num-seqs", str(service["max_num_seqs"]),
        "--gpu-memory-utilization", str(service["gpu_memory_utilization"]),
        "--attention-backend", str(service["attention_backend"]),
        "--tensor-parallel-size", str(gpu["tensor_parallel_size"]),
    ]
    if service.get("quantization"):
        cmd.extend(["--quantization", str(service["quantization"])])
    return cmd


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()


def owned_groups(state: Mapping[str, Any]) -> set[int]:
    return {
        int(item["process_group_id"])
        for item in state.get("services", {}).values()
        if item.get("owned_by_this_manager") and item.get("process_group_id") is not None
        and process_group(int(item["pid"])) == int(item["process_group_id"])
    }


def gate_gpu(config: Mapping[str, Any], min_free_mib: int, max_util: int,
             allowed_groups: set[int], require_unoccupied: bool,
             allowed_pids: set[int] | None = None) -> dict[str, Any]:
    snapshot = gpu_snapshot(config)
    pids = gpu_compute_pids(config)
    allowed_pids = allowed_pids or set()
    foreign = []
    for pid in pids:
        pgid = process_group(pid)
        if pid not in allowed_pids and pgid not in allowed_groups:
            foreign.append({"pid": pid, "process_group_id": pgid, "cmdline": process_cmdline(pid)[:300]})
    if foreign and (require_unoccupied or not config["gpu"].get("allow_foreign_compute_processes", False)):
        raise RuntimeError(f"Refusing to share physical GPU {snapshot['physical_index']} with foreign compute PIDs: {foreign}")
    if snapshot["free_mib"] < int(min_free_mib):
        raise RuntimeError(f"GPU {snapshot['physical_index']} has {snapshot['free_mib']} MiB free; require {min_free_mib} MiB")
    if snapshot["utilization_pct"] > int(max_util):
        raise RuntimeError(f"GPU {snapshot['physical_index']} utilization {snapshot['utilization_pct']}% exceeds {max_util}% gate")
    return {**snapshot, "compute_pids": pids, "foreign_processes": foreign, "gate_min_free_mib": min_free_mib,
            "gate_max_utilization_pct": max_util, "gate_passed": True}


def preflight(config: dict[str, Any], config_path: Path) -> int:
    runtime = validate_runtime(config)
    gpu_runtime = verify_torch_gpu(config)
    assets = {role: verify_model_assets(role, service) for role, service in config["services"].items()}
    pending = []
    current_services = {}
    for role in ("base", "expert"):
        reused = reuse_or_refuse(config, role)
        current_services[role] = reused
        if reused is None:
            pending.append(role)
    gpu_gate = None
    if pending:
        gpu_gate = gate_gpu(
            config,
            int(config["gpu"]["min_free_mib_before_any_new_service"]),
            int(config["gpu"]["max_utilization_pct_before_start"]),
            set(), require_unoccupied=True, allowed_pids={os.getpid()},
        )
    result = {
        "status": "ready_to_start" if pending and gpu_gate and gpu_gate["gate_passed"] else "services_reusable" if not pending else "blocked",
        "config_path": str(config_path.resolve()),
        "runtime": runtime, "cuda_mapping": gpu_runtime, "assets": assets,
        "pending_services": pending, "reusable_services": current_services,
        "gpu_gate": gpu_gate,
        "estimated_combined_vllm_budget_mib": sum(
            int(gpu_snapshot(config)["total_mib"] * float(config["services"][role]["gpu_memory_utilization"]))
            for role in ("base", "expert")
        ),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] in ("ready_to_start", "services_reusable") else 1


def create_manifest(config: Mapping[str, Any], role: str, assets: Mapping[str, Any],
                    runtime: Mapping[str, Any], gpu_runtime: Mapping[str, Any],
                    start_gpu: Mapping[str, Any], process: subprocess.Popen[Any] | None,
                    run_id: str, launched: bool) -> dict[str, Any]:
    service = config["services"][role]
    manifest = {
        "created_at_utc": utc_now(), "experiment": "VI-EA-ATPO Phase 2",
        "ownership_run_id": run_id, "owned_by_this_manager": launched,
        "service_role": role, "model_id": service["model_id"],
        "model_revision": service["model_revision"], "model_path": str(Path(service["model_path"]).resolve()),
        "served_model_name": service["served_model_name"], "host": config["network"]["host"],
        "port": int(service["port"]), "base_url": service_url(config, service),
        "framework": "vLLM", "framework_version": runtime["packages"]["vllm"],
        "torch_version": runtime["packages"]["torch"], "python": runtime["python"],
        "physical_gpu_index": int(config["gpu"]["physical_index"]),
        "cuda_visible_devices": str(config["gpu"]["cuda_visible_devices"]),
        "logical_cuda_device": 0, "gpu_name": gpu_runtime["device_name"],
        "tensor_parallel_size": int(config["gpu"]["tensor_parallel_size"]),
        "dtype": service["dtype"], "quantization": service.get("quantization"),
        "quantization_config": assets["model_config_quantization"],
        "max_model_len": int(service["max_model_len"]), "max_num_seqs": int(service["max_num_seqs"]),
        "gpu_memory_utilization": float(service["gpu_memory_utilization"]),
        "attention_backend": service["attention_backend"],
        "temperature": float(config["sampling"]["temperature"]), "top_p": float(config["sampling"]["top_p"]),
        "chat_template": config["sampling"]["chat_template"],
        "role_chat_template_kwargs": service.get("role_chat_template_kwargs", {}),
        "command": build_command(config, service),
        "pid": process.pid if process is not None else None,
        "process_group_id": process.pid if process is not None else None,
        "gpu_before_start": dict(start_gpu), "asset_verification": dict(assets),
        "model_config_sha256": assets["config_sha256"], "model_index_sha256": assets["index_sha256"],
        "service_config_sha256": sha256_file(PHASE2.parent / "configs/model_services.yaml"),
        "v1_models_verified": False,
    }
    return manifest


def wait_ready(config: Mapping[str, Any], role: str, process: subprocess.Popen[Any]) -> dict[str, Any]:
    service = config["services"][role]
    url = service_url(config, service)
    deadline = time.monotonic() + int(config["artifacts"].get("start_timeout_seconds", 1200))
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"{role} vLLM process exited with code {return_code}; see {config['artifacts'][role + '_log']}")
        reachable, ids, error = get_models(url, timeout=4)
        if reachable:
            expected = str(service["served_model_name"])
            if expected not in ids:
                raise RuntimeError(f"{role} endpoint returned unexpected model IDs {ids}; expected {expected}")
            return {"url": url + "/models", "reachable": True, "model_ids": ids, "alias_verified": True}
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for {role} vLLM at {url}: {error if 'error' in locals() else 'no response'}")


def reuse_or_refuse(config: Mapping[str, Any], role: str) -> dict[str, Any] | None:
    service = config["services"][role]
    reachable, ids, error = get_models(service_url(config, service))
    if not reachable:
        if port_open(str(config["network"]["host"]), int(service["port"])):
            raise RuntimeError(f"Port {service['port']} is occupied but not a usable OpenAI /v1/models endpoint: {error}")
        return None
    if str(service["served_model_name"]) not in ids:
        raise RuntimeError(f"Port {service['port']} is already serving different model IDs {ids}; refusing to replace it")
    path = manifest_path(config, role)
    if not path.is_file():
        raise RuntimeError(f"Port {service['port']} has the expected alias but no VI service manifest to verify checkpoint; refusing unverified reuse")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (manifest.get("model_revision") != service["model_revision"]
            or manifest.get("model_path") != str(Path(service["model_path"]).resolve())
            or manifest.get("served_model_name") != service["served_model_name"]):
        raise RuntimeError(f"Port {service['port']} manifest identity mismatch; refusing to reuse/terminate")
    pid = manifest.get("pid")
    if pid and manifest.get("owned_by_this_manager"):
        cmdline = process_cmdline(int(pid))
        if (process_group(int(pid)) != int(manifest.get("process_group_id") or -1)
                or str(service["model_path"]) not in cmdline or str(service["port"]) not in cmdline):
            raise RuntimeError(f"Existing service manifest PID identity failed; refusing to trust {pid}")
    return {"pid": pid, "process_group_id": manifest.get("process_group_id"),
            "owned_by_this_manager": bool(manifest.get("owned_by_this_manager")),
            "reused": True, "manifest_path": str(path),
            "model_revision": manifest["model_revision"], "served_model_name": manifest["served_model_name"]}


def run_service_health_checks(config: Mapping[str, Any], run_id: str) -> dict[str, Any]:
    endpoints = {role: service_url(config, service) for role, service in config["services"].items()}
    base_cfg, expert_cfg = config["services"]["base"], config["services"]["expert"]
    top_p = float(config["sampling"]["top_p"])
    async def execute() -> dict[str, Any]:
        base = ChatAPI(endpoints["base"], base_cfg["served_model_name"],
                       enable_thinking=False, top_p=top_p)
        patient = ChatAPI(endpoints["base"], base_cfg["served_model_name"],
                          enable_thinking=False, top_p=top_p)
        expert = ChatAPI(endpoints["expert"], expert_cfg["served_model_name"],
                         enable_thinking=False, top_p=top_p)
        record: dict[str, Any] = {"checked_at_utc": utc_now(), "run_id": run_id, "roles": {}}
        state = PublicState("service-health-check", ({
            "role": "user", "content": "Select the best answer. A. option A. B. option B. C. option C. D. option D."
        },))
        try:
            answer, call = await _base_action(base, state, Action.ANSWER, temperature=0.0)
            record["roles"]["base"] = {"ok": True, "answer": answer, **_call_metrics(call)}
        except Exception as exc:
            record["roles"]["base"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        try:
            response = await patient.generate(
                _patient_messages(["The patient has a mild headache today."], "Is a headache present today?"),
                temperature=0.0, max_tokens=64,
            )
            visible = visible_response(response.content)
            record["roles"]["patient"] = {"ok": bool(visible.strip()), "response": visible, **_call_metrics(response)}
        except Exception as exc:
            record["roles"]["patient"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        try:
            answer, call = await _defer_to_expert(expert, state, [], temperature=0.0)
            record["roles"]["expert"] = {"ok": answer in {"A", "B", "C", "D"}, "answer": answer, **_call_metrics(call)}
        except Exception as exc:
            record["roles"]["expert"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        finally:
            await asyncio.gather(base.close(), patient.close(), expert.close())
        return record
    result = asyncio.run(execute())
    logs = config["artifacts"]
    for role in ("base", "patient", "expert"):
        append_jsonl(Path(logs["base_log"] if role == "base" else logs["patient_role_log"] if role == "patient" else logs["expert_log"]),
                     {"event": "minimal_inference_health_check", **result["roles"][role]})
    if not all(item.get("ok") for item in result["roles"].values()):
        result["status"] = "failed"
    else:
        result["status"] = "passed"
    atomic_json(Path(logs["service_dir"]) / "health_check.json", result)
    return result


def start_services(config: dict[str, Any], config_path: Path) -> int:
    service_dir = Path(config["artifacts"]["service_dir"])
    service_dir.mkdir(parents=True, exist_ok=True)
    (service_dir / "logs").mkdir(parents=True, exist_ok=True)
    (service_dir / "pids").mkdir(parents=True, exist_ok=True)
    config["artifacts"]["config_path"] = str(config_path.resolve())
    run_id = f"vi-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{os.getpid()}"

    runtime = validate_runtime(config)
    gpu_runtime = verify_torch_gpu(config)
    assets = {role: verify_model_assets(role, service) for role, service in config["services"].items()}
    state = read_state(config)
    state.pop("error", None)
    state.update({"status": "starting", "run_id": run_id, "updated_at_utc": utc_now(), "services": state.get("services", {})})

    pending = []
    for role in ("base", "expert"):
        reused = reuse_or_refuse(config, role)
        if reused is not None:
            state["services"][role] = reused
            print(f"REUSE {role}: {reused['served_model_name']} revision={reused['model_revision']}", flush=True)
        else:
            pending.append(role)
    if pending:
        min_free = int(config["gpu"]["min_free_mib_before_any_new_service"])
        gate_gpu(config, min_free, int(config["gpu"]["max_utilization_pct_before_start"]),
                 owned_groups(state), require_unoccupied=True, allowed_pids={os.getpid()})

    new_processes: list[tuple[str, subprocess.Popen[Any]]] = []
    try:
        for role in pending:
            service = config["services"][role]
            if role == "expert":
                gate_gpu(config, int(config["gpu"]["min_free_mib_before_expert"]),
                         int(config["gpu"]["max_utilization_pct_before_expert"]),
                         owned_groups(state), require_unoccupied=False, allowed_pids={os.getpid()})
            elif role == "base":
                gate_gpu(config, int(config["gpu"]["min_free_mib_before_any_new_service"]),
                         int(config["gpu"]["max_utilization_pct_before_start"]),
                         owned_groups(state), require_unoccupied=not bool(state["services"]),
                         allowed_pids={os.getpid()})
            occupied = reuse_or_refuse(config, role)
            if occupied is not None:
                state["services"][role] = occupied
                continue
            log_path = Path(config["artifacts"]["base_log"] if role == "base" else config["artifacts"]["expert_log"])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            command = build_command(config, service)
            service_gpu_before = gpu_snapshot(config)
            with log_path.open("a", encoding="utf-8") as log_stream:
                log_stream.write(f"\n[{utc_now()}] START run_id={run_id} command={json.dumps(command)}\n")
                log_stream.flush()
                process = subprocess.Popen(
                    command, cwd=str(ROOT), env=build_environment(config, role),
                    stdin=subprocess.DEVNULL, stdout=log_stream, stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            new_processes.append((role, process))
            manifest = create_manifest(config, role, assets[role], runtime, gpu_runtime,
                                       service_gpu_before, process, run_id, launched=True)
            atomic_json(manifest_path(config, role), manifest)
            service_state = {"pid": process.pid, "process_group_id": process.pid,
                             "owned_by_this_manager": True, "reused": False,
                             "model_revision": service["model_revision"],
                             "served_model_name": service["served_model_name"],
                             "model_path": str(Path(service["model_path"]).resolve()),
                             "port": int(service["port"]), "started_at_utc": utc_now()}
            state["services"][role] = service_state
            service_pid_path(config, role).write_text(f"{process.pid}\n", encoding="utf-8")
            state.update({"status": "starting", "updated_at_utc": utc_now()})
            write_state(config, state)
            endpoint_result = wait_ready(config, role, process)
            manifest["v1_models_verified"] = True
            manifest["v1_models_result"] = endpoint_result
            atomic_json(manifest_path(config, role), manifest)
            service_state["ready"] = True
            write_state(config, state)
            print(f"READY {role}: {service_url(config, service)} alias={service['served_model_name']} pid={process.pid}", flush=True)

        for role in ("base", "expert"):
            ok, details = check_service_identity(config, role)
            if not ok:
                raise RuntimeError(f"Service identity verification failed: {details}")
        health = run_service_health_checks(config, run_id)
        state.update({"status": "ready" if health["status"] == "passed" else "health_check_failed",
                      "updated_at_utc": utc_now(), "service_health": health})
        write_state(config, state)
        if health["status"] != "passed":
            raise RuntimeError(f"Minimal inference health check failed; see {service_dir / 'health_check.json'}")
        print(json.dumps({"status": "ready", "run_id": run_id, "services": state["services"],
                          "health": health, "gpu": gpu_snapshot(config)}, ensure_ascii=False, indent=2), flush=True)
        return 0
    except Exception as exc:
        state.update({"status": "startup_or_health_failed", "updated_at_utc": utc_now(),
                      "error": f"{type(exc).__name__}: {exc}"})
        write_state(config, state)
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        print("Owned services, if any, remain recorded in services/state.json; use stop_vi_services.sh to stop only this deployment.", file=sys.stderr, flush=True)
        return 1


def check_services(config: dict[str, Any]) -> int:
    runtime = validate_runtime(config)
    gpu_runtime = verify_torch_gpu(config)
    identity: dict[str, Any] = {}
    good = True
    for role in ("base", "expert"):
        ok, details = check_service_identity(config, role)
        identity[role] = details
        good = good and ok
    if not good:
        print(json.dumps({"status": "not_ready", "identity": identity}, ensure_ascii=False, indent=2))
        return 1
    run_id = read_state(config).get("run_id", "check-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    health = run_service_health_checks(config, run_id)
    current = read_state(config)
    current.pop("error", None)
    current.update({"status": "ready" if health["status"] == "passed" else "health_check_failed",
                    "updated_at_utc": utc_now(), "service_health": health})
    write_state(config, current)
    result = {"status": current["status"], "identity": identity, "runtime": runtime,
              "cuda_mapping": gpu_runtime, "gpu": gpu_snapshot(config), "health": health}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if health["status"] == "passed" else 1


def terminate_owned(config: Mapping[str, Any], role: str, entry: Mapping[str, Any]) -> dict[str, Any]:
    if not entry.get("owned_by_this_manager"):
        return {"role": role, "stopped": False, "reason": "reused service is not owned by this manager"}
    pid = int(entry["pid"])
    pgid = int(entry.get("process_group_id", pid))
    service = config["services"][role]
    cmdline = process_cmdline(pid)
    if not cmdline:
        return {"role": role, "stopped": False, "reason": "owned process already exited"}
    if process_group(pid) != pgid or str(service["model_path"]) not in cmdline or str(service["port"]) not in cmdline:
        raise RuntimeError(f"Refusing to signal PID {pid}: process identity does not match saved VI service state")
    os.killpg(pgid, signal.SIGTERM)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if process_group(pid) != pgid:
            break
        time.sleep(0.5)
    if process_group(pid) == pgid:
        refreshed = process_cmdline(pid)
        if str(service["model_path"]) not in refreshed or str(service["port"]) not in refreshed:
            raise RuntimeError(f"PID {pid} identity changed during stop; refusing SIGKILL")
        os.killpg(pgid, signal.SIGKILL)
    return {"role": role, "stopped": True, "pid": pid, "process_group_id": pgid}


def stop_services(config: dict[str, Any]) -> int:
    state = read_state(config)
    results = []
    for role in ("expert", "base"):
        entry = state.get("services", {}).get(role)
        if entry:
            results.append(terminate_owned(config, role, entry))
            if entry.get("owned_by_this_manager"):
                try:
                    service_pid_path(config, role).unlink()
                except FileNotFoundError:
                    pass
    state.update({"status": "stopped", "updated_at_utc": utc_now(), "stop_results": results})
    write_state(config, state)
    print(json.dumps({"status": "stopped", "results": results}, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("preflight", "start", "check", "stop"))
    parser.add_argument("--config", type=Path, default=PHASE2.parent / "configs/model_services.yaml")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    if args.action == "preflight":
        return preflight(config, config_path)
    if args.action == "start":
        return start_services(config, config_path)
    if args.action == "check":
        return check_services(config)
    return stop_services(config)


if __name__ == "__main__":
    raise SystemExit(main())
