#!/usr/bin/env python3
"""Record a lightweight environment/GPU/disk/model-asset audit in the project tree."""
from __future__ import annotations

import importlib.metadata
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/expert_sft"
CONFIG = EXP / "configs/domain_sft.json"


def sha_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    model_dir = ROOT / config["model"]["local_path"]
    packages = {}
    for name in ("torch", "transformers", "peft", "accelerate", "datasets", "pandas", "pyarrow", "huggingface_hub", "trl", "bitsandbytes"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    gpu_raw = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True)
    gpus = []
    for line in gpu_raw.strip().splitlines():
        fields = [x.strip() for x in line.split(",")]
        gpus.append({"index": int(fields[0]), "name": fields[1], "uuid": fields[2], "total_mib": int(fields[3]), "used_mib": int(fields[4]), "free_mib": int(fields[5]), "utilization_pct": int(fields[6])})
    app_raw = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True)
    app_counts: dict[str, int] = {}
    for line in app_raw.strip().splitlines():
        fields = [x.strip() for x in line.split(",")]
        if len(fields) == 2 and fields[1].upper() != "N/A":
            app_counts[fields[0]] = app_counts.get(fields[0], 0) + 1
    for gpu in gpus:
        gpu["compute_process_count"] = app_counts.get(gpu["uuid"], 0)
    usage = shutil.disk_usage(ROOT)
    safetensors = sorted(model_dir.glob("*.safetensors"))
    partial = sorted(set(model_dir.rglob("*.incomplete")) | set((EXP / "cache/hf_home").rglob("*.incomplete")))
    all_model_files = [x for x in model_dir.rglob("*") if x.is_file()]
    expected_shards = None
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        expected_shards = len(set(json.loads(index.read_text(encoding="utf-8")).get("weight_map", {}).values()))
    state = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "workspace_root": str(ROOT), "experiment_root": str(EXP),
        "python": os.sys.version, "platform": os.sys.platform,
        "packages": packages,
        "cuda": {"nvidia_smi_available": True, "torch_package_cuda_build": packages.get("torch", "").split("+")[-1] if "+" in str(packages.get("torch", "")) else None, "bf16_supported": None, "bf16_support_check": "deferred to CUDA smoke because importing torch for inventory is slow in this environment"},
        "gpus": gpus,
        "disk_at_workspace": {"total_bytes": usage.total, "used_bytes": usage.used, "free_bytes": usage.free},
        "model": {"repo_id": config["model"]["repo_id"], "revision": config["model"]["revision"], "configured_local_path": str(model_dir), "local_safetensors_count": len(safetensors), "expected_shards_from_index": expected_shards, "local_safetensors_bytes": sum(x.stat().st_size for x in safetensors), "model_directory_bytes_including_partial": sum(x.stat().st_size for x in all_model_files), "incomplete_hf_files": len(partial), "incomplete_hf_bytes": sum(x.stat().st_size for x in partial), "config_present": (model_dir / "config.json").is_file(), "index_present": index.is_file(), "tokenizer_config_present": (model_dir / "tokenizer_config.json").is_file(), "download_complete": bool(index.is_file() and expected_shards is not None and len(safetensors) == expected_shards and not partial)},
        "data": {"train_jsonl_sha256": sha_file(EXP / "data/train_sft.jsonl"), "validation_jsonl_sha256": sha_file(EXP / "data/validation_sft.jsonl"), "fresh_test_selection_sha256": sha_file(EXP / "data/fresh_test_selection.json")},
    }
    out_json = EXP / "reports/environment_audit_20261119.json"
    out_md = EXP / "reports/environment_audit_20261119.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    busy = [f"GPU {x['index']}: {x['free_mib']} MiB free, {x['utilization_pct']}% util, {x['compute_process_count']} compute processes" for x in gpus if x["free_mib"] < 150000 or x["utilization_pct"] > 20 or x["compute_process_count"] > 0]
    eligible = [x["index"] for x in gpus if x["free_mib"] >= 150000 and x["utilization_pct"] <= 20 and x["compute_process_count"] == 0]
    text = f"""# Environment Audit — 2026-11-19

- Qwen3-32B BF16 revision: `{state['model']['revision']}`
- Model download complete: **{state['model']['download_complete']}** ({state['model']['local_safetensors_count']} / {state['model']['expected_shards_from_index']} shards; {state['model']['local_safetensors_bytes']:,} complete shard bytes; {state['model']['incomplete_hf_files']} partial files, {state['model']['incomplete_hf_bytes']:,} partial bytes; {state['model']['model_directory_bytes_including_partial']:,} bytes in the model directory)
- NVIDIA runtime visible through `nvidia-smi`: **{state['cuda']['nvidia_smi_available']}**; PyTorch build: `{state['cuda']['torch_package_cuda_build']}`; BF16 capability: **not yet verified** (will be checked by the CUDA smoke).
- Workspace free disk: **{usage.free / (1024**4):.2f} TiB**
- GPUs failing the reserved SFT gate (>=150,000 MiB free, <=20% utilization): {', '.join(busy) if busy else 'none'}
- All download, HF, datasets, XDG, TRITON and temporary caches are configured under `{EXP.relative_to(ROOT)}`.

## Package Versions

```json
{json.dumps(packages, ensure_ascii=False, indent=2)}
```

This audit records availability only. It does not start or stop any model service or GPU process.
"""
    out_md.write_text(text, encoding="utf-8")
    print(json.dumps({"status": "recorded", "json": str(out_json.relative_to(ROOT)), "markdown": str(out_md.relative_to(ROOT)), "download_complete": state["model"]["download_complete"], "eligible_gpu_indices": eligible}, ensure_ascii=False))


if __name__ == "__main__":
    main()
