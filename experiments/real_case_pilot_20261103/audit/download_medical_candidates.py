#!/usr/bin/env python3
"""Download pinned public medical Qwen candidates with resumable HF snapshots."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


AUDIT_DIR = Path(__file__).resolve().parent
PILOT_DIR = AUDIT_DIR.parent
MODEL_DIR = PILOT_DIR / "assets" / "models" / "medical_candidates"
CACHE_DIR = PILOT_DIR / "cache" / "huggingface"
MODELS = {
    "huatuo_gpt_o1_7b": {
        "repo_id": "FreedomIntelligence/HuatuoGPT-o1-7B",
        "revision": "6487c8071a3f87702a6816ba7a09f38921b685ed",
        "use": "Medical reasoning/QA candidate; Qwen2.5-7B base.",
    },
    "qwen3_4b_remedi": {
        "repo_id": "VioletAmethyst/Qwen3-4B-ReMeDi",
        "revision": "dfa418ca1c3c0f8879afd6d356b88083f01ca032",
        "use": "Multi-turn medical-dialogue candidate; Qwen3-4B.",
    },
    "popit_qwen3_4b_medical_sft": {
        "repo_id": "Hoshino-Yumetsuki/PopiT-Qwen3-4B-Medical-SFT-1128",
        "revision": "f7f95e22f5fd2c8c015abfba9fe249d6b80b6f97",
        "use": "Chinese medical-dialogue/QA SFT candidate; Qwen3-4B Instruct base.",
    },
}


def download_one(name: str, spec: dict[str, str]) -> dict[str, object]:
    destination = MODEL_DIR / name
    path = snapshot_download(
        repo_id=spec["repo_id"],
        revision=spec["revision"],
        local_dir=destination,
        cache_dir=CACHE_DIR,
        max_workers=4,
    )
    config = Path(path) / "config.json"
    if not config.is_file():
        raise RuntimeError(f"download completed without config.json: {spec['repo_id']}")
    snapshot = Path(path)
    shards = sorted(snapshot.glob("model-*.safetensors"))
    if not shards:
        raise RuntimeError(f"download completed without safetensors weights: {spec['repo_id']}")
    index_path = snapshot / "model.safetensors.index.json"
    if index_path.is_file():
        expected_shards = set(json.loads(index_path.read_text(encoding="utf-8"))["weight_map"].values())
        found_shards = {shard.name for shard in shards}
        if expected_shards != found_shards:
            raise RuntimeError(
                f"incomplete safetensors snapshot for {spec['repo_id']}: "
                f"missing={sorted(expected_shards - found_shards)}, unexpected={sorted(found_shards - expected_shards)}"
            )

    def file_sha256(file_path: Path) -> str:
        digest = hashlib.sha256()
        with file_path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()

    return {
        "name": name,
        "path": str(snapshot.resolve()),
        "config_sha256": file_sha256(config),
        "safetensors_sha256": {
            shard.name: {"sha256": file_sha256(shard), "size_bytes": shard.stat().st_size}
            for shard in shards
        },
    }


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    for name, spec in MODELS.items():
        info = api.model_info(spec["repo_id"], revision=spec["revision"], files_metadata=True)
        if info.sha != spec["revision"]:
            raise RuntimeError(f"revision changed for {name}: expected {spec['revision']}, got {info.sha}")

    completed: dict[str, dict[str, object]] = {}
    with ThreadPoolExecutor(max_workers=len(MODELS)) as pool:
        futures = {
            pool.submit(download_one, name, spec): name
            for name, spec in MODELS.items()
        }
        for future in as_completed(futures):
            name = futures[future]
            completed[name] = future.result()
            print(f"downloaded {name}: {completed[name]['path']}", flush=True)

    manifest = {
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "download_method": "huggingface_hub.snapshot_download; reruns resume from local cache",
        "models": {
            name: {**spec, **completed[name]}
            for name, spec in MODELS.items()
        },
    }
    manifest_path = MODEL_DIR / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
