#!/usr/bin/env python3
"""Record model revisions and content hashes used by the DEFER validation."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs/qwen32_defer_validation_20261119_attempt01"
BASE = Path("/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/experiments/real_case_pilot_20261103/assets/models/Qwen3-8B")
EXPERT = Path("/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/experiments/real_case_pilot_20261103/assets/models/Qwen3-32B-AWQ")
BASE_REV = "b968826d9c46dd6066d109eabc6255188de91218"
EXPERT_REV = "0499c3ac83fdef8810b907a23894ba91e95eddd8"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def file_map(root: Path, names: list[str]) -> dict[str, dict[str, object]]:
    out = {}
    for name in names:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        out[name] = {"size_bytes": path.stat().st_size, "sha256": digest(path)}
    return out


def main() -> None:
    if (OUT / "model_fingerprint.json").exists():
        raise FileExistsError(f"Model fingerprint is already frozen: {OUT / 'model_fingerprint.json'}")
    OUT.mkdir(parents=True, exist_ok=True)
    lock = json.loads((EXPERT / "model_source_lock.json").read_text(encoding="utf-8"))
    status = json.loads((EXPERT / "range_download_status.json").read_text(encoding="utf-8"))
    index = json.loads((EXPERT / "model.safetensors.index.json").read_text(encoding="utf-8"))
    expected = {item["name"]: item["sha256"] for item in lock["weight_files"]}
    shards = sorted(set(index["weight_map"].values()))
    if lock.get("huggingface_revision") != EXPERT_REV or status.get("revision") != EXPERT_REV or status.get("status") != "all_shards_verified":
        raise RuntimeError("Pinned Qwen3-32B-AWQ revision is not fully verified")
    expert_files = file_map(EXPERT, shards + ["config.json", "generation_config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json"])
    for name, wanted in expected.items():
        if expert_files[name]["sha256"] != wanted:
            raise RuntimeError(f"Qwen3-32B-AWQ hash mismatch: {name}")
    base_index = json.loads((BASE / "model.safetensors.index.json").read_text(encoding="utf-8"))
    base_shards = sorted(set(base_index["weight_map"].values()))
    base_names = base_shards + ["config.json", "generation_config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json"]
    base_files = file_map(BASE, base_names)
    obj = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "base": {"repo_id": "Qwen/Qwen3-8B", "revision": BASE_REV, "model_type": "qwen3", "files": base_files},
        "expert": {"repo_id": "Qwen/Qwen3-32B-AWQ", "revision": EXPERT_REV, "quantization": "AWQ", "files": expert_files, "source_lock_sha256": digest(EXPERT / "model_source_lock.json"), "download_status_sha256": digest(EXPERT / "range_download_status.json"), "weight_shards_match_locked_sha256": True},
        "note": "Content hashes identify the exact local model and tokenizer files; no model weights are copied or modified.",
    }
    path = OUT / "model_fingerprint.json"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(path), "base_shards": len(base_shards), "expert_shards": len(shards), "base_revision": BASE_REV, "expert_revision": EXPERT_REV, "expert_locked_shards_verified": True}, indent=2))


if __name__ == "__main__":
    main()
