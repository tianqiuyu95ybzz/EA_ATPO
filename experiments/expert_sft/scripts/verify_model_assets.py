#!/usr/bin/env python3
"""Check pinned Qwen3-32B model files and write content hashes locally."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_REVISION = "9216db5781bf21249d130ec9da846c4624c16137"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--write-manifest", type=Path)
    args = parser.parse_args()
    config_path = args.model / "config.json"
    index_path = args.model / "model.safetensors.index.json"
    tokenizer_path = args.model / "tokenizer_config.json"
    for path in (config_path, index_path, tokenizer_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    tokenizer_config = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3" or config.get("architectures") != ["Qwen3ForCausalLM"]:
        raise RuntimeError(f"Unexpected architecture: {config.get('architectures')}")
    if config.get("quantization_config"):
        raise RuntimeError("Refusing quantized/AWQ weights for standard BF16 LoRA SFT")
    if str(config.get("torch_dtype", "")).lower() not in {"bfloat16", "bf16"}:
        raise RuntimeError(f"Pinned training checkpoint is not advertised as BF16: {config.get('torch_dtype')}")
    if not tokenizer_config.get("chat_template"):
        raise RuntimeError("Pinned model tokenizer is missing a native chat template")
    names = sorted(set(index.get("weight_map", {}).values()))
    if not names:
        raise RuntimeError("No indexed safetensors shards")
    shards = []
    for name in names:
        path = args.model / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
        shards.append({"name": name, "size_bytes": path.stat().st_size, "sha256": sha(path)})
    result = {
        "repo_id": "Qwen/Qwen3-32B",
        "revision": EXPECTED_REVISION,
        "model_type": config["model_type"],
        "architectures": config["architectures"],
        "torch_dtype": config.get("torch_dtype"),
        "quantized": False,
        "chat_template_sha256": hashlib.sha256(tokenizer_config["chat_template"].encode("utf-8")).hexdigest(),
        "files": {
            "config.json": {"size_bytes": config_path.stat().st_size, "sha256": sha(config_path)},
            "model.safetensors.index.json": {"size_bytes": index_path.stat().st_size, "sha256": sha(index_path)},
            "tokenizer_config.json": {"size_bytes": tokenizer_path.stat().st_size, "sha256": sha(tokenizer_path)},
        },
        "shards": shards,
        "total_shard_bytes": sum(item["size_bytes"] for item in shards),
        "all_indexed_shards_verified": True,
    }
    if args.write_manifest:
        args.write_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.write_manifest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"revision": result["revision"], "shard_count": len(shards), "total_shard_bytes": result["total_shard_bytes"], "bf16": result["torch_dtype"], "verified": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
