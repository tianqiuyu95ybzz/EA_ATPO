#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PILOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_DIR="$(cd "$PILOT_DIR/../.." && pwd)"
PYTHON="$PROJECT_DIR/.venv-real-pilot/bin/python"
HF_CLI="$PROJECT_DIR/.venv-real-pilot/bin/hf"
MODEL_DIR="$PILOT_DIR/assets/models/Qwen3-32B"
HF_HOME="$PILOT_DIR/cache/huggingface"
REVISION="9216db5781bf21249d130ec9da846c4624c16137"

mkdir -p "$MODEL_DIR" "$HF_HOME" "$PILOT_DIR/cache/xdg" "$PROJECT_DIR/.tmp"
export HF_HOME HF_HUB_CACHE="$HF_HOME/hub" XDG_CACHE_HOME="$PILOT_DIR/cache/xdg"
export HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_XET=1
# Large safetensors can take longer than the Hub's short default read timeout
# on this shared environment. Keep the CLI resumable and avoid saturating the
# route with eight simultaneous shard downloads.
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}"
export TMPDIR="$PROJECT_DIR/.tmp"

"$HF_CLI" download Qwen/Qwen3-32B --repo-type model --revision "$REVISION" --local-dir "$MODEL_DIR" --max-workers "${HF_DOWNLOAD_WORKERS:-2}"

"$PYTHON" - "$SCRIPT_DIR" "$MODEL_DIR" "$REVISION" <<'PY'
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

audit_dir, model_dir = map(Path, sys.argv[1:3])
revision = sys.argv[3]
files = {}
for path in sorted(model_dir.rglob("*")):
    if not path.is_file() or ".cache" in path.parts:
        continue
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    files[str(path.relative_to(model_dir))] = {
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }
manifest = {
    "repository": "Qwen/Qwen3-32B",
    "revision": revision,
    "local_path": str(model_dir),
    "files": files,
    "total_bytes": sum(item["bytes"] for item in files.values()),
    "vllm_version": importlib.metadata.version("vllm"),
    "transformers_version": importlib.metadata.version("transformers"),
}
(audit_dir / "qwen3_32b_model_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps({key: manifest[key] for key in ("repository", "revision", "local_path", "total_bytes", "vllm_version")}, indent=2))
PY
