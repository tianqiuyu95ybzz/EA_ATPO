#!/usr/bin/env bash
set -euo pipefail

EA_ATPO_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EA_ATPO_PILOT_DIR="$EA_ATPO_PROJECT_DIR/experiments/real_case_pilot_20261103"
EA_ATPO_PYTHON="$EA_ATPO_PROJECT_DIR/.venv-real-pilot/bin/python"
EA_ATPO_HF_HOME="$EA_ATPO_PILOT_DIR/cache/huggingface"
EA_ATPO_MODEL_DIR="$EA_ATPO_PILOT_DIR/assets/models/Qwen3-8B"
EA_ATPO_MODEL_REPO="Qwen/Qwen3-8B"
EA_ATPO_MODEL_REVISION="b968826d9c46dd6066d109eabc6255188de91218"

if [[ ! -x "$EA_ATPO_PYTHON" ]]; then
  echo "Missing project runtime: $EA_ATPO_PYTHON" >&2
  exit 2
fi
mkdir -p "$EA_ATPO_HF_HOME" "$EA_ATPO_MODEL_DIR" "$EA_ATPO_PROJECT_DIR/.tmp" \
  "$EA_ATPO_PILOT_DIR/cache/datasets" "$EA_ATPO_PILOT_DIR/cache/xdg"
export HF_HOME="$EA_ATPO_HF_HOME"
export HF_HUB_CACHE="$EA_ATPO_HF_HOME/hub"
export HF_DATASETS_CACHE="$EA_ATPO_PILOT_DIR/cache/datasets"
export XDG_CACHE_HOME="$EA_ATPO_PILOT_DIR/cache/xdg"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_XET_HIGH_PERFORMANCE=1
export HF_XET_NUM_CONCURRENT_RANGE_GETS=16
export TMPDIR="$EA_ATPO_PROJECT_DIR/.tmp"

"$EA_ATPO_PROJECT_DIR/.venv-real-pilot/bin/hf" download "$EA_ATPO_MODEL_REPO" \
  --repo-type model --revision "$EA_ATPO_MODEL_REVISION" --local-dir "$EA_ATPO_MODEL_DIR"

"$EA_ATPO_PYTHON" - "$EA_ATPO_PILOT_DIR" "$EA_ATPO_MODEL_DIR" <<'PY'
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

pilot_dir, model_dir = map(Path, sys.argv[1:])
files = {}
for path in sorted(model_dir.rglob("*")):
    if not path.is_file() or ".cache" in path.parts:
        continue
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    files[str(path.relative_to(model_dir))] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
manifest = {
    "repository": "Qwen/Qwen3-8B",
    "revision": "b968826d9c46dd6066d109eabc6255188de91218",
    "local_path": str(model_dir),
    "files": files,
    "vllm_version": importlib.metadata.version("vllm"),
    "transformers_version": importlib.metadata.version("transformers"),
}
(pilot_dir / "assets/models/model_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps({"model": manifest["repository"], "revision": manifest["revision"], "files": len(files), "bytes": sum(x["bytes"] for x in files.values())}, indent=2))
PY
