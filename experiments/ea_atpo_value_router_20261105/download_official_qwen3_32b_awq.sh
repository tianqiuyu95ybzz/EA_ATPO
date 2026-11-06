#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODEL_DIR="$PROJECT_DIR/experiments/real_case_pilot_20261103/assets/models/Qwen3-32B-AWQ"
mkdir -p "$MODEL_DIR"
# Metadata and weight shards are fetched by the same pinned, hash-verified
# downloader. Override EA_ATPO_MODEL_SOURCE=huggingface when that endpoint is
# available; the default is Qwen's official ModelScope mirror with SHA-256
# checks against the pinned Hugging Face LFS manifest.
"$PROJECT_DIR/.venv-real-pilot/bin/python" "$SCRIPT_DIR/download_official_qwen3_32b_awq_ranged.py" \
  --model-dir "$MODEL_DIR" --source "${EA_ATPO_MODEL_SOURCE:-modelscope}" \
  --max-workers "${EA_ATPO_AWQ_MAX_WORKERS:-8}" --chunk-mib 16 --read-timeout-s 90
