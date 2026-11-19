#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXP="$ROOT/experiments/expert_sft"
DATA_PYTHON_DEFAULT="$ROOT/../EA_ATPO/.venv-real-pilot/bin/python"
if [[ ! -x "$DATA_PYTHON_DEFAULT" ]]; then DATA_PYTHON_DEFAULT="$ROOT/.venv/bin/python"; fi
PYTHON="${EA_ATPO_DATA_PYTHON:-$DATA_PYTHON_DEFAULT}"
COMMIT="25af83688c06917f94245a4db06be7d0924c090c"
BASE="https://raw.githubusercontent.com/Quark-Medical/ATPO/$COMMIT"
SOURCE="$EXP/data/source"
RECON="${EA_ATPO_RECON_DIR:-$EXP/data/reconstructed}"

export HF_HOME="$EXP/cache/hf_home"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$EXP/cache/datasets"
export XDG_CACHE_HOME="$EXP/cache/xdg"
export TMPDIR="$EXP/cache/tmp"
export UV_CACHE_DIR="$EXP/cache/uv"
export PIP_CACHE_DIR="$EXP/cache/pip"
export HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "$SOURCE/dataset" "$SOURCE/codes/recipe/atpo/data_utils" "$RECON" "$TMPDIR" \
  "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME" "$UV_CACHE_DIR" "$PIP_CACHE_DIR"

download_pinned() {
  local rel="$1" dest="$SOURCE/$1" partial="$SOURCE/$1.part"
  mkdir -p "$(dirname "$dest")"
  if [[ -s "$dest" ]]; then
    echo "Keeping existing file: $rel"
    return
  fi
  curl --fail --location --retry 8 --retry-all-errors --retry-delay 2 \
    --connect-timeout 30 --continue-at - "$BASE/$rel" --output "$partial"
  mv "$partial" "$dest"
}

download_pinned "dataset/merged_train_dataset.jsonl"
download_pinned "codes/recipe/atpo/data_utils/mix_data_jsonl.py"

"$PYTHON" "$SOURCE/codes/recipe/atpo/data_utils/mix_data_jsonl.py" \
  train "$RECON/train.parquet" "$SOURCE/dataset/merged_train_dataset"

"$PYTHON" - "$ROOT" "$RECON" "$SOURCE" <<'PY'
import hashlib, importlib.util, json, sys
from pathlib import Path

root, out, source = map(Path, sys.argv[1:])
adapter_path = root / "experiments/real_case_pilot_20261103/adapt_atpo_parquet.py"
spec = importlib.util.spec_from_file_location("atpo_data_adapter", adapter_path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
assert spec.loader is not None
spec.loader.exec_module(module)
input_path = out / "train.parquet"
pilot_path = out / "train.pilot.parquet"
adapted = module.adapt_split(input_path, pilot_path, "train")

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

source_jsonl = source / "dataset/merged_train_dataset.jsonl"
preprocessor = source / "codes/recipe/atpo/data_utils/mix_data_jsonl.py"
def line_count(path):
    with path.open("rb") as f:
        return sum(1 for row in f if row.strip())

manifest = {
    "repository": "https://github.com/Quark-Medical/ATPO",
    "commit": "25af83688c06917f94245a4db06be7d0924c090c",
    "preprocessor": {"path": str(preprocessor.relative_to(root)), "sha256": sha(preprocessor)},
    "source_jsonl": {"path": str(source_jsonl.relative_to(root)), "sha256": sha(source_jsonl), "records": line_count(source_jsonl)},
    "reconstructed_train_parquet": {"path": str(input_path.relative_to(root)), "sha256": sha(input_path), "bytes": input_path.stat().st_size},
    "adapted_train_parquet": {"path": str(pilot_path.relative_to(root)), "sha256": sha(pilot_path), "bytes": pilot_path.stat().st_size, "rows": adapted["rows"]},
    "adapter_audit": {k: adapted[k] for k in ("missing_prompt", "missing_ground_truth", "invalid_ground_truth_format", "missing_atomic_facts", "prompt_option_label_count_distribution")},
    "test_data_downloaded_or_read": False,
}
(out / "source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"train_rows": adapted["rows"], "train_pilot_sha256": manifest["adapted_train_parquet"]["sha256"], "missing_prompt": adapted["missing_prompt"], "missing_gold": adapted["missing_ground_truth"], "invalid_gold": adapted["invalid_ground_truth_format"], "missing_atomic_facts": adapted["missing_atomic_facts"]}, ensure_ascii=False))
PY
