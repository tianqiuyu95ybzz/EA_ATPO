#!/usr/bin/env bash
set -euo pipefail

EA_ATPO_PILOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EA_ATPO_ASSET_DIR="${1:-$EA_ATPO_PILOT_DIR/assets}"
EA_ATPO_PILOT_PYTHON="${EA_ATPO_PILOT_PYTHON:-python3}"
EA_ATPO_SOURCE_COMMIT="25af83688c06917f94245a4db06be7d0924c090c"
EA_ATPO_SOURCE_BASE="https://raw.githubusercontent.com/Quark-Medical/ATPO/$EA_ATPO_SOURCE_COMMIT"
EA_ATPO_PROJECT_DIR="$(cd "$EA_ATPO_PILOT_DIR/../.." && pwd)"
export HF_HOME="$EA_ATPO_PILOT_DIR/cache/huggingface"
export HF_DATASETS_CACHE="$EA_ATPO_PILOT_DIR/cache/datasets"
export XDG_CACHE_HOME="$EA_ATPO_PILOT_DIR/cache/xdg"
export TMPDIR="$EA_ATPO_PROJECT_DIR/.tmp"

mkdir -p "$EA_ATPO_ASSET_DIR/source/dataset" "$HF_HOME" "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME" "$TMPDIR" \
  "$EA_ATPO_ASSET_DIR/source/codes/recipe/atpo/data_utils"

download_pinned_file() {
  local relative_path="$1"
  local destination="$EA_ATPO_ASSET_DIR/source/$relative_path"
  local partial="$destination.part"
  mkdir -p "$(dirname "$destination")"
  if [[ -s "$destination" ]]; then
    echo "Keeping existing pinned source file: $relative_path"
    return
  fi
  curl --fail --location --retry 5 --retry-all-errors --retry-delay 2 \
    --connect-timeout 30 --continue-at - \
    "$EA_ATPO_SOURCE_BASE/$relative_path" --output "$partial"
  mv "$partial" "$destination"
}

download_pinned_file "dataset/merged_train_dataset.jsonl"
download_pinned_file "dataset/medqa_test_dataset.jsonl"
download_pinned_file "codes/recipe/atpo/data_utils/mix_data_jsonl.py"

"$EA_ATPO_PILOT_PYTHON" - "$EA_ATPO_ASSET_DIR" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

asset_dir = Path(sys.argv[1])
source_dir = asset_dir / "source"
manifest = {
    "repository": "https://github.com/Quark-Medical/ATPO",
    "commit": "25af83688c06917f94245a4db06be7d0924c090c",
    "files": {},
}
for rel in (
    "dataset/merged_train_dataset.jsonl",
    "dataset/medqa_test_dataset.jsonl",
    "codes/recipe/atpo/data_utils/mix_data_jsonl.py",
):
    path = source_dir / rel
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    file_record = {"bytes": path.stat().st_size, "sha256": digest}
    if rel.endswith(".jsonl"):
        with path.open("r", encoding="utf-8") as handle:
            file_record["records"] = sum(1 for line in handle if line.strip())
    manifest["files"][rel] = file_record
(asset_dir / "source_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
)
PY

"$EA_ATPO_PILOT_PYTHON" "$EA_ATPO_ASSET_DIR/source/codes/recipe/atpo/data_utils/mix_data_jsonl.py" \
  train "$EA_ATPO_ASSET_DIR/train.parquet" \
  "$EA_ATPO_ASSET_DIR/source/dataset/merged_train_dataset"
"$EA_ATPO_PILOT_PYTHON" "$EA_ATPO_ASSET_DIR/source/codes/recipe/atpo/data_utils/mix_data_jsonl.py" \
  test "$EA_ATPO_ASSET_DIR/test.parquet" \
  "$EA_ATPO_ASSET_DIR/source/dataset/medqa_test_dataset"

"$EA_ATPO_PILOT_PYTHON" "$EA_ATPO_PILOT_DIR/adapt_atpo_parquet.py" \
  --train "$EA_ATPO_ASSET_DIR/train.parquet" \
  --test "$EA_ATPO_ASSET_DIR/test.parquet" \
  --output-dir "$EA_ATPO_ASSET_DIR"

"$EA_ATPO_PILOT_PYTHON" - "$EA_ATPO_ASSET_DIR" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

asset_dir = Path(sys.argv[1])
manifest_path = asset_dir / "source_manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
manifest["reconstructed_parquet"] = {}
manifest["pilot_parquet"] = {}
for split in ("train", "test"):
    path = asset_dir / f"{split}.parquet"
    manifest["reconstructed_parquet"][split] = {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    pilot_path = asset_dir / f"{split}.pilot.parquet"
    manifest["pilot_parquet"][split] = {
        "path": pilot_path.name,
        "bytes": pilot_path.stat().st_size,
        "sha256": hashlib.sha256(pilot_path.read_bytes()).hexdigest(),
    }
manifest["adapter_manifest_sha256"] = hashlib.sha256(
    (asset_dir / "adapter_manifest.json").read_bytes()
).hexdigest()
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
PY

printf 'Created reconstructed Parquet copies under %s\n' "$EA_ATPO_ASSET_DIR"
printf 'Source files and preprocessor were downloaded from pinned upstream commit %s; inputs were not edited.\n' \
  "$EA_ATPO_SOURCE_COMMIT"
