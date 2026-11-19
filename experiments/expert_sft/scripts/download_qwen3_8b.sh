#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXP="$ROOT/experiments/expert_sft"
HF_CLI="$ROOT/.venv/bin/hf"
REPO="Qwen/Qwen3-8B"
REVISION="b968826d9c46dd6066d109eabc6255188de91218"
MODEL_DIR="$EXP/assets/models/Qwen3-8B"
export HF_HOME="$EXP/cache/hf_home" HF_HUB_CACHE="$EXP/cache/hf_home/hub"
export HF_DATASETS_CACHE="$EXP/cache/datasets" XDG_CACHE_HOME="$EXP/cache/xdg"
export TMPDIR="$EXP/cache/tmp" TRITON_CACHE_DIR="$EXP/cache/triton" TORCH_HOME="$EXP/cache/torch"
export HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "$MODEL_DIR" "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME" "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCH_HOME"
[[ -x "$HF_CLI" ]] || { echo "Missing workspace HF CLI: $HF_CLI" >&2; exit 2; }
"$HF_CLI" download "$REPO" --repo-type model --revision "$REVISION" --local-dir "$MODEL_DIR"
"$ROOT/.venv/bin/python" - "$MODEL_DIR" <<'PY'
import json, sys
from pathlib import Path
p=Path(sys.argv[1]); c=json.loads((p/'config.json').read_text()); t=json.loads((p/'tokenizer_config.json').read_text())
if c.get('model_type')!='qwen3' or c.get('quantization_config') or not t.get('chat_template'):
 raise SystemExit('Base checkpoint architecture/quantization/template mismatch')
print(json.dumps({'model_type':c['model_type'],'architecture':c.get('architectures'),'revision':'b968826d9c46dd6066d109eabc6255188de91218','chat_template_present':True,'quantized':False}))
PY
