#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/setup_cuda_driver_link.sh"
PYTHON_BIN="${EA_ATPO_PYTHON_BIN:-/usr/bin/python3.12}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/.cache/uv}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-10}"

if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  uv venv --python "$PYTHON_BIN" "$ROOT/.venv"
fi

uv pip install \
  --python "$ROOT/.venv/bin/python" \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match \
  'setuptools>=77' wheel 'packaging>=24'

uv pip install \
  --python "$ROOT/.venv/bin/python" \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match \
  --no-build-isolation \
  -r "$ROOT/requirements-ea-atpo-v1-cu128.txt"

# SGLang's historical dependency set resolves sgl-kernel 0.2.8, which has no
# SM100 cubins. Replace it with the official Torch 2.7 / CUDA 12.8 wheel. Keep
# the artifact on PFS, resume range downloads, and verify its published digest.
SGL_KERNEL_WHEEL="$ROOT/experiments/ea_atpo_v1_smoke/kernel_wheels/candidate_029/sgl_kernel-0.2.9+cu128-cp39-abi3-manylinux2014_x86_64.whl"
SGL_KERNEL_URL='https://github.com/sgl-project/whl/releases/download/v0.2.9/sgl_kernel-0.2.9+cu128-cp39-abi3-manylinux2014_x86_64.whl'
SGL_KERNEL_SIZE=355669905
SGL_KERNEL_SHA256='5f875f1d3b48be1f2d15b50ec2ddf4e8dfb9b839be50292717954057e80af1ec'
if [[ -f "$SGL_KERNEL_WHEEL" ]] && { \
  [[ "$(stat -c '%s' "$SGL_KERNEL_WHEEL")" -ne "$SGL_KERNEL_SIZE" ]] || \
  [[ "$(sha256sum "$SGL_KERNEL_WHEEL" | awk '{print $1}')" != "$SGL_KERNEL_SHA256" ]]; \
}; then
  "$ROOT/.venv/bin/python" -c 'import sys; from pathlib import Path; Path(sys.argv[1]).unlink(missing_ok=True)' "$SGL_KERNEL_WHEEL"
fi
if [[ ! -f "$SGL_KERNEL_WHEEL" ]]; then
  "$ROOT/.venv/bin/python" "$ROOT/experiments/ea_atpo_v1_smoke/download_curl_range_wheel.py" \
    "$SGL_KERNEL_URL" "$SGL_KERNEL_WHEEL" \
    --size "$SGL_KERNEL_SIZE" --sha256 "$SGL_KERNEL_SHA256" --workers 16
fi
printf '%s  %s\n' "$SGL_KERNEL_SHA256" "$SGL_KERNEL_WHEEL" | sha256sum --check --status
uv pip install --python "$ROOT/.venv/bin/python" --no-deps "$SGL_KERNEL_WHEEL"

# Use the official prebuilt wheel to avoid compiling against this host's CUDA 13.2
# toolkit while the environment uses PyTorch cu128. The artifact is cached in the
# project workspace and verified against the official GitHub release digest.
FLASH_ATTN_WHEEL="$ROOT/.cache/wheels/flash_attn-2.8.3.post1+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
FLASH_ATTN_URL='https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1%2Bcu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl'
FLASH_ATTN_SIZE=256029263
FLASH_ATTN_SHA256='f87164bb919f5597cb94f7485196be6ac7842b71d7bf8e9f8e9133b9045ab5ea'
mkdir -p "$(dirname "$FLASH_ATTN_WHEEL")"
if [[ ! -f "$FLASH_ATTN_WHEEL" || "$(stat -c '%s' "$FLASH_ATTN_WHEEL")" -lt "$FLASH_ATTN_SIZE" ]]; then
  curl --fail --location --retry 5 --retry-all-errors --continue-at - \
    --output "$FLASH_ATTN_WHEEL" "$FLASH_ATTN_URL"
fi
if [[ "$(stat -c '%s' "$FLASH_ATTN_WHEEL")" -ne "$FLASH_ATTN_SIZE" ]] || \
   [[ "$(sha256sum "$FLASH_ATTN_WHEEL" | awk '{print $1}')" != "$FLASH_ATTN_SHA256" ]]; then
  rm -f "$FLASH_ATTN_WHEEL"
  curl --fail --location --retry 5 --retry-all-errors \
    --output "$FLASH_ATTN_WHEEL" "$FLASH_ATTN_URL"
fi
printf '%s  %s\n' "$FLASH_ATTN_SHA256" "$FLASH_ATTN_WHEEL" | sha256sum --check --status
uv pip install --python "$ROOT/.venv/bin/python" --no-deps "$FLASH_ATTN_WHEEL"

uv pip freeze --python "$ROOT/.venv/bin/python" > "$ROOT/requirements-ea-atpo-v1-resolved.txt"
