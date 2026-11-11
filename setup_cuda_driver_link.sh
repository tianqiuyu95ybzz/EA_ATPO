#!/usr/bin/env bash
# Source this helper before starting SGLang/Triton. The host exposes libcuda.so.1
# but not the unversioned linker name expected by Triton's runtime compiler.
set -euo pipefail

EA_ATPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$EA_ATPO_ROOT/.venv/bin:$PATH"
EA_ATPO_CUDA_DRIVER_LIB="${EA_ATPO_CUDA_DRIVER_LIB:-/lib/x86_64-linux-gnu/libcuda.so.1}"
if [[ ! -r "$EA_ATPO_CUDA_DRIVER_LIB" ]]; then
  echo "CUDA driver library not readable: $EA_ATPO_CUDA_DRIVER_LIB" >&2
  return 1
fi

EA_ATPO_CUDA_LINK_DIR="$EA_ATPO_ROOT/.runtime/cuda-driver-link"
mkdir -p "$EA_ATPO_CUDA_LINK_DIR"
ln -sfn "$EA_ATPO_CUDA_DRIVER_LIB" "$EA_ATPO_CUDA_LINK_DIR/libcuda.so"
export LIBRARY_PATH="$EA_ATPO_CUDA_LINK_DIR${LIBRARY_PATH:+:$LIBRARY_PATH}"
export HOME="$EA_ATPO_ROOT/.cache/home"
export XDG_CACHE_HOME="$EA_ATPO_ROOT/.cache/xdg"
export TMPDIR="$EA_ATPO_ROOT/.cache/tmp"
EA_ATPO_RAY_LINK="/tmp/ea-atpo-v1-ray-${UID:-$(id -u)}"
if [[ -L "$EA_ATPO_RAY_LINK" ]]; then
  if [[ "$(readlink -f "$EA_ATPO_RAY_LINK")" != "$(readlink -f "$EA_ATPO_ROOT/.cache/ray")" ]]; then
    echo "Ray temp link points elsewhere: $EA_ATPO_RAY_LINK" >&2
    return 1
  fi
elif [[ -e "$EA_ATPO_RAY_LINK" ]]; then
  echo "Ray temp path already exists and is not this project's link: $EA_ATPO_RAY_LINK" >&2
  return 1
else
  ln -s "$EA_ATPO_ROOT/.cache/ray" "$EA_ATPO_RAY_LINK"
fi
export RAY_TMPDIR="$EA_ATPO_RAY_LINK"
unset RAY_ADDRESS
export TRITON_CACHE_DIR="$EA_ATPO_ROOT/.cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$EA_ATPO_ROOT/.cache/torchinductor"
export TORCHINDUCTOR_COMPILE_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
mkdir -p "$HOME" "$XDG_CACHE_HOME" "$TMPDIR" "$RAY_TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
