#!/usr/bin/env bash
# Set up the GPU environment without sudo by extracting the matching
# nvidia 580.126 userspace libs from the apt cache (the installed
# userspace is 580.142 but the kernel module is 580.126.09, so cuInit
# fails with CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE on every JAX
# invocation).  Source this file:
#
#   source scripts/gpu_env.sh
#   .venv/bin/python -c "import jax; print(jax.devices())"
#
# Idempotent.  Safe to source multiple times in the same shell.

set -e

NVIDIA_LIB_DIR="${NVIDIA_LIB_DIR:-/tmp/nvidia-580.126/lib}"
NVIDIA_DEB="/var/cache/apt/archives/libnvidia-compute-580_580.126.09-0ubuntu0.24.04.2_amd64.deb"

if [[ ! -d "$NVIDIA_LIB_DIR" || ! -f "$NVIDIA_LIB_DIR/libcuda.so.1" ]]; then
    if [[ ! -f "$NVIDIA_DEB" ]]; then
        echo "[gpu_env] ERROR: matching nvidia .deb not in apt cache:" >&2
        echo "  $NVIDIA_DEB" >&2
        echo "[gpu_env] Run 'sudo apt install --reinstall libnvidia-compute-580=580.126.09-0ubuntu0.24.04.2'" >&2
        return 1 2>/dev/null || exit 1
    fi
    rm -rf "$NVIDIA_LIB_DIR"
    mkdir -p "$NVIDIA_LIB_DIR"
    tmp_dir=$(mktemp -d)
    trap "rm -rf $tmp_dir" EXIT
    dpkg-deb -x "$NVIDIA_DEB" "$tmp_dir"
    cp -P "$tmp_dir"/usr/lib/x86_64-linux-gnu/libcuda.so* "$NVIDIA_LIB_DIR"/
    cp -P "$tmp_dir"/usr/lib/x86_64-linux-gnu/libnvidia-*.so* "$NVIDIA_LIB_DIR"/ 2>/dev/null || true
    # libnvidia-ptxjitcompiler might be packaged elsewhere — copy if present.
    find "$tmp_dir" -name "libnvidia-ptxjitcompiler.so*" -exec cp -P {} "$NVIDIA_LIB_DIR"/ \; 2>/dev/null || true
    echo "[gpu_env] extracted matching 580.126 libs to $NVIDIA_LIB_DIR"
fi

# Prepend to LD_LIBRARY_PATH (avoid duplicates)
case ":${LD_LIBRARY_PATH:-}:" in
    *:"$NVIDIA_LIB_DIR":*) ;;
    *) export LD_LIBRARY_PATH="$NVIDIA_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
esac

export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"
echo "[gpu_env] JAX_PLATFORMS=$JAX_PLATFORMS  LD_LIBRARY_PATH=$NVIDIA_LIB_DIR:..."
