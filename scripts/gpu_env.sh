#!/bin/bash
# Source this to point JAX at the GPU on this host.
#
# Workaround for a kernel-userspace driver-version mismatch: the loaded
# nvidia kernel module is 580.126.09 but the system libnvidia-compute-580
# was upgraded to 580.142, breaking ``cuInit()``.  The matching 580.126.09
# userspace libs are extracted from the apt cache to /tmp/nvidia-580.126/lib;
# LD_LIBRARY_PATH points JAX at them.  No sudo needed.
#
# Permanent fix would be ``sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset
# nvidia && sudo modprobe nvidia`` (no reboot) or a reboot.
#
# Usage: source scripts/gpu_env.sh

NVIDIA_COMPAT_DIR="/tmp/nvidia-580.126/lib"

if [ ! -f "${NVIDIA_COMPAT_DIR}/libcuda.so.1" ]; then
    mkdir -p "${NVIDIA_COMPAT_DIR}"
    EXTRACT_DIR="$(mktemp -d)"
    dpkg -x /var/cache/apt/archives/libnvidia-compute-580_580.126.09-0ubuntu0.24.04.2_amd64.deb "${EXTRACT_DIR}"
    cp -L "${EXTRACT_DIR}/usr/lib/x86_64-linux-gnu/libcuda.so.580.126.09" "${NVIDIA_COMPAT_DIR}/libcuda.so.1"
    cp -L "${EXTRACT_DIR}/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.126.09" "${NVIDIA_COMPAT_DIR}/libnvidia-ml.so.1"
    rm -rf "${EXTRACT_DIR}"
    echo "Staged matching 580.126 nvidia libs at ${NVIDIA_COMPAT_DIR}"
fi

export LD_LIBRARY_PATH="${NVIDIA_COMPAT_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
echo "GPU env ready: JAX_PLATFORMS=${JAX_PLATFORMS}, LD_LIBRARY_PATH includes ${NVIDIA_COMPAT_DIR}"
