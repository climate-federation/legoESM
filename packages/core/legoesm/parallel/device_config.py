"""Device configuration and hardware-aware optimization.

Auto-detects available accelerators (CPU, GPU, TPU, Apple MPS).  JAX/XLA
backend configuration lives in :mod:`legoesm.runtime`.  Provides:

- Hardware detection and capability reporting
- Mixed-precision dtype policies

Usage
-----
>>> from legoesm.parallel.device_config import detect_devices, get_optimal_dtype
>>> config = detect_devices()
>>> dtype = get_optimal_dtype(config, precision='single')
"""

from __future__ import annotations

import logging
import os
from typing import NamedTuple

import jax
import jax.numpy as jnp

logger = logging.getLogger(__name__)


# ============================================================================
# Hardware configuration
# ============================================================================

class HardwareConfig(NamedTuple):
    """Hardware configuration for the current execution environment.

    Attributes
    ----------
    backend : str
        One of ``'cpu'``, ``'gpu'``, ``'tpu'``, ``'mps'``.
    device_count : int
        Total number of devices visible to this process.
    devices_per_host : int
        Number of devices on the local host (equals ``device_count`` for
        single-host setups; less for multi-host TPU pods).
    num_hosts : int
        Number of JAX processes (hosts).  1 for single-host.
    supports_float64 : bool
        Whether the backend supports float64 arithmetic natively.
    supports_complex128 : bool
        Whether the backend supports complex128 (needed for spectral solver).
    memory_per_device_gb : float
        Estimated HBM/VRAM per device in GB.  Used for batch size heuristics.
    recommended_batch_size : int
        Heuristic batch size for cubed-sphere faces based on device memory.
    """
    backend: str
    device_count: int
    devices_per_host: int
    num_hosts: int
    supports_float64: bool
    supports_complex128: bool
    memory_per_device_gb: float
    recommended_batch_size: int


# Backends that lack float64/complex128 support.  ``mps`` is the Apple GPU
# (jax-mps / MLX, float32-only); ``metal`` is kept as the legacy Apple-GPU
# platform alias so a stale backend string can never bypass the no-f64 guard.
_NO_F64_BACKENDS = frozenset({"mps", "metal"})

# Known GPU memory sizes (GB) by platform string.
# NVIDIA GPUs.
_GPU_MEMORY_TABLE = {
    "a100": 40.0,
    "a100-80": 80.0,
    "h100": 80.0,
    "h200": 141.0,
    "v100": 16.0,
    "t4": 16.0,
    "l4": 24.0,
    "rtx3090": 24.0,
    "rtx4090": 24.0,
    # AMD Instinct GPUs.
    "mi250x": 128.0,  # 2×64 GB HBM2e
    "mi250": 128.0,
    "mi210": 64.0,
    "mi300x": 192.0,  # 8×24 GB HBM3
    "mi300a": 128.0,  # APU: 8×16 GB HBM3
    "mi100": 32.0,
    # AMD Radeon Pro / consumer.
    "w7900": 48.0,
    "w7800": 32.0,
    "rx7900xtx": 24.0,
}

# TPU memory per chip (GB) by generation.
_TPU_MEMORY_TABLE = {
    "v2": 8.0,
    "v3": 16.0,
    "v4": 32.0,
    "v5e": 16.0,
    "v5p": 95.0,
    "v6e": 32.0,
}


def _estimate_device_memory(backend: str, devices: list) -> float:
    """Estimate per-device memory in GB from device metadata."""
    if backend == "tpu":
        # Try to infer TPU generation from device kind string.
        if devices:
            kind = str(getattr(devices[0], "device_kind", "")).lower()
            for gen, mem in _TPU_MEMORY_TABLE.items():
                if gen in kind:
                    return mem
        # Default for unknown TPU.
        return 16.0

    if backend == "gpu":
        if devices:
            kind = str(getattr(devices[0], "device_kind", "")).lower()
            for gpu_name, mem in _GPU_MEMORY_TABLE.items():
                if gpu_name.replace("-", "") in kind.replace("-", "").replace(" ", ""):
                    return mem
        # Default for unknown GPU.
        return 16.0

    if backend == "mps":
        # Apple Silicon unified memory; conservatively assume half
        # of system RAM is available for GPU.
        return 8.0

    # CPU: effectively unlimited relative to accelerators.
    return 64.0


def _recommended_batch_size(memory_gb: float) -> int:
    """Heuristic batch size based on per-device memory.

    For cubed-sphere C48/L40 in float32, one full state is ~50 MB.
    We want enough headroom for XLA workspace, gradients, and halos.
    """
    if memory_gb >= 80.0:
        return 6   # All 6 faces per device
    elif memory_gb >= 32.0:
        return 4
    elif memory_gb >= 16.0:
        return 2
    else:
        return 1


def detect_devices() -> HardwareConfig:
    """Auto-detect hardware and return configuration.

    Queries JAX for available devices and infers backend capabilities.

    Side-effect: applies environment-variable-only XLA scheduler flags
    *before* the first ``jax.devices()`` query.  Setting XLA flags
    after PJRT initialisation is silently ineffective on most JAX
    versions, which would defeat the latency-hiding scheduler flag
    that is critical for multi-GPU strong scaling.  We therefore
    sniff the GPU vendor from env-vars only (no ``jax.*`` calls)
    and prime ``XLA_FLAGS`` here, before the device query.  The
    follow-up :func:`legoesm.runtime.configure_backend` call sets
    backend-specific JAX-level options (matmul precision etc.), which
    are safe to set post-init.

    Returns
    -------
    HardwareConfig
    """
    # Pre-init: set XLA scheduler flags based on env-var-only vendor
    # detection, before jax.devices() is called.  After PJRT init,
    # mutating XLA_FLAGS is silently ineffective on most JAX versions.
    try:
        from legoesm.runtime.backend import (
            detect_gpu_vendor_pre_init,
            NVIDIA_GPU_XLA_FLAGS,
            AMD_GPU_XLA_FLAGS,
            TPU_XLA_FLAGS,
            set_xla_flags,
        )

        _platforms = os.environ.get("JAX_PLATFORMS", "").lower()
        if "tpu" in _platforms:
            set_xla_flags(TPU_XLA_FLAGS)
        else:
            _vendor = detect_gpu_vendor_pre_init()
            if _vendor == "nvidia":
                set_xla_flags(NVIDIA_GPU_XLA_FLAGS)
            elif _vendor == "amd":
                set_xla_flags(AMD_GPU_XLA_FLAGS)
    except Exception:
        # Non-fatal: if the runtime helpers are unavailable for any
        # reason, fall through to plain JAX detection.
        pass

    backend = jax.default_backend().lower()
    devices = jax.devices()
    device_count = len(devices)
    num_hosts = jax.process_count()
    devices_per_host = len(jax.local_devices())

    supports_f64 = backend not in _NO_F64_BACKENDS
    supports_c128 = supports_f64  # complex128 requires float64 support

    memory_gb = _estimate_device_memory(backend, devices)
    batch_size = _recommended_batch_size(memory_gb)

    config = HardwareConfig(
        backend=backend,
        device_count=device_count,
        devices_per_host=devices_per_host,
        num_hosts=num_hosts,
        supports_float64=supports_f64,
        supports_complex128=supports_c128,
        memory_per_device_gb=memory_gb,
        recommended_batch_size=batch_size,
    )
    logger.info(
        "Detected %s backend: %d device(s), %.0f GB/device, "
        "f64=%s, %d host(s)",
        backend, device_count, memory_gb, supports_f64, num_hosts,
    )
    return config


# ============================================================================
# Optimal dtype selection
# ============================================================================

def get_optimal_dtype(
    config: HardwareConfig,
    precision: str = "single",
) -> jnp.dtype:
    """Return the optimal floating-point dtype for the given device.

    Parameters
    ----------
    config : HardwareConfig
        Output of :func:`detect_devices`.
    precision : str
        ``'half'`` — bfloat16 on TPU, float16 on GPU, float32 on CPU/MPS.
        ``'single'`` — float32 everywhere.
        ``'double'`` — float64 on backends that support it, float32 otherwise.

    Returns
    -------
    jnp.dtype
    """
    backend = config.backend

    if precision == "half":
        if backend == "tpu":
            return jnp.bfloat16
        elif backend == "gpu":
            return jnp.float16
        else:
            # CPU and MPS: float16/bfloat16 are emulated, not faster.
            return jnp.float32

    elif precision == "double":
        if config.supports_float64:
            return jnp.float64
        return jnp.float32

    # Default: single precision.
    return jnp.float32


# ============================================================================
# Mixed-precision policy
# ============================================================================

class MixedPrecisionPolicy(NamedTuple):
    """Per-device mixed-precision policy.

    Attributes
    ----------
    compute_dtype : jnp.dtype
        Dtype for forward-pass computation (matmuls, stencils).
    param_dtype : jnp.dtype
        Dtype for model parameters and state storage.
    output_dtype : jnp.dtype
        Dtype for outputs and loss computation.
    """
    compute_dtype: jnp.dtype
    param_dtype: jnp.dtype
    output_dtype: jnp.dtype


def mixed_precision_policy(config: HardwareConfig) -> MixedPrecisionPolicy:
    """Return a mixed-precision policy for the given device.

    TPU:   compute in bfloat16, params/output in float32.
    GPU:   compute in float32, params/output in float32.
           (float16 with loss scaling is left to user-level code.)
    MPS:   all float32 (no float64, no fast float16 path).
    CPU:   all float32 (no hardware acceleration for reduced precision).

    Parameters
    ----------
    config : HardwareConfig
        Output of :func:`detect_devices`.

    Returns
    -------
    MixedPrecisionPolicy
    """
    backend = config.backend

    if backend == "tpu":
        return MixedPrecisionPolicy(
            compute_dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
            output_dtype=jnp.float32,
        )

    # GPU, MPS, CPU: float32 across the board.
    return MixedPrecisionPolicy(
        compute_dtype=jnp.float32,
        param_dtype=jnp.float32,
        output_dtype=jnp.float32,
    )
