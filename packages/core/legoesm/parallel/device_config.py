"""Device configuration and hardware-aware optimization.

Auto-detects available accelerators (CPU, GPU, TPU, Metal) and configures
JAX for optimal performance on each platform.  Provides a unified interface
for device-specific settings including:

- Hardware detection and capability reporting
- XLA compiler flags tuned per backend
- Mixed-precision dtype policies
- Optimal device mesh construction for cubed-sphere and spectral grids

Usage
-----
>>> from legoesm.parallel.device_config import detect_devices, configure_jax_for_device
>>> config = detect_devices()
>>> configure_jax_for_device(config)
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
        One of ``'cpu'``, ``'gpu'``, ``'tpu'``, ``'metal'``.
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


# Backends that lack float64/complex128 support.
_NO_F64_BACKENDS = frozenset({"metal"})

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

    if backend == "metal":
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
    follow-up :func:`configure_jax_for_device` call still sets
    backend-specific JAX-level options (matmul precision, SPMD mode,
    etc.) which are safe to set post-init.

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
# XLA flags — canonical definitions live in runtime.backend.  They are imported
# *function-scope* inside the _configure_* helpers below (not at module top):
# a top-level ``from legoesm.runtime.backend import ...`` here closes a
# parallel->runtime->parallel import cycle (runtime.devices re-exports
# MixedPrecisionPolicy from this module, which is defined further down), so a
# cold import of device_config crashed with "cannot import name
# 'MixedPrecisionPolicy' from partially initialized module".
# ============================================================================


def configure_jax_for_device(config: HardwareConfig) -> None:
    """Apply JAX configuration optimized for the detected hardware.

    This function sets XLA flags, memory configuration, and JAX-level
    options appropriate for the backend.  It should be called once at
    startup, before any JAX computation.

    Parameters
    ----------
    config : HardwareConfig
        Output of :func:`detect_devices`.
    """
    backend = config.backend

    if backend == "tpu":
        _configure_tpu(config)
    elif backend == "gpu":
        _configure_gpu(config)
    elif backend == "metal":
        _configure_metal(config)
    else:
        _configure_cpu(config)

    logger.info("JAX configured for %s backend", backend)


def _configure_tpu(config: HardwareConfig) -> None:
    """Apply TPU-specific JAX and XLA configuration."""
    from legoesm.runtime.backend import set_xla_flags, TPU_XLA_FLAGS
    # Set XLA flags for TPU optimization.
    set_xla_flags(TPU_XLA_FLAGS)

    # TPU natively supports bfloat16 — enable matmul precision control
    # to allow XLA to choose bf16 for non-critical ops.
    jax.config.update("jax_default_matmul_precision", "bfloat16")

    # NOTE: the legacy `jax_spmd_mode='allow_all'` toggle was removed
    # in modern JAX (0.9+) — `jax.config.update("jax_spmd_mode", ...)`
    # raises AttributeError.  Under the unified sharding model SPMD
    # partitioning is the default for sharded arrays, so no explicit
    # toggle is required for multi-host TPU pods.


def _configure_gpu(config: HardwareConfig) -> None:
    """Apply GPU-specific JAX and XLA configuration.

    Detects NVIDIA vs AMD GPUs and applies vendor-appropriate flags.
    NVIDIA: cuDNN GEMM fusion + TensorFloat32 matmul precision.
    AMD (ROCm): no cuDNN flags, default float32 matmul precision.

    Uses the pre-init detector first so XLA scheduler flags land
    *before* the PJRT client is initialised — once ``jax.devices()``
    runs, mutating ``XLA_FLAGS`` is silently ineffective on most JAX
    versions, which would defeat the latency-hiding flags critical
    for multi-GPU scaling.
    """
    from legoesm.runtime.backend import (
        detect_gpu_vendor_pre_init, gpu_vendor,
        set_xla_flags, NVIDIA_GPU_XLA_FLAGS, AMD_GPU_XLA_FLAGS,
    )

    # Pre-init detection (env-var only, no jax.devices()) to land
    # XLA_FLAGS on time.
    pre_vendor = detect_gpu_vendor_pre_init()
    if pre_vendor == "nvidia":
        set_xla_flags(NVIDIA_GPU_XLA_FLAGS)
    elif pre_vendor == "amd":
        set_xla_flags(AMD_GPU_XLA_FLAGS)

    # Pre-allocate 90% of GPU memory to avoid fragmentation.
    # Only set if not already configured by the user.
    if "XLA_PYTHON_CLIENT_MEM_FRACTION" not in os.environ:
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"

    # Now safe to query JAX about the GPU.
    vendor = gpu_vendor()

    # If pre-init detection missed the vendor, apply best-effort flags
    # post-init (may not take effect, but won't make things worse) and
    # warn so the user can fix the env for next run.
    if pre_vendor is None and vendor != "unknown":
        if vendor == "nvidia":
            set_xla_flags(NVIDIA_GPU_XLA_FLAGS)
        elif vendor == "amd":
            set_xla_flags(AMD_GPU_XLA_FLAGS)
        logger.warning(
            "GPU vendor detected post-init from device_config; XLA "
            "scheduler flags may not take effect this run.  Set "
            "CUDA_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES or "
            "JAX_PLATFORMS before import to enable latency-hiding flags.",
        )

    # TensorFloat32 is an NVIDIA Ampere+ feature (19-bit mantissa) —
    # faster than FP32 with negligible accuracy loss for weather/climate.
    # AMD GPUs do not have TF32 hardware; leave matmul at default float32.
    if vendor == "nvidia":
        jax.config.update("jax_default_matmul_precision", "tensorfloat32")

    logger.info("GPU vendor: %s (%d device(s))", vendor, config.device_count)


def _configure_metal(config: HardwareConfig) -> None:
    """Apply Metal-specific JAX configuration.

    Metal does not support float64; spectral solvers are routed to CPU
    automatically.  We enable multi-threading for CPU fallback operations
    to ensure the spectral transforms (which run on CPU) use all cores.
    """
    _enable_cpu_multithreading()


def _configure_cpu(config: HardwareConfig) -> None:
    """Apply CPU-specific JAX configuration."""
    _enable_cpu_multithreading()


def _enable_cpu_multithreading() -> None:
    """Enable multi-threaded Eigen on the CPU backend.

    XLA dropped the ``--intra_op_parallelism_threads`` flag in jaxlib
    0.10; passing it now aborts the process at backend init.  Multi-thread
    Eigen is still enabled via ``--xla_cpu_multi_thread_eigen``, and the
    Eigen worker count is controlled by ``OMP_NUM_THREADS`` (defaults to
    hardware concurrency).
    """
    if "XLA_FLAGS" not in os.environ:
        try:
            from legoesm.runtime.backend import set_xla_flags
            # NOTE: the legacy `intra_op_parallelism_threads=N` XLA flag
            # is NOT recognized by current XLA and crashes JAX at first
            # use with a fatal `Unknown flag in XLA_FLAGS` error from
            # parse_flags_from_env.cc.  Drop it; rely on XLA's default
            # CPU thread-pool autoscaling from os.cpu_count().
            set_xla_flags({"xla_cpu_multi_thread_eigen": "true"})
        except Exception:
            pass  # Non-critical; XLA will use defaults.
    if "OMP_NUM_THREADS" not in os.environ:
        try:
            os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() or 4)
        except Exception:
            pass


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
        ``'half'`` — bfloat16 on TPU, float16 on GPU, float32 on CPU/Metal.
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
            # CPU and Metal: float16/bfloat16 are emulated, not faster.
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
    Metal: all float32 (no float64, no fast float16 path).
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

    # GPU, Metal, CPU: float32 across the board.
    return MixedPrecisionPolicy(
        compute_dtype=jnp.float32,
        param_dtype=jnp.float32,
        output_dtype=jnp.float32,
    )


def cast_for_device(
    x: jax.Array,
    config: HardwareConfig,
    role: str = "compute",
) -> jax.Array:
    """Cast array to appropriate dtype for current device and role.

    Parameters
    ----------
    x : jax.Array
        Input array.
    config : HardwareConfig
        Output of :func:`detect_devices`.
    role : str
        ``'compute'`` — use the compute dtype (may be bfloat16 on TPU).
        ``'param'`` — use the parameter dtype (always float32).
        ``'output'`` — use the output dtype (always float32).

    Returns
    -------
    jax.Array
        Array cast to the target dtype.  Same object if no cast needed.
    """
    policy = mixed_precision_policy(config)
    if role == "compute":
        target = policy.compute_dtype
    elif role == "param":
        target = policy.param_dtype
    elif role == "output":
        target = policy.output_dtype
    else:
        raise ValueError(
            f"Unknown role {role!r}. Use 'compute', 'param', or 'output'."
        )

    if x.dtype == target:
        return x
    return x.astype(target)


# ============================================================================
# Optimal device mesh
# ============================================================================

def get_optimal_mesh(
    config: HardwareConfig,
    grid_type: str = "cubed_sphere",
    nlev: int = 40,
    *,
    nCells: int = 0,
    nEdges: int = 0,
    nVertices: int = 0,
):
    """Create the optimal device mesh for the given hardware and grid.

    This is a high-level convenience that dispatches to the appropriate
    mesh constructor in :mod:`legoesm.parallel.mesh` based on device
    count and grid type.

    Parameters
    ----------
    config : HardwareConfig
        Output of :func:`detect_devices`.
    grid_type : str
        ``'cubed_sphere'``, ``'latlon'``, ``'spectral'``, or ``'voronoi'``.
    nlev : int
        Number of vertical levels (used for level-parallel spectral mesh).
    nCells : int
        Number of Voronoi cells (required for ``grid_type='voronoi'``).
    nEdges : int
        Number of Voronoi edges (required for ``grid_type='voronoi'``).
    nVertices : int
        Number of Voronoi vertices (required for ``grid_type='voronoi'``).

    Returns
    -------
    DeviceConfig
        From :mod:`legoesm.parallel.mesh`.

    Notes
    -----
    Sharding strategies by device count and grid:

    **Cubed-sphere:**
    - 1 device: no mesh (single device fast path)
    - 2-6 devices: face sharding (6 faces across devices)
    - >6 devices (multiples of 6): sub-face tiling

    **TPU v4-8 (4 chips):**
      4 chips for cubed-sphere means face sharding with 3 chips active
      (6 faces / 2 faces per chip).  Alternatively, use 2 chips if the
      grid is small enough to fit.

    **Multi-GPU (4x A100, 8x H100):**
      Face sharding for <= 6 GPUs; sub-face tiling for 24+ GPUs.

    **Spectral:**
    - Level-parallel sharding when nlev > device_count.
    - Falls back to single-device when device_count == 1.

    **Voronoi (icosahedral):**
    - Cell/edge dimension sharding across all devices.
    - Reorder mesh with ``reorder_voronoi_for_sharding`` for best locality.
    """
    from legoesm.parallel.mesh import (
        create_device_mesh,
        create_latlon_mesh,
        create_level_mesh,
        create_voronoi_device_mesh,
    )

    n_dev = config.devices_per_host

    if grid_type == "cubed_sphere":
        return create_device_mesh(n_devices=n_dev)
    elif grid_type == "latlon":
        return create_latlon_mesh(n_devices=n_dev)
    elif grid_type == "spectral":
        # For spectral grids, level-parallel sharding is used.
        # Clamp device count to not exceed nlev.
        effective = min(n_dev, nlev)
        return create_level_mesh(n_devices=effective)
    elif grid_type == "voronoi":
        if nCells <= 0 or nEdges <= 0 or nVertices <= 0:
            raise ValueError(
                "For grid_type='voronoi', nCells, nEdges, and nVertices "
                "must be provided (positive integers)."
            )
        return create_voronoi_device_mesh(
            nCells=nCells, nEdges=nEdges, nVertices=nVertices,
            n_devices=n_dev,
        )
    else:
        raise ValueError(
            f"Unknown grid_type {grid_type!r}. "
            f"Use 'cubed_sphere', 'latlon', 'spectral', or 'voronoi'."
        )
