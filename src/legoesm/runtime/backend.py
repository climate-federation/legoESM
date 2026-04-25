"""Backend detection, XLA configuration, and X64 policy.

This is the single source of truth for all backend-related queries and
environment setup.  It consolidates logic previously split across
``core.hardware`` and ``parallel.device_config``.

Key design rules:

* **No JAX import at module scope** — JAX is imported lazily inside
  functions so that environment variables (``JAX_PLATFORMS``,
  ``XLA_FLAGS``, etc.) can be set *before* JAX initialises.
* All public helpers are pure queries or idempotent setters.
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import NamedTuple

logger = logging.getLogger(__name__)

# Backends that lack float64/complex128 hardware support.
_NO_F64_BACKENDS = frozenset({"metal"})

# Cached result of the Metal health check (None = not yet tested).
_metal_healthy: bool | None = None
# Whether we have already applied the CPU fallback.
_cpu_fallback_applied: bool = False


# ---------------------------------------------------------------------------
# Metal health check and CPU fallback
# ---------------------------------------------------------------------------

def _metal_is_functional() -> bool:
    """Test whether the Metal backend can execute a trivial operation.

    Returns ``False`` when an incompatible ``jax-metal`` plugin is installed
    (e.g. jax-metal 0.1.x with JAX 0.9.x).  The result is cached so the
    probe runs at most once per process.
    """
    global _metal_healthy
    if _metal_healthy is not None:
        return _metal_healthy

    try:
        import jax
        import jax.numpy as jnp
        x = jax.device_put(1.0, jax.devices()[0])
        _ = float(x + x)
        _metal_healthy = True
    except Exception:
        _metal_healthy = False
    return _metal_healthy


def ensure_metal_or_fallback() -> None:
    """If Metal is the default backend but non-functional, fall back to CPU.

    This sets ``jax.default_device`` to the CPU device so all subsequent
    array creation and computation runs on CPU transparently.  Call this
    once at startup (e.g. from ``configure_backend``).
    """
    global _cpu_fallback_applied
    if _cpu_fallback_applied:
        return

    import jax
    if jax.default_backend().lower() != "metal":
        return

    if _metal_is_functional():
        return

    cpu = jax.devices("cpu")[0]
    jax.config.update("jax_default_device", cpu)
    _cpu_fallback_applied = True
    warnings.warn(
        "Metal backend detected but non-functional (likely jax-metal / JAX "
        "version mismatch). All computation will run on CPU.  To silence "
        "this warning, either upgrade jax-metal or set JAX_PLATFORMS=cpu.",
        RuntimeWarning,
        stacklevel=2,
    )
    logger.warning(
        "Metal backend broken — forced CPU fallback via jax.default_device"
    )


def metal_fell_back_to_cpu() -> bool:
    """Return ``True`` if Metal was detected but we fell back to CPU."""
    return _cpu_fallback_applied


# ---------------------------------------------------------------------------
# Backend query
# ---------------------------------------------------------------------------

def get_backend() -> str:
    """Return the current JAX default backend name (lowercase).

    Common values: ``"cpu"``, ``"gpu"``, ``"tpu"``, ``"metal"``.

    If the Metal backend was detected but is non-functional and we fell
    back to CPU, this returns ``"cpu"``.
    """
    import jax
    backend = jax.default_backend().lower()
    if backend == "metal":
        # Lazy health check — triggers at most once.
        ensure_metal_or_fallback()
        if _cpu_fallback_applied:
            return "cpu"
    return backend


def supports_float64(backend: str | None = None) -> bool:
    """Return whether *backend* supports float64/complex128 natively."""
    if backend is None:
        backend = get_backend()
    return backend.lower() not in _NO_F64_BACKENDS


# ---------------------------------------------------------------------------
# X64 policy
# ---------------------------------------------------------------------------

def enable_x64(*, quiet: bool = False) -> None:
    """Enable JAX float64 support (idempotent).

    Must be called **before** any JAX computation for the flag to
    take effect globally.
    """
    import jax
    if not jax.config.jax_enable_x64:
        jax.config.update("jax_enable_x64", True)
        if not quiet:
            logger.debug("Enabled JAX x64 mode")


def is_x64_enabled() -> bool:
    """Return ``True`` if JAX x64 mode is active."""
    import jax
    return bool(jax.config.jax_enable_x64)


def require_x64(component: str) -> None:
    """Raise ``RuntimeError`` if JAX x64 is not enabled.

    Parameters
    ----------
    component : str
        Human-readable name of the component that needs x64
        (used in the error message).
    """
    if not is_x64_enabled():
        raise RuntimeError(
            f"{component} requires float64 precision, but JAX is running "
            f"in 32-bit mode (jax_enable_x64 is not set).\n\n"
            f"Enable 64-bit mode before constructing the mesh:\n"
            f"  - environment variable: JAX_ENABLE_X64=1\n"
            f"  - Python:  jax.config.update('jax_enable_x64', True)\n"
            f"  - legoESM: from legoesm.runtime import enable_x64; enable_x64()"
        )


# ---------------------------------------------------------------------------
# XLA flag helpers (from parallel.device_config)
# ---------------------------------------------------------------------------

_TPU_XLA_FLAGS = {
    "xla_tpu_enable_async_collective_fusion": "true",
    "xla_tpu_enable_data_parallel_all_reduce_opt": "true",
    "xla_tpu_enable_latency_hiding_scheduler": "LHS_DEFAULT",
}

_NVIDIA_GPU_XLA_FLAGS = {
    "xla_gpu_cudnn_gemm_fusion_level": "3",
    # Overlap compute with collective communication (halo exchange, allreduce).
    "xla_gpu_enable_latency_hiding_scheduler": "true",
    "xla_gpu_enable_async_all_reduce": "true",
    # Enable async for ALL collectives (ppermute, all-gather, etc.),
    # not just allreduce.  Critical for icosahedral grids that use
    # ppermute-based halo exchange — without this flag, ppermute blocks
    # until completion, leaving the GPU idle during communication.
    "xla_gpu_enable_async_collectives": "true",
    "xla_gpu_enable_highest_priority_async_stream": "true",
}

_AMD_GPU_XLA_FLAGS: dict[str, str] = {
    # ROCm does not use cuDNN; no vendor-specific flags needed yet.
}


def _set_xla_flags(flags: dict[str, str]) -> None:
    """Append XLA flags to ``XLA_FLAGS``, without duplicating existing keys."""
    existing = os.environ.get("XLA_FLAGS", "")
    new_parts = []
    for key, value in flags.items():
        flag = f"--{key}={value}"
        if key not in existing:
            new_parts.append(flag)
    if new_parts:
        combined = (existing + " " + " ".join(new_parts)).strip()
        os.environ["XLA_FLAGS"] = combined


def _detect_gpu_vendor_pre_init() -> str | None:
    """Detect GPU vendor *without* triggering ``jax.devices()``.

    Once ``jax.devices()`` runs, the XLA client is initialised and any
    further mutations of ``os.environ['XLA_FLAGS']`` no longer take
    effect.  The latency-hiding scheduler flags applied below
    (``_NVIDIA_GPU_XLA_FLAGS``, ``_AMD_GPU_XLA_FLAGS``) are critical for
    multi-GPU scaling, so we must set them before JAX is initialised.

    Returns ``"nvidia"`` / ``"amd"`` if a vendor is unambiguously
    identifiable from environment hints, otherwise ``None`` and the
    caller should fall back to the post-init JAX-based detection (which
    will at least pick the right matmul precision even if ``XLA_FLAGS``
    is too late to alter).
    """
    # SLURM and Nvidia commonly set ``CUDA_VISIBLE_DEVICES=-1`` or
    # ``CUDA_VISIBLE_DEVICES=NoDevFiles`` to mask GPUs on CPU-only
    # allocations.  Those values are non-empty strings, so a naive
    # truthiness check would mis-detect NVIDIA and inject GPU XLA flags
    # into the CPU backend.  Reject them explicitly.
    _CUDA_DISABLED = {"-1", "NoDevFiles", ""}

    cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    hip = os.environ.get("HIP_VISIBLE_DEVICES", "")
    rocr = os.environ.get("ROCR_VISIBLE_DEVICES", "")

    has_cuda = cuda not in _CUDA_DISABLED and not cuda.startswith("-")
    has_hip = (hip not in _CUDA_DISABLED and not hip.startswith("-")) or (
        rocr not in _CUDA_DISABLED and not rocr.startswith("-")
    )

    if has_cuda and not has_hip:
        return "nvidia"
    if has_hip:
        return "amd"
    platforms = os.environ.get("JAX_PLATFORMS", "").lower()
    if platforms.startswith("cuda"):
        return "nvidia"
    if platforms.startswith("rocm"):
        return "amd"
    return None


def _is_nvidia_gpu() -> bool:
    """Return ``True`` if the default GPU device is an NVIDIA (CUDA) GPU."""
    import jax
    devices = jax.devices()
    if not devices:
        return False
    kind = str(getattr(devices[0], "device_kind", "")).lower()
    return "nvidia" in kind


def _is_amd_gpu() -> bool:
    """Return ``True`` if the default GPU device is an AMD (ROCm) GPU."""
    import jax
    devices = jax.devices()
    if not devices:
        return False
    kind = str(getattr(devices[0], "device_kind", "")).lower()
    return "amd" in kind or "instinct" in kind


def gpu_vendor() -> str:
    """Return the GPU vendor: ``"nvidia"``, ``"amd"``, or ``"unknown"``.

    Only meaningful when ``get_backend() == "gpu"``.
    """
    if _is_nvidia_gpu():
        return "nvidia"
    if _is_amd_gpu():
        return "amd"
    return "unknown"


def _detect_backend_pre_init() -> str | None:
    """Detect the backend without calling ``jax.default_backend()``.

    Mirrors :func:`_detect_gpu_vendor_pre_init`: we cannot afford to
    initialise the PJRT client (via ``jax.default_backend()`` or
    ``jax.devices()``) before the GPU-specific XLA scheduler flags are
    set, otherwise ``XLA_FLAGS`` mutations no-op.  Returns one of
    ``"tpu"``, ``"gpu"``, ``"metal"``, ``"cpu"``, or ``None`` when no
    hint is available (caller must fall back to ``get_backend()``).
    """
    platforms = os.environ.get("JAX_PLATFORMS", "").lower()
    if platforms:
        if platforms.startswith("tpu"):
            return "tpu"
        if platforms.startswith(("cuda", "rocm", "gpu")):
            return "gpu"
        if platforms.startswith("metal"):
            return "metal"
        if platforms.startswith("cpu"):
            return "cpu"
    if _detect_gpu_vendor_pre_init() is not None:
        return "gpu"
    if os.environ.get("TPU_NAME") or os.environ.get("COLAB_TPU_ADDR"):
        return "tpu"
    return None


def configure_backend(backend: str | None = None) -> str:
    """Apply backend-specific XLA flags and JAX options.

    This should be called **once at startup**, before any JAX computation.
    If *backend* is ``None`` the current default backend is detected
    from environment hints (``JAX_PLATFORMS``, ``CUDA_VISIBLE_DEVICES``,
    ``TPU_NAME``).  We deliberately avoid ``jax.default_backend()`` /
    ``jax.devices()`` until **after** the GPU XLA scheduler flags are
    set — otherwise PJRT initialises with the wrong flags.

    Returns the resolved backend name (lowercase).
    """
    import jax

    if backend is None:
        backend = _detect_backend_pre_init()
    if backend is None:
        # No hint at all — fall through to the JAX default.  We still
        # try to apply NVIDIA flags pre-init in case the heuristic
        # missed something (env var weirdness).  This branch is the
        # last resort and will warn from the GPU path below if the
        # client is already up.
        backend = get_backend()
    backend = backend.lower()

    if backend == "tpu":
        _set_xla_flags(_TPU_XLA_FLAGS)
        jax.config.update("jax_default_matmul_precision", "bfloat16")
        num_hosts = jax.process_count()
        if num_hosts > 1:
            jax.config.update("jax_spmd_mode", "allow_all")

    elif backend == "gpu":
        # Detect vendor BEFORE ``jax.devices()`` so the XLA scheduler
        # flags below land in the env var that the XLA client will
        # consume on first init.  Once ``jax.devices()`` runs, mutating
        # ``os.environ['XLA_FLAGS']`` is silently ineffective on most
        # JAX versions — the latency-hiding flags are critical for
        # multi-GPU scaling, so we cannot afford that race.
        if "XLA_PYTHON_CLIENT_MEM_FRACTION" not in os.environ:
            os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"

        pre_init_vendor = _detect_gpu_vendor_pre_init()
        if pre_init_vendor == "nvidia":
            _set_xla_flags(_NVIDIA_GPU_XLA_FLAGS)
        elif pre_init_vendor == "amd":
            _set_xla_flags(_AMD_GPU_XLA_FLAGS)

        devices = jax.devices()
        vendor = gpu_vendor()
        if pre_init_vendor is None:
            # Vendor was only known after JAX init; XLA_FLAGS already
            # locked in.  Still apply best-effort matmul precision
            # below, and warn so the user can pre-set ``CUDA_VISIBLE_DEVICES``
            # / ``HIP_VISIBLE_DEVICES`` for the next run.
            if vendor == "nvidia":
                _set_xla_flags(_NVIDIA_GPU_XLA_FLAGS)
                logger.warning(
                    "GPU vendor detected post-init; XLA scheduler flags "
                    "may not take effect this run.  Set "
                    "CUDA_VISIBLE_DEVICES or JAX_PLATFORMS=cuda before "
                    "import to enable latency-hiding flags.",
                )
            elif vendor == "amd":
                _set_xla_flags(_AMD_GPU_XLA_FLAGS)
                logger.warning(
                    "GPU vendor detected post-init; XLA scheduler flags "
                    "may not take effect this run.  Set "
                    "HIP_VISIBLE_DEVICES before import to enable "
                    "latency-hiding flags.",
                )

        # TensorFloat32 is an NVIDIA Ampere+ feature (19-bit mantissa).
        # AMD GPUs do not have TF32 hardware; use default float32.
        if vendor == "nvidia":
            jax.config.update("jax_default_matmul_precision", "tensorfloat32")
        logger.info("GPU vendor: %s (%d device(s))", vendor, len(devices))

    elif backend == "metal":
        ensure_metal_or_fallback()
        if _cpu_fallback_applied:
            backend = "cpu"

    else:  # cpu
        if "XLA_FLAGS" not in os.environ:
            try:
                n_cores = os.cpu_count() or 4
                _set_xla_flags({
                    "xla_cpu_multi_thread_eigen": "true",
                    "intra_op_parallelism_threads": str(n_cores),
                })
            except Exception:
                pass

    logger.info("Configured XLA for %s backend", backend)
    return backend


# ---------------------------------------------------------------------------
# Spectral-backend guard (from core.hardware)
# ---------------------------------------------------------------------------

def check_spectral_backend(*, allow_unsupported: bool = False) -> None:
    """Verify the current backend supports float64/complex128.

    The spectral solver requires these types.  Backends like Metal
    do not provide them and will produce incorrect results.

    Raises ``ValueError`` unless *allow_unsupported* is ``True``
    (in which case a warning is emitted instead).
    """
    import jax

    if not jax.config.jax_enable_x64:
        msg = (
            "The spectral solver requires float64 and complex128 arithmetic, "
            "but JAX is running in 32-bit mode (jax_enable_x64 is not set).\n\n"
            "Remediation: set the environment variable JAX_ENABLE_X64=True "
            "or call jax.config.update('jax_enable_x64', True) before "
            "importing any spectral modules."
        )
        if allow_unsupported:
            warnings.warn(msg, RuntimeWarning, stacklevel=3)
        else:
            raise ValueError(msg)

    backend = get_backend()
    if backend.lower() not in _NO_F64_BACKENDS:
        return

    msg = (
        f"The spectral solver requires float64 and complex128, "
        f"but the current backend is '{backend}', which does not "
        f"support these types.\n\n"
        f"Remediation options:\n"
        f"  1. Force CPU: JAX_PLATFORMS=cpu python your_script.py\n"
        f"  2. Use the finite-volume solver (cubed-sphere), which "
        f"works in float32 on all backends.\n"
        f"  3. Set allow_unsupported=true to bypass (expert only)."
    )
    if allow_unsupported:
        warnings.warn(
            f"Spectral solver on unsupported backend '{backend}'. "
            f"Results may be incorrect.\n\n" + msg,
            RuntimeWarning,
            stacklevel=3,
        )
        return
    raise ValueError(msg)
