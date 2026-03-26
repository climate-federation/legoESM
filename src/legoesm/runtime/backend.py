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


# ---------------------------------------------------------------------------
# Backend query
# ---------------------------------------------------------------------------

def get_backend() -> str:
    """Return the current JAX default backend name (lowercase).

    Common values: ``"cpu"``, ``"gpu"``, ``"tpu"``, ``"metal"``.
    """
    import jax
    return jax.default_backend().lower()


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

_GPU_XLA_FLAGS = {
    "xla_gpu_enable_async_collectives": "true",
    "xla_gpu_cudnn_gemm_fusion_level": "3",
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


def configure_backend(backend: str | None = None) -> str:
    """Apply backend-specific XLA flags and JAX options.

    This should be called **once at startup**, before any JAX computation.
    If *backend* is ``None`` the current default backend is detected.

    Returns the resolved backend name (lowercase).
    """
    import jax

    if backend is None:
        backend = get_backend()
    backend = backend.lower()

    if backend == "tpu":
        _set_xla_flags(_TPU_XLA_FLAGS)
        jax.config.update("jax_default_matmul_precision", "bfloat16")
        num_hosts = jax.process_count()
        if num_hosts > 1:
            jax.config.update("jax_spmd_mode", "allow_all")

    elif backend == "gpu":
        devices = jax.devices()
        if len(devices) > 1:
            _set_xla_flags(_GPU_XLA_FLAGS)
        if "XLA_PYTHON_CLIENT_MEM_FRACTION" not in os.environ:
            os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"
        jax.config.update("jax_default_matmul_precision", "tensorfloat32")

    elif backend == "metal":
        pass  # No special flags needed.

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
