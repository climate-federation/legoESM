"""Hardware backend detection and compatibility guards for legoESM.

Provides runtime checks to ensure that code requiring specific numerical
capabilities (e.g., float64, complex128) is not accidentally run on
backends that lack those capabilities (e.g., Apple Metal/MPS).
"""

from __future__ import annotations

import warnings

import jax


# Backends known to lack float64 / complex128 support.
_UNSUPPORTED_F64_BACKENDS = frozenset({"METAL"})


def get_backend() -> str:
    """Return the current JAX default backend name (uppercase).

    Common values: ``"CPU"``, ``"GPU"``, ``"TPU"``, ``"METAL"``.
    """
    return jax.default_backend().upper()


def check_spectral_backend(
    *,
    allow_unsupported: bool = False,
) -> None:
    """Verify the current JAX backend supports float64 and complex128.

    The spectral solver (Gaussian grid + spherical harmonic transforms)
    requires float64 real arithmetic and complex128 FFTs.  Backends that
    do not support these types (e.g., Apple Metal/MPS) will produce
    incorrect results or crash with opaque errors.

    Parameters
    ----------
    allow_unsupported : bool
        If ``True``, emit a warning instead of raising.  Intended for
        expert users who have forced a compatible backend via
        ``JAX_PLATFORMS=cpu``.

    Raises
    ------
    ValueError
        If the backend is unsupported and *allow_unsupported* is False.
    """
    backend = get_backend()

    # Check that float64/complex128 is actually enabled in JAX config.
    # Without jax_enable_x64, JAX silently truncates float64 to float32,
    # which corrupts spectral transforms.
    if not jax.config.jax_enable_x64:
        x64_message = (
            "The spectral solver requires float64 and complex128 arithmetic, "
            "but JAX is running in 32-bit mode (jax_enable_x64 is not set).\n\n"
            "Remediation: set the environment variable JAX_ENABLE_X64=True "
            "or call jax.config.update('jax_enable_x64', True) before "
            "importing any spectral modules."
        )
        if allow_unsupported:
            warnings.warn(
                x64_message,
                RuntimeWarning,
                stacklevel=3,
            )
        else:
            raise ValueError(x64_message)

    if backend not in _UNSUPPORTED_F64_BACKENDS:
        return

    message = (
        f"The spectral solver requires float64 and complex128 arithmetic, "
        f"but the current JAX backend is '{backend}', which does not support "
        f"these types.\n\n"
        f"Remediation options:\n"
        f"  1. Force CPU backend:  JAX_PLATFORMS=cpu python your_script.py\n"
        f"  2. Use the finite-volume solver (cubed-sphere) instead, which "
        f"works in float32 on all backends including Metal.\n"
        f"  3. Set atmosphere.spectral.allow_unsupported=true in your config "
        f"to bypass this check (expert only, results may be incorrect)."
    )

    if allow_unsupported:
        warnings.warn(
            f"Spectral solver running on unsupported backend '{backend}'. "
            f"Results may be incorrect.\n\n" + message,
            RuntimeWarning,
            stacklevel=3,
        )
        return

    raise ValueError(message)


def detect_devices() -> dict:
    """Detect available JAX devices and their capabilities.

    Returns
    -------
    dict
        Keys:

        - ``backend`` (*str*) — default backend name (uppercase).
        - ``n_devices`` (*int*) — number of available devices.
        - ``devices`` (*list*) — list of ``jax.Device`` objects.
        - ``supports_f64`` (*bool*) — whether float64 is available.
        - ``distributed`` (*bool*) — whether ``jax.distributed``
          has been initialized.
    """
    backend = get_backend()
    devices = jax.devices()
    return {
        "backend": backend,
        "n_devices": len(devices),
        "devices": devices,
        "supports_f64": backend not in _UNSUPPORTED_F64_BACKENDS,
        "distributed": jax.process_count() > 1,
    }
