"""Hardware backend detection and compatibility guards for legoESM.

Provides runtime checks to ensure that code requiring specific numerical
capabilities (e.g., float64, complex128) is not accidentally run on
backends that lack those capabilities (e.g., Apple Metal/MPS).
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp


# Backends known to lack float64 / complex128 support.
_UNSUPPORTED_F64_BACKENDS = frozenset({"METAL"})

_PRECISION_NAME_TO_DTYPE = {
    "float16": jnp.float16,
    "fp16": jnp.float16,
    "half": jnp.float16,
    "bfloat16": jnp.bfloat16,
    "bf16": jnp.bfloat16,
    "float32": jnp.float32,
    "fp32": jnp.float32,
    "single": jnp.float32,
    "float64": jnp.float64,
    "fp64": jnp.float64,
    "double": jnp.float64,
}

_RUNTIME_PRECISION_POLICY = {
    "dynamics": jnp.float32,
    "ml": jnp.bfloat16,
    "conservation": None,  # None => auto-select widest supported accumulator.
}
_UNSET = object()


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


def _parse_precision_dtype(value, *, field_name: str, allow_none: bool = False):
    """Parse a precision config value into a JAX dtype."""
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{field_name} precision cannot be None")

    if value in (jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64):
        return value

    if isinstance(value, str):
        key = value.strip().lower()
        if key in _PRECISION_NAME_TO_DTYPE:
            return _PRECISION_NAME_TO_DTYPE[key]

    allowed = sorted(_PRECISION_NAME_TO_DTYPE.keys())
    raise ValueError(
        f"Unknown {field_name} precision {value!r}. "
        f"Use one of {allowed}."
    )


def set_runtime_precision_policy(
    *,
    dynamics=_UNSET,
    ml=_UNSET,
    conservation=_UNSET,
) -> dict:
    """Set runtime precision policy from config-like values.

    Returns the resolved dtype policy as a dict.
    """
    if dynamics is not _UNSET:
        _RUNTIME_PRECISION_POLICY["dynamics"] = _parse_precision_dtype(
            dynamics, field_name="dynamics",
        )
    if ml is not _UNSET:
        _RUNTIME_PRECISION_POLICY["ml"] = _parse_precision_dtype(
            ml, field_name="ml",
        )
    if conservation is not _UNSET:
        cons = _parse_precision_dtype(
            conservation, field_name="conservation", allow_none=True,
        )
        if cons not in (None, jnp.float32, jnp.float64):
            raise ValueError(
                "conservation precision must be float32, float64, or None."
            )
        _RUNTIME_PRECISION_POLICY["conservation"] = cons

    return get_runtime_precision_policy()


def get_runtime_precision_policy() -> dict:
    """Return the active runtime precision policy."""
    return dict(_RUNTIME_PRECISION_POLICY)


def get_runtime_precision_dtype(component: str):
    """Return configured dtype for one precision component."""
    if component not in _RUNTIME_PRECISION_POLICY:
        raise KeyError(
            f"Unknown precision component {component!r}. "
            f"Use one of {tuple(_RUNTIME_PRECISION_POLICY.keys())}."
        )
    return _RUNTIME_PRECISION_POLICY[component]


def apply_hardware_config(config) -> dict:
    """Apply ``hardware.*`` runtime options from a legoESM Config object.

    This consumes:
    - ``hardware.precision.*``
    - ``hardware.devices`` (legacy alias)
    - ``hardware.parallelism.*``
    """
    dynamics_precision = config.get("hardware.precision.dynamics")
    ml_precision = config.get("hardware.precision.ml")
    conservation_precision = config.get("hardware.precision.conservation")

    policy = set_runtime_precision_policy(
        dynamics=dynamics_precision,
        ml=ml_precision,
        conservation=conservation_precision,
    )

    # Enable x64 when dynamics precision requires it. We intentionally avoid
    # forcing x64 off for lower-precision settings, since spectral workflows
    # may rely on x64 being enabled elsewhere.
    if policy["dynamics"] == jnp.float64 and not jax.config.jax_enable_x64:
        jax.config.update("jax_enable_x64", True)

    n_devices = config.get("hardware.parallelism.n_devices", "auto")
    backend = config.get("hardware.parallelism.backend", None)
    distributed = bool(config.get("hardware.parallelism.distributed", False))

    # Legacy compatibility: hardware.devices acts as n_devices when
    # parallelism.n_devices is not set explicitly.
    legacy_devices = config.get("hardware.devices", "auto")
    if n_devices in (None, "auto") and legacy_devices not in (None, "auto"):
        n_devices = legacy_devices
    if n_devices is None:
        n_devices = "auto"

    if distributed:
        if backend is not None or n_devices not in (None, "auto"):
            warnings.warn(
                "hardware.parallelism.distributed=true ignores "
                "hardware.parallelism.backend and n_devices; using "
                "MPI rank-local device topology.",
                RuntimeWarning,
                stacklevel=2,
            )
        from legoesm.parallel.distributed import initialize_distributed
        device_config, topology = initialize_distributed(return_topology=True)
        return {
            "precision": policy,
            "distributed": True,
            "device_config": device_config,
            "topology": topology,
        }

    from legoesm.parallel.mesh import create_device_mesh
    device_config = create_device_mesh(n_devices=n_devices, backend=backend)
    return {
        "precision": policy,
        "distributed": False,
        "device_config": device_config,
        "topology": None,
    }
