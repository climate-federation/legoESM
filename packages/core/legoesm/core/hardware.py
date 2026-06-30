"""Hardware backend detection and compatibility guards for legoESM.

.. deprecated::
    This module is **legacy**.  New code should import from
    :mod:`legoesm.runtime` instead.  Every public function here is now a
    thin wrapper that delegates to the canonical runtime layer.  These
    wrappers exist only for backward compatibility and will be removed in
    a future release.
"""

from __future__ import annotations


import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Legacy constants — kept so that ``from legoesm.core.hardware import
# _UNSUPPORTED_F64_BACKENDS`` in conservation.py et al. keeps working.
# The Apple GPU backend ``mps`` (jax-mps / MLX) is float32-only.  Include
# both lowercase and uppercase spellings: some callers (and
# ``test_runtime_bootstrap``) match against ``"MPS"``; newer callers
# normalise to lowercase.  Carrying both keys avoids subtle cross-version
# mismatches without changing the lookup semantics elsewhere (every
# existing call lowercases its query).  ``metal``/``METAL`` are kept as the
# legacy Apple-GPU platform aliases (that backend is no longer wired in) so a
# stale backend string can never silently bypass the no-f64 guard — those
# Metal GPUs also lacked f64.
# ---------------------------------------------------------------------------
_UNSUPPORTED_F64_BACKENDS = frozenset({"mps", "MPS", "metal", "METAL"})

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

# Legacy mutable precision dict — still the backing store for the 3-component
# policy used by conservation.py.  Kept in sync by set_runtime_precision_policy
# and by runtime.precision.apply_precision.
_RUNTIME_PRECISION_POLICY = {
    "dynamics": jnp.float32,
    "ml": jnp.bfloat16,
    "conservation": None,  # None => auto-select widest supported accumulator.
}
_UNSET = object()


# ---------------------------------------------------------------------------
# Delegating wrappers
# ---------------------------------------------------------------------------

def get_backend() -> str:
    """Return the current JAX default backend name (**uppercase**).

    .. deprecated:: Use ``legoesm.runtime.get_backend()`` (lowercase) instead.
    """
    from legoesm.runtime.backend import get_backend as _get_backend
    return _get_backend().upper()          # legacy callers expect uppercase


def check_spectral_backend(
    *,
    allow_unsupported: bool = False,
) -> None:
    """Verify the backend supports float64/complex128.

    .. deprecated:: Use ``legoesm.runtime.check_spectral_backend()`` instead.
    """
    from legoesm.runtime.backend import check_spectral_backend as _check
    _check(allow_unsupported=allow_unsupported)


def detect_devices() -> dict:
    """Detect available JAX devices and capabilities.

    .. deprecated:: Use ``legoesm.runtime.detect_hardware()`` instead.
    """
    backend = get_backend()
    devices = jax.devices()
    return {
        "backend": backend,
        "n_devices": len(devices),
        "devices": devices,
        "supports_f64": backend.lower() not in _UNSUPPORTED_F64_BACKENDS,
        "distributed": jax.process_count() > 1,
    }


# ---------------------------------------------------------------------------
# Precision helpers — still the canonical implementation for the 3-component
# legacy dict because several modules import _parse_precision_dtype and
# get_runtime_precision_dtype directly.
# ---------------------------------------------------------------------------

def _parse_precision_dtype(value, *, field_name: str, allow_none: bool = False):
    """Parse a precision config value into a JAX dtype.

    .. deprecated:: Delegates to ``legoesm.core.precision.parse_dtype``.
    """
    from legoesm.core.precision import parse_dtype
    return parse_dtype(value, field_name=field_name, allow_none=allow_none)


def set_runtime_precision_policy(
    *,
    dynamics=_UNSET,
    ml=_UNSET,
    conservation=_UNSET,
) -> dict:
    """Set the legacy 3-component runtime precision policy.

    .. deprecated:: Use ``legoesm.runtime.apply_precision()`` instead.
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
    """Return the active legacy 3-component precision policy."""
    return dict(_RUNTIME_PRECISION_POLICY)


def get_runtime_precision_dtype(component: str):
    """Return configured dtype for one legacy precision component."""
    if component not in _RUNTIME_PRECISION_POLICY:
        raise KeyError(
            f"Unknown precision component {component!r}. "
            f"Use one of {tuple(_RUNTIME_PRECISION_POLICY.keys())}."
        )
    return _RUNTIME_PRECISION_POLICY[component]


def apply_hardware_config(config) -> dict:
    """Apply ``hardware.*`` runtime options from a legoESM Config object.

    .. deprecated::
        Use ``legoesm.runtime.bootstrap_from_yaml_config(config)`` instead.
        This wrapper delegates to the canonical runtime layer and syncs
        the legacy 3-component precision dict for backward compatibility.
    """
    # First, consume the precision keys into the legacy dict so that
    # callers reading get_runtime_precision_dtype() see the right values.
    dynamics_precision = config.get("hardware.precision.dynamics", None)
    ml_precision = config.get("hardware.precision.ml", None)
    conservation_precision = config.get("hardware.precision.conservation", None)
    kwargs = {}
    if dynamics_precision is not None:
        kwargs["dynamics"] = dynamics_precision
    if ml_precision is not None:
        kwargs["ml"] = ml_precision
    if conservation_precision is not None:
        kwargs["conservation"] = conservation_precision
    if kwargs:
        set_runtime_precision_policy(**kwargs)

    # Delegate to canonical bootstrap.
    from legoesm.runtime.config import bootstrap_from_yaml_config
    rc = bootstrap_from_yaml_config(config)

    return {
        "precision": get_runtime_precision_policy(),
        "distributed": rc.distributed,
        "device_config": rc.device_config,
        "topology": None,
    }
