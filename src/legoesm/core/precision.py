"""Precision management for legoESM.

Provides a unified precision policy system that controls storage, compute,
accumulation, and control dtypes across all model components. Supports
FP32, FP64, and mixed-precision modes for GPU-optimized century-scale
climate simulations.

Three precision modes
---------------------
1. **FP32**: All computation in float32 (fastest, GPU-optimal)
2. **FP64**: Full scientific reference mode (most accurate)
3. **Mixed**: Storage/compute in float32, accumulation/solvers in float64
   (best balance of performance and stability)

Usage
-----
>>> from legoesm.core.precision import PrecisionPolicy, get_policy, cast, const
>>> policy = PrecisionPolicy.mixed()
>>> x_compute = cast(x, "tracer_advection", "compute")
>>> one = const(1.0, "pressure_gradient", "control")

References
----------
- Váňa et al. (2017): Single precision in weather forecasting.
- Klöwer et al. (2020): Number formats, error mitigation, and scope for
  16-bit arithmetics in weather and climate modeling.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Precision roles
# ---------------------------------------------------------------------------

class PrecisionPolicy(NamedTuple):
    """Global precision configuration.

    Attributes
    ----------
    storage : jnp.dtype
        Dtype for state arrays in memory. Determines GPU memory footprint.
    compute : jnp.dtype
        Dtype for arithmetic in kernels (matmuls, stencils, tendencies).
    accumulate : jnp.dtype
        Dtype for reductions (global sums, budget integrals, norms).
        Must be >= compute width to avoid catastrophic cancellation.
    control : jnp.dtype
        Dtype for solvers, implicit systems, and diagnostics that
        require tight tolerances (semi-implicit, barotropic, EOS).
    """
    storage: jnp.dtype = jnp.float32
    compute: jnp.dtype = jnp.float32
    accumulate: jnp.dtype = jnp.float32
    control: jnp.dtype = jnp.float32

    @staticmethod
    def fp32() -> PrecisionPolicy:
        """All-float32 mode. Fastest, GPU-optimal."""
        return PrecisionPolicy(
            storage=jnp.float32,
            compute=jnp.float32,
            accumulate=jnp.float32,
            control=jnp.float32,
        )

    @staticmethod
    def fp64() -> PrecisionPolicy:
        """All-float64 mode. Full scientific reference."""
        return PrecisionPolicy(
            storage=jnp.float64,
            compute=jnp.float64,
            accumulate=jnp.float64,
            control=jnp.float64,
        )

    @staticmethod
    def mixed() -> PrecisionPolicy:
        """Mixed-precision mode.

        Storage and compute in float32 for GPU bandwidth.
        Accumulation and control in float64 for stability.
        """
        return PrecisionPolicy(
            storage=jnp.float32,
            compute=jnp.float32,
            accumulate=jnp.float64,
            control=jnp.float64,
        )


# Module-level recommended dtypes: maps (module_name, role) -> role override.
# When a module has a specific precision requirement, it overrides the global
# policy for that role. None means "use the global policy default".
_MODULE_OVERRIDES: dict[str, dict[str, str | None]] = {}

# The active global policy.
_ACTIVE_POLICY: list[PrecisionPolicy] = [PrecisionPolicy.fp32()]


# ---------------------------------------------------------------------------
# Policy management
# ---------------------------------------------------------------------------

def set_policy(policy: PrecisionPolicy) -> None:
    """Set the active global precision policy.

    Also enables JAX x64 if any dtype is float64.
    """
    _ACTIVE_POLICY[0] = policy
    if jnp.float64 in (policy.storage, policy.compute,
                        policy.accumulate, policy.control):
        if not jax.config.jax_enable_x64:
            jax.config.update("jax_enable_x64", True)


def get_policy() -> PrecisionPolicy:
    """Return the active global precision policy."""
    return _ACTIVE_POLICY[0]


def set_module_override(module: str, **role_overrides: str) -> None:
    """Override precision roles for a specific module.

    Parameters
    ----------
    module : str
        Module name (e.g., "pressure_gradient", "barotropic_solver").
    **role_overrides
        Role name -> precision mode string. Valid roles: storage, compute,
        accumulate, control. Valid values: "fp32", "fp64", or None to
        clear the override.

    Examples
    --------
    >>> set_module_override("pressure_gradient", compute="fp64", control="fp64")
    >>> set_module_override("tracer_advection", compute="fp32")
    """
    valid_roles = {"storage", "compute", "accumulate", "control"}
    for role in role_overrides:
        if role not in valid_roles:
            raise ValueError(
                f"Unknown role {role!r}. Valid: {sorted(valid_roles)}"
            )

    if module not in _MODULE_OVERRIDES:
        _MODULE_OVERRIDES[module] = {}

    from legoesm.core.hardware import _parse_precision_dtype
    for role, value in role_overrides.items():
        if value is None:
            _MODULE_OVERRIDES[module].pop(role, None)
        else:
            _MODULE_OVERRIDES[module][role] = _parse_precision_dtype(
                value, field_name=f"{module}.{role}",
            )


def clear_module_overrides() -> None:
    """Remove all per-module precision overrides."""
    _MODULE_OVERRIDES.clear()


def get_module_overrides() -> dict[str, dict[str, str | None]]:
    """Return current per-module overrides (copy)."""
    return {k: dict(v) for k, v in _MODULE_OVERRIDES.items()}


def _resolve_dtype(module: str | None, role: str) -> jnp.dtype:
    """Resolve the effective dtype for a (module, role) pair.

    Priority: module override > global policy > fallback to float32.
    """
    # Check module-level override first.
    if module is not None and module in _MODULE_OVERRIDES:
        overrides = _MODULE_OVERRIDES[module]
        if role in overrides:
            return overrides[role]

    # Fall back to global policy.
    policy = get_policy()
    return getattr(policy, role, jnp.float32)


# ---------------------------------------------------------------------------
# Casting helpers — the main API for kernels
# ---------------------------------------------------------------------------

def cast(x: jax.Array, module: str | None, role: str) -> jax.Array:
    """Cast array to the effective dtype for (module, role).

    This is the primary entry point for precision-aware kernels.
    No-op if the array is already in the target dtype.

    Parameters
    ----------
    x : jax.Array
        Input array.
    module : str or None
        Module name for override lookup (e.g., "tracer_advection").
        None uses the global policy only.
    role : str
        Precision role: "storage", "compute", "accumulate", or "control".

    Returns
    -------
    jax.Array
        Array cast to the resolved dtype. Same object if no cast needed.
    """
    target = _resolve_dtype(module, role)
    if x.dtype == target:
        return x
    return x.astype(target)


def const(value: float, module: str | None, role: str) -> jax.Array:
    """Create a scalar constant in the effective dtype for (module, role).

    Parameters
    ----------
    value : float
        Scalar value.
    module : str or None
        Module name.
    role : str
        Precision role.

    Returns
    -------
    jax.Array
        Scalar array in the resolved dtype.
    """
    target = _resolve_dtype(module, role)
    return jnp.array(value, dtype=target)


def cast_pytree(pytree, module: str | None, role: str):
    """Cast all float arrays in a pytree to the effective dtype.

    Non-float leaves (int, bool) and non-array leaves are left unchanged.
    """
    target = _resolve_dtype(module, role)

    def _maybe_cast(leaf):
        if isinstance(leaf, jax.Array) and jnp.issubdtype(leaf.dtype, jnp.floating):
            return leaf.astype(target) if leaf.dtype != target else leaf
        return leaf

    return jax.tree.map(_maybe_cast, pytree)


# ---------------------------------------------------------------------------
# Reduction wrappers — accumulation-precision aware
# ---------------------------------------------------------------------------

def global_sum(x: jax.Array, module: str | None = None) -> jax.Array:
    """Sum with accumulation precision.

    Upcasts to accumulation dtype, sums, then returns in that dtype.
    """
    acc_dtype = _resolve_dtype(module, "accumulate")
    return jnp.sum(x.astype(acc_dtype))


def global_max(x: jax.Array) -> jax.Array:
    """Global maximum (precision-independent, returns in input dtype)."""
    return jnp.max(x)


def norm(x: jax.Array, module: str | None = None, ord: int = 2) -> jax.Array:
    """Vector norm with accumulation precision.

    Parameters
    ----------
    x : jax.Array
        Input array.
    module : str or None
        Module for accumulation dtype resolution.
    ord : int
        Norm order (1 = L1, 2 = L2/RMS, inf = Linf).

    Returns
    -------
    jax.Array
        Scalar norm value in accumulation dtype.
    """
    acc_dtype = _resolve_dtype(module, "accumulate")
    x_acc = x.astype(acc_dtype)
    if ord == 1:
        return jnp.sum(jnp.abs(x_acc))
    elif ord == 2:
        return jnp.sqrt(jnp.sum(x_acc ** 2))
    else:
        return jnp.max(jnp.abs(x_acc))


def weighted_mean(
    x: jax.Array,
    weights: jax.Array,
    module: str | None = None,
) -> jax.Array:
    """Area-weighted mean with accumulation precision."""
    acc_dtype = _resolve_dtype(module, "accumulate")
    x_acc = x.astype(acc_dtype)
    w_acc = weights.astype(acc_dtype)
    return jnp.sum(x_acc * w_acc) / jnp.sum(w_acc)


def compensated_sum(x: jax.Array, axis: int = 0) -> jax.Array:
    """Kahan compensated summation via jax.lax.scan.

    Achieves near-float64 accuracy for float32 reductions by tracking
    a running compensation term. Useful for budget integrals.

    Parameters
    ----------
    x : jax.Array
        Array to sum along *axis*.
    axis : int
        Axis to reduce.

    Returns
    -------
    jax.Array
        Compensated sum along the specified axis.
    """
    x = jnp.moveaxis(x, axis, 0)
    n = x.shape[0]
    rest_shape = x.shape[1:]

    def _step(carry, xi):
        s, c = carry
        y = xi - c
        t = s + y
        c = (t - s) - y
        return (t, c), None

    init = (jnp.zeros(rest_shape, dtype=x.dtype),
            jnp.zeros(rest_shape, dtype=x.dtype))
    (result, _), _ = jax.lax.scan(_step, init, x)
    return result


# ---------------------------------------------------------------------------
# Precision-aware kernel decorators
# ---------------------------------------------------------------------------

def with_precision(module: str):
    """Decorator that casts inputs to compute dtype and output to storage dtype.

    Usage
    -----
    @with_precision("tracer_advection")
    def tracer_advection(T, u, v, dx, dt):
        # All inputs arrive in compute dtype
        dT = -u * grad_x(T, dx) - v * grad_y(T, dx)
        return T + dt * dT
        # Output is automatically cast to storage dtype
    """
    def decorator(fn):
        def wrapper(*args, **kwargs):
            # Cast all array args to compute dtype.
            compute_dtype = _resolve_dtype(module, "compute")
            storage_dtype = _resolve_dtype(module, "storage")

            def _to_compute(leaf):
                if isinstance(leaf, jax.Array) and jnp.issubdtype(
                    leaf.dtype, jnp.floating
                ):
                    return leaf.astype(compute_dtype)
                return leaf

            cast_args = jax.tree.map(_to_compute, args)
            cast_kwargs = jax.tree.map(_to_compute, kwargs)

            result = fn(*cast_args, **cast_kwargs)

            # Cast output back to storage dtype.
            def _to_storage(leaf):
                if isinstance(leaf, jax.Array) and jnp.issubdtype(
                    leaf.dtype, jnp.floating
                ):
                    return leaf.astype(storage_dtype)
                return leaf

            return jax.tree.map(_to_storage, result)

        wrapper.__name__ = fn.__name__
        wrapper.__doc__ = fn.__doc__
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Recommended per-module precision strategies
# ---------------------------------------------------------------------------

# These are applied via set_recommended_overrides() — they encode domain
# knowledge about which ESM kernels are precision-sensitive.

_OCEAN_OVERRIDES = {
    # Barotropic solver: iterative, very sensitive to rounding in SSH
    "barotropic_solver": {"compute": jnp.float64, "control": jnp.float64},
    # Pressure gradient: hydrostatic cancellation is catastrophic in fp32
    "pressure_gradient": {"compute": jnp.float64},
    # EOS: density differences O(0.01 kg/m³) require > 7 sig figs
    "equation_of_state": {"compute": jnp.float64},
    # Coriolis: f*u difference sensitive at high latitudes
    "coriolis": {"compute": jnp.float64},
    # Tracer advection: safe in fp32 with flux limiters
    "tracer_advection": {},
    # Tracer diffusion: safe in fp32
    "tracer_diffusion": {},
    # Vertical mixing: safe in fp32
    "vertical_mixing": {},
    # Time stepping: accumulation in fp64 for long runs
    "ocean_timestepping": {"accumulate": jnp.float64},
    # Global budgets: always fp64
    "ocean_diagnostics": {"accumulate": jnp.float64, "control": jnp.float64},
}

_ATMOSPHERE_OVERRIDES = {
    # Spectral transforms: require fp64 (complex128 FFTs)
    "spectral_transform": {
        "storage": jnp.float64, "compute": jnp.float64,
        "accumulate": jnp.float64, "control": jnp.float64,
    },
    # Pressure gradient: hydrostatic balance subtraction
    "atm_pressure_gradient": {"compute": jnp.float64},
    # Semi-implicit solver: matrix inversion needs fp64
    "semi_implicit": {"compute": jnp.float64, "control": jnp.float64},
    # Gravity wave propagation: vertical eigenvalue problem
    "gravity_wave": {"control": jnp.float64},
    # Radiation: safe in fp32 (optical depths are O(1))
    "radiation": {},
    # Convection: safe in fp32
    "convection": {},
    # Turbulence: safe in fp32
    "turbulence": {},
    # Microphysics: safe in fp32
    "microphysics": {},
    # Tracer transport: safe in fp32 with limiters
    "atm_tracer_transport": {},
}

_LAND_OVERRIDES = {
    # Soil moisture: Richards equation — safe in fp32
    "soil_moisture": {},
    # Soil thermal: diffusion — safe in fp32
    "soil_thermal": {},
    # Carbon pools: century timescale, small fluxes → fp64 accumulation
    "carbon_pools": {"accumulate": jnp.float64},
    # Runoff: safe in fp32
    "runoff": {},
    # ET: safe in fp32
    "evapotranspiration": {},
}

_ICE_OVERRIDES = {
    # EVP stress solver: subcycled, sensitive to convergence
    "evp_solver": {"compute": jnp.float64, "control": jnp.float64},
    # Ice thickness: conservation-critical, small increments
    "ice_thickness": {"accumulate": jnp.float64},
    # Energy balance: safe in fp32
    "ice_energy": {},
    # Mass balance: fp64 accumulation for century runs
    "ice_mass_balance": {"accumulate": jnp.float64},
}


def set_recommended_overrides(mode: str = "mixed") -> None:
    """Apply recommended per-module overrides for a precision mode.

    Parameters
    ----------
    mode : str
        "fp32" — no overrides (everything fp32).
        "fp64" — no overrides needed (global policy handles it).
        "mixed" — apply domain-knowledge overrides for sensitive kernels.
    """
    clear_module_overrides()

    if mode == "fp32":
        set_policy(PrecisionPolicy.fp32())
        return
    elif mode == "fp64":
        set_policy(PrecisionPolicy.fp64())
        return
    elif mode != "mixed":
        raise ValueError(f"Unknown mode {mode!r}. Use 'fp32', 'fp64', or 'mixed'.")

    set_policy(PrecisionPolicy.mixed())

    # Apply all domain-specific overrides.
    for overrides_dict in [
        _OCEAN_OVERRIDES,
        _ATMOSPHERE_OVERRIDES,
        _LAND_OVERRIDES,
        _ICE_OVERRIDES,
    ]:
        for module_name, roles in overrides_dict.items():
            if roles:
                _MODULE_OVERRIDES[module_name] = dict(roles)


# ---------------------------------------------------------------------------
# Sync with legacy hardware.py policy
# ---------------------------------------------------------------------------

def sync_from_hardware() -> PrecisionPolicy:
    """Create a PrecisionPolicy from the existing hardware.py config.

    This bridges the legacy 3-component policy (dynamics, ml, conservation)
    to the new 4-role policy.
    """
    from legoesm.core.hardware import get_runtime_precision_policy

    legacy = get_runtime_precision_policy()
    dynamics_dtype = legacy.get("dynamics", jnp.float32)
    conservation_dtype = legacy.get("conservation")

    if dynamics_dtype == jnp.float64:
        policy = PrecisionPolicy.fp64()
    elif conservation_dtype == jnp.float64:
        policy = PrecisionPolicy.mixed()
    else:
        policy = PrecisionPolicy.fp32()

    set_policy(policy)
    return policy


def sync_to_hardware() -> None:
    """Push the active PrecisionPolicy back to hardware.py's legacy config."""
    from legoesm.core.hardware import set_runtime_precision_policy

    policy = get_policy()
    # Map: compute → dynamics, accumulate → conservation
    set_runtime_precision_policy(
        dynamics=policy.compute,
        conservation=policy.accumulate,
    )
