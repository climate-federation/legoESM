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

from typing import NamedTuple

import jax
import jax.numpy as jnp

# Deferred import to break cycle: runtime/__init__.py imports
# runtime.precision, which re-exports from this module — so
# importing runtime.backend at module load time would mid-init
# this very file. Both helpers are only called from function
# bodies, so a lazy import is safe.
def _backend():
    from legoesm.runtime import backend as _b
    return _b


# ---------------------------------------------------------------------------
# Dtype parsing helper
# ---------------------------------------------------------------------------

_PRECISION_NAME_TO_DTYPE = {
    "float16": jnp.float16, "fp16": jnp.float16, "half": jnp.float16,
    "bfloat16": jnp.bfloat16, "bf16": jnp.bfloat16,
    "float32": jnp.float32, "fp32": jnp.float32, "single": jnp.float32,
    "float64": jnp.float64, "fp64": jnp.float64, "double": jnp.float64,
}


def parse_dtype(value, *, field_name: str, allow_none: bool = False):
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
    raise ValueError(
        f"Unknown {field_name} precision {value!r}. "
        f"Use one of {sorted(_PRECISION_NAME_TO_DTYPE.keys())}."
    )


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

    @staticmethod
    def mixed_fp64_storage() -> PrecisionPolicy:
        """Mixed mode with float64 storage, float32 compute.

        State arrays are stored in float64 for maximum precision in
        I/O, checkpointing, and long-run accumulation.  Compute kernels
        run in float32 for GPU throughput.  Accumulation and control
        remain float64.
        """
        return PrecisionPolicy(
            storage=jnp.float64,
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

    Also enables JAX x64 if any dtype is float64.  On backends that
    lack float64 hardware (e.g. Apple Metal), x64 is still enabled
    because spectral solvers route computation to CPU and need float64
    there.  ``resolve_dtype`` handles clamping float64 → float32 for
    non-spectral code that runs on the default (Metal) device.
    """
    _ACTIVE_POLICY[0] = policy
    if jnp.float64 in (policy.storage, policy.compute,
                        policy.accumulate, policy.control):
        if not jax.config.jax_enable_x64:
            jax.config.update("jax_enable_x64", True)


def get_policy() -> PrecisionPolicy:
    """Return the active global precision policy."""
    return _ACTIVE_POLICY[0]


def validate_policy(policy: PrecisionPolicy | None = None) -> None:
    """Verify the active precision policy is actually achievable.

    On backends that lack float64 (e.g. Metal), float64 requests in the
    policy are silently clamped to float32 by ``resolve_dtype``, so the
    policy is always achievable — this function is a no-op in that case.

    Raises
    ------
    RuntimeError
        If the policy requires float64, the backend supports it, but
        JAX x64 mode is not enabled.
    """
    if policy is None:
        policy = get_policy()
    needs_x64 = jnp.float64 in (
        policy.storage, policy.compute, policy.accumulate, policy.control,
    )
    if not needs_x64:
        return
    if not _backend().supports_float64():
        # Backend cannot do float64; resolve_dtype will clamp to float32.
        return
    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            "Precision policy requires float64 but JAX x64 mode is not enabled. "
            "Set JAX_ENABLE_X64=1 or call jax.config.update('jax_enable_x64', True) "
            "before bootstrapping."
        )


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

    for role, value in role_overrides.items():
        if value is None:
            _MODULE_OVERRIDES[module].pop(role, None)
        else:
            _MODULE_OVERRIDES[module][role] = parse_dtype(
                value, field_name=f"{module}.{role}",
            )


def clear_module_overrides() -> None:
    """Remove all per-module precision overrides."""
    _MODULE_OVERRIDES.clear()


def get_module_overrides() -> dict[str, dict[str, str | None]]:
    """Return current per-module overrides (copy)."""
    return {k: dict(v) for k, v in _MODULE_OVERRIDES.items()}


def _clamp_to_backend(dtype: jnp.dtype) -> jnp.dtype:
    """Clamp *dtype* to what the current backend actually supports.

    On backends that lack float64 (e.g. Apple Metal) or when JAX x64
    mode is disabled, float64 is silently downgraded to float32 so that
    the precision policy never requests an impossible dtype.
    """
    if dtype == jnp.float64:
        b = _backend()
        if not (b.supports_float64() and b.is_x64_enabled()):
            return jnp.float32
    return dtype


def resolve_dtype(module: str | None, role: str) -> jnp.dtype:
    """Resolve the effective dtype for a (module, role) pair.

    Priority: module override > global policy > fallback to float32.
    The result is clamped to what the backend supports (float64 is
    downgraded to float32 on Metal or when x64 is disabled).
    """
    # Check module-level override first.
    if module is not None and module in _MODULE_OVERRIDES:
        overrides = _MODULE_OVERRIDES[module]
        if role in overrides:
            return _clamp_to_backend(overrides[role])

    # Fall back to global policy.
    policy = get_policy()
    return _clamp_to_backend(getattr(policy, role, jnp.float32))


# ---------------------------------------------------------------------------
# Casting helpers — the main API for kernels
# ---------------------------------------------------------------------------

def cast(x: jax.Array, module: str | None, role: str, *,
         allow_downcast: bool = False) -> jax.Array:
    """Cast array to the effective dtype for (module, role).

    This is the primary entry point for precision-aware kernels.
    No-op if the array is already in the target dtype.

    By default, only upcasts and no-ops are performed.  Downcasts are
    skipped unless *allow_downcast* is True.  This prevents silent
    precision loss when the default fp32 policy is active but arrays
    were created in float64 (e.g. under JAX_ENABLE_X64).

    Parameters
    ----------
    x : jax.Array
        Input array.
    module : str or None
        Module name for override lookup (e.g., "tracer_advection").
        None uses the global policy only.
    role : str
        Precision role: "storage", "compute", "accumulate", or "control".
    allow_downcast : bool
        If False (default), skip casts that would reduce precision.

    Returns
    -------
    jax.Array
        Array cast to the resolved dtype. Same object if no cast needed.
    """
    target = resolve_dtype(module, role)
    if x.dtype == target:
        return x
    if allow_downcast or jnp.dtype(x.dtype).itemsize <= jnp.dtype(target).itemsize:
        return x.astype(target)
    return x  # skip downcast


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
    target = resolve_dtype(module, role)
    return jnp.array(value, dtype=target)


def cast_pytree(pytree, module: str | None, role: str, *,
                allow_downcast: bool = False):
    """Cast all float arrays in a pytree to the effective dtype.

    Non-float leaves (int, bool) and non-array leaves are left unchanged.

    By default, only upcasts (e.g. float32 -> float64) and no-ops are
    performed.  Downcasts are skipped unless *allow_downcast* is True.
    This prevents silent precision loss when the default fp32 policy is
    active but arrays were created in float64 (e.g. under JAX_ENABLE_X64).
    """
    target = resolve_dtype(module, role)
    target_size = jnp.dtype(target).itemsize

    def _maybe_cast(leaf):
        if isinstance(leaf, jax.Array) and jnp.issubdtype(leaf.dtype, jnp.floating):
            if leaf.dtype == target:
                return leaf
            if allow_downcast or jnp.dtype(leaf.dtype).itemsize <= target_size:
                return leaf.astype(target)
            return leaf  # skip downcast
        return leaf

    return jax.tree.map(_maybe_cast, pytree)


# ---------------------------------------------------------------------------
# Step-boundary storage re-cast (#1675)
# ---------------------------------------------------------------------------

#: Prognostic leaves that are DELIBERATELY carried at the accumulate dtype and
#: must survive :func:`finalize_to_storage`.  Surface pressure is the
#: conservation field: the mass fixer adds an exact accumulate-dtype correction
#: to it and rounding that back to storage would destroy the machine-precision
#: fixing (see ``conservation.conservation_accumulator``).  Owner decision on
#: #1675: ``p_s`` stays float64 everywhere in ``mixed``; the memory saving comes
#: from the 3-D bulk state, not from one 2-D field.
ACCUMULATE_ROLE_LEAVES: tuple[str, ...] = ("p_s",)


def finalize_to_storage(state, *, accumulate_leaves=ACCUMULATE_ROLE_LEAVES):
    """Re-cast a state pytree to the STORAGE dtype at a step boundary.

    Why this exists (#1675).  In ``mixed`` the mass fixer's correction is
    computed at the accumulate dtype (float64) and added to ``p_s`` without
    rounding back — deliberately, because that exactness is what delivers the
    machine-precision mass fixing.  But the promoted ``p_s`` then flows into
    the tracer mass-conservation rescale (``q * dp_pre/dp_post``), so the
    float64 leaks into the 3-D bulk state and the run silently stops being
    fp32-storage.  Measured on the lat-lon PE eager step: all three tracers
    come back float64 after one step.  The compiled/production carry already
    had this re-cast (``compiled_segments._match_dtype``); the eager step did
    not, which is why the leaf-dtype gate could only be run in ``fp64``.

    Leaves named in *accumulate_leaves* are cast to the ACCUMULATE dtype
    instead; every other float leaf is cast to STORAGE.  Non-float and
    non-array leaves are untouched.

    NO-OP UNLESS THE POLICY SPLITS THE TWO ROLES.  When ``storage ==
    accumulate`` — which is every mode except ``mixed`` — this returns the
    pytree unchanged, so ``fp64`` and strict ``fp32`` are byte-identical and,
    importantly, the ``fp32``-policy-with-x64 path keeps its intentional
    float64 ``p_s`` (rounding it back there would regress that path's mass
    fixing from ~1e-12 to the fp32 ceiling — the review finding recorded on
    #1675).
    """
    storage = resolve_dtype(None, "storage")
    accumulate = resolve_dtype(None, "accumulate")
    if jnp.dtype(storage) == jnp.dtype(accumulate):
        return state

    names = frozenset(accumulate_leaves)

    def _components(path):
        """Every path component's own name -- EXACT, never a substring.

        Substring matching on ``keystr`` would sweep in every leaf whose path
        merely contains the word (``h`` would match ``h_s``, ``phis``, and any
        dict key with an h in it), which is how an exclusion list quietly
        becomes an exclusion of everything.  Matching ANY component rather
        than only the last one is what makes a ``Field``-wrapped state work:
        its leaf sits at ``.p_s.data``, so the last component is ``data``.
        """
        out = []
        for entry in path:
            for attr in ("name", "key", "idx"):
                if hasattr(entry, attr):
                    out.append(str(getattr(entry, attr)))
                    break
            else:
                out.append(str(entry))
        return out

    def _cast(path, leaf):
        if not (isinstance(leaf, jax.Array)
                and jnp.issubdtype(leaf.dtype, jnp.floating)):
            return leaf
        keep = any(c in names for c in _components(path))
        target = accumulate if keep else storage
        return leaf if leaf.dtype == target else leaf.astype(target)

    return jax.tree_util.tree_map_with_path(_cast, state)


# ---------------------------------------------------------------------------
# Reduction wrappers — accumulation-precision aware
# ---------------------------------------------------------------------------

def global_sum(x: jax.Array, module: str | None = None) -> jax.Array:
    """Sum with accumulation precision.

    Upcasts to accumulation dtype, sums, then returns in that dtype.
    """
    acc_dtype = resolve_dtype(module, "accumulate")
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
    acc_dtype = resolve_dtype(module, "accumulate")
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
    acc_dtype = resolve_dtype(module, "accumulate")
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
            compute_dtype = resolve_dtype(module, "compute")
            storage_dtype = resolve_dtype(module, "storage")

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
# knowledge about which ESM kernels are precision-sensitive.  Every non-empty
# entry must name a module some kernel actually passes to this API (gated by
# tests/unit/test_precision.py); an unconsumed name promises fp64 it never
# delivers.

_OCEAN_OVERRIDES = {
    # Barotropic solver: iterative, very sensitive to rounding in SSH
    "barotropic_solver": {"compute": jnp.float64, "control": jnp.float64},
    # EOS: density differences O(0.01 kg/m³) require > 7 sig figs
    "equation_of_state": {"compute": jnp.float64},
    # Tracer advection: safe in fp32 with flux limiters
    "tracer_advection": {},
    # Tracer diffusion: safe in fp32
    "tracer_diffusion": {},
    # Vertical mixing: safe in fp32
    "vertical_mixing": {},
    # Global budgets: always fp64
    "ocean_diagnostics": {"accumulate": jnp.float64, "control": jnp.float64},
}

_ATMOSPHERE_OVERRIDES = {
    # Pressure gradient: hydrostatic balance subtraction
    "atm_pressure_gradient": {"compute": jnp.float64},
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
    # Runoff: safe in fp32
    "runoff": {},
    # ET: safe in fp32
    "evapotranspiration": {},
}

_ICE_OVERRIDES = {
    # Energy balance: safe in fp32
    "ice_energy": {},
}


def set_recommended_overrides(mode: str = "mixed") -> None:
    """Apply recommended per-module overrides for a precision mode.

    This function only manages module overrides — it does NOT reset the
    global policy.  The caller (``apply_precision``) is responsible for
    setting the policy before calling this.

    Parameters
    ----------
    mode : str
        "fp32" — clear overrides (global policy handles it).
        "fp64" — clear overrides (global policy handles it).
        "mixed" — apply domain-knowledge overrides for sensitive kernels.
    """
    clear_module_overrides()

    if mode in ("fp32", "fp64"):
        return
    elif mode != "mixed":
        raise ValueError(f"Unknown mode {mode!r}. Use 'fp32', 'fp64', or 'mixed'.")

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
