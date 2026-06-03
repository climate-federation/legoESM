"""Conservation fixers for legoESM.

These enforce hard constraints on conserved quantities (mass, energy, momentum)
after each timestep. The fixers use uniform additive corrections to preserve
gradients for automatic differentiation.

References
----------
- Sha et al. (2025): Global mass and energy conservation schemes for AI weather models.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

def _tiny(x=None):
    """Smallest normal float for the given array's dtype (or active accumulate dtype)."""
    if x is not None and hasattr(x, 'dtype'):
        return float(jnp.finfo(x.dtype).tiny)
    return float(jnp.finfo(resolve_dtype(None, "accumulate")).tiny)
# Epsilon for energy fixers: prevents sqrt(0) which has infinite gradient,
# causing 0*Inf=NaN in the backward pass when jnp.maximum clamps KE_target to 0.
_EPS_ENERGY = 1e-20

from legoesm import constants
from legoesm.core.operators import global_integral, _is_distributed
from legoesm.core.operators_voronoi import kinetic_energy_cell
from legoesm.core.precision import resolve_dtype, get_policy
from legoesm.core.state import ShallowWaterState, HydrostaticState
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.grids.vertical import compute_geopotential

# legoesm.parallel.reductions (batch_allreduce_mpi, global_sum_mpi) and
# legoesm.runtime.backend (is_x64_enabled, supports_float64) are imported at
# function scope below: core/ must not import parallel/ or runtime/ at module
# top level (re-enters their __init__ mid-load, breaks isolated pytest; CLAUDE.md).


def _accumulation_dtype():
    """Return the dtype for accumulation in conservation fixers.

    Uses the global precision policy's ``accumulate`` role when available,
    clamped to what the backend actually supports. On backends that lack
    float64 (e.g. Apple Metal) or when JAX x64 mode is disabled, the
    result is clamped to float32 even if the policy requests float64.
    """
    from legoesm.runtime.backend import is_x64_enabled, supports_float64
    target = get_policy().accumulate
    if target == jnp.float64 and not (supports_float64() and is_x64_enabled()):
        return jnp.float32
    return target


def conservation_accumulator():
    """Accumulator dtype for *budget* sums (mass, energy, tracer).

    Distinct from :func:`_accumulation_dtype` — promotes to ``float64``
    whenever JAX has x64 enabled, regardless of the active precision
    policy.  Mass/energy budgets involve subtracting two near-equal
    extensive quantities (e.g. ``mass_old - mass_new``), so even when
    storage/compute are intentionally float32 we want the budget sum
    to use the highest precision JAX is willing to give us.  This
    restores end-step :func:`fix_mass_hydrostatic` to ~machine
    precision and lets the "skip per-stage ``zero_mean_tendency`` when
    end-step fixer is on" scaling optimisation be lossless even in
    float32 storage/compute mode.
    """
    if jax.config.read("jax_enable_x64"):
        return jnp.float64
    return _accumulation_dtype()


def _global_area_sum(
    array: jax.Array,
    grid,
    owned_mask: jax.Array | None = None,
) -> jax.Array:
    """Area-weighted global sum of a raw array, distributed-aware.

    Works on any grid with a ``.area`` attribute (CubedSphereGrid,
    LatLonGrid, etc.).

    Parameters
    ----------
    array : jax.Array
        The field to integrate (e.g. surface pressure).
    grid : grid object
        Must have ``.area`` attribute.
    owned_mask : jax.Array, optional
        Shape ``(n_faces,)`` boolean/float mask indicating which faces
        this rank owns.  Required for **replicated-dynamics MPI** where
        each rank holds full ``(6, n, n)`` data but only owned faces
        are authoritative.  Non-owned faces are zeroed before local
        summation; ``global_sum_mpi`` then combines owned portions.
        If ``None``, all faces are summed (single-rank or SPMD).

    Execution modes:

    - **Single device**: plain ``jnp.sum``.
    - **Multi-device SPMD** (NamedSharding): ``jnp.sum`` on a
      face-sharded array already produces the correct global sum --
      JAX/XLA automatically inserts an all-reduce when the reduction
      spans a sharded axis.  No explicit ``psum`` is needed.
    - **MPI distributed** (replicated dynamics): mask to owned faces,
      local sum, then ``allreduce(SUM)``.
    """
    acc = conservation_accumulator()
    prod = array.astype(acc) * grid.area.astype(acc)
    if owned_mask is not None:
        # Broadcast (n_faces,) → match prod shape: (6,) → (6,1,1,...)
        mask = owned_mask.astype(acc)
        while mask.ndim < prod.ndim:
            mask = mask[..., None]
        prod = prod * mask
    local_sum = jnp.sum(prod)
    if _is_distributed():
        from legoesm.parallel.reductions import global_sum_mpi
        return global_sum_mpi(local_sum)
    return local_sum


def _batch_global_area_sums(
    arrays: list[jax.Array],
    grid,
    owned_mask: jax.Array | None = None,
) -> list[jax.Array]:
    """Compute multiple area-weighted global sums in a single MPI call.

    Same semantics as calling :func:`_global_area_sum` on each array
    individually, but batches all reductions into one ``allreduce``
    when running under MPI, reducing latency from O(N) to O(1).

    Falls back to individual ``jnp.sum`` when not distributed.
    """
    acc = conservation_accumulator()
    area_acc = grid.area.astype(acc)
    weight = area_acc
    if owned_mask is not None:
        mask = owned_mask.astype(acc)
        while mask.ndim < area_acc.ndim:
            mask = mask[..., None]
        weight = area_acc * mask

    # All inputs share the same horizontal axes and weight ``weight``;
    # stack them along a new trailing axis and reduce once locally so
    # XLA fuses the N independent sum kernels into one.
    stacked = jnp.stack([arr.astype(acc) for arr in arrays], axis=-1)
    summed = jnp.sum(
        stacked * weight[..., None], axis=tuple(range(area_acc.ndim)),
    )
    local_sums = [summed[..., i] for i in range(len(arrays))]

    if _is_distributed():
        from legoesm.parallel.reductions import batch_allreduce_mpi
        return batch_allreduce_mpi(local_sums, op="sum")
    return local_sums


def _total_area(grid) -> jax.Array:
    """Total area for any grid, promoted to the fp64 budget accumulator.

    iter-13: ``grid.grid_total_area`` is computed as
    ``jnp.sum(self.area)`` on grid classes that store ``area`` in the
    storage dtype (fp32 by default).  That reduction leaks ~N·eps into
    the divisor of every mass-fixer correction — visible as a residual
    ~10^-13 drift on the cube hydro PE even with the anchored
    fixer + fp64 ``mass_target``.  Cast to the conservation
    accumulator here so every fixer division sees an fp64 denominator.
    """
    acc = conservation_accumulator()
    return grid.grid_total_area.astype(acc)


def fix_mass_shallow_water(
    state_new: ShallowWaterState,
    state_old: ShallowWaterState,
    grid,
) -> ShallowWaterState:
    """Fix mass conservation for shallow water equations (any grid).

    Applies a uniform additive correction to the fluid depth h
    so that the global integral of h is preserved exactly.

    The uniform correction preserves gradients of h (important
    for differentiability) while enforcing exact conservation.

    Parameters
    ----------
    state_new : ShallowWaterState
        State after time integration (may not conserve mass exactly).
    state_old : ShallowWaterState
        State before time integration (reference mass).
    grid : CubedSphereGrid or LatLonGrid
        The grid (any grid with .area and .total_area).

    Returns
    -------
    ShallowWaterState : Mass-conserving state.
    """
    mass_old, mass_new = _batch_global_area_sums(
        [state_old.h.data, state_new.h.data], grid,
    )
    correction = (mass_old - mass_new) / _total_area(grid)
    h_fixed = state_new.h.replace(data=state_new.h.data + correction)

    return state_new._replace(h=h_fixed)


def fix_energy_shallow_water(
    state_new: ShallowWaterState,
    state_old: ShallowWaterState,
    grid,
    g: float = constants.g,
) -> ShallowWaterState:
    """Fix total energy conservation for shallow water equations (any grid).

    Total energy = kinetic + potential:
        E = 0.5 * h * (u^2 + v^2) + 0.5 * g * (h + h_s)^2

    Applies a uniform scaling to velocities to restore total energy.

    Parameters
    ----------
    state_new : ShallowWaterState
        State after time integration.
    state_old : ShallowWaterState
        State before time integration.
    grid : CubedSphereGrid or LatLonGrid
        The grid (any grid with .area).
    g : float
        Gravitational acceleration.

    Returns
    -------
    ShallowWaterState : Energy-conserving state.
    """
    # Compute all three energy integrals as local sums, then batch
    # into a single MPI allreduce (3 separate allreduces -> 1).
    # iter-42: promote the per-cell energy *field* computation to the
    # fp64 budget accumulator before the area-weighted sum.  The pre-
    # iter-42 path computed ``0.5 * h * (u² + v²)`` in fp32 (input
    # storage dtype), so the square+multiply lost ~7 bits of precision
    # before ``_batch_global_area_sums`` ever cast to fp64 — the same
    # fp32-field bug iter-1/4/5 fixed for the mass diagnostic, just on
    # the energy path.
    acc = conservation_accumulator()
    h_old = state_old.h.data.astype(acc)
    u_old = state_old.u.data.astype(acc)
    v_old = state_old.v.data.astype(acc)
    h_s_old = state_old.h_s.data.astype(acc)
    g_acc = jnp.asarray(g, dtype=acc)
    E_old_field = 0.5 * h_old * (u_old**2 + v_old**2) + 0.5 * g_acc * (h_old + h_s_old)**2

    h_new = state_new.h.data.astype(acc)
    u_new = state_new.u.data.astype(acc)
    v_new = state_new.v.data.astype(acc)
    h_s_new = state_new.h_s.data.astype(acc)
    KE_new_field = 0.5 * h_new * (u_new**2 + v_new**2)
    PE_new_field = 0.5 * g_acc * (h_new + h_s_new)**2

    E_old, KE_new, PE_new = _batch_global_area_sums(
        [E_old_field, KE_new_field, PE_new_field], grid,
    )

    KE_target = E_old - PE_new
    KE_target = jnp.maximum(KE_target, _EPS_ENERGY)
    scale = jnp.where(KE_new > _tiny(KE_new), jnp.sqrt(KE_target / KE_new), 1.0)

    u_fixed = state_new.u.replace(data=u_new * scale)
    v_fixed = state_new.v.replace(data=v_new * scale)

    return state_new._replace(u=u_fixed, v=v_fixed)


def apply_conservation_fixer(
    state_new: ShallowWaterState,
    state_old: ShallowWaterState,
    grid,
    fix_mass: bool = True,
    fix_energy: bool = True,
    g: float = constants.g,
) -> ShallowWaterState:
    """Apply all conservation fixers in sequence (any grid).

    Order: mass first, then energy (energy fix adjusts velocities
    without changing mass).
    """
    if fix_mass:
        state_new = fix_mass_shallow_water(state_new, state_old, grid)
    if fix_energy:
        state_new = fix_energy_shallow_water(state_new, state_old, grid, g)
    return state_new


def fix_mass_hydrostatic(
    state_new: HydrostaticState,
    state_old: HydrostaticState,
    grid: CubedSphereGrid,
) -> HydrostaticState:
    """Fix mass conservation for the hydrostatic primitive equations.

    Applies a uniform additive correction to surface pressure p_s
    so that the global dry air mass is preserved exactly.

    Global dry air mass: M = (1/g) * ∫ p_s * dA

    Since g and dA are constant, we just need ∫ p_s * dA to be conserved.

    The uniform correction preserves gradients of p_s (important
    for differentiability) while enforcing exact conservation.

    Parameters
    ----------
    state_new : HydrostaticState
        State after time integration.
    state_old : HydrostaticState
        State before time integration (reference mass).
    grid : CubedSphereGrid
        The grid.

    Returns
    -------
    HydrostaticState : Mass-conserving state.
    """
    mass_old, mass_new = _batch_global_area_sums(
        [state_old.p_s.data, state_new.p_s.data], grid,
    )
    correction = (mass_old - mass_new) / _total_area(grid)
    p_s_fixed = state_new.p_s.replace(data=state_new.p_s.data + correction)

    return state_new._replace(p_s=p_s_fixed)


def zero_mean_tendency(
    tendency: jax.Array,
    grid,
) -> jax.Array:
    """Remove the area-weighted global mean from a tendency field (any grid).

    After this correction, ``sum(tendency * area) == 0`` to machine
    precision, enforcing exact conservation for explicit FV mass equations.

    Works for 2D, 3D, and 4D arrays. The area dimensions are inferred
    from ``grid.area.ndim``:
    - Cubed-sphere: area is (6,n,n), tendency is (6,n,n) or (6,n,n,nlev).
    - Lat-lon: area is (n_lat,n_lon), tendency is (n_lat,n_lon) or (n_lat,n_lon,nlev).

    For arrays with a trailing level axis, the correction is applied
    independently at each level.

    Parameters
    ----------
    tendency : jax.Array
        The mass/tracer tendency field.
    grid : CubedSphereGrid or LatLonGrid
        Grid with cell areas.

    Returns
    -------
    jax.Array : Corrected tendency with zero global integral.
    """
    acc = conservation_accumulator()
    area = grid.area
    area_acc = area.astype(acc)
    total_area_acc = jnp.sum(area_acc)
    orig_dtype = tendency.dtype
    area_ndim = area.ndim  # 3 for cubed-sphere, 2 for lat-lon

    if tendency.ndim == area_ndim:
        # 2D tendency (lat-lon) or 3D tendency (cubed-sphere) — no level axis
        global_sum = _global_area_sum(tendency, grid)
        correction = global_sum / total_area_acc
        return (tendency.astype(acc) - correction).astype(orig_dtype)
    elif tendency.ndim == area_ndim + 1:
        # Has a trailing level axis — per-level correction
        tend_acc = tendency.astype(acc)
        prod = tend_acc * area_acc[..., None]
        # Sum over all spatial axes (all except the last)
        spatial_axes = tuple(range(area_ndim))
        level_sums = jnp.sum(prod, axis=spatial_axes)  # (nlev,)
        if _is_distributed():
            from legoesm.parallel.reductions import global_sum_mpi
            level_sums = global_sum_mpi(level_sums)
        corrections = level_sums / total_area_acc  # (nlev,)
        # Broadcast corrections to match tendency shape
        for _ in range(area_ndim):
            corrections = jnp.expand_dims(corrections, 0)
        return (tend_acc - corrections).astype(orig_dtype)
    else:
        return tendency


# ==============================================================================
# Moisture conservation
# ==============================================================================

def compute_global_moisture(
    q_v: jax.Array,
    p_s: jax.Array,
    dsigma: jax.Array,
    grid,
    owned_mask: jax.Array | None = None,
) -> jax.Array:
    """Compute global column-integrated water vapor.

    Integral: (1/g) * ∫∫ q_v · p_s · dσ · dA

    Parameters
    ----------
    q_v : jax.Array, shape (..., nlev)
        Specific humidity [kg/kg].
    p_s : jax.Array, shape (...)
        Surface pressure [Pa].
    dsigma : jax.Array, shape (nlev,)
        Sigma layer thicknesses.
    grid : CubedSphereGrid or similar
        Grid with ``.area`` attribute.
    owned_mask : jax.Array, optional
        Shape ``(n_faces,)`` for MPI replicated dynamics.

    Returns
    -------
    jax.Array : Scalar global moisture integral [kg].
    """
    # Column water vapor: ∫ q_v dp/g = q_v * p_s * dsigma / g
    # iter-44: promote field computation to fp64 budget accumulator
    # (same fp32-field bug as iter-42/43 energy diagnostics).
    acc = conservation_accumulator()
    cwv = jnp.sum(
        q_v.astype(acc) * p_s.astype(acc)[..., None] * dsigma.astype(acc),
        axis=-1,
    ) / jnp.asarray(constants.g, dtype=acc)
    return _global_area_sum(cwv, grid, owned_mask=owned_mask)


def fix_moisture_hydrostatic(
    q_v: jax.Array,
    target_moisture: jax.Array,
    p_s: jax.Array,
    dsigma: jax.Array,
    grid,
    owned_mask: jax.Array | None = None,
) -> jax.Array:
    """Fix global moisture conservation via multiplicative scaling.

    Scales q_v uniformly so that the global column-integrated water vapor
    matches *target_moisture*.  Uses multiplicative (not additive) correction
    to preserve spatial gradients and guarantee non-negativity.

    Parameters
    ----------
    q_v : jax.Array, shape (..., nlev)
        Specific humidity after physics [kg/kg].
    target_moisture : jax.Array
        Target global moisture integral [kg] (from initial state).
    p_s : jax.Array, shape (...)
        Surface pressure [Pa].
    dsigma : jax.Array, shape (nlev,)
        Sigma layer thicknesses.
    grid : CubedSphereGrid or similar
    owned_mask : jax.Array, optional
        Shape ``(n_faces,)`` for MPI replicated dynamics.

    Returns
    -------
    jax.Array : Moisture-conserving q_v with same shape as input.
    """
    current = compute_global_moisture(q_v, p_s, dsigma, grid, owned_mask=owned_mask)
    scale = jnp.where(current > _tiny(current), target_moisture / current, 1.0)
    return q_v * scale


def diagnose_moisture_correction(
    q_v: jax.Array,
    target_moisture: jax.Array,
    p_s: jax.Array,
    dsigma: jax.Array,
    grid,
    owned_mask: jax.Array | None = None,
) -> dict[str, jax.Array]:
    """Diagnose the moisture correction that ``fix_moisture_hydrostatic``
    would apply, without actually applying it.

    Returns a dict of:
      - ``current_mass`` [kg]: current global column water-vapor integral
      - ``target_mass`` [kg]: prescribed target
      - ``correction_mass`` [kg]: target - current (positive = mass added by
        the fixer; negative = mass removed)
      - ``scale``: multiplicative factor that ``fix_moisture_hydrostatic``
        would multiply q_v by.  Equals 1.0 when target == current.

    Use to track silent corrections that would otherwise compound with
    other untracked sources/sinks (advection clipping, microphysics
    saturation adjustment, etc.) and bias the moist energy budget.
    Iter-87 audit follow-up to the deferred multiplicative-fixer
    tracking finding.
    """
    current = compute_global_moisture(q_v, p_s, dsigma, grid, owned_mask=owned_mask)
    scale = jnp.where(current > _tiny(current), target_moisture / current, 1.0)
    return {
        "current_mass": current,
        "target_mass": target_moisture,
        "correction_mass": target_moisture - current,
        "scale": scale,
    }


def fix_total_water(
    tracers: dict[str, jax.Array],
    target_total_water: jax.Array,
    p_s: jax.Array,
    dsigma: jax.Array,
    grid,
    water_names: tuple[str, ...] = ("q_v", "q_c", "q_r"),
) -> dict[str, jax.Array]:
    """Fix total water (vapor + condensate) conservation.

    Scales all water tracers by a single uniform factor so that
    ``∫(q_v + q_c + q_r + ...) dp/g dA = target_total_water``.

    Parameters
    ----------
    tracers : dict[str, jax.Array]
        Tracer dict; only entries whose keys are in *water_names* are scaled.
    target_total_water : jax.Array
        Target global total water integral [kg].
    p_s : jax.Array
        Surface pressure [Pa].
    dsigma : jax.Array
        Sigma layer thicknesses.
    grid : CubedSphereGrid or similar
    water_names : tuple[str, ...]
        Names of water-species tracers to include.

    Returns
    -------
    dict[str, jax.Array] : Tracers with water species scaled to conserve total water.
    """
    total_q = sum(tracers[n] for n in water_names if n in tracers)
    current = compute_global_moisture(total_q, p_s, dsigma, grid)
    scale = jnp.where(current > _tiny(current), target_total_water / current, 1.0)

    result = dict(tracers)
    for name in water_names:
        if name in result:
            result[name] = result[name] * scale
    return result


def fix_mass_hydrostatic_target(
    state_new: HydrostaticState,
    target_mass: jax.Array,
    grid: CubedSphereGrid,
    owned_mask: jax.Array | None = None,
) -> HydrostaticState:
    """Fix mass conservation anchored to a fixed target mass.

    Unlike ``fix_mass_hydrostatic`` which anchors to the previous step,
    this anchors to a fixed target (typically the initial global mass),
    preventing slow drift accumulation over many steps.

    Parameters
    ----------
    state_new : HydrostaticState
        State after time integration.
    target_mass : jax.Array
        Target global mass integral (∫ p_s * dA at t=0).
    grid : CubedSphereGrid
    owned_mask : jax.Array, optional
        Shape ``(n_faces,)`` for MPI replicated dynamics — pass-through
        to ``fix_ps_mass_target`` so the mass integral correctly
        avoids double-counting under replicated MPI.  Iter-99 audit
        fix: previously this function called the Field-API
        ``global_integral`` directly without an owned_mask escape
        hatch.

    Returns
    -------
    HydrostaticState : Mass-conserving state.
    """
    if owned_mask is not None:
        # Use the raw-array path that handles owned_mask for MPI
        # replicated dynamics.  Same correction formula but the
        # global integral correctly weights by owned_mask.
        p_s_fixed_data = fix_ps_mass_target(
            state_new.p_s.data, target_mass, grid, owned_mask=owned_mask,
        )
        return state_new._replace(
            p_s=state_new.p_s.replace(data=p_s_fixed_data),
        )
    mass_new = global_integral(state_new.p_s, grid)
    correction = (target_mass - mass_new) / _total_area(grid)
    p_s_fixed = state_new.p_s.replace(data=state_new.p_s.data + correction)
    return state_new._replace(p_s=p_s_fixed)


def fix_ps_mass(
    p_s_new: jax.Array,
    p_s_old: jax.Array,
    grid: CubedSphereGrid,
    owned_mask: jax.Array | None = None,
) -> jax.Array:
    """Fix dry mass on raw p_s arrays — non-anchor variant.

    Raw-array equivalent of :func:`fix_mass_hydrostatic`.  Lets
    callers (e.g. the FV3 D-grid dycore) avoid round-tripping the
    full prognostic state through ``fv3_to_hydrostatic`` just to
    extract ``p_s``: D-grid → cell-centre wind interpolation is
    expensive and gets thrown away because only ``p_s`` is touched.

    Both ``mass_old`` and ``mass_new`` are computed in one
    batched allreduce, matching :func:`fix_mass_hydrostatic`'s
    communication pattern.
    """
    mass_old, mass_new = _batch_global_area_sums(
        [p_s_old, p_s_new], grid, owned_mask=owned_mask,
    )
    correction = (mass_old - mass_new) / _total_area(grid)
    return p_s_new + correction


def fix_ps_mass_target(
    p_s: jax.Array,
    target_mass: jax.Array,
    grid: CubedSphereGrid,
    owned_mask: jax.Array | None = None,
) -> jax.Array:
    """Fix dry mass conservation on raw p_s array, anchored to a fixed target.

    This is the raw-array version of ``fix_mass_hydrostatic_target``,
    suitable for use inside ``jax.lax.scan`` where we work with raw
    arrays rather than Field-wrapped NamedTuples.

    Parameters
    ----------
    p_s : jax.Array, shape (6, n, n)
        Surface pressure after dynamics.
    target_mass : jax.Array (scalar)
        Target global mass integral (∫ p_s * dA at t=0).
    grid : CubedSphereGrid
    owned_mask : jax.Array, optional
        Shape ``(6,)`` float mask for MPI replicated dynamics.
        See :func:`_global_area_sum` for details.

    Returns
    -------
    jax.Array : Corrected p_s with same shape.
    """
    mass_new = _global_area_sum(p_s, grid, owned_mask=owned_mask)
    correction = (target_mass - mass_new) / _total_area(grid)
    return p_s + correction


def compute_nh_dry_mass(
    rho_prime: jax.Array,
    height_coord,
    terrain_metric,
    grid: CubedSphereGrid,
) -> jax.Array:
    """Compute global dry-air mass for the non-hydrostatic model.

    M = ∫ J · rho_total · dz · dA  summed over all levels.

    Parameters
    ----------
    rho_prime : jax.Array, shape (6, n, n, nlev)
    height_coord : HeightCoordinate
    terrain_metric : TerrainMetric
    grid : CubedSphereGrid

    Returns
    -------
    jax.Array : Scalar global dry mass [kg].
    """
    rho_total = height_coord.rho_ref + rho_prime  # (6,n,n,nlev)
    J = terrain_metric.jacobian  # (6, n, n)
    dz = height_coord.dz  # (nlev,)
    # Column mass: sum_k(J * rho_total_k * dz_k)
    col_mass = jnp.sum(
        J[..., None] * rho_total * dz[None, None, None, :],
        axis=-1,
    )  # (6, n, n)
    return _global_area_sum(col_mass, grid)


def fix_mass_nonhydrostatic(
    state,
    target_mass: jax.Array,
    height_coord,
    terrain_metric,
    grid: CubedSphereGrid,
):
    """Fix dry-mass conservation for the non-hydrostatic model.

    Applies a uniform additive correction to rho_prime so that the
    global dry mass matches the target.

    Parameters
    ----------
    state : NonHydrostaticState
    target_mass : jax.Array
        Target global mass.
    height_coord : HeightCoordinate
    terrain_metric : TerrainMetric
    grid : CubedSphereGrid

    Returns
    -------
    NonHydrostaticState : Mass-conserving state.
    """
    rho_total = height_coord.rho_ref + state.rho_prime.data
    J = terrain_metric.jacobian
    dz = height_coord.dz
    col_mass = jnp.sum(
        J[..., None] * rho_total * dz[None, None, None, :],
        axis=-1,
    )
    col_vol = J * jnp.sum(dz)
    current_mass, total_vol = _batch_global_area_sums(
        [col_mass, col_vol], grid,
    )
    correction = (target_mass - current_mass) / total_vol
    rho_fixed = state.rho_prime.replace(
        data=state.rho_prime.data + correction,
    )
    return state._replace(rho_prime=rho_fixed)


def compute_hydrostatic_energy(
    state: HydrostaticState,
    grid: CubedSphereGrid,
    sigma_coord,
) -> dict[str, jax.Array]:
    """Compute energy diagnostics for the hydrostatic PE model.

    Returns
    -------
    dict with:
        'kinetic_energy': ∫ 0.5·p_s·(u²+v²)·dσ·dA / g
        'internal_energy': ∫ c_v·T·p_s·dσ·dA / g
        'potential_energy': ∫ Φ·p_s·dσ·dA / g
        'total_energy': sum of all three
    """
    # iter-43: promote energy fields to the fp64 budget accumulator
    # before the column + area-weighted sums.  Same fp32-field bug as
    # iter-1/4/5 mass path and iter-42 SW/MPAS energy fixer — the
    # 0.5·(u²+v²) square+multiply lost ~7 bits of precision when the
    # state was stored in fp32.
    acc = conservation_accumulator()
    u = state.u.data.astype(acc)
    v = state.v.data.astype(acc)
    T = state.T.data.astype(acc)
    p_s = state.p_s.data.astype(acc)
    phis = state.phis.data.astype(acc)
    g = jnp.asarray(constants.g, dtype=acc)
    c_v = jnp.asarray(constants.c_vd, dtype=acc)
    dsigma = sigma_coord.dsigma.astype(acc)

    # Mass weight per layer: p_s * dsigma / g
    # Use [..., None] broadcasting so this works for both cubed-sphere
    # (6, n, n, nlev) and lat-lon (n_lat, n_lon, nlev) state shapes.
    mass_weight = p_s[..., None] * dsigma / g

    # Kinetic + internal + potential energy column reductions all share
    # the level axis and ``mass_weight``; stack the integrands and reduce
    # once.  ``mass_weight`` is factored into the stack so each integrand
    # contributes only its own value field.
    Phi = compute_geopotential(T, p_s, sigma_coord, phis)
    _ke_intg = 0.5 * (u**2 + v**2)
    _ie_intg = c_v * T
    _pe_intg = Phi
    _col_triple = jnp.sum(
        jnp.stack([_ke_intg, _ie_intg, _pe_intg], axis=-1)
        * mass_weight[..., None],
        axis=-2,
    )
    ke_col = _col_triple[..., 0]
    ie_col = _col_triple[..., 1]
    pe_col = _col_triple[..., 2]

    ke, ie, pe = _batch_global_area_sums([ke_col, ie_col, pe_col], grid)

    total = ke + ie + pe
    return {
        'kinetic_energy': ke,
        'internal_energy': ie,
        'potential_energy': pe,
        'total_energy': total,
    }


def compute_nh_energy(
    state,
    grid: CubedSphereGrid,
    height_coord,
    terrain_metric,
) -> dict[str, jax.Array]:
    """Compute energy diagnostics for the non-hydrostatic CE model.

    Returns
    -------
    dict with:
        'kinetic_energy': ∫ 0.5·rho·(u²+v²+w²)·J·dz·dA
        'internal_energy': ∫ c_v·T·rho·J·dz·dA
        'potential_energy': ∫ g·z·rho·J·dz·dA
        'total_energy': sum of all three
    """
    # iter-43: promote energy fields to fp64 budget accumulator (see
    # compute_hydrostatic_energy docstring above).
    acc = conservation_accumulator()
    u = state.u.data.astype(acc)
    v = state.v.data.astype(acc)
    w = state.w.data.astype(acc)
    theta_p = state.theta_prime.data.astype(acc)
    rho_p = state.rho_prime.data.astype(acc)

    g = jnp.asarray(constants.g, dtype=acc)
    c_v = jnp.asarray(constants.c_vd, dtype=acc)
    theta_0 = height_coord.theta_ref.astype(acc)
    rho_0 = height_coord.rho_ref.astype(acc)
    exner_0 = height_coord.exner_ref.astype(acc)
    dz = height_coord.dz.astype(acc)
    z_full = height_coord.z_full.astype(acc)
    J = terrain_metric.jacobian.astype(acc)

    rho_total = rho_0 + rho_p
    theta_total = theta_0 + theta_p
    T = theta_total * exner_0  # approximate T from theta * exner_ref

    # w at full levels
    w_full = 0.5 * (w[..., :-1] + w[..., 1:])

    weight = J[..., None] * dz[None, None, None, :] * rho_total  # (6,n,n,nlev)

    # KE+IE+PE column reductions all share the level axis and
    # ``weight``; stack the integrands and reduce once.
    _ke_intg = 0.5 * (u**2 + v**2 + w_full**2)
    _ie_intg = c_v * T
    _pe_intg = g * jnp.broadcast_to(z_full[None, None, None, :], T.shape)
    _col_triple = jnp.sum(
        jnp.stack([_ke_intg, _ie_intg, _pe_intg], axis=-1)
        * weight[..., None],
        axis=-2,
    )
    ke_col = _col_triple[..., 0]
    ie_col = _col_triple[..., 1]
    pe_col = _col_triple[..., 2]

    ke, ie, pe = _batch_global_area_sums([ke_col, ie_col, pe_col], grid)

    total = ke + ie + pe
    return {
        'kinetic_energy': ke,
        'internal_energy': ie,
        'potential_energy': pe,
        'total_energy': total,
    }


def compute_conservation_diagnostics(
    state: ShallowWaterState,
    grid,
    g: float = constants.g,
) -> dict[str, jax.Array]:
    """Compute conservation diagnostic quantities (any grid).

    Returns
    -------
    dict with:
        'total_mass': Global integral of h*area
        'total_energy': Global integral of (KE + PE)*area
    """
    # iter-44: promote energy field to fp64 budget accumulator before
    # the area-sum (same fp32-field bug as iter-42/43).  The mass
    # integrand (h alone) is already correct via _batch_global_area_sums
    # internal cast.
    acc = conservation_accumulator()
    h = state.h.data.astype(acc)
    u = state.u.data.astype(acc)
    v = state.v.data.astype(acc)
    h_s = state.h_s.data.astype(acc)
    g_acc = jnp.asarray(g, dtype=acc)

    ke = 0.5 * h * (u**2 + v**2)
    pe = 0.5 * g_acc * (h + h_s)**2
    total_mass, total_energy = _batch_global_area_sums(
        [h, ke + pe], grid,
    )

    return {
        'total_mass': total_mass,
        'total_energy': total_energy,
    }


# ==============================================================================
# Voronoi mesh conservation
# ==============================================================================

def global_integral_voronoi(field, mesh) -> jax.Array:
    """Area-weighted global integral on a Voronoi mesh.

    Parameters
    ----------
    field : jax.Array, shape (nCells,)
    mesh : VoronoiMesh

    Returns
    -------
    jax.Array : scalar
    """
    acc = conservation_accumulator()
    return jnp.sum(field.astype(acc) * mesh.areaCell.astype(acc))


def fix_mass_mpas(state, target_mass, mesh):
    """Fix mass conservation on Voronoi mesh via uniform h correction.

    Parameters
    ----------
    state : MPASShallowWaterState
    target_mass : jax.Array
        Target global mass (∫ h * dA).
    mesh : VoronoiMesh

    Returns
    -------
    MPASShallowWaterState
    """
    current_mass = global_integral_voronoi(state.h.data, mesh)
    # ``mesh.grid_total_area`` is precomputed at mesh construction —
    # avoid recomputing the global ``jnp.sum(areaCell)`` every step
    # (one extra reduction in serial; one extra allreduce under
    # multi-rank Voronoi sharding).  Iter-13: cast to fp64 so the
    # divisor matches the fp64 ``current_mass`` and ``target_mass``;
    # otherwise an fp32 ``total_area`` leaks ~N·eps into ``correction``.
    total_area = mesh.grid_total_area.astype(conservation_accumulator())
    correction = (target_mass - current_mass) / total_area
    h_fixed = state.h.replace(data=state.h.data + correction)
    return state._replace(h=h_fixed)


def compute_nh_dry_mass_mpas(
    rho_prime: jax.Array,
    height_coord,
    terrain_metric,
    mesh,
) -> jax.Array:
    """Global dry mass on a Voronoi mesh for the non-hydrostatic model.

    M = sum_k(sum_cells J · (rho_ref_k + rho_prime_k) · dz_k · areaCell).

    Parameters
    ----------
    rho_prime : jax.Array, shape (nCells, nlev)
    height_coord : HeightCoordinate (rho_ref, dz attrs as in
        ``compute_nh_dry_mass`` cubed-sphere variant).
    terrain_metric : TerrainMetric (jacobian shape ``(nCells,)``).
    mesh : VoronoiMesh.

    Returns
    -------
    jax.Array : scalar dry mass [kg].
    """
    rho_total = height_coord.rho_ref + rho_prime  # (nCells, nlev)
    J = terrain_metric.jacobian                   # (nCells,)
    dz = height_coord.dz                          # (nlev,)
    col_mass = jnp.sum(
        J[:, None] * rho_total * dz[None, :], axis=-1,
    )                                             # (nCells,)
    return global_integral_voronoi(col_mass, mesh)


def fix_mass_nonhydrostatic_mpas(
    state,
    target_mass: jax.Array,
    height_coord,
    terrain_metric,
    mesh,
):
    """Fix dry-mass conservation on a Voronoi mesh.

    Applies a uniform additive correction to ``rho_prime`` so the
    global dry mass matches ``target_mass``.  Volume normalisation is
    ``∫ J · dz · dA`` so the correction has units of density (kg/m³)
    and is broadcast across levels — matches the cubed-sphere
    ``fix_mass_nonhydrostatic`` convention.

    Parameters
    ----------
    state : MPASNonHydrostaticState
    target_mass : jax.Array
        Target dry mass (e.g. computed once via
        ``compute_nh_dry_mass_mpas`` on the initial state).
    height_coord, terrain_metric, mesh : as for
        :func:`compute_nh_dry_mass_mpas`.
    """
    rho_total = height_coord.rho_ref + state.rho_prime.data
    J = terrain_metric.jacobian
    dz = height_coord.dz
    col_mass = jnp.sum(
        J[:, None] * rho_total * dz[None, :], axis=-1,
    )                                             # (nCells,)
    col_vol = J * jnp.sum(dz)                     # (nCells,)
    current_mass, total_vol = _batch_global_area_sums_voronoi(
        [col_mass, col_vol], mesh,
    )
    correction = (target_mass - current_mass) / total_vol
    rho_fixed = state.rho_prime.replace(
        data=state.rho_prime.data + correction,
    )
    return state._replace(rho_prime=rho_fixed)


def _batch_global_area_sums_voronoi(
    arrays: list[jax.Array],
    mesh,
) -> list[jax.Array]:
    """Voronoi analogue of :func:`_batch_global_area_sums`.

    Stacks the arrays along a new trailing axis, multiplies by
    ``mesh.areaCell`` (cast to the fp64 conservation accumulator),
    reduces once locally, then applies the multi-rank allreduce when
    the mesh is sharded.
    """
    acc = conservation_accumulator()
    area_acc = mesh.areaCell.astype(acc)
    stacked = jnp.stack([arr.astype(acc) for arr in arrays], axis=-1)
    summed = jnp.sum(stacked * area_acc[..., None], axis=0)  # (n_arrays,)
    local_sums = [summed[..., i] for i in range(len(arrays))]
    if _is_distributed():
        from legoesm.parallel.reductions import batch_allreduce_mpi
        return batch_allreduce_mpi(local_sums, op="sum")
    return local_sums


def fix_energy_mpas(state, target_energy, mesh, g=constants.g):
    """Fix energy conservation on Voronoi mesh via velocity scaling.

    Batches the KE / PE sums into one stacked reduction so the helper
    issues a single ``jnp.sum`` per accumulator pair instead of two
    separate ones — half the allreduce traffic when the Voronoi mesh
    is sharded across ranks (matching the ``shallow_water_mpas`` /
    ``conservation_mpas`` fixers).

    Parameters
    ----------
    state : MPASShallowWaterState
    target_energy : jax.Array
        Target total energy.
    mesh : VoronoiMesh
    g : float

    Returns
    -------
    MPASShallowWaterState
    """
    # iter-42: promote energy fields to the fp64 budget accumulator
    # before the area-weighted sum (same fix as
    # ``fix_energy_shallow_water`` for the SW-on-any-grid path).
    acc = conservation_accumulator()
    h = state.h.data.astype(acc)
    u = state.u.data.astype(acc)
    h_s = state.h_s.data.astype(acc)
    area = mesh.areaCell.astype(acc)
    g_acc = jnp.asarray(g, dtype=acc)

    KE_cells = kinetic_energy_cell(u, mesh)
    # Both KE and PE share the ``area`` weight on the horizontal axes —
    # stack the two integrands and reduce once locally so XLA fires one
    # sum kernel; the stacked result still yields a 2-vector for the
    # downstream ``KE / PE`` split (and a future allreduce, if any).
    _energy_intg = jnp.stack(
        [KE_cells * h, 0.5 * g_acc * (h + h_s) ** 2], axis=-1,
    ) * area[..., None]
    energy_terms = jnp.sum(_energy_intg, axis=tuple(range(area.ndim)))
    KE, PE = energy_terms[0], energy_terms[1]

    KE_target = jnp.maximum(target_energy - PE, _EPS_ENERGY)
    scale = jnp.where(KE > _tiny(KE), jnp.sqrt(KE_target / KE), 1.0)
    u_fixed = state.u.replace(data=u * scale)
    return state._replace(u=u_fixed)
