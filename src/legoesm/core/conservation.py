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

from legoesm.core.operators import global_integral, _is_distributed
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.core.state import ShallowWaterState, HydrostaticState


def _accumulation_dtype():
    """Return the best dtype for accumulation: float64 if available, else float32.

    On backends that lack float64 (e.g. Apple Metal), JAX silently
    truncates .astype(float64) to float32, which is harmless but wastes
    a cast.  This helper lets callers use the widest available type
    explicitly, and makes the intent clear.
    """
    from legoesm.core.hardware import _UNSUPPORTED_F64_BACKENDS, get_backend
    if get_backend() in _UNSUPPORTED_F64_BACKENDS or not jax.config.jax_enable_x64:
        return jnp.float32
    return jnp.float64


def _global_area_sum(array: jax.Array, grid: CubedSphereGrid) -> jax.Array:
    """Area-weighted global sum of a raw array, MPI-aware.

    Like ``global_integral`` but works on raw arrays instead of
    ``Field`` objects.  Under MPI, local sums are combined via
    ``allreduce(SUM)`` to produce the true global total.

    The accumulation is performed in float64 (if available) to avoid
    precision loss in large-scale global integrals.
    """
    acc = _accumulation_dtype()
    prod = array.astype(acc) * grid.area.astype(acc)
    local_sum = jnp.sum(prod)
    if _is_distributed():
        from legoesm.parallel.reductions import global_sum_mpi
        return global_sum_mpi(local_sum)
    return local_sum


def fix_mass_shallow_water(
    state_new: ShallowWaterState,
    state_old: ShallowWaterState,
    grid: CubedSphereGrid,
) -> ShallowWaterState:
    """Fix mass conservation for shallow water equations.

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
    grid : CubedSphereGrid
        The grid.

    Returns
    -------
    ShallowWaterState : Mass-conserving state.
    """
    mass_old = global_integral(state_old.h, grid)
    mass_new = global_integral(state_new.h, grid)

    # Uniform correction
    correction = (mass_old - mass_new) / grid.total_area
    h_fixed = state_new.h.replace(data=state_new.h.data + correction)

    return state_new._replace(h=h_fixed)


def fix_energy_shallow_water(
    state_new: ShallowWaterState,
    state_old: ShallowWaterState,
    grid: CubedSphereGrid,
    g: float = 9.80616,
) -> ShallowWaterState:
    """Fix total energy conservation for shallow water equations.

    Total energy = kinetic + potential:
        E = 0.5 * h * (u^2 + v^2) + 0.5 * g * (h + h_s)^2

    Applies a uniform scaling to velocities to restore total energy.

    Parameters
    ----------
    state_new : ShallowWaterState
        State after time integration.
    state_old : ShallowWaterState
        State before time integration.
    grid : CubedSphereGrid
        The grid.
    g : float
        Gravitational acceleration.

    Returns
    -------
    ShallowWaterState : Energy-conserving state.
    """
    def total_energy(state):
        h = state.h.data
        u = state.u.data
        v = state.v.data
        h_s = state.h_s.data
        ke = 0.5 * h * (u**2 + v**2)
        pe = 0.5 * g * (h + h_s)**2
        return _global_area_sum(ke + pe, grid)

    E_old = total_energy(state_old)

    # Compute current kinetic energy
    h_new = state_new.h.data
    u_new = state_new.u.data
    v_new = state_new.v.data
    KE_new = _global_area_sum(0.5 * h_new * (u_new**2 + v_new**2), grid)

    # PE is already set by h (which was fixed by mass fixer)
    PE_new = _global_area_sum(0.5 * g * (h_new + state_new.h_s.data)**2, grid)

    # Scale KE to match target: KE_target = E_old - PE_new
    KE_target = E_old - PE_new
    # Guard against negative KE_target (PE_new > E_old after mass fixer)
    KE_target = jnp.maximum(KE_target, 0.0)
    # Avoid division by zero when KE_new is negligible
    scale = jnp.where(KE_new > 1e-30, jnp.sqrt(KE_target / KE_new), 1.0)

    u_fixed = state_new.u.replace(data=u_new * scale)
    v_fixed = state_new.v.replace(data=v_new * scale)

    return state_new._replace(u=u_fixed, v=v_fixed)


def apply_conservation_fixer(
    state_new: ShallowWaterState,
    state_old: ShallowWaterState,
    grid: CubedSphereGrid,
    fix_mass: bool = True,
    fix_energy: bool = True,
    g: float = 9.80616,
) -> ShallowWaterState:
    """Apply all conservation fixers in sequence.

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
    mass_old = global_integral(state_old.p_s, grid)
    mass_new = global_integral(state_new.p_s, grid)

    # Uniform correction to p_s
    correction = (mass_old - mass_new) / grid.total_area
    p_s_fixed = state_new.p_s.replace(data=state_new.p_s.data + correction)

    return state_new._replace(p_s=p_s_fixed)


def fix_mass_hydrostatic_latlon(
    state_new: HydrostaticState,
    state_old: HydrostaticState,
    grid,
) -> HydrostaticState:
    """Fix mass conservation for the hydrostatic PE on a lat-lon grid.

    Same logic as fix_mass_hydrostatic but uses lat-lon global integral.

    Parameters
    ----------
    state_new : HydrostaticState
        State after time integration.
    state_old : HydrostaticState
        State before time integration (reference mass).
    grid : LatLonGrid
        The lat-lon grid.

    Returns
    -------
    HydrostaticState : Mass-conserving state.
    """
    acc = _accumulation_dtype()
    mass_old = jnp.sum(state_old.p_s.data.astype(acc) * grid.area.astype(acc))
    mass_new = jnp.sum(state_new.p_s.data.astype(acc) * grid.area.astype(acc))

    correction = (mass_old - mass_new) / grid.total_area
    p_s_fixed = state_new.p_s.replace(data=state_new.p_s.data + correction)

    return state_new._replace(p_s=p_s_fixed)


def zero_mean_tendency(
    tendency: jax.Array,
    grid: CubedSphereGrid,
) -> jax.Array:
    """Remove the area-weighted global mean from a tendency field.

    After this correction, ``sum(tendency * area) == 0`` to machine
    precision, enforcing exact conservation for explicit FV mass equations.

    Works for both 2D ``(6, n, n)`` and 3D ``(6, n, n, nlev)`` arrays.
    For 3D arrays the correction is applied independently at each level.

    Parameters
    ----------
    tendency : jax.Array
        The mass/tracer tendency field.
    grid : CubedSphereGrid
        Grid with cell areas and total_area.

    Returns
    -------
    jax.Array : Corrected tendency with zero global integral.
    """
    acc = _accumulation_dtype()
    area = grid.area  # (6, n, n)
    area_acc = area.astype(acc)
    total_area_acc = jnp.sum(area_acc)
    orig_dtype = tendency.dtype
    if tendency.ndim == 3:
        global_sum = _global_area_sum(tendency, grid)
        correction = global_sum / total_area_acc
        # Compute in accumulation dtype then cast back to original dtype
        return (tendency.astype(acc) - correction).astype(orig_dtype)
    elif tendency.ndim == 4:
        # Per-level correction
        tend_acc = tendency.astype(acc)
        prod = tend_acc * area_acc[..., None]
        level_sums = jnp.sum(prod, axis=(0, 1, 2))  # (nlev,)
        if _is_distributed():
            from legoesm.parallel.reductions import global_sum_mpi
            level_sums = global_sum_mpi(level_sums)
        corrections = level_sums / total_area_acc  # (nlev,)
        return (tend_acc - corrections[None, None, None, :]).astype(orig_dtype)
    else:
        return tendency


def fix_mass_hydrostatic_target(
    state_new: HydrostaticState,
    target_mass: jax.Array,
    grid: CubedSphereGrid,
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

    Returns
    -------
    HydrostaticState : Mass-conserving state.
    """
    mass_new = global_integral(state_new.p_s, grid)
    correction = (target_mass - mass_new) / grid.total_area
    p_s_fixed = state_new.p_s.replace(data=state_new.p_s.data + correction)
    return state_new._replace(p_s=p_s_fixed)


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
    current_mass = compute_nh_dry_mass(
        state.rho_prime.data, height_coord, terrain_metric, grid,
    )
    J = terrain_metric.jacobian
    dz = height_coord.dz
    # Total weighted volume: ∫ J * sum(dz) * dA
    col_vol = J * jnp.sum(dz)  # (6, n, n)
    total_vol = _global_area_sum(col_vol, grid)
    # Uniform correction to rho_prime
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
    from legoesm import constants
    from legoesm.grids.vertical import compute_geopotential

    u = state.u.data
    v = state.v.data
    T = state.T.data
    p_s = state.p_s.data
    phis = state.phis.data
    g = constants.g
    c_v = constants.c_vd
    dsigma = sigma_coord.dsigma

    # Kinetic energy
    ke_3d = 0.5 * (u**2 + v**2) * p_s[..., None] * dsigma[None, None, None, :]
    ke = _global_area_sum(jnp.sum(ke_3d, axis=-1), grid) / g

    # Internal energy
    ie_3d = c_v * T * p_s[..., None] * dsigma[None, None, None, :]
    ie = _global_area_sum(jnp.sum(ie_3d, axis=-1), grid) / g

    # Potential energy
    Phi = compute_geopotential(T, p_s, sigma_coord, phis)
    pe_3d = Phi * p_s[..., None] * dsigma[None, None, None, :]
    pe = _global_area_sum(jnp.sum(pe_3d, axis=-1), grid) / g

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
    from legoesm import constants

    u = state.u.data
    v = state.v.data
    w = state.w.data
    theta_p = state.theta_prime.data
    rho_p = state.rho_prime.data

    g = constants.g
    c_v = constants.c_vd
    theta_0 = height_coord.theta_ref
    rho_0 = height_coord.rho_ref
    exner_0 = height_coord.exner_ref
    dz = height_coord.dz
    z_full = height_coord.z_full
    J = terrain_metric.jacobian

    rho_total = rho_0 + rho_p
    theta_total = theta_0 + theta_p
    T = theta_total * exner_0  # approximate T from theta * exner_ref

    # w at full levels
    w_full = 0.5 * (w[..., :-1] + w[..., 1:])

    weight = J[..., None] * dz[None, None, None, :] * rho_total  # (6,n,n,nlev)

    # Kinetic
    ke_3d = 0.5 * (u**2 + v**2 + w_full**2) * weight
    ke = _global_area_sum(jnp.sum(ke_3d, axis=-1), grid)

    # Internal
    ie_3d = c_v * T * weight
    ie = _global_area_sum(jnp.sum(ie_3d, axis=-1), grid)

    # Potential
    pe_3d = g * z_full[None, None, None, :] * weight
    pe = _global_area_sum(jnp.sum(pe_3d, axis=-1), grid)

    total = ke + ie + pe
    return {
        'kinetic_energy': ke,
        'internal_energy': ie,
        'potential_energy': pe,
        'total_energy': total,
    }


def compute_conservation_diagnostics(
    state: ShallowWaterState,
    grid: CubedSphereGrid,
    g: float = 9.80616,
) -> dict[str, jax.Array]:
    """Compute conservation diagnostic quantities.

    Returns
    -------
    dict with:
        'total_mass': Global integral of h*area
        'total_energy': Global integral of (KE + PE)*area
        'total_enstrophy': Global integral of 0.5*q^2*h*area (q = abs vorticity / h)
    """
    h = state.h.data
    u = state.u.data
    v = state.v.data
    h_s = state.h_s.data

    total_mass = _global_area_sum(h, grid)
    ke = 0.5 * h * (u**2 + v**2)
    pe = 0.5 * g * (h + h_s)**2
    total_energy = _global_area_sum(ke + pe, grid)

    return {
        'total_mass': total_mass,
        'total_energy': total_energy,
    }
