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


def _global_area_sum(array: jax.Array, grid: CubedSphereGrid) -> jax.Array:
    """Area-weighted global sum of a raw array, MPI-aware.

    Like ``global_integral`` but works on raw arrays instead of
    ``Field`` objects.  Under MPI, local sums are combined via
    ``allreduce(SUM)`` to produce the true global total.
    """
    local_sum = jnp.sum(array * grid.area)
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
    mass_old = jnp.sum(state_old.p_s.data * grid.area)
    mass_new = jnp.sum(state_new.p_s.data * grid.area)

    correction = (mass_old - mass_new) / grid.total_area
    p_s_fixed = state_new.p_s.replace(data=state_new.p_s.data + correction)

    return state_new._replace(p_s=p_s_fixed)


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
