"""Barotropic (depth-averaged) solver for ocean split-explicit stepping.

Solves the 2D free-surface equations via forward-backward substeps:

    d(eta)/dt = -div(H_total * U_bar, H_total * V_bar)
    d(U_bar)/dt = f * V_bar - g * d(eta)/dx + F_slow_x
    d(V_bar)/dt = -f * U_bar - g * d(eta)/dy + F_slow_y

where U_bar, V_bar are depth-averaged velocities and F_slow is the
baroclinic forcing held constant during substeps.

This parallels acoustic_substeps() in compressible_euler.py but for
barotropic ocean gravity waves instead of atmospheric sound waves.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.operators import gradient_x, gradient_y, divergence, laplacian
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.ocean.vertical import OceanZStarCoordinate, compute_layer_thickness
from legoesm.ocean.state import OceanState, OceanTendencies, OceanConfig


# ==============================================================================
# Raw-array operator wrappers for use inside fori_loop
# ==============================================================================

def _divergence_raw(u: jnp.ndarray, v: jnp.ndarray, grid: CubedSphereGrid) -> jnp.ndarray:
    """Divergence on raw arrays (avoids Field construction in fori_loop)."""
    u_f = Field(data=u, name="u", dims=("face", "x", "y"), units="m/s")
    v_f = Field(data=v, name="v", dims=("face", "x", "y"), units="m/s")
    return divergence(u_f, v_f, grid).data


def _gradient_x_raw(data: jnp.ndarray, grid: CubedSphereGrid) -> jnp.ndarray:
    """X-gradient on raw arrays."""
    f = Field(data=data, name="f", dims=("face", "x", "y"), units="")
    return gradient_x(f, grid).data


def _gradient_y_raw(data: jnp.ndarray, grid: CubedSphereGrid) -> jnp.ndarray:
    """Y-gradient on raw arrays."""
    f = Field(data=data, name="f", dims=("face", "x", "y"), units="")
    return gradient_y(f, grid).data


def _laplacian_raw(data: jnp.ndarray, grid: CubedSphereGrid) -> jnp.ndarray:
    """Compact Laplacian on raw arrays (sees 2dx mode)."""
    f = Field(data=data, name="f", dims=("face", "x", "y"), units="")
    return laplacian(f, grid).data


# ==============================================================================
# Barotropic substeps
# ==============================================================================

def barotropic_substeps(
    state: OceanState,
    dt_s: float,
    n_substeps: int,
    grid: CubedSphereGrid,
    z_coord: OceanZStarCoordinate,
    config: OceanConfig,
) -> OceanState:
    """Run barotropic substeps via jax.lax.fori_loop.

    Forward-backward time stepping for 2D free-surface equations:
      1. Forward: update eta from continuity
      2. Backward: update U_bar, V_bar from momentum + updated eta

    After all substeps, correct 3D velocities to match new depth-average.

    Parameters
    ----------
    state : OceanState
        State after slow tendency application.
    dt_s : float
        Substep size [seconds].
    n_substeps : int
        Number of barotropic substeps.
    grid : CubedSphereGrid
        Horizontal grid.
    z_coord : OceanZStarCoordinate
        Vertical coordinate.
    config : OceanConfig
        Model configuration.

    Returns
    -------
    OceanState : State with updated eta and velocity correction.
    """
    g = jnp.asarray(config.g)
    H_bathy = state.H_bathy.data
    mask = state.land_mask.data
    u = state.u.data
    v = state.v.data
    eta_raw = state.eta.data
    min_water_col = jnp.asarray(config.min_water_column_m, dtype=eta_raw.dtype)
    dt_s = jnp.asarray(dt_s, dtype=eta_raw.dtype)
    g = g.astype(eta_raw.dtype)
    eta_floor = min_water_col - H_bathy
    eta = jnp.maximum(eta_raw, eta_floor) * mask

    # Compute depth-averaged velocity from current 3D state
    h_k = compute_layer_thickness(
        eta, H_bathy, z_coord, min_water_column_m=config.min_water_column_m,
    )
    H_total = jnp.sum(h_k, axis=-1)                        # (6, n, n)
    # Keep depth averages finite if a column becomes too thin.
    H_total = jnp.maximum(H_total, min_water_col)
    U_bar = jnp.sum(u * h_k, axis=-1) / H_total * mask     # (6, n, n)
    V_bar = jnp.sum(v * h_k, axis=-1) / H_total * mask

    # No explicit F_slow here: the 3D slow tendency has already been applied
    # to u, v by _pytree_axpy in split_explicit_step before this function is
    # called. Including F_slow would double-count the depth-averaged part of
    # the slow tendency (once in U_bar initial condition, once via F_slow).
    # Semi-implicit (Crank-Nicolson) Coriolis.
    # Coriolis MUST be applied at the substep level (small rotations per
    # substep, f*dt_s ~ 0.01 rad) rather than at the baroclinic level
    # (large rotation, f*dt ~ 0.5 rad) to avoid time-splitting errors
    # that generate spurious gravity waves.
    alpha = (0.5 * grid.f.astype(eta.dtype) * dt_s).astype(eta.dtype)   # (6, n, n)
    denom = 1.0 + alpha ** 2

    if config.barotropic_diffusion_alpha < 0.0:
        raise ValueError(
            "barotropic_diffusion_alpha must be >= 0, got "
            f"{config.barotropic_diffusion_alpha!r}",
        )
    if config.barotropic_diffusion_dt_ref <= 0.0:
        raise ValueError(
            "barotropic_diffusion_dt_ref must be > 0, got "
            f"{config.barotropic_diffusion_dt_ref!r}",
        )
    # Tunable, dt-scaled barotropic diffusion.
    # Per-substep coefficient is alpha * (dt_s / dt_ref) * area.
    baro_alpha = jnp.asarray(
        config.barotropic_diffusion_alpha, dtype=eta.dtype,
    ) * (
        dt_s / jnp.asarray(config.barotropic_diffusion_dt_ref, dtype=eta.dtype)
    )
    nu_dt = (baro_alpha * grid.area).astype(eta.dtype)

    # Forward-backward substeps via fori_loop
    def substep_body(i, carry):
        eta_c, U_bar_c, V_bar_c = carry

        H_total_c = jnp.maximum(eta_c + H_bathy, min_water_col) * mask

        # Forward: update eta from continuity
        # deta/dt = -div(H_total * U_bar, H_total * V_bar)
        flux_u = H_total_c * U_bar_c
        flux_v = H_total_c * V_bar_c
        div_flux = _divergence_raw(flux_u, flux_v, grid).astype(eta.dtype)
        eta_unfloored = (eta_c - dt_s * div_flux) * mask
        eta_new = jnp.maximum(eta_unfloored, eta_floor) * mask
        # Redistribute mass added by floor clamping to preserve continuity.
        mass_added = jnp.sum((eta_new - eta_unfloored) * grid.area * mask)
        ocean_area = jnp.sum(grid.area * mask)
        eta_new = (eta_new - mass_added / jnp.maximum(ocean_area, 1.0) * mask)

        # Backward: update U_bar, V_bar with UPDATED eta
        deta_dx = _gradient_x_raw(eta_new, grid).astype(eta.dtype)
        deta_dy = _gradient_y_raw(eta_new, grid).astype(eta.dtype)

        # Semi-implicit Coriolis + backward PGF
        rhs_u = U_bar_c + alpha * V_bar_c - dt_s * g * deta_dx
        rhs_v = V_bar_c - alpha * U_bar_c - dt_s * g * deta_dy
        U_bar_new = (rhs_u + alpha * rhs_v) / denom * mask
        V_bar_new = (rhs_v - alpha * rhs_u) / denom * mask

        if config.barotropic_diffusion_alpha > 0.0:
            # Compact Laplacian diffusion (damps modes amplified by
            # Coriolis-PGF interaction on non-adjoint cubed-sphere operators).
            eta_new = (
                eta_new + nu_dt * _laplacian_raw(eta_new, grid).astype(eta.dtype)
            ) * mask
            eta_new = jnp.maximum(eta_new, eta_floor) * mask
            U_bar_new = (
                U_bar_new + nu_dt * _laplacian_raw(U_bar_new, grid).astype(eta.dtype)
            ) * mask
            V_bar_new = (
                V_bar_new + nu_dt * _laplacian_raw(V_bar_new, grid).astype(eta.dtype)
            ) * mask

        return (eta_new, U_bar_new, V_bar_new)

    if config.differentiable_barotropic:
        # lax.scan stores intermediates for reverse-mode AD (jax.grad).
        # Slower forward pass, but enables full differentiability.
        def scan_body(carry, _):
            new_carry = substep_body(0, carry)
            return new_carry, None

        (eta_f, U_bar_f, V_bar_f), _ = jax.lax.scan(
            scan_body, (eta, U_bar, V_bar), xs=None, length=n_substeps,
        )
    else:
        # fori_loop is faster (no intermediate storage) but not
        # reverse-mode differentiable. Use for production runs.
        eta_f, U_bar_f, V_bar_f = jax.lax.fori_loop(
            0, n_substeps, substep_body, (eta, U_bar, V_bar),
        )

    # Correct 3D velocities: preserve baroclinic structure, update depth-average
    # u_new = u_baroclinic_prime + U_bar_new
    # where u_baroclinic_prime = u - U_bar_old (deviations from old depth-average)
    u_baro_prime = u - U_bar[..., jnp.newaxis]
    v_baro_prime = v - V_bar[..., jnp.newaxis]
    u_new = (u_baro_prime + U_bar_f[..., jnp.newaxis]) * mask[..., jnp.newaxis]
    v_new = (v_baro_prime + V_bar_f[..., jnp.newaxis]) * mask[..., jnp.newaxis]

    return state._replace(
        eta=state.eta.replace(data=eta_f),
        u=state.u.replace(data=u_new),
        v=state.v.replace(data=v_new),
    )
