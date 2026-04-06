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
from legoesm.core.precision import cast
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.grids.halo import pad_halo
from legoesm.ocean.vertical import OceanZStarCoordinate, compute_layer_thickness
from legoesm.ocean.state import OceanState, OceanTendencies, OceanConfig


# ==============================================================================
# Land-fill helper (Neumann BC at coastlines)
# ==============================================================================

def _fill_land_cells(
    field: jnp.ndarray,
    mask: jnp.ndarray,
    grid: CubedSphereGrid,
    n_passes: int = 3,
) -> jnp.ndarray:
    """Replace land values with ocean-neighbor average (zero-gradient BC).

    For each land cell (mask < 0.5) that has at least one ocean neighbor,
    set its value to the area-unweighted mean of ocean neighbors.  This
    removes the sharp jump between physical ocean values and masked zeros
    that would otherwise produce spurious pressure gradients and diffusive
    fluxes at the coastline.

    Multiple passes propagate through consecutive land cells (e.g. narrow
    land strips 2-3 cells wide between ocean basins), matching the
    lat-lon solver's 3-pass approach.

    Uses ``pad_halo`` for correct cross-face halo exchange.

    Parameters
    ----------
    field : array, shape (6, n, n)
    mask  : array, shape (6, n, n), 1 = ocean, 0 = land
    grid  : CubedSphereGrid
    n_passes : int
        Number of fill passes (default 3).

    Returns
    -------
    field_filled : array, same shape, ocean cells unchanged, coastal
        land cells filled.
    """
    filled = field
    effective_mask = mask
    for _ in range(n_passes):
        f_pad = pad_halo(filled, interp_offsets=grid.halo_interp_offsets)
        m_pad = pad_halo(effective_mask, interp_offsets=grid.halo_interp_offsets)

        # 4-connected neighbour sum, weighted by ocean mask
        nbr_sum = (
            f_pad[:, 2:, 1:-1] * m_pad[:, 2:, 1:-1]
            + f_pad[:, :-2, 1:-1] * m_pad[:, :-2, 1:-1]
            + f_pad[:, 1:-1, 2:] * m_pad[:, 1:-1, 2:]
            + f_pad[:, 1:-1, :-2] * m_pad[:, 1:-1, :-2]
        )
        nbr_count = (
            m_pad[:, 2:, 1:-1]
            + m_pad[:, :-2, 1:-1]
            + m_pad[:, 1:-1, 2:]
            + m_pad[:, 1:-1, :-2]
        )
        nbr_avg = nbr_sum / jnp.maximum(nbr_count, 1.0)

        is_land = mask < 0.5
        has_ocean_nbr = nbr_count > 0.0
        filled = jnp.where(is_land & has_ocean_nbr, nbr_avg, filled)
        # Expand effective mask so next pass can propagate further
        effective_mask = jnp.where(is_land & has_ocean_nbr, 1.0, effective_mask)

    return jnp.where(mask > 0.5, field, filled)


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
    # --- Precision management ---
    # Upcast key arrays to barotropic solver compute precision (fp64 in mixed
    # mode). The fori_loop/scan carry must have uniform dtype, so we cast
    # before any loop setup and downcast after the loop exits.
    _M = "barotropic_solver"
    g = cast(jnp.asarray(config.g), _M, "compute")
    H_bathy = cast(state.H_bathy.data, _M, "compute")
    mask = cast(state.land_mask.data, _M, "compute")
    u = cast(state.u.data, _M, "compute")
    v = cast(state.v.data, _M, "compute")
    eta_raw = cast(state.eta.data, _M, "compute")
    min_water_col = cast(jnp.asarray(config.min_water_column_m), _M, "compute")
    dt_s = cast(jnp.asarray(dt_s), _M, "compute")
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
        # Mask fluxes before divergence so that no transport crosses
        # land-ocean boundaries.  Without this, the centered-difference
        # stencil in divergence() sees a nonzero-to-zero transition at
        # coastal cells, generating spurious divergence.  (The latlon
        # solver achieves this via mask= kwarg in fv_divergence_latlon.)
        flux_u = H_total_c * U_bar_c * mask
        flux_v = H_total_c * V_bar_c * mask
        div_flux = _divergence_raw(flux_u, flux_v, grid).astype(eta.dtype)
        eta_unfloored = (eta_c - dt_s * div_flux) * mask
        eta_new = jnp.maximum(eta_unfloored, eta_floor) * mask
        # Redistribute mass added by floor clamping to preserve continuity.
        mass_added = jnp.sum((eta_new - eta_unfloored) * grid.area * mask)
        ocean_area = jnp.sum(grid.area * mask)
        eta_new = (eta_new - mass_added / jnp.maximum(ocean_area, 1.0) * mask)
        # Re-floor after correction to prevent negative water columns
        eta_new = jnp.maximum(eta_new, eta_floor) * mask

        # Backward: update U_bar, V_bar with UPDATED eta
        # Fill land cells with ocean-neighbour average so that the
        # centred-difference gradient sees a smooth field at coastlines
        # instead of the sharp ocean-to-zero transition from masking.
        # Without this, coastal ocean cells experience a spurious pressure
        # gradient force that drives flow toward/away from land.
        eta_filled = _fill_land_cells(eta_new, mask, grid)
        deta_dx = _gradient_x_raw(eta_filled, grid).astype(eta.dtype)
        deta_dy = _gradient_y_raw(eta_filled, grid).astype(eta.dtype)

        # Semi-implicit Coriolis + backward PGF
        rhs_u = U_bar_c + alpha * V_bar_c - dt_s * g * deta_dx
        rhs_v = V_bar_c - alpha * U_bar_c - dt_s * g * deta_dy
        U_bar_new = (rhs_u + alpha * rhs_v) / denom * mask
        V_bar_new = (rhs_v - alpha * rhs_u) / denom * mask

        if config.barotropic_diffusion_alpha > 0.0:
            # Compact Laplacian diffusion (damps modes amplified by
            # Coriolis-PGF interaction on non-adjoint cubed-sphere operators).
            # Fill land cells before Laplacian to avoid diffusing against
            # the masked discontinuity at coastlines.
            eta_new = (
                eta_new + nu_dt * _laplacian_raw(
                    _fill_land_cells(eta_new, mask, grid), grid,
                ).astype(eta.dtype)
            ) * mask
            eta_new = jnp.maximum(eta_new, eta_floor) * mask
            U_bar_new = (
                U_bar_new + nu_dt * _laplacian_raw(
                    _fill_land_cells(U_bar_new, mask, grid), grid,
                ).astype(eta.dtype)
            ) * mask
            V_bar_new = (
                V_bar_new + nu_dt * _laplacian_raw(
                    _fill_land_cells(V_bar_new, mask, grid), grid,
                ).astype(eta.dtype)
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

    # Downcast results back to storage precision
    eta_f = cast(eta_f, _M, "storage")
    u_new = cast(u_new, _M, "storage")
    v_new = cast(v_new, _M, "storage")

    return state._replace(
        eta=state.eta.replace(data=eta_f),
        u=state.u.replace(data=u_new),
        v=state.v.replace(data=v_new),
    )
