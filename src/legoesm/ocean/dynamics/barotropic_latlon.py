"""Barotropic solver for the lat-lon FV ocean model.

Forward-backward substeps for 2D free-surface gravity waves:

    d(eta)/dt = -div(H_total * U_bar, H_total * V_bar)
    d(U_bar)/dt = f * V_bar - g * d(eta)/dx
    d(V_bar)/dt = -f * U_bar - g * d(eta)/dy

Parallels barotropic.py (cubed-sphere) but uses lat-lon operators.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.grids.latlon import LatLonGrid
from legoesm.ocean.vertical import OceanZStarCoordinate, compute_layer_thickness
from legoesm.ocean.state import LatLonOceanState, LatLonOceanConfig
from legoesm.ocean.dynamics.latlon_operators import (
    fv_divergence_latlon,
    gradient_x_latlon,
    gradient_y_latlon,
    laplacian_latlon,
)


def barotropic_substeps_latlon(
    state: LatLonOceanState,
    dt_s: float,
    n_substeps: int,
    grid: LatLonGrid,
    z_coord: OceanZStarCoordinate,
    config: LatLonOceanConfig,
) -> LatLonOceanState:
    """Run barotropic substeps on a lat-lon grid.

    Parameters
    ----------
    state : LatLonOceanState
        State after slow tendency application.
    dt_s : float
        Substep size [seconds].
    n_substeps : int
        Number of barotropic substeps.
    grid : LatLonGrid
    z_coord : OceanZStarCoordinate
    config : LatLonOceanConfig

    Returns
    -------
    LatLonOceanState with updated eta and velocity.
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

    # Depth-averaged velocity
    h_k = compute_layer_thickness(
        eta, H_bathy, z_coord, min_water_column_m=config.min_water_column_m,
    )
    H_total = jnp.maximum(jnp.sum(h_k, axis=-1), min_water_col)
    U_bar = jnp.sum(u * h_k, axis=-1) / H_total * mask
    V_bar = jnp.sum(v * h_k, axis=-1) / H_total * mask

    # Semi-implicit Coriolis
    alpha = (0.5 * grid.f.astype(eta.dtype) * dt_s).astype(eta.dtype)
    denom = 1.0 + alpha ** 2

    # Barotropic diffusion
    baro_alpha = jnp.asarray(
        config.barotropic_diffusion_alpha, dtype=eta.dtype,
    ) * (dt_s / jnp.asarray(config.barotropic_diffusion_dt_ref, dtype=eta.dtype))
    nu_dt = (baro_alpha * grid.area).astype(eta.dtype)

    def substep_body(i, carry):
        eta_c, U_bar_c, V_bar_c = carry

        H_total_c = jnp.maximum(eta_c + H_bathy, min_water_col) * mask

        # Forward: update eta from continuity
        div_flux = fv_divergence_latlon(
            H_total_c * U_bar_c, H_total_c * V_bar_c, grid,
        ).astype(eta.dtype)
        eta_new = jnp.maximum(eta_c - dt_s * div_flux, eta_floor) * mask

        # Backward: update velocity with UPDATED eta
        deta_dx = gradient_x_latlon(eta_new, grid).astype(eta.dtype)
        deta_dy = gradient_y_latlon(eta_new, grid).astype(eta.dtype)

        # Semi-implicit Coriolis + backward PGF
        rhs_u = U_bar_c + alpha * V_bar_c - dt_s * g * deta_dx
        rhs_v = V_bar_c - alpha * U_bar_c - dt_s * g * deta_dy
        U_bar_new = (rhs_u + alpha * rhs_v) / denom * mask
        V_bar_new = (rhs_v - alpha * rhs_u) / denom * mask

        # Optional Laplacian damping
        if config.barotropic_diffusion_alpha > 0.0:
            eta_new = (
                eta_new + nu_dt * laplacian_latlon(eta_new, grid, mask=mask).astype(eta.dtype)
            ) * mask
            eta_new = jnp.maximum(eta_new, eta_floor) * mask
            U_bar_new = (
                U_bar_new + nu_dt * laplacian_latlon(U_bar_new, grid, mask=mask).astype(eta.dtype)
            ) * mask
            V_bar_new = (
                V_bar_new + nu_dt * laplacian_latlon(V_bar_new, grid, mask=mask).astype(eta.dtype)
            ) * mask

        return (eta_new, U_bar_new, V_bar_new)

    if config.differentiable_barotropic:
        def scan_body(carry, _):
            new_carry = substep_body(0, carry)
            return new_carry, None

        (eta_f, U_bar_f, V_bar_f), _ = jax.lax.scan(
            scan_body, (eta, U_bar, V_bar), xs=None, length=n_substeps,
        )
    else:
        eta_f, U_bar_f, V_bar_f = jax.lax.fori_loop(
            0, n_substeps, substep_body, (eta, U_bar, V_bar),
        )

    # Correct 3D velocities: preserve baroclinic structure
    u_baro_prime = u - U_bar[..., jnp.newaxis]
    v_baro_prime = v - V_bar[..., jnp.newaxis]
    u_new = (u_baro_prime + U_bar_f[..., jnp.newaxis]) * mask[..., jnp.newaxis]
    v_new = (v_baro_prime + V_bar_f[..., jnp.newaxis]) * mask[..., jnp.newaxis]

    return state._replace(
        eta=state.eta.replace(data=eta_f),
        u=state.u.replace(data=u_new),
        v=state.v.replace(data=v_new),
    )
