"""C-grid barotropic solver for the cubed-sphere ocean model.

Replaces the A-grid barotropic solver when ``OceanConfig.barotropic_staggering
== "c_grid"``.  Uses the C-grid infrastructure from ``operators_cdgrid.py``
and ``cubed_sphere_cdgrid.py`` to eliminate the 2*dx checkerboard null space
inherent in the A-grid formulation.

Forward-backward (Matsuno) substeps:

    d(eta)/dt = -div(H * u_c, H * v_c)             [cell centres]
    d(u_c)/dt = f * v_at_u - g * d(eta)/dx          [u-edge midpoints]
    d(v_c)/dt = -f * u_at_v - g * d(eta)/dy         [v-edge midpoints]

Coriolis is computed at cell centres (where halo exchange is well-tested)
and projected to C-grid edges via ``fv3_cc2c`` (vector halo exchange with
non-orthogonality correction), avoiding cross-face staggered interpolation.

External API is cell-centre: A-grid velocities are projected to C-grid
on entry and back to cell centres on exit.  No changes to OceanState,
physics, coupler, or ML interfaces.

See issue #182 for motivation and design rationale.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.core.operators_cdgrid import cgrid_divergence, cgrid_gradient_2d, fv3_cc2c
from legoesm.core.precision import cast
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.grids.cubed_sphere_cdgrid import CubedSphereCDGrid
from legoesm.grids.halo import pad_halo
from legoesm.ocean.vertical import OceanZStarCoordinate, compute_layer_thickness
from legoesm.ocean.state import OceanState, OceanConfig
from legoesm.ocean.dynamics.eta_floor import clamp_and_redistribute as _clamp_redistribute


# ==============================================================================
# Edge mask derivation
# ==============================================================================

def _derive_edge_masks(mask, grid):
    """Derive C-grid edge masks from cell-centre land mask.

    A u-edge (between cells i-1 and i) is ocean if both flanking cells
    are ocean.  Same logic for v-edges.

    Parameters
    ----------
    mask : jax.Array, shape (6, n, n)
        Cell-centre land mask (1 = ocean, 0 = land).
    grid : CubedSphereGrid

    Returns
    -------
    u_mask : jax.Array, shape (6, n+1, n)
    v_mask : jax.Array, shape (6, n, n+1)
    """
    mask_pad = pad_halo(mask, interp_offsets=None)  # (6, n+2, n+2)
    # u-edge at (i, j): between cell (i-1, j) and cell (i, j)
    u_mask = mask_pad[:, :-1, 1:-1] * mask_pad[:, 1:, 1:-1]   # (6, n+1, n)
    # v-edge at (i, j): between cell (i, j-1) and cell (i, j)
    v_mask = mask_pad[:, 1:-1, :-1] * mask_pad[:, 1:-1, 1:]   # (6, n, n+1)
    return u_mask, v_mask


# ==============================================================================
# C-grid ↔ cell-centre velocity helpers
# ==============================================================================

def _cgrid_to_cell_centre(u_c, v_c):
    """Average C-grid edge-normal velocities back to cell centres.

    Simple 2nd-order average of flanking edge values.  Used for
    reconstructing cell-centre velocity from C-grid for Coriolis
    computation and for the final 3D velocity correction.

    Parameters
    ----------
    u_c : jax.Array, shape (6, n+1, n)
    v_c : jax.Array, shape (6, n, n+1)

    Returns
    -------
    u_cc, v_cc : jax.Array, shape (6, n, n)
    """
    u_cc = 0.5 * (u_c[:, :-1] + u_c[:, 1:])
    v_cc = 0.5 * (v_c[:, :, :-1] + v_c[:, :, 1:])
    return u_cc, v_cc


# ==============================================================================
# Flux-form diffusion on eta (conservative)
# ==============================================================================

def _diffuse_eta_flux_form(eta, u_mask, v_mask, nu_dt_u, nu_dt_v, cdgrid):
    """Conservative flux-form Laplacian diffusion on eta.

    div(nu * grad(eta)) computed with C-grid compact operators so that
    the divergence theorem is exactly satisfied (sum of div*area = 0).

    Parameters
    ----------
    eta : shape (6, n, n)
    u_mask, v_mask : edge masks
    nu_dt_u : shape (6, n+1, n)
    nu_dt_v : shape (6, n, n+1)
    cdgrid : CubedSphereCDGrid

    Returns
    -------
    eta_diffused : shape (6, n, n)
    """
    deta_dx, deta_dy = cgrid_gradient_2d(eta, cdgrid)
    flux_x = nu_dt_u * deta_dx * u_mask
    flux_y = nu_dt_v * deta_dy * v_mask
    return eta + cgrid_divergence(flux_x, flux_y, cdgrid)


# ==============================================================================
# C-grid barotropic substeps
# ==============================================================================

def barotropic_substeps_cgrid(
    state: OceanState,
    dt_s: float,
    n_substeps: int,
    grid: CubedSphereGrid,
    cdgrid: CubedSphereCDGrid,
    z_coord: OceanZStarCoordinate,
    config: OceanConfig,
) -> OceanState:
    """Run barotropic substeps on the cubed-sphere C-grid.

    Forward-backward (Matsuno) time stepping with C-grid staggering:
      1. Forward: update eta from C-grid flux divergence
      2. Backward: update u_c, v_c with compact PGF + Coriolis

    Coriolis is evaluated at cell centres and projected to edges via
    ``fv3_cc2c`` (vector halo exchange + non-orthogonality correction),
    avoiding cross-face staggered velocity interpolation.

    Cell-centre API: converts A-grid -> C-grid on entry, C-grid -> A-grid
    on exit.  No changes to OceanState or external interfaces.

    Parameters
    ----------
    state : OceanState
    dt_s : float
        Substep size [seconds].
    n_substeps : int
    grid : CubedSphereGrid
    cdgrid : CubedSphereCDGrid
    z_coord : OceanZStarCoordinate
    config : OceanConfig

    Returns
    -------
    OceanState with updated eta and velocity.
    """
    # --- Precision management ---
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
    _area = grid.area

    # --- Derive edge masks ---
    u_mask, v_mask = _derive_edge_masks(mask, grid)

    # --- Compute depth-averaged velocity at cell centres ---
    h_k = compute_layer_thickness(
        eta, H_bathy, z_coord, min_water_column_m=config.min_water_column_m,
    )
    H_total = jnp.maximum(jnp.sum(h_k, axis=-1), min_water_col)
    U_bar_cc = jnp.sum(u * h_k, axis=-1) / H_total * mask
    V_bar_cc = jnp.sum(v * h_k, axis=-1) / H_total * mask

    # --- Project to C-grid via vector halo exchange ---
    U_bar, V_bar = fv3_cc2c(U_bar_cc, V_bar_cc, cdgrid)
    U_bar = U_bar * u_mask
    V_bar = V_bar * v_mask

    # --- Coriolis parameter at cell centres ---
    f_cc = grid.f.astype(eta.dtype)  # (6, n, n)

    # --- Barotropic diffusion coefficients ---
    baro_alpha = jnp.asarray(
        config.barotropic_diffusion_alpha, dtype=eta.dtype,
    ) * (dt_s / jnp.asarray(config.barotropic_diffusion_dt_ref, dtype=eta.dtype))

    if config.barotropic_diffusion_alpha > 0.0:
        # Face-centred diffusion coefficients from adjacent cell areas
        area = grid.area.astype(eta.dtype)
        area_pad = pad_halo(area, interp_offsets=None)
        nu_dt_u = baro_alpha * 0.5 * (
            area_pad[:, :-1, 1:-1] + area_pad[:, 1:, 1:-1]
        )  # (6, n+1, n)
        nu_dt_v = baro_alpha * 0.5 * (
            area_pad[:, 1:-1, :-1] + area_pad[:, 1:-1, 1:]
        )  # (6, n, n+1)

    # --- Forward-backward substep body ---
    def substep_body(i, carry):
        eta_c, U_bar_c, V_bar_c = carry

        H_total_c = jnp.maximum(eta_c + H_bathy, min_water_col) * mask

        # -- Forward: update eta from C-grid continuity --
        H_pad = pad_halo(H_total_c, interp_offsets=grid.halo_interp_offsets)
        H_u = 0.5 * (H_pad[:, :-1, 1:-1] + H_pad[:, 1:, 1:-1])  # (6, n+1, n)
        H_v = 0.5 * (H_pad[:, 1:-1, :-1] + H_pad[:, 1:-1, 1:])  # (6, n, n+1)

        flux_u = H_u * U_bar_c * u_mask
        flux_v = H_v * V_bar_c * v_mask

        div_flux = cgrid_divergence(flux_u, flux_v, cdgrid).astype(eta.dtype)
        eta_unfloored = (eta_c - dt_s * div_flux) * mask
        eta_new = _clamp_redistribute(eta_unfloored, eta_floor, mask, _area)

        # -- Backward: update velocity with compact PGF + Coriolis --
        deta_dx, deta_dy = cgrid_gradient_2d(eta_new, cdgrid)
        deta_dx = deta_dx.astype(eta.dtype)
        deta_dy = deta_dy.astype(eta.dtype)

        # Coriolis at cell centres from C-grid velocities, then project
        # to edges via fv3_cc2c (vector halo + non-orthogonality).
        # f*v goes into the u-equation, -f*u into the v-equation.
        u_cc, v_cc = _cgrid_to_cell_centre(U_bar_c, V_bar_c)
        cor_u_cc = f_cc * v_cc * mask    # (6, n, n)
        cor_v_cc = -f_cc * u_cc * mask   # (6, n, n)
        cor_u, cor_v = fv3_cc2c(cor_u_cc, cor_v_cc, cdgrid)

        # Matsuno step 1: update u with old v (forward)
        U_bar_new = (U_bar_c + dt_s * (cor_u - g * deta_dx)) * u_mask

        # Matsuno step 2: update v with NEW u (backward)
        # Recompute full Coriolis vector with updated U but old V, then
        # project via fv3_cc2c.  Both components are needed for correct
        # vector rotation at face boundaries.
        u_cc_new, _ = _cgrid_to_cell_centre(U_bar_new, V_bar_c)
        cor_u_step2_cc = f_cc * v_cc * mask       # u-tendency unchanged (old V)
        cor_v_step2_cc = -f_cc * u_cc_new * mask   # v-tendency from new U
        _, cor_v_new = fv3_cc2c(cor_u_step2_cc, cor_v_step2_cc, cdgrid)
        V_bar_new = (V_bar_c + dt_s * (cor_v_new - g * deta_dy)) * v_mask

        # -- Diffusion on eta only (flux-form, conservative) --
        if config.barotropic_diffusion_alpha > 0.0:
            eta_new = _diffuse_eta_flux_form(
                eta_new, u_mask, v_mask, nu_dt_u, nu_dt_v, cdgrid,
            ) * mask
            eta_new = _clamp_redistribute(eta_new, eta_floor, mask, _area)

        return (eta_new, U_bar_new, V_bar_new)

    # --- Time integration ---
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

    # --- Convert C-grid velocity back to cell centres ---
    U_bar_cc_new, V_bar_cc_new = _cgrid_to_cell_centre(U_bar_f, V_bar_f)
    U_bar_cc_new = U_bar_cc_new * mask
    V_bar_cc_new = V_bar_cc_new * mask

    # --- Correct 3D velocities: preserve baroclinic structure ---
    u_baro_prime = u - U_bar_cc[..., jnp.newaxis]
    v_baro_prime = v - V_bar_cc[..., jnp.newaxis]
    u_new = (u_baro_prime + U_bar_cc_new[..., jnp.newaxis]) * mask[..., jnp.newaxis]
    v_new = (v_baro_prime + V_bar_cc_new[..., jnp.newaxis]) * mask[..., jnp.newaxis]

    # Downcast to storage precision
    eta_f = cast(eta_f, _M, "storage")
    u_new = cast(u_new, _M, "storage")
    v_new = cast(v_new, _M, "storage")

    return state._replace(
        eta=state.eta.replace(data=eta_f),
        u=state.u.replace(data=u_new),
        v=state.v.replace(data=v_new),
    )
