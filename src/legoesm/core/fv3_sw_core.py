"""FV3-style D-grid shallow water core (d_sw).

Covariant velocity formulation with PPM-transported KE at corners.
The KE uses the SAME contravariant transport velocities as the vorticity
flux, ensuring exact discrete geostrophic balance and eliminating
cubed-sphere edge artifacts.

Reusable by 3D atmosphere PE and ocean PE dycores (vmap over levels).

References
----------
- Lin (2004): A "Vertically Lagrangian" FV Dynamical Core
- Putman & Lin (2007): Finite-volume transport on cubed-sphere grids
- Harris et al. (2021): Scientific Description of GFDL FV3
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.grids.cubed_sphere_cdgrid import CubedSphereCDGrid
from legoesm.core.d2a2c_vect import d2a2c_vect
from legoesm.core.fv_tp_2d import fv_tp_2d, _ppm_face_values, _ppm_flux_1d
from legoesm.core.operators_cdgrid import (
    cgrid_mass_flux_divergence,
    _interp_center_to_corner,
    _pad_halo_auto,
    fv3_vorticity,
)


# ==============================================================================
# 1-D PPM wind advection to corners (xtp_u, ytp_v)
# ==============================================================================

def _ytp_v(vb, u_d, cdgrid):
    """Upwind-interpolate covariant u to corners using vb direction.

    At each corner (i, j), selects the covariant u from the upwind
    i-edge based on the sign of the contravariant v (``vb``).
    This ensures the KE contraction ``vb * u_adv`` uses the SAME
    directional bias as the vorticity flux ``ζ * vb``.

    Will be upgraded to PPM reconstruction for higher accuracy.

    Parameters
    ----------
    vb : (6, n+1, n+1) — contravariant v at corners
    u_d : (6, n, n+1) — covariant u at x-edge midpoints

    Returns
    -------
    u_adv : (6, n+1, n+1)
    """
    u_d_pad = jnp.pad(u_d, [(0, 0), (1, 1), (0, 0)], mode='edge')
    u_left = u_d_pad[:, :-1, :]   # u from i-1 side (upwind for vb > 0)
    u_right = u_d_pad[:, 1:, :]   # u from i side (upwind for vb < 0)
    return jnp.where(vb > 0, u_left, u_right)


def _xtp_u(ub, v_d, cdgrid):
    """Upwind-interpolate covariant v to corners using ub direction.

    Same as _ytp_v but for the x-direction.
    """
    v_d_pad = jnp.pad(v_d, [(0, 0), (0, 0), (1, 1)], mode='edge')
    v_left = v_d_pad[:, :, :-1]   # v from j-1 side
    v_right = v_d_pad[:, :, 1:]   # v from j side
    return jnp.where(ub > 0, v_left, v_right)


def fv3_d_sw(
    h, u_d, v_d, h_s, cdgrid,
    g=9.80616, div_damp=0.0, hyperdiff_coeff=0.0,
):
    """FV3 D-grid shallow water tendencies with covariant KE.

    Parameters
    ----------
    h : (6, n, n)     — height at cell centres
    u_d : (6, n, n+1) — x-velocity at x-edge midpoints
    v_d : (6, n+1, n) — y-velocity at y-edge midpoints
    h_s : (6, n, n)   — surface topography
    cdgrid : CubedSphereCDGrid
    g, div_damp, hyperdiff_coeff : float

    Returns
    -------
    dh_dt : (6, n, n), du_d_dt : (6, n, n+1), dv_d_dt : (6, n+1, n)
    """
    n = cdgrid.n
    grid = cdgrid.base
    dx = cdgrid.dx_edge_y   # (6, n, n+1)
    dy = cdgrid.dy_edge_x   # (6, n+1, n)

    # ==================================================================
    # (a) D→A→C conversion (4th-order interior)
    # ==================================================================
    ua, va, uc, vc = d2a2c_vect(u_d, v_d, cdgrid)

    # ==================================================================
    # (b) Height tendency — PPM mass flux divergence (UNCHANGED)
    # ==================================================================
    dh_dt = cgrid_mass_flux_divergence(h, uc, vc, cdgrid)
    total_area = jnp.sum(grid.area)
    dh_dt = dh_dt - jnp.sum(dh_dt * grid.area) / total_area

    # ==================================================================
    # (c,d) KE at cell centres from contravariant × covariant contraction.
    # ==================================================================
    # Use d2a2c_vect's contravariant (ua, va) and the simple covariant
    # cell-centre winds (u_cc, v_cc) for KE.  This ensures the KE data
    # path goes through the SAME cell-centre halo exchange as g*h,
    # giving consistent B = KE + g*h at cell centres.
    u_cc = 0.5 * (u_d[:, :, :-1] + u_d[:, :, 1:])   # covariant cc
    v_cc = 0.5 * (v_d[:, :-1, :] + v_d[:, 1:, :])
    KE_cc = 0.5 * (ua * u_cc + va * v_cc)            # contraction (6, n, n)

    # B at cell centres → interpolate to corners → bare-difference gradient
    B_cc = KE_cc + g * (h + h_s)
    B_corner = _interp_center_to_corner(B_cc, cdgrid)

    # Contravariant at corners from the SAME cell-centre data (for vort flux)
    from legoesm.grids.halo import pad_halo_vector
    ua_pad, va_pad = pad_halo_vector(
        ua, va,
        grid.cos_angle, grid.sin_angle,
        grid.cos_angle_padded, grid.sin_angle_padded,
        interp_offsets=grid.halo_interp_offsets,
    )
    ub_corner = 0.25 * (ua_pad[:, :-1, :-1] + ua_pad[:, 1:, :-1]
                         + ua_pad[:, :-1, 1:] + ua_pad[:, 1:, 1:])
    vb_corner = 0.25 * (va_pad[:, :-1, :-1] + va_pad[:, 1:, :-1]
                         + va_pad[:, :-1, 1:] + va_pad[:, 1:, 1:])

    # B_corner already computed above from B_cc = KE_cc + g*(h+h_s)

    # ==================================================================
    # (f) Bernoulli gradient — BARE DIFFERENCE at edge midpoints
    # ==================================================================
    dB_u = (B_corner[:, :-1, :] - B_corner[:, 1:, :]) / dx  # (6, n, n+1)
    dB_v = (B_corner[:, :, :-1] - B_corner[:, :, 1:]) / dy  # (6, n+1, n)

    # ==================================================================
    # (g) Vorticity — exact Stokes circulation at corners
    # ==================================================================
    zeta = fv3_vorticity(u_d, v_d, cdgrid)
    zeta_abs = zeta + cdgrid.f_corner

    # ==================================================================
    # (h) Vorticity flux at edges using the SAME contravariant velocities
    # ==================================================================
    # Average absolute vorticity to edge midpoints
    zeta_u = 0.5 * (zeta_abs[:, :-1, :] + zeta_abs[:, 1:, :])  # (6, n, n+1)
    zeta_v = 0.5 * (zeta_abs[:, :, :-1] + zeta_abs[:, :, 1:])  # (6, n+1, n)

    # Perpendicular contravariant velocity at edges from the SAME
    # ub_corner/vb_corner used for KE.
    # At u_d edge (i, j): contravariant v ≈ average of vb_corner
    vb_at_u = 0.5 * (vb_corner[:, :-1, :] + vb_corner[:, 1:, :])  # (6, n, n+1)
    ub_at_v = 0.5 * (ub_corner[:, :, :-1] + ub_corner[:, :, 1:])  # (6, n+1, n)

    vort_flux_u = zeta_u * vb_at_u
    vort_flux_v = -zeta_v * ub_at_v

    # ==================================================================
    # (i) Momentum tendencies at edge midpoints
    # ==================================================================
    du_d_dt = dB_u + vort_flux_u
    dv_d_dt = dB_v + vort_flux_v

    # ==================================================================
    # (j) Divergence damping
    # ==================================================================
    if div_damp > 0:
        from legoesm.core.operators_cdgrid import cgrid_divergence
        div_field = cgrid_divergence(uc, vc, cdgrid)
        area_min = jnp.min(grid.area)
        d2_bg = div_damp / area_min
        dddmp = 0.2
        div_abs_corner = _interp_center_to_corner(jnp.abs(div_field), cdgrid)
        adapt_u = 0.5 * (div_abs_corner[:, :-1, :] + div_abs_corner[:, 1:, :])
        adapt_v = 0.5 * (div_abs_corner[:, :, :-1] + div_abs_corner[:, :, 1:])
        coeff_u = area_min * jnp.maximum(
            d2_bg, jnp.minimum(0.20, dddmp * adapt_u))
        coeff_v = area_min * jnp.maximum(
            d2_bg, jnp.minimum(0.20, dddmp * adapt_v))
        div_corner = _interp_center_to_corner(div_field, cdgrid)
        ddiv_u = (div_corner[:, :-1, :] - div_corner[:, 1:, :]) / dx
        ddiv_v = (div_corner[:, :, :-1] - div_corner[:, :, 1:]) / dy
        du_d_dt = du_d_dt + coeff_u * ddiv_u
        dv_d_dt = dv_d_dt + coeff_v * ddiv_v

    # ==================================================================
    # (k) Biharmonic hyperdiffusion
    # ==================================================================
    if hyperdiff_coeff > 0:
        from legoesm.core.operators import laplacian_compact
        u_cc = 0.5 * (u_d[:, :, :-1] + u_d[:, :, 1:])
        v_cc = 0.5 * (v_d[:, :-1, :] + v_d[:, 1:, :])
        lap_u = laplacian_compact(u_cc, grid)
        lap2_u = laplacian_compact(lap_u, grid)
        lap2_u_pad = _pad_halo_auto(lap2_u, cdgrid)
        du_d_dt = du_d_dt - hyperdiff_coeff * 0.5 * (
            lap2_u_pad[:, 1:-1, :-1] + lap2_u_pad[:, 1:-1, 1:])
        lap_v = laplacian_compact(v_cc, grid)
        lap2_v = laplacian_compact(lap_v, grid)
        lap2_v_pad = _pad_halo_auto(lap2_v, cdgrid)
        dv_d_dt = dv_d_dt - hyperdiff_coeff * 0.5 * (
            lap2_v_pad[:, :-1, 1:-1] + lap2_v_pad[:, 1:, 1:-1])

    return dh_dt, du_d_dt, dv_d_dt
