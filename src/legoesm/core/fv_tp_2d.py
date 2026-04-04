"""FV3-faithful 2D finite-volume transport on the cubed sphere.

Implements Putman & Lin (2007) Lin-Rood transport with monotone PPM
(hord=8) and Courant-number flux integration.  Face-boundary edges
use position-aware weighted averages derived from the known halo
interpolation offsets, replacing the equal-spacing assumption.

References
----------
- Putman & Lin (2007): Finite-volume transport on various cubed-sphere grids
- Lin (2004): A "Vertically Lagrangian" FV Dynamical Core
- Colella & Woodward (1984): The Piecewise Parabolic Method (PPM)
- GFDL tp_core.F90
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.grids.cubed_sphere_cdgrid import CubedSphereCDGrid
from legoesm.grids.halo import pad_halo

_R3 = 1.0 / 3.0


def _pert_ppm(bl, br):
    """FV3 pert_ppm iv=1: monotonicity at face boundaries."""
    is_ext = bl * br >= 0.0
    da1 = bl - br
    da2 = da1 ** 2
    a6da = 3.0 * (bl + br) * da1
    bl_out = jnp.where(a6da < -da2, -2.0 * br, bl)
    br_out = jnp.where(a6da > da2, -2.0 * bl, br)
    bl_out = jnp.where(is_ext, 0.0, bl_out)
    br_out = jnp.where(is_ext, 0.0, br_out)
    return bl_out, br_out


def _ppm_1d(q, n, off_left=None, off_right=None,
            off_left_d1=None, off_right_d1=None):
    """PPM bl/br along axis=1 with hord=8 + position-aware boundaries.

    Parameters
    ----------
    q : (6, n+4, M)  field with halo=2 in sweep direction
    n : int           number of interior cells
    off_left : (6, M) or None — halo interp offset at left boundary (depth=0)
    off_right : (6, M) or None — halo interp offset at right boundary (depth=0)
    off_left_d1 : (6, M) or None — depth=1 offset (outer halo)
    off_right_d1 : (6, M) or None — depth=1 offset (outer halo)

    Returns
    -------
    bl, br   : (6, n+2, M)
    q_cells  : (6, n+2, M)
    """
    qe = jnp.pad(q, [(0, 0), (1, 1), (0, 0)], mode='edge')  # (6, n+6, M)

    # Monotone slopes
    xt = 0.25 * (qe[:, 2:, :] - qe[:, :-2, :])
    qm = qe[:, 1:-1, :]
    q_hi = jnp.maximum(jnp.maximum(qe[:, :-2, :], qm), qe[:, 2:, :])
    q_lo = jnp.minimum(jnp.minimum(qe[:, :-2, :], qm), qe[:, 2:, :])
    dm = jnp.sign(xt) * jnp.minimum(
        jnp.abs(xt), jnp.minimum(q_hi - qm, qm - q_lo))
    # dm[:, k, :] = slope at qe cell (k+1)

    # Correct dm at boundary cells for non-uniform halo spacing.
    # Standard dm = 0.25*(q[i+1]-q[i-1]) assumes span=2.
    # Actual span at boundary cells differs by halo offsets.
    def _correct_dm(dm_arr, idx, q_hi_arr, q_lo_arr, qm_arr, scale):
        """Correct dm at index idx by scale factor, re-apply monotone limit."""
        dm_scaled = dm_arr[:, idx, :] * scale
        pmp = q_hi_arr[:, idx, :] - qm_arr[:, idx, :]
        pmm = qm_arr[:, idx, :] - q_lo_arr[:, idx, :]
        dm_lim = jnp.sign(dm_scaled) * jnp.minimum(
            jnp.abs(dm_scaled), jnp.minimum(pmp, pmm))
        return dm_arr.at[:, idx, :].set(dm_lim)

    if off_left is not None:
        # dm at halo cell -1 (dm index 1): spans from halo(-2) to interior(0)
        if off_left_d1 is not None:
            span_halo = 2.0 - off_left_d1 + off_left
            dm = _correct_dm(dm, 1, q_hi, q_lo, qm,
                             2.0 / jnp.maximum(span_halo, 0.5))
        # dm at first interior cell 0 (dm index 2): spans from halo(-1) to interior(1)
        # halo(-1) is at position (-1+off0), interior(1) at position 1 → span = 2-off0
        span_int0 = 2.0 - off_left
        dm = _correct_dm(dm, 2, q_hi, q_lo, qm,
                         2.0 / jnp.maximum(span_int0, 0.5))

    if off_right is not None:
        # dm at halo cell n (dm index n+2): spans from interior(n-1) to halo(n+1)
        if off_right_d1 is not None:
            span_halo_r = 2.0 + off_right - off_right_d1
            dm = _correct_dm(dm, n + 2, q_hi, q_lo, qm,
                             2.0 / jnp.maximum(span_halo_r, 0.5))
        # dm at last interior cell n-1 (dm index n+1): spans from interior(n-2) to halo(n)
        # halo(n) at position (n+off0), interior(n-2) at n-2 → span = 2+off0
        span_int_nm1 = 2.0 + off_right
        dm = _correct_dm(dm, n + 1, q_hi, q_lo, qm,
                         2.0 / jnp.maximum(span_int_nm1, 0.5))

    # Edge values al (dm-corrected)
    al = (0.5 * (qe[:, 1:-2, :] + qe[:, 2:-1, :])
          + _R3 * (dm[:, :-1, :] - dm[:, 1:, :]))

    # Position-aware correction at face boundary EDGES:
    # The halo cell is at position (-1 + offset), not -1.
    # The face boundary edge is at position -0.5.
    # Correct edge value using actual distances to the boundary.
    if off_left is not None:
        # Left face-boundary edge: al[:, 1, :] between halo(-1) and interior(0)
        q_hm1 = qe[:, 2, :]   # halo -1 at position (-1 + off0)
        q_i0 = qe[:, 3, :]    # interior 0 at position 0
        off0 = off_left
        h_L = jnp.maximum(0.5 - off0, 0.01)
        al_L0 = (0.5 * q_hm1 + h_L * q_i0) / (h_L + 0.5)
        al_L0 = jnp.clip(al_L0, jnp.minimum(q_hm1, q_i0),
                          jnp.maximum(q_hm1, q_i0))
        al = al.at[:, 1, :].set(al_L0)

        # Second edge: al[:, 0, :] between halo(-2) and halo(-1)
        if off_left_d1 is not None:
            q_hm2 = qe[:, 1, :]   # halo -2 at position (-2 + off1)
            off1 = off_left_d1
            h_outer = jnp.maximum(0.5 - off1, 0.01)  # dist from halo-2 to midpoint
            h_inner = jnp.maximum(0.5 + off0, 0.01)   # dist from midpoint to halo-1
            al_L1 = (h_inner * q_hm2 + h_outer * q_hm1) / (h_outer + h_inner)
            al_L1 = jnp.clip(al_L1, jnp.minimum(q_hm2, q_hm1),
                              jnp.maximum(q_hm2, q_hm1))
            al = al.at[:, 0, :].set(al_L1)

    if off_right is not None:
        # Right face-boundary edge: al[:, n+1, :] between interior(n-1) and halo(n)
        q_inm1 = qe[:, n + 2, :]  # interior n-1
        q_hn = qe[:, n + 3, :]    # halo n at position (n + off0)
        off0r = off_right
        h_R = jnp.maximum(0.5 + off0r, 0.01)
        al_R0 = (h_R * q_inm1 + 0.5 * q_hn) / (0.5 + h_R)
        al_R0 = jnp.clip(al_R0, jnp.minimum(q_inm1, q_hn),
                          jnp.maximum(q_inm1, q_hn))
        al = al.at[:, n + 1, :].set(al_R0)

        # Second edge: al[:, n+2, :] between halo(n) and halo(n+1)
        if off_right_d1 is not None:
            q_hnp1 = qe[:, n + 4, :]
            off1r = off_right_d1
            h_inner = jnp.maximum(0.5 - off0r, 0.01)
            h_outer = jnp.maximum(0.5 + off1r, 0.01)
            al_R1 = (h_outer * q_hn + h_inner * q_hnp1) / (h_inner + h_outer)
            al_R1 = jnp.clip(al_R1, jnp.minimum(q_hn, q_hnp1),
                              jnp.maximum(q_hn, q_hnp1))
            al = al.at[:, n + 2, :].set(al_R1)

    # hord=8 fast monotone limiter
    dm_c = dm[:, 1:-1, :]
    q_c = qe[:, 2:-2, :]
    al_L = al[:, :-1, :]
    al_R = al[:, 1:, :]
    two_dm = 2.0 * dm_c
    bl = -jnp.sign(two_dm) * jnp.minimum(jnp.abs(two_dm), jnp.abs(al_L - q_c))
    br = jnp.sign(two_dm) * jnp.minimum(jnp.abs(two_dm), jnp.abs(al_R - q_c))

    # pert_ppm at boundary cells
    for k in [0, 1, 2, -3, -2, -1]:
        bl_k, br_k = _pert_ppm(bl[:, k, :], br[:, k, :])
        bl = bl.at[:, k, :].set(bl_k)
        br = br.at[:, k, :].set(br_k)

    return bl, br, q_c


def _xppm(q_h2, crx, n, off_left=None, off_right=None,
          off_left_d1=None, off_right_d1=None):
    """PPM in x with hord=8 Courant-number integration."""
    bl, br, q_c = _ppm_1d(q_h2, n, off_left, off_right,
                           off_left_d1, off_right_d1)
    bl_L, br_L, q_L = bl[:, :n+1, :], br[:, :n+1, :], q_c[:, :n+1, :]
    bl_R, br_R, q_R = bl[:, 1:n+2, :], br[:, 1:n+2, :], q_c[:, 1:n+2, :]

    # Correct Courant number at face boundaries
    crx_adj = crx
    if off_left is not None:
        crx_adj = crx_adj.at[:, 0, :].set(
            crx[:, 0, :] / jnp.maximum(1.0 - off_left, 0.3))
    if off_right is not None:
        crx_adj = crx_adj.at[:, n, :].set(
            crx[:, n, :] / jnp.maximum(1.0 + off_right, 0.3))

    fx_pos = q_L + (1.0 - crx_adj) * (br_L - crx_adj * (bl_L + br_L))
    fx_neg = q_R + (1.0 + crx_adj) * (bl_R + crx_adj * (bl_R + br_R))
    return jnp.where(crx_adj > 0, fx_pos, fx_neg)


def _yppm(q_h2, cry, n, off_left=None, off_right=None,
          off_left_d1=None, off_right_d1=None):
    """PPM in y with hord=8 Courant-number integration."""
    q_t = jnp.swapaxes(q_h2, 1, 2)
    c_t = jnp.swapaxes(cry, 1, 2)
    bl, br, q_c = _ppm_1d(q_t, n, off_left, off_right,
                           off_left_d1, off_right_d1)
    bl_L, br_L, q_L = bl[:, :n+1, :], br[:, :n+1, :], q_c[:, :n+1, :]
    bl_R, br_R, q_R = bl[:, 1:n+2, :], br[:, 1:n+2, :], q_c[:, 1:n+2, :]
    fy_pos = q_L + (1.0 - c_t) * (br_L - c_t * (bl_L + br_L))
    fy_neg = q_R + (1.0 + c_t) * (bl_R + c_t * (bl_R + br_R))
    return jnp.swapaxes(jnp.where(c_t > 0, fy_pos, fy_neg), 1, 2)


# =========================================================================
# Transport quantities
# =========================================================================

def compute_transport_quantities(ut, vt, dt, cdgrid):
    """Compute Courant numbers, area fluxes, swept areas."""
    n = cdgrid.n
    grid = cdgrid.base
    sin_sg = cdgrid.sin_sg

    crx = dt * ut * cdgrid.rdxc
    cry = dt * vt * cdgrid.rdyc

    dy = cdgrid.dy_edge_x
    dx = cdgrid.dx_edge_y

    sin_east = sin_sg[:, :, :, 2]
    sin_west = sin_sg[:, :, :, 0]
    se_pad = pad_halo(sin_east, interp_offsets=grid.halo_interp_offsets)
    sw_pad = pad_halo(sin_west, interp_offsets=grid.halo_interp_offsets)
    sin_x = jnp.where(ut > 0, se_pad[:, :n+1, 1:-1], sw_pad[:, 1:n+2, 1:-1])
    xfx = dt * ut * dy * sin_x

    sin_north = sin_sg[:, :, :, 3]
    sin_south = sin_sg[:, :, :, 1]
    sn_pad = pad_halo(sin_north, interp_offsets=grid.halo_interp_offsets)
    ss_pad = pad_halo(sin_south, interp_offsets=grid.halo_interp_offsets)
    sin_y = jnp.where(vt > 0, sn_pad[:, 1:-1, :n+1], ss_pad[:, 1:-1, 1:n+2])
    yfx = dt * vt * dx * sin_y

    area = grid.area
    ra_x = area + xfx[:, :-1, :] - xfx[:, 1:, :]
    ra_y = area + yfx[:, :, :-1] - yfx[:, :, 1:]
    return crx, cry, xfx, yfx, ra_x, ra_y


# =========================================================================
# fv_tp_2d: Lin-Rood 2D transport
# =========================================================================

def fv_tp_2d(q, crx, cry, xfx, yfx, ra_x, ra_y, cdgrid):
    """Lin-Rood operator-split 2D transport (Putman & Lin 2007)."""
    n = cdgrid.n
    grid = cdgrid.base
    area = grid.area
    offsets_h2 = grid.halo_interp_offsets_h2

    # Extract boundary offsets for sweep directions
    # offsets_h2: (6, 4, 2, n) — [face, edge, depth, cell_along_edge]
    # WEST=0, EAST=1, SOUTH=2, NORTH=3; depth 0 = adjacent to interior
    ox_L0 = offsets_h2[:, 0, 0, :]   # WEST depth=0
    ox_R0 = offsets_h2[:, 1, 0, :]   # EAST depth=0
    oy_L0 = offsets_h2[:, 2, 0, :]   # SOUTH depth=0
    oy_R0 = offsets_h2[:, 3, 0, :]   # NORTH depth=0
    ox_L1 = offsets_h2[:, 0, 1, :]   # WEST depth=1
    ox_R1 = offsets_h2[:, 1, 1, :]   # EAST depth=1
    oy_L1 = offsets_h2[:, 2, 1, :]   # SOUTH depth=1
    oy_R1 = offsets_h2[:, 3, 1, :]   # NORTH depth=1

    q_full = pad_halo(q, halo=2, interp_offsets=offsets_h2)

    # Pass 1: Y-sweep on q, X-sweep on cross-corrected q_i
    fy2 = _yppm(q_full[:, 2:-2, :], cry, n, oy_L0, oy_R0, oy_L1, oy_R1)
    fyy = yfx * fy2
    q_i = (q * area + fyy[:, :, :-1] - fyy[:, :, 1:]) / ra_y

    # Reuse q's x-halo for q_i instead of a separate halo exchange.
    # This avoids the interpolation errors from an additional exchange.
    # The x-halo of q_i differs from q's by O(dt) — acceptable.
    q_i_x = q_full[:, :, 2:-2].at[:, 2:-2, :].set(q_i)
    fx1 = _xppm(q_i_x, crx, n, ox_L0, ox_R0, ox_L1, ox_R1)

    # Pass 2: X-sweep on q, Y-sweep on cross-corrected q_j
    fx2 = _xppm(q_full[:, :, 2:-2], crx, n, ox_L0, ox_R0, ox_L1, ox_R1)
    fxx = xfx * fx2
    q_j = (q * area + fxx[:, :-1, :] - fxx[:, 1:, :]) / ra_x

    # Reuse q's y-halo for q_j
    q_j_y = q_full[:, 2:-2, :].at[:, :, 2:-2].set(q_j)
    fy1 = _yppm(q_j_y, cry, n, oy_L0, oy_R0, oy_L1, oy_R1)

    fx = 0.5 * (fx1 + fx2) * xfx
    fy = 0.5 * (fy1 + fy2) * yfx
    return fx, fy


def transport_step(h, ut, vt, dt, cdgrid, **_kwargs):
    """Single FV3-style transport step."""
    area = cdgrid.base.area
    crx, cry, xfx, yfx, ra_x, ra_y = compute_transport_quantities(
        ut, vt, dt, cdgrid)
    fx, fy = fv_tp_2d(h, crx, cry, xfx, yfx, ra_x, ra_y, cdgrid)
    return h + (fx[:, :-1, :] - fx[:, 1:, :]
                + fy[:, :, :-1] - fy[:, :, 1:]) / area
