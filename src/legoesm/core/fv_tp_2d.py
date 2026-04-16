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
    """FV3 pert_ppm iv=1: standard PPM constraint (tp_core.F90:1193-1212).

    Prevents new extrema in the reconstruction.  When bl and br have
    opposite signs (parabola crosses cell value), clips the overshoot.
    When both have the same sign (cell value is already an extremum),
    zeros both to flatten the reconstruction.
    """
    is_ext = bl * br >= 0.0
    da1 = bl - br
    da2 = da1 ** 2
    a6da = 3.0 * (bl + br) * da1
    # Fortran: if a6da < -da2: ar = -2*al → br = -2*bl
    #          if a6da >  da2: al = -2*ar → bl = -2*br
    br_out = jnp.where(a6da < -da2, -2.0 * bl, br)
    bl_out = jnp.where(a6da > da2, -2.0 * br, bl)
    bl_out = jnp.where(is_ext, 0.0, bl_out)
    br_out = jnp.where(is_ext, 0.0, br_out)
    return bl_out, br_out


def _pert_ppm_iv0(q, bl, br):
    """FV3 pert_ppm iv=0: positive definite constraint (tp_core.F90:1169-1192).

    Ensures the PPM parabola does not produce negative values when the
    cell mean ``q`` is positive.  When ``q <= 0``, zeroes the reconstruction.
    When ``q > 0`` and the parabola minimum is negative, clips bl/br.

    This is the limiter used by hord=9 (FV3 default for mass, vorticity,
    and momentum transport).
    """
    r12 = 1.0 / 12.0
    zero = jnp.zeros_like(bl)

    # Parabola coefficients: a4 = -3*(bl+br), da1 = br-bl
    a4 = -3.0 * (br + bl)
    da1 = br - bl

    # Condition: parabola has an extremum in [0,1] ↔ |da1| < -a4
    has_extremum = jnp.abs(da1) < -a4

    # Minimum of parabola: q + 0.25/a4 * da1² + a4/12
    # Guard against a4=0 (flat parabola — no extremum anyway)
    a4_safe = jnp.where(jnp.abs(a4) < 1e-30, -1e-30, a4)
    fmin = q + 0.25 / a4_safe * da1 ** 2 + a4_safe * r12
    is_negative = fmin < 0.0

    # When both conditions met and q>0: apply fix
    needs_fix = has_extremum & is_negative & (q > 0.0)
    both_positive = (br > 0.0) & (bl > 0.0)
    da1_positive = da1 > 0.0

    bl_fix = jnp.where(both_positive, zero,
                       jnp.where(da1_positive, bl, -2.0 * br))
    br_fix = jnp.where(both_positive, zero,
                       jnp.where(da1_positive, -2.0 * bl, br))

    bl_out = jnp.where(q <= 0.0, zero, jnp.where(needs_fix, bl_fix, bl))
    br_out = jnp.where(q <= 0.0, zero, jnp.where(needs_fix, br_fix, br))

    return bl_out, br_out


def _ppm_1d(q, n, off_left=None, off_right=None,
            off_left_d1=None, off_right_d1=None,
            use_duogrid=False):
    """PPM bl/br along axis=1 with hord=9 + position-aware boundaries.

    Parameters
    ----------
    q : (6, n+4, M)  field with halo=2 in sweep direction
    n : int           number of interior cells
    off_left : (6, M) or None — halo interp offset at left boundary (depth=0)
    off_right : (6, M) or None — halo interp offset at right boundary (depth=0)
    off_left_d1 : (6, M) or None — depth=1 offset (outer halo)
    off_right_d1 : (6, M) or None — depth=1 offset (outer halo)
    use_duogrid : bool — when True, skip pert_ppm(iv=1) at face boundaries
        (Fortran gates this on .not. (bounded_domain .or. duogrid), tp_core.F90:612)

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

    # hord=9: FV3 default (fv_arrays.F90:339,343: hord_dp=9, hord_vt=9).
    # Simple PPM reconstruction (bl = al - q, br = al - q) with positive-
    # definite constraint via pert_ppm(iv=0).  This is LESS restrictive
    # than hord=8 (2*dm monotone) or hord=10 (pmp/lac), preserving more
    # sub-grid structure.  FV3 tp_core.F90 xppm lines 603-610.
    q_c = qe[:, 2:-2, :]       # (6, n+2, M) — cells at padded indices 2..n+3
    al_L = al[:, :-1, :]        # al at left edge of each cell
    al_R = al[:, 1:, :]         # al at right edge of each cell

    bl = al_L - q_c
    br = al_R - q_c

    # pert_ppm(iv=0): positive definite constraint (tp_core.F90:610)
    bl, br = _pert_ppm_iv0(q_c, bl, br)

    # pert_ppm(iv=1) at face-boundary cells: extra monotonicity for cells
    # whose PPM stencil crosses a face boundary (halo-quality guard).
    # Fortran gates on (.not. (bounded_domain .or. duogrid)) at tp_core.F90:612.
    # For duogrid, boundary halo quality is sufficient — skip extra constraint.
    if not use_duogrid:
        for k in [0, 1, 2, -3, -2, -1]:
            bl_k, br_k = _pert_ppm(bl[:, k, :], br[:, k, :])
            bl = bl.at[:, k, :].set(bl_k)
            br = br.at[:, k, :].set(br_k)

    return bl, br, q_c


def _xppm(q_h2, crx, n, off_left=None, off_right=None,
          off_left_d1=None, off_right_d1=None, use_duogrid=False):
    """PPM in x with hord=9 Courant-number integration.

    FV3 tp_core.F90 xppm lines 670-677: uses raw Courant number ``crx``
    in the standard PPM flux formula.  Boundary non-uniformity is handled
    entirely through bl/br corrections in ``_ppm_1d``, NOT by scaling crx
    (the Fortran does not adjust the Courant number at face boundaries).
    """
    bl, br, q_c = _ppm_1d(q_h2, n, off_left, off_right,
                           off_left_d1, off_right_d1,
                           use_duogrid=use_duogrid)
    bl_L, br_L, q_L = bl[:, :n+1, :], br[:, :n+1, :], q_c[:, :n+1, :]
    bl_R, br_R, q_R = bl[:, 1:n+2, :], br[:, 1:n+2, :], q_c[:, 1:n+2, :]

    fx_pos = q_L + (1.0 - crx) * (br_L - crx * (bl_L + br_L))
    fx_neg = q_R + (1.0 + crx) * (bl_R + crx * (bl_R + br_R))
    return jnp.where(crx > 0, fx_pos, fx_neg)


def _yppm(q_h2, cry, n, off_left=None, off_right=None,
          off_left_d1=None, off_right_d1=None, use_duogrid=False):
    """PPM in y with hord=9 Courant-number integration."""
    q_t = jnp.swapaxes(q_h2, 1, 2)
    c_t = jnp.swapaxes(cry, 1, 2)
    bl, br, q_c = _ppm_1d(q_t, n, off_left, off_right,
                           off_left_d1, off_right_d1,
                           use_duogrid=use_duogrid)
    bl_L, br_L, q_L = bl[:, :n+1, :], br[:, :n+1, :], q_c[:, :n+1, :]
    bl_R, br_R, q_R = bl[:, 1:n+2, :], br[:, 1:n+2, :], q_c[:, 1:n+2, :]
    fy_pos = q_L + (1.0 - c_t) * (br_L - c_t * (bl_L + br_L))
    fy_neg = q_R + (1.0 + c_t) * (bl_R + c_t * (bl_R + br_R))
    return jnp.swapaxes(jnp.where(c_t > 0, fy_pos, fy_neg), 1, 2)


# =========================================================================
# Transport quantities
# =========================================================================

def compute_transport_quantities(ut, vt, dt, cdgrid):
    """Compute Courant numbers, area fluxes, swept areas.

    Follows FV3 sw_core.F90:830-862:
    - xfx = dt * ut * dy * sin_sg(upwind)
    - crx = xfx * rdxa(upwind_cell)  (NOT rdxc at face!)
    The Fortran uses upwind-selected cell-centre rdxa for the Courant
    number, which gives the CFL as a fraction of the upwind CELL WIDTH.
    """
    n = cdgrid.n
    grid = cdgrid.base
    sin_sg = cdgrid.sin_sg

    dy = cdgrid.dy_edge_x
    dx = cdgrid.dx_edge_y

    # --- Transport distance (Fortran: xfx_adv = dt*ut before dy*sin scaling) ---
    xfx_raw = dt * ut   # (6, n+1, n) distance in contravariant coords
    yfx_raw = dt * vt   # (6, n, n+1)

    # --- x-direction Courant number (FV3 sw_core.F90:849-853) ---
    # crx = (dt*ut) * rdxa(upwind_cell)  where rdxa = 1/cell_width.
    # FV3 computes rdxa from exact face-to-face distance (fv_grid_tools.F90).
    rdxa_pad = pad_halo(cdgrid.rdxa, interp_offsets=grid.halo_interp_offsets)
    rdxa_upwind = jnp.where(ut > 0,
                            rdxa_pad[:, :n+1, 1:-1],    # cell i-1
                            rdxa_pad[:, 1:n+2, 1:-1])   # cell i
    crx = xfx_raw * rdxa_upwind

    # --- x-direction area flux (xfx = dt*ut*dy*sin_sg_upwind) ---
    sin_east = sin_sg[:, :, :, 2]
    sin_west = sin_sg[:, :, :, 0]
    se_pad = pad_halo(sin_east, interp_offsets=grid.halo_interp_offsets)
    sw_pad = pad_halo(sin_west, interp_offsets=grid.halo_interp_offsets)
    sin_x = jnp.where(ut > 0, se_pad[:, :n+1, 1:-1], sw_pad[:, 1:n+2, 1:-1])
    xfx = xfx_raw * dy * sin_x

    # --- y-direction Courant number ---
    rdya_pad = pad_halo(cdgrid.rdya, interp_offsets=grid.halo_interp_offsets)
    rdya_upwind = jnp.where(vt > 0,
                            rdya_pad[:, 1:-1, :n+1],
                            rdya_pad[:, 1:-1, 1:n+2])
    cry = yfx_raw * rdya_upwind

    # --- y-direction area flux ---
    sin_north = sin_sg[:, :, :, 3]
    sin_south = sin_sg[:, :, :, 1]
    sn_pad = pad_halo(sin_north, interp_offsets=grid.halo_interp_offsets)
    ss_pad = pad_halo(sin_south, interp_offsets=grid.halo_interp_offsets)
    sin_y = jnp.where(vt > 0, sn_pad[:, 1:-1, :n+1], ss_pad[:, 1:-1, 1:n+2])
    yfx = yfx_raw * dx * sin_y

    area = grid.area
    ra_x = area + xfx[:, :-1, :] - xfx[:, 1:, :]
    ra_y = area + yfx[:, :, :-1] - yfx[:, :, 1:]
    return crx, cry, xfx, yfx, ra_x, ra_y


# =========================================================================
# fv_tp_2d: Lin-Rood 2D transport
# =========================================================================

def _deln_flux(nord, damp, q, fx, fy, cdgrid, mass=None):
    """FV3 deln_flux: del-n damping for cell-mean values (tp_core.F90:1217-1365).

    Adds diffusive fluxes to the transport fluxes ``fx``/``fy`` to provide
    del-2 (nord=0), del-4 (nord=1), or del-6 (nord=2) damping.

    Currently implements nord=0 (del-2).  Higher orders require iterative
    Laplacian application with intermediate halo exchanges.

    Parameters
    ----------
    nord : int — damping order (0=del-2, 1=del-4, ...)
    damp : float — damping coefficient (pre-scaled: (damp_c * da_min)^(nord+1))
    q : (6, n, n) — transported field (cell-mean values)
    fx : (6, n+1, n) — x-direction transport flux (modified in-place)
    fy : (6, n, n+1) — y-direction transport flux (modified in-place)
    cdgrid : CubedSphereCDGrid
    mass : (6, n, n) or None — mass field for mass-weighted damping

    Returns
    -------
    fx, fy : modified fluxes with diffusive contribution added
    """
    n = cdgrid.n
    grid = cdgrid.base
    sg = cdgrid.sin_sg
    dy = cdgrid.dy_edge_x   # (6, n+1, n) — edge length at u-faces
    dx = cdgrid.dx_edge_y   # (6, n, n+1) — edge length at v-faces
    rdxc = cdgrid.rdxc       # (6, n+1, n)
    rdyc = cdgrid.rdyc       # (6, n, n+1)
    rarea = 1.0 / grid.area  # (6, n, n)

    # Step 1: initialize d2 (tp_core.F90:1253-1265)
    if mass is None:
        d2 = damp * q
    else:
        d2 = q

    # Step 2: Laplacian diffusive fluxes (tp_core.F90:1270-1290, USE_SG path)
    # fx2 = 0.5*(sin_sg(i-1,j,E)+sin_sg(i,j,W)) * dy * (d2[i-1]-d2[i]) * rdxc
    d2_pad = pad_halo(d2, interp_offsets=grid.halo_interp_offsets)
    sin_E = sg[:, :, :, 2]   # E-edge
    sin_W = sg[:, :, :, 0]   # W-edge
    sin_E_pad = pad_halo(sin_E, interp_offsets=grid.halo_interp_offsets)
    sin_W_pad = pad_halo(sin_W, interp_offsets=grid.halo_interp_offsets)
    sin_uv_x = 0.5 * (sin_E_pad[:, :n+1, 1:-1] + sin_W_pad[:, 1:n+2, 1:-1])
    fx2 = sin_uv_x * dy * (d2_pad[:, :-1, 1:-1] - d2_pad[:, 1:, 1:-1]) * rdxc

    sin_N = sg[:, :, :, 3]   # N-edge
    sin_S = sg[:, :, :, 1]   # S-edge
    sin_N_pad = pad_halo(sin_N, interp_offsets=grid.halo_interp_offsets)
    sin_S_pad = pad_halo(sin_S, interp_offsets=grid.halo_interp_offsets)
    sin_uv_y = 0.5 * (sin_N_pad[:, 1:-1, :n+1] + sin_S_pad[:, 1:-1, 1:n+2])
    fy2 = sin_uv_y * dx * (d2_pad[:, 1:-1, :-1] - d2_pad[:, 1:-1, 1:]) * rdyc

    # Step 3: Higher-order iteration (nord > 0, tp_core.F90:1298-1331)
    for _it in range(nord):
        # Compute divergence of diffusive fluxes
        d2 = (fx2[:, :-1, :] - fx2[:, 1:, :] + fy2[:, :, :-1] - fy2[:, :, 1:]) * rarea
        # Re-exchange and recompute fluxes with sign flip (d2[i]-d2[i-1])
        d2_pad = pad_halo(d2, interp_offsets=grid.halo_interp_offsets)
        fx2 = sin_uv_x * dy * (d2_pad[:, 1:, 1:-1] - d2_pad[:, :-1, 1:-1]) * rdxc
        fy2 = sin_uv_y * dx * (d2_pad[:, 1:-1, 1:] - d2_pad[:, 1:-1, :-1]) * rdyc

    # Step 4: Add diffusive fluxes to transport fluxes (tp_core.F90:1339-1363)
    if mass is not None:
        mass_pad = pad_halo(mass, interp_offsets=grid.halo_interp_offsets)
        mass_u = 0.5 * (mass_pad[:, :-1, 1:-1] + mass_pad[:, 1:, 1:-1])  # (6, n+1, n)
        mass_v = 0.5 * (mass_pad[:, 1:-1, :-1] + mass_pad[:, 1:-1, 1:])  # (6, n, n+1)
        fx = fx + 0.5 * damp * mass_u * fx2
        fy = fy + 0.5 * damp * mass_v * fy2
    else:
        fx = fx + fx2
        fy = fy + fy2

    return fx, fy


def fv_tp_2d(q, crx, cry, xfx, yfx, ra_x, ra_y, cdgrid,
             nord=None, damp_c=None, mass=None):
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

    dg = cdgrid.base.duogrid
    _use_dg = dg is not None and dg.ng >= 2

    q_full = pad_halo(q, halo=2, interp_offsets=offsets_h2)

    # Pass 1: Y-sweep on q, X-sweep on cross-corrected q_i
    fy2 = _yppm(q_full[:, 2:-2, :], cry, n, oy_L0, oy_R0, oy_L1, oy_R1,
                use_duogrid=_use_dg)
    fyy = yfx * fy2
    q_i = (q * area + fyy[:, :, :-1] - fyy[:, :, 1:]) / ra_y

    # Proper halo exchange for q_i (required for mass conservation)
    q_i_pad = pad_halo(q_i, halo=2, interp_offsets=offsets_h2)
    fx1 = _xppm(q_i_pad[:, :, 2:-2], crx, n, ox_L0, ox_R0, ox_L1, ox_R1,
                use_duogrid=_use_dg)

    # Pass 2: X-sweep on q, Y-sweep on cross-corrected q_j
    fx2 = _xppm(q_full[:, :, 2:-2], crx, n, ox_L0, ox_R0, ox_L1, ox_R1,
                use_duogrid=_use_dg)
    fxx = xfx * fx2
    q_j = (q * area + fxx[:, :-1, :] - fxx[:, 1:, :]) / ra_x

    # Proper halo exchange for q_j (required for mass conservation)
    q_j_pad = pad_halo(q_j, halo=2, interp_offsets=offsets_h2)
    fy1 = _yppm(q_j_pad[:, 2:-2, :], cry, n, oy_L0, oy_R0, oy_L1, oy_R1,
                use_duogrid=_use_dg)

    if mass is not None:
        # With mass: fx = 0.5*(fx1+fx2)*mfx, fy = 0.5*(fy1+fy2)*mfy
        # (tp_core.F90:188-196).  Here mfx/mfy are the mass fluxes = xfx/yfx.
        fx = 0.5 * (fx1 + fx2) * xfx
        fy = 0.5 * (fy1 + fy2) * yfx
    else:
        # Without mass: fx = 0.5*(fx1+fx2)*xfx, fy = 0.5*(fy1+fy2)*yfx
        # (tp_core.F90:207-216)
        fx = 0.5 * (fx1 + fx2) * xfx
        fy = 0.5 * (fy1 + fy2) * yfx

    # Del-n damping (tp_core.F90:197-201 and 217-222)
    if nord is not None and damp_c is not None and damp_c > 1e-4:
        damp = (damp_c * jnp.min(cdgrid.base.area)) ** (nord + 1)
        fx, fy = _deln_flux(nord, damp, q, fx, fy, cdgrid, mass=mass)

    # Duogrid flux synchronization (see cgrid_mass_flux_divergence for rationale).
    dg = cdgrid.base.duogrid
    if dg is not None and dg.ng >= 2:
        from legoesm.grids.halo import synchronize_cgrid_fluxes
        fx, fy = synchronize_cgrid_fluxes(fx, fy, n)

    return fx, fy


def transport_step(h, ut, vt, dt, cdgrid, mass_target=None, **_kwargs):
    """Single FV3-style transport step with mass conservation.

    Parameters
    ----------
    h : (6, n, n) height field
    ut, vt : contravariant velocities
    dt : timestep
    cdgrid : CubedSphereCDGrid
    mass_target : float or None — if provided, enforce exact mass conservation
    """
    area = cdgrid.base.area
    crx, cry, xfx, yfx, ra_x, ra_y = compute_transport_quantities(
        ut, vt, dt, cdgrid)
    fx, fy = fv_tp_2d(h, crx, cry, xfx, yfx, ra_x, ra_y, cdgrid)
    h_new = h + (fx[:, :-1, :] - fx[:, 1:, :]
                 + fy[:, :, :-1] - fy[:, :, 1:]) / area

    # Mass conservation fixer: clip negative values and rescale
    # positive values to conserve total mass.  This compensates for
    # flux mismatches at face boundaries while maintaining non-negativity.
    if mass_target is not None:
        # Step 1: clip negatives to zero
        h_pos = jnp.maximum(h_new, 0.0)
        mass_pos = jnp.sum(h_pos * area)
        # Step 2: scale positive values to match target mass
        scale = mass_target / jnp.maximum(mass_pos, 1.0)
        h_new = h_pos * scale

    return h_new
