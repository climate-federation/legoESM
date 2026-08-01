"""FV3-inspired SW forward-backward core (EXPERIMENTAL).

GFDL sw_core.F90 port (c_sw 79-488, d2a2c_vect 3006-3345). Covariant velocity + sin_sg flux scaling.
Use fv3_sw_tendencies (operators_cdgrid.py) for production.
Lin 2004; Mouallem, Harris & Chen 2023.

Wind conventions (2026-07-10 audit): the MODEL's prognostic D winds are an
ORTHOGONAL pair (u_d = V·x̂ with x̂ the unit i-tangent from angle_edge_x/
angle_edge_y; v_d = V·rot90(x̂)); the FV3 Fortran chain assumes COVARIANT
winds (u identical; v = V·ŷ, the j-line tangent).  The FB entry points
(fv3_fb_sw_step / fv3_forward_backward_step) convert v_d to covariant at
entry (fb_v_d_to_covariant) and back at exit, so every formula between
(d2a2c cosa/rsin conversions, KE contravariant·covariant products, corner
circulation, B-grid Courant numbers, along-line gradients, one_grad_p) is
Fortran-verbatim on the convention it assumes.  O(cosa)=0.5 at cube
vertices — the pre-fix orthogonal-input/covariant-formula mismatch was the
FB panel-edge instability root cause (one-step vertex v_d kick 0.71 m/s ->
0.03; W2 C36 2-day max|u| 49 -> 41.6).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from legoesm.core.fv_tp_2d import (
    compute_transport_quantities,
    fv_tp_2d,
    pert_ppm,
    transport_step,
)
from legoesm.core.operators_cdgrid import (
    interp_center_to_corner_a2b_ord4,
    pad_halo_auto,
)
from legoesm.grids.duogrid import ext_vector_dgrid
from legoesm.grids.halo import (
    CONNECTIVITY,
    EAST,
    NORTH,
    SOUTH,
    WEST,
    pad_halo,
    pad_halo_vector,
    pad_halo_vector_4d,
    synchronize_bgrid_ne_corner_geo,
    synchronize_cgrid_fluxes,
)

from legoesm import constants

_EPS = float(jnp.finfo(jnp.float32).eps)


def _pad_halo_dgrid_for_ppm(u_d, v_d, cdgrid, halo: int = 2,
                            basis: str = "covariant"):
    """iter-945: cross-face halo for D-grid winds via ext_vector_dgrid (rotated cube_rmp).

    Output: u_d_ihalo (6, n+2h, n+1) and v_d_jhalo (6, n+1, n+2h) for d_sw3 PPM sweeps.
    Equivalent to single-rank mpp_update_domains(DGRID_NE). Interior preserved exactly.
    Requires duogrid with ng >= halo.

    ``basis`` must match the convention of the (u_d, v_d) actually passed in
    (see ``ext_vector_dgrid``): "covariant" for the FB-internal winds
    (post 2026-07-10 the FB chain converts v_d to true covariant at entry),
    "orthogonal" for the model's prognostic pair (u=V.x, v=V.rot90(x)).
    """
    n = cdgrid.n
    h = halo
    grid = cdgrid.base
    dg = grid.duogrid
    if dg is None or dg.ng < h:
        raise ValueError(
            f"_pad_halo_dgrid_for_ppm: requires duogrid with ng>={h}, "
            f"got ng={None if dg is None else dg.ng}."
        )

    # A-grid prep: 2nd-order length-weighted D→A avg (mirror of _d2a2c_vect_duogrid step 1)
    dx_u = cdgrid.dx_edge_y  # (6, n, n+1)
    dy_v = cdgrid.dy_edge_x  # (6, n+1, n) — y-length at v_d positions
    wu = u_d * dx_u
    wv = v_d * dy_v
    utmp_2nd = (wu[:, :, :-1] + wu[:, :, 1:]) / (
        dx_u[:, :, :-1] + dx_u[:, :, 1:])  # (6, n, n)
    vtmp_2nd = (wv[:, :-1, :] + wv[:, 1:, :]) / (
        dy_v[:, :-1, :] + dy_v[:, 1:, :])  # (6, n, n)
    cos_sg5 = cdgrid.cos_sg[:, :, :, 4]

    # 2026-07-10 FIX (FB panel-edge instability root cause): the halo basis
    # must match the wind convention of the inputs.  The model's prognostic
    # D winds are u_d = V·x̂, v_d = V·x̂⊥ (angle_edge_y is the i-tangent
    # angle) — basis="orthogonal"; the FB chain's INTERNAL winds are true
    # FV3 covariant (v converted at fv3_fb_sw_step entry) — basis="covariant".
    # Using the covariant machinery on orthogonal inputs put an
    # O(cosa_s·|V|) sign-flipping convention error in the seam halo
    # (~100% of v_d at panel edges).
    u_d_full, v_d_full = ext_vector_dgrid(
        utmp_2nd, vtmp_2nd, dg,
        grid.cos_angle, grid.sin_angle,
        cos_sg5,
        halo=h,
        basis=basis,
    )  # u_d_full: (6, n+2h, n+2h-1); v_d_full: (6, n+2h-1, n+2h)

    # Preserve interior exactly (mirror Fortran mpp_update_domains: halo cells only)
    u_d_full = u_d_full.at[:, h:h + n, h - 1:h + n].set(u_d)
    v_d_full = v_d_full.at[:, h - 1:h + n, h:h + n].set(v_d)

    # Slice j-stagger axis (n+2h-1 → n+1) for PPM transport
    u_d_ihalo = u_d_full[:, :, h - 1:h + n]
    v_d_jhalo = v_d_full[:, h - 1:h + n, :]

    return u_d_ihalo, v_d_jhalo


def _pad_halo_uc_vc_via_d2a2c(u_d, v_d, cdgrid):
    """iter-946 NEGATIVE-RESULT: 4th-order d2a2c uc/vc halo from OLD u_d/v_d.

    OLD-derived halo lacks c_sw+p_grad_c increment → OLD/NEW mismatch in d_sw 4-cell avg.
    Measured C36 W2 1d: |v|=156 m/s vs mode='edge' baseline 81 m/s (WORSE). Reverted.
    Retained for iter-947+ work (needs c_sw+p_grad_c halo propagation).
    Requires duogrid ng>=3.
    """
    n = cdgrid.n
    grid = cdgrid.base
    dg = grid.duogrid
    if dg is None or dg.ng < 3:
        raise ValueError(
            f"_pad_halo_uc_vc_via_d2a2c: requires duogrid with ng>=3 "
            f"(j-halo extension on the 4-point j-stencil); "
            f"got ng={None if dg is None else dg.ng}."
        )
    h = 3

    # Step 1: A-grid prep (mirror _d2a2c_vect_duogrid: length-weighted 2nd-order D→A avg)
    dx_u = cdgrid.dx_edge_y
    dy_v = cdgrid.dy_edge_x
    wu = u_d * dx_u
    wv = v_d * dy_v
    utmp_2nd = (wu[:, :, :-1] + wu[:, :, 1:]) / (
        dx_u[:, :, :-1] + dx_u[:, :, 1:])  # (6, n, n)
    vtmp_2nd = (wv[:, :-1, :] + wv[:, 1:, :]) / (
        dy_v[:, :-1, :] + dy_v[:, 1:, :])  # (6, n, n)
    cos_sg5 = cdgrid.cos_sg[:, :, :, 4]

    # Covariant halo: the FB-internal winds are true FV3 covariant post the
    # 2026-07-10 entry conversion (see _pad_halo_dgrid_for_ppm note).
    u_d_full, v_d_full = ext_vector_dgrid(
        utmp_2nd, vtmp_2nd, dg,
        grid.cos_angle, grid.sin_angle,
        cos_sg5,
        halo=h,
        basis="covariant",
    )  # u_d_full: (6, n+2h, n+2h-1); v_d_full: (6, n+2h-1, n+2h)

    # Preserve interior exactly (mirror Fortran mpp_update_domains)
    u_d_full = u_d_full.at[:, h:h + n, h - 1:h + n].set(u_d)
    v_d_full = v_d_full.at[:, h - 1:h + n, h:h + n].set(v_d)

    # Step 2: utmp at i-halo full + j-halo=1 (j ∈ [-1, n]). 4-pt j-stencil needs h>=3
    utmp_T = (
        _A2 * (u_d_full[:, :, h - 3:h - 3 + n + 2]
               + u_d_full[:, :, h:h + n + 2])
        + _A1 * (u_d_full[:, :, h - 2:h - 2 + n + 2]
                 + u_d_full[:, :, h - 1:h - 1 + n + 2])
    )  # (6, n+2h, n+2)

    # Step 3: 4th-order A→C i-stencil → uc with j-halo=1
    uc_jhalo = (
        _A2 * (utmp_T[:, h - 2:h - 2 + n + 1, :]
               + utmp_T[:, h + 1:h + 1 + n + 1, :])
        + _A1 * (utmp_T[:, h - 1:h - 1 + n + 1, :]
                 + utmp_T[:, h:h + n + 1, :])
    )  # (6, n+1, n+2)

    # Step 4: vtmp at j-halo full + i-halo=1 (symmetric to step 2)
    vtmp_T = (
        _A2 * (v_d_full[:, h - 3:h - 3 + n + 2, :]
               + v_d_full[:, h:h + n + 2, :])
        + _A1 * (v_d_full[:, h - 2:h - 2 + n + 2, :]
                 + v_d_full[:, h - 1:h - 1 + n + 2, :])
    )  # (6, n+2, n+2h)

    # Step 5: 4th-order A→C j-stencil → vc with i-halo=1
    vc_ihalo = (
        _A2 * (vtmp_T[:, :, h - 2:h - 2 + n + 1]
               + vtmp_T[:, :, h + 1:h + 1 + n + 1])
        + _A1 * (vtmp_T[:, :, h - 1:h - 1 + n + 1]
                 + vtmp_T[:, :, h:h + n + 1])
    )  # (6, n+2, n+1)

    return uc_jhalo, vc_ihalo


def _pad_halo_uc_vc_new_via_neighbor_delta(uc, vc, u_d, v_d, cdgrid):
    """Faithful ext_vector halo semantics for the updated C winds.

    Oracle (dyn_core.F90:655) exchanges the UPDATED (uc, vc) after
    p_grad_c, so the d_sw halo carries the NEIGHBOUR's c_sw+p_grad_c
    increment.  iter-947's old-delta halo anchors the NEW local
    boundary but transports the LOCAL increment — a per-step
    time-correlated seam error (candidate #2 of the 2026-07-10 edge
    review).  Here: halo = OLD cross-face halo (iter-946 d2a2c
    machinery) + the NEIGHBOUR-side increment, reconstructed by
    exchanging the cc-averaged increment vector (Δuc, Δvc) through the
    duogrid cc vector halo and re-staggering to the C halo row
    (2nd-order A→C; Δ is O(dt) small).  Requires duogrid ng>=3.
    """
    n = cdgrid.n
    grid = cdgrid.base
    dg = grid.duogrid

    uc_old_jhalo, vc_old_ihalo = _pad_halo_uc_vc_via_d2a2c(u_d, v_d, cdgrid)
    _, _, uc_old_int, vc_old_int, _, _ = d2a2c_vect(u_d, v_d, cdgrid)

    duc = uc - uc_old_int   # (6, n+1, n) x-faces
    dvc = vc - vc_old_int   # (6, n, n+1) y-faces
    duc_cc = 0.5 * (duc[:, :-1, :] + duc[:, 1:, :])   # (6, n, n)
    dvc_cc = 0.5 * (dvc[:, :, :-1] + dvc[:, :, 1:])   # (6, n, n)
    # 2026-07-10: the FB-internal winds (hence Δuc/Δvc) are COVARIANT —
    # exchange through pad_halo_vector's covariant branch (cos_theta), not
    # the orthogonal rotation.
    duc_pad, dvc_pad = pad_halo_vector(
        duc_cc, dvc_cc,
        grid.cos_angle, grid.sin_angle,
        grid.cos_angle_padded, grid.sin_angle_padded,
        interp_offsets=(None if dg is not None
                        else grid.halo_interp_offsets),
        duogrid=dg,
        cos_theta=cdgrid.cosa_cell,
        sin_theta=cdgrid.sina_cell,
    )   # (6, n+2, n+2), cc index -1..n both dims

    # uc j-halo rows (j=-1, j=n): A→C i-average of the neighbour Δ row.
    duc_row_s = 0.5 * (duc_pad[:, :-1, 0] + duc_pad[:, 1:, 0])    # (6, n+1)
    duc_row_n = 0.5 * (duc_pad[:, :-1, -1] + duc_pad[:, 1:, -1])
    uc_new_south = uc_old_jhalo[:, :, 0] + duc_row_s
    uc_new_north = uc_old_jhalo[:, :, n + 1] + duc_row_n
    uc_pad = jnp.concatenate(
        [uc_new_south[:, :, None], uc, uc_new_north[:, :, None]], axis=2)

    # vc i-halo cols (i=-1, i=n): A→C j-average of the neighbour Δ col.
    dvc_col_w = 0.5 * (dvc_pad[:, 0, :-1] + dvc_pad[:, 0, 1:])    # (6, n+1)
    dvc_col_e = 0.5 * (dvc_pad[:, -1, :-1] + dvc_pad[:, -1, 1:])
    vc_new_west = vc_old_ihalo[:, 0, :] + dvc_col_w
    vc_new_east = vc_old_ihalo[:, n + 1, :] + dvc_col_e
    vc_pad = jnp.concatenate(
        [vc_new_west[:, None, :], vc, vc_new_east[:, None, :]], axis=1)
    return uc_pad, vc_pad


def _d_sw1_recompute_ut_vt(uc, vc, cdgrid, dt,
                            u_d_old=None, v_d_old=None):
    """FV3 d_sw1 ut/vt recomputation (sw_core.F90:618-812).

    4-cell cross-vel avg + face-boundary sin_sg upwind + adjacent strip + corner 2x2 solve.
    iter-947: u_d_old/v_d_old enable NEW-corrected duogrid halo (ng>=3).
    """
    n = cdgrid.n
    dg = cdgrid.base.duogrid
    use_duogrid = dg is not None and dg.ng >= 2

    cosa_u = cdgrid.cosa_u  # (6, n+1, n)
    cosa_v = cdgrid.cosa_v  # (6, n, n+1)
    rsin_u = cdgrid.rsin_u  # (6, n+1, n)
    rsin_v = cdgrid.rsin_v  # (6, n, n+1)
    sg = cdgrid.sin_sg

    # Part 1: Interior ut/vt from 4-cell vc/uc avg.
    # ut(I,j) = (uc - 0.25*cosa_u*(vc(I-1,j)+vc(I,j)+vc(I-1,j+1)+vc(I,j+1)))*rsin_u
    # iter-947 → 2026-07-10: NEW-corrected duogrid halo via _pad_halo_uc_vc_new_via_neighbor_delta (neighbour-side increment).
    # iter-946 d2a2c-only halo regressed W2 (OLD/NEW mismatch with c_sw+p_grad_c increments).
    if (use_duogrid and dg.ng >= 3
            and u_d_old is not None and v_d_old is not None):
        uc_pad, vc_pad = _pad_halo_uc_vc_new_via_neighbor_delta(
            uc, vc, u_d_old, v_d_old, cdgrid)
    else:
        vc_pad = jnp.pad(vc, [(0, 0), (1, 1), (0, 0)], mode='edge')
        uc_pad = jnp.pad(uc, [(0, 0), (0, 0), (1, 1)], mode='edge')
    # 4-cell averages at u-face/v-face
    vc_avg = (vc_pad[:, :-1, :-1] + vc_pad[:, 1:, :-1]
              + vc_pad[:, :-1, 1:] + vc_pad[:, 1:, 1:])  # (6, n+1, n)
    ut = (uc - 0.25 * cosa_u * vc_avg) * rsin_u

    uc_avg = (uc_pad[:, :-1, :-1] + uc_pad[:, 1:, :-1]
              + uc_pad[:, :-1, 1:] + uc_pad[:, 1:, 1:])  # (6, n, n+1)
    vt = (vc - 0.25 * cosa_v * uc_avg) * rsin_v

    # iter-953/954 NEGATIVE: Parts 2-4 on duogrid regress v_ll_Linf (sin_sg padding conflicts; Part 2 drops cosa_u term)
    if use_duogrid:
        return ut, vt

    # Part 2: Non-duogrid face-boundary overrides (ut = uc / sin_sg(upwind))
    sin_w_left = sg[:, :, :, 2]
    sin_w_right = sg[:, :, :, 0]
    grid = cdgrid.base
    offsets = grid.halo_interp_offsets
    se_pad = pad_halo(sin_w_left, interp_offsets=offsets)
    sw_pad = pad_halo(sin_w_right, interp_offsets=offsets)

    # West face (I=0)
    sin_upwind_w = jnp.where(uc[:, 0, :] * dt > 0,
                             se_pad[:, :n+1, 1:-1][:, 0, :],
                             sw_pad[:, 1:n+2, 1:-1][:, 0, :])
    ut = ut.at[:, 0, :].set(uc[:, 0, :] / jnp.maximum(jnp.abs(sin_upwind_w), _EPS))

    # East face (I=n)
    sin_upwind_e = jnp.where(uc[:, n, :] * dt > 0,
                             se_pad[:, :n+1, 1:-1][:, n, :],
                             sw_pad[:, 1:n+2, 1:-1][:, n, :])
    ut = ut.at[:, n, :].set(uc[:, n, :] / jnp.maximum(jnp.abs(sin_upwind_e), _EPS))

    # South face (J=0)
    sin_s_below = sg[:, :, :, 3]
    sin_s_above = sg[:, :, :, 1]
    sn_pad = pad_halo(sin_s_below, interp_offsets=offsets)
    ss_pad = pad_halo(sin_s_above, interp_offsets=offsets)

    sin_upwind_s = jnp.where(vc[:, :, 0] * dt > 0,
                             sn_pad[:, 1:-1, :n+1][:, :, 0],
                             ss_pad[:, 1:-1, 1:n+2][:, :, 0])
    vt = vt.at[:, :, 0].set(vc[:, :, 0] / jnp.maximum(jnp.abs(sin_upwind_s), _EPS))

    # North face (J=n)
    sin_upwind_n = jnp.where(vc[:, :, n] * dt > 0,
                             sn_pad[:, 1:-1, :n+1][:, :, n],
                             ss_pad[:, 1:-1, 1:n+2][:, :, n])
    vt = vt.at[:, :, n].set(vc[:, :, n] / jnp.maximum(jnp.abs(sin_upwind_n), _EPS))

    # Part 3: adjacent strip recomputation (FV3 sw_core.F90:666-726)
    if n > 4:
        jlo = 2; jhi = n - 1

        # West rows 0, 1
        for row in [0, 1]:
            avg = (ut[:, row, jlo-1:jhi] + ut[:, row+1, jlo-1:jhi]
                   + ut[:, row, jlo:jhi+1] + ut[:, row+1, jlo:jhi+1])  # (6, jhi-jlo+1)
            vt = vt.at[:, row, jlo:jhi+1].set(
                vc[:, row, jlo:jhi+1] - 0.25 * cosa_v[:, row, jlo:jhi+1] * avg)

        # East rows n-2, n-1
        for row in [n-2, n-1]:
            if row >= 0 and row + 1 <= n:
                avg = (ut[:, row, jlo-1:jhi] + ut[:, row+1, jlo-1:jhi]
                       + ut[:, row, jlo:jhi+1] + ut[:, row+1, jlo:jhi+1])
                vt = vt.at[:, row, jlo:jhi+1].set(
                    vc[:, row, jlo:jhi+1] - 0.25 * cosa_v[:, row, jlo:jhi+1] * avg)

        # South cols 0, 1
        ilo = 2; ihi = n - 1
        for col in [0, 1]:
            avg = (vt[:, ilo-1:ihi, col] + vt[:, ilo:ihi+1, col]
                   + vt[:, ilo-1:ihi, col+1] + vt[:, ilo:ihi+1, col+1])
            ut = ut.at[:, ilo:ihi+1, col].set(
                uc[:, ilo:ihi+1, col] - 0.25 * cosa_u[:, ilo:ihi+1, col] * avg)

        # North cols n-1, n
        for col in [n-1, n]:
            if col - 1 >= 0 and col <= n:
                avg = (vt[:, ilo-1:ihi, col-1] + vt[:, ilo:ihi+1, col-1]
                       + vt[:, ilo-1:ihi, col] + vt[:, ilo:ihi+1, col])
                ut = ut.at[:, ilo:ihi+1, col].set(
                    uc[:, ilo:ihi+1, col] - 0.25 * cosa_u[:, ilo:ihi+1, col] * avg)

    # Part 4: Corner 2x2 coupled solve at cube vertices (FV3 sw_core.F90:739-811)
    # Fortran (2,1) → Python (1,0). damp = 1/(1 - 0.0625*cu*cv)
    cu = cosa_u  # (6, n+1, n)
    cv = cosa_v  # (6, n, n+1)

    # SW corner
    damp = 1.0 / (1.0 - 0.0625 * cu[:, 1, 0] * cv[:, 0, 1])
    ut = ut.at[:, 1, 0].set(
        (uc[:, 1, 0] - 0.25 * cu[:, 1, 0] * (
            vt[:, 0, 0] + vt[:, 1, 0] + vt[:, 1, 1] + vc[:, 0, 1]
            - 0.25 * cv[:, 0, 1] * (ut[:, 0, 0] + ut[:, 0, 1] + ut[:, 1, 1])
        )) * damp)
    vt = vt.at[:, 0, 1].set(
        (vc[:, 0, 1] - 0.25 * cv[:, 0, 1] * (
            ut[:, 0, 0] + ut[:, 0, 1] + ut[:, 1, 1] + uc[:, 1, 0]
            - 0.25 * cu[:, 1, 0] * (vt[:, 0, 0] + vt[:, 1, 0] + vt[:, 1, 1])
        )) * damp)

    # SE corner
    damp = 1.0 / (1.0 - 0.0625 * cu[:, n-1, 0] * cv[:, n-2, 1])
    ut = ut.at[:, n-1, 0].set(
        (uc[:, n-1, 0] - 0.25 * cu[:, n-1, 0] * (
            vt[:, n-2, 0] + vt[:, n-3, 0] + vt[:, n-3, 1] + vc[:, n-2, 1]
            - 0.25 * cv[:, n-2, 1] * (ut[:, n, 0] + ut[:, n, 1] + ut[:, n-1, 1])
        )) * damp)
    vt = vt.at[:, n-2, 1].set(
        (vc[:, n-2, 1] - 0.25 * cv[:, n-2, 1] * (
            ut[:, n, 0] + ut[:, n, 1] + ut[:, n-1, 1] + uc[:, n-1, 0]
            - 0.25 * cu[:, n-1, 0] * (vt[:, n-2, 0] + vt[:, n-3, 0] + vt[:, n-3, 1])
        )) * damp)

    # NE corner
    damp = 1.0 / (1.0 - 0.0625 * cu[:, n-1, n-1] * cv[:, n-2, n-1])
    ut = ut.at[:, n-1, n-1].set(
        (uc[:, n-1, n-1] - 0.25 * cu[:, n-1, n-1] * (
            vt[:, n-2, n] + vt[:, n-3, n] + vt[:, n-3, n-1] + vc[:, n-2, n-1]
            - 0.25 * cv[:, n-2, n-1] * (ut[:, n, n-1] + ut[:, n, n-2] + ut[:, n-1, n-2])
        )) * damp)
    vt = vt.at[:, n-2, n-1].set(
        (vc[:, n-2, n-1] - 0.25 * cv[:, n-2, n-1] * (
            ut[:, n, n-1] + ut[:, n, n-2] + ut[:, n-1, n-2] + uc[:, n-1, n-1]
            - 0.25 * cu[:, n-1, n-1] * (vt[:, n-2, n] + vt[:, n-3, n] + vt[:, n-3, n-1])
        )) * damp)

    # NW corner
    damp = 1.0 / (1.0 - 0.0625 * cu[:, 1, n-1] * cv[:, 0, n-1])
    ut = ut.at[:, 1, n-1].set(
        (uc[:, 1, n-1] - 0.25 * cu[:, 1, n-1] * (
            vt[:, 0, n] + vt[:, 1, n] + vt[:, 1, n-1] + vc[:, 0, n-1]
            - 0.25 * cv[:, 0, n-1] * (ut[:, 0, n-1] + ut[:, 0, n-2] + ut[:, 1, n-2])
        )) * damp)
    vt = vt.at[:, 0, n-1].set(
        (vc[:, 0, n-1] - 0.25 * cv[:, 0, n-1] * (
            ut[:, 0, n-1] + ut[:, 0, n-2] + ut[:, 1, n-2] + uc[:, 1, n-1]
            - 0.25 * cu[:, 1, n-1] * (vt[:, 0, n] + vt[:, 1, n] + vt[:, 1, n-1])
        )) * damp)

    return ut, vt


def _edge_interpolate4(ua4, dxa4):
    """FV3 non-uniform 4-point interpolation to the interface between cells 2 and 3.

    Exact port of ``edge_interpolate4`` from sw_core.F90.
    """
    t1 = dxa4[..., 0] + dxa4[..., 1]
    t2 = dxa4[..., 2] + dxa4[..., 3]
    return 0.5 * (
        ((t1 + dxa4[..., 1]) * ua4[..., 1] - dxa4[..., 1] * ua4[..., 0]) / t1
        + ((t2 + dxa4[..., 2]) * ua4[..., 2] - dxa4[..., 2] * ua4[..., 3]) / t2
    )


# FV3 interpolation constants (from sw_core.F90)
_A1 = 0.5625     # 4th-order Lagrange
_A2 = -0.0625
_C1 = -2.0 / 14.0  # one-sided cubic at face boundary
_C2 = 11.0 / 14.0
_C3 = 5.0 / 14.0


# ==============================================================================
# d2a2c_vect: D-grid → A-grid → C-grid (FV3 covariant convention)
# ==============================================================================

def _d2a2c_vect_duogrid(u_d, v_d, cdgrid):
    """D→A→C duogrid (FV3 sw_core.F90:3419-3454 gridstruct%dg%is_initialized branch).

    1. ext_vector pipeline (c2l_ord2 + lat/lon halo + cubed_a2d_halo) = FV3 mpp_update_domains(DGRID_NE)
    2. 4th-order D→A on fully-haloed domain
    3. Covariant → contravariant via cosa_s/rsin2 (FV3:3451-3452)
    4. 4th-order A→C on full-halo utmp/vtmp
    """
    n = cdgrid.n
    grid = cdgrid.base
    dg = grid.duogrid
    # iter-654: h=3 when ng>=3 (c_sw cube-edge upwind needs 3-ring halo); h=2 fallback
    h = 3 if (dg is not None and dg.ng >= 3) else 2
    cos_sg5 = cdgrid.cos_sg[:, :, :, 4]
    rsin2 = cdgrid.rsin2_cell

    # Step 1: ext_vector D-grid halo (FV3 fv_duogrid.F90:741-826): c2l_ord2 → A-cov → lat/lon →
    # scalar halo cube_rmp → cubed_a2d_halo. Length-weighted utmp = (u*dx)|sum / dx|sum
    # (preserves constant state). Fortran c2l_ord2 has leading 2 absorbed into a11/a22.
    dx_u = cdgrid.dx_edge_y  # (6, n, n+1)
    dy_v = cdgrid.dy_edge_x  # (6, n+1, n) — y-length at v_d positions
    wu = u_d * dx_u
    wv = v_d * dy_v
    utmp_2nd = (wu[:, :, :-1] + wu[:, :, 1:]) / (
        dx_u[:, :, :-1] + dx_u[:, :, 1:])  # (6, n, n)
    vtmp_2nd = (wv[:, :-1, :] + wv[:, 1:, :]) / (
        dy_v[:, :-1, :] + dy_v[:, 1:, :])  # (6, n, n)
    u_d_full, v_d_full = ext_vector_dgrid(
        utmp_2nd, vtmp_2nd, dg,
        grid.cos_angle, grid.sin_angle,
        cos_sg5,
        halo=h,
    )  # u_d_full: (6, n+2h, n+2h-1), v_d_full: (6, n+2h-1, n+2h)

    # Preserve interior exactly (mirror Fortran mpp_update_domains)
    u_d_full = u_d_full.at[:, h:h + n, h - 1:h + n].set(u_d)
    v_d_full = v_d_full.at[:, h - 1:h + n, h:h + n].set(v_d)

    # Step 2: 4th-order D→A on fully-haloed domain (FV3 sw_core.F90:3421-3435 duogrid).
    # utmp(i, j_cell) = a2*(u(j-1)+u(j+2)) + a1*(u(j)+u(j+1)). iter-654: h>=2 generalised.
    if n > 3:
        utmp_full = (
            _A2 * (u_d_full[:, :, h - 2:h - 2 + n]
                   + u_d_full[:, :, h + 1:h + 1 + n])
            + _A1 * (u_d_full[:, :, h - 1:h - 1 + n]
                     + u_d_full[:, :, h:h + n])
        )  # (6, n+2h, n) — i-halo full, j-interior only
        vtmp_full = (
            _A2 * (v_d_full[:, h - 2:h - 2 + n, :]
                   + v_d_full[:, h + 1:h + 1 + n, :])
            + _A1 * (v_d_full[:, h - 1:h - 1 + n, :]
                     + v_d_full[:, h:h + n, :])
        )  # (6, n, n+2h) — j-halo full, i-interior only
    else:
        # Tiny grid fallback: 2-point average without 4th-order stencil.
        # Read edges [j_cell, j_cell+1] → padded [j_cell+(h-1), j_cell+h].
        utmp_full = 0.5 * (u_d_full[:, :, h - 1:h - 1 + n]
                            + u_d_full[:, :, h:h + n])
        vtmp_full = 0.5 * (v_d_full[:, h - 1:h - 1 + n, :]
                            + v_d_full[:, h:h + n, :])

    # ---- Step 3: Covariant→contravariant at cell centres (interior) ----
    # FV3 sw_core.F90:3451-3452:
    #   ua(i,j) = (utmp(i,j)-vtmp(i,j)*cosa_s(i,j)) * rsin2(i,j)
    # Take the i-interior slice of utmp_full / j-interior of vtmp_full
    # (ua/va only needs interior for downstream operators).
    utmp_int = utmp_full[:, h:h + n, :]
    vtmp_int = vtmp_full[:, :, h:h + n]
    ua = (utmp_int - vtmp_int * cos_sg5) * rsin2
    va = (vtmp_int - utmp_int * cos_sg5) * rsin2

    # ---- Step 4: 4th-order A→C interpolation ----
    # uc(i+1/2, j) = a2*(utmp(i-1,j)+utmp(i+2,j)) + a1*(utmp(i,j)+utmp(i+1,j))
    # utmp_full has shape (6, n+2h, n).  Padded i-index p maps to cell
    # (p - h).  For u-face k in [0, n], the stencil reads cells
    # [k-2, k-1, k, k+1] → padded [k-2+h, k-1+h, k+h, k+1+h].
    # For k in [0, n] (n+1 faces): padded ranges
    #   [h-2:h-1+n, h-1:h+n, h:h+n+1, h+1:h+2+n]
    # At h=2 these collapse to [0:n+1, 1:n+2, 2:n+3, 3:n+4] == the
    # original `[:-3, 1:-2, 2:-1, 3:]` slicing of length (n+4).
    uc = (_A2 * (utmp_full[:, h - 2:h - 1 + n, :]
                 + utmp_full[:, h + 1:h + 2 + n, :])
          + _A1 * (utmp_full[:, h - 1:h + n, :]
                   + utmp_full[:, h:h + n + 1, :]))  # (6, n+1, n)

    ut = (uc - v_d * cdgrid.cosa_u) * cdgrid.rsin_u

    vc = (_A2 * (vtmp_full[:, :, h - 2:h - 1 + n]
                 + vtmp_full[:, :, h + 1:h + 2 + n])
          + _A1 * (vtmp_full[:, :, h - 1:h + n]
                   + vtmp_full[:, :, h:h + n + 1]))  # (6, n, n+1)

    vt = (vc - u_d * cdgrid.cosa_v) * cdgrid.rsin_v

    return ua, va, uc, vc, ut, vt


def _apply_fortran_d2a2c_corner_overrides(utmp_pad, vtmp_pad, n):
    """Iter-938 port of Fortran sw_core.F90:3527-3545 + 3620-3639
    cube-corner sign-flip overrides on utmp_pad / vtmp_pad.

    Fortran applies four corner overrides per axis (SW/SE/NE/NW),
    each writing 3 halo cells (i=-2..0 or i=0..2) at the boundary
    halo row/col.  Our Python pad_halo_vector(halo=2) only provides
    2 halo cells per side, so we port the 2 deepest cells (depth
    -1 and -0 in Fortran indexing → padded-index depth-2 and
    depth-1).  The third Fortran cell (depth-3) is OUT of our
    halo=2 reach and is skipped — a documented partial port.

    Index map (with halo=2, padded shape (n+4, n+4)):
      Fortran i=-1 → padded 0   (depth-2 west halo)
      Fortran i=0  → padded 1   (depth-1 west halo)
      Fortran i=1  → padded 2   (first interior, west boundary)
      Fortran i=npx → padded n+2 (depth-1 east halo)
      Fortran i=npx+1 → padded n+3 (depth-2 east halo)

    The Fortran utmp/vtmp interior values are NOT modified — only
    halo cells.  This matches our intent of correcting `pad_halo_vector`'s
    `fill_corners_h2` 2-point AVERAGE with Fortran's sign-flipped
    cross-component copy at the cube vertex.

    The vtmp overrides read from utmp_pad INTERIOR cells (which the
    utmp overrides do NOT modify), so the two override blocks are
    independent in input/output and may be applied in either order.

    Parameters
    ----------
    utmp_pad : (6, n+4, n+4) — pad_halo_vector output
    vtmp_pad : (6, n+4, n+4) — pad_halo_vector output
    n : int — interior grid size

    Returns
    -------
    utmp_pad, vtmp_pad : same shapes, with cube-vertex halo cells
        overwritten using Fortran's sign-flip cross-component values.
    """
    # ---- utmp x-direction overrides (Fortran 3527-3545) ----
    # SW corner: utmp(i=-1..0, j=0) = -vtmp(0, 1-i)
    utmp_pad = utmp_pad.at[:, 0, 1].set(-vtmp_pad[:, 1, 3])  # i=-1
    utmp_pad = utmp_pad.at[:, 1, 1].set(-vtmp_pad[:, 1, 2])  # i=0
    # SE corner: utmp(npx+i, 0) = +vtmp(npx, i+1)  for i in {0, 1}
    utmp_pad = utmp_pad.at[:, n+2, 1].set(+vtmp_pad[:, n+2, 2])
    utmp_pad = utmp_pad.at[:, n+3, 1].set(+vtmp_pad[:, n+2, 3])
    # NE corner: utmp(npx+i, npy) = -vtmp(npx, npy-1-i) for i in {0, 1}
    utmp_pad = utmp_pad.at[:, n+2, n+2].set(-vtmp_pad[:, n+2, n+1])
    utmp_pad = utmp_pad.at[:, n+3, n+2].set(-vtmp_pad[:, n+2, n])
    # NW corner: utmp(i=-1..0, npy) = +vtmp(0, npy-1+i+1)
    utmp_pad = utmp_pad.at[:, 0, n+2].set(+vtmp_pad[:, 1, n])     # i=-1
    utmp_pad = utmp_pad.at[:, 1, n+2].set(+vtmp_pad[:, 1, n+1])   # i=0

    # ---- vtmp y-direction overrides (Fortran 3620-3639) ----
    # SW corner: vtmp(0, j=-1..0) = -utmp(1-j, 0)
    vtmp_pad = vtmp_pad.at[:, 1, 0].set(-utmp_pad[:, 3, 1])  # j=-1, reads utmp(2, 0)
    vtmp_pad = vtmp_pad.at[:, 1, 1].set(-utmp_pad[:, 2, 1])  # j=0,  reads utmp(1, 0)
    # NW corner: vtmp(0, npy+j) = +utmp(j+1, npy)  for j in {0, 1}
    vtmp_pad = vtmp_pad.at[:, 1, n+2].set(+utmp_pad[:, 2, n+2])
    vtmp_pad = vtmp_pad.at[:, 1, n+3].set(+utmp_pad[:, 3, n+2])
    # SE corner: vtmp(npx, j=-1..0) = +utmp(ie+j, 0)
    vtmp_pad = vtmp_pad.at[:, n+2, 0].set(+utmp_pad[:, n,   1])   # j=-1 → utmp(npx-2, 0)
    vtmp_pad = vtmp_pad.at[:, n+2, 1].set(+utmp_pad[:, n+1, 1])   # j=0  → utmp(npx-1, 0)
    # NE corner: vtmp(npx, npy+j) = -utmp(ie-j, npy) for j in {0, 1}
    vtmp_pad = vtmp_pad.at[:, n+2, n+2].set(-utmp_pad[:, n+1, n+2])
    vtmp_pad = vtmp_pad.at[:, n+2, n+3].set(-utmp_pad[:, n,   n+2])

    return utmp_pad, vtmp_pad


def d2a2c_d_to_a(u_d, v_d, cdgrid):
    """D-grid → A-grid covariant step of d2a2c (Steps 1+2), verbatim.

    utmp/vtmp = covariant cell-centre winds (2nd-order base, 4th-order
    interior npt-band) then a halo=2 vector exchange.  Returns
    ``(utmp_pad, vtmp_pad)`` each ``(6, n+4, n+4)``.  P4 phase-1b
    approach C runs this in the GLOBAL GSPMD view (cheap) and feeds the
    padded result into the tiled A→C; the per-tile block is a
    ``tiled_padded_block`` (h2) slice — no staggered D-wind halo needed.
    """
    n = cdgrid.n
    npt = min(4, n // 2)
    utmp = 0.5 * (u_d[:, :, :-1] + u_d[:, :, 1:])   # (6, n, n)
    vtmp = 0.5 * (v_d[:, :-1, :] + v_d[:, 1:, :])   # (6, n, n)
    if n > 2 * npt and npt > 0:
        u4 = (_A2 * (u_d[:, :, :-3] + u_d[:, :, 3:])
              + _A1 * (u_d[:, :, 1:-2] + u_d[:, :, 2:-1]))
        utmp = utmp.at[:, :, npt:n - npt].set(u4[:, :, npt - 1:n - npt - 1])
        v4 = (_A2 * (v_d[:, :-3, :] + v_d[:, 3:, :])
              + _A1 * (v_d[:, 1:-2, :] + v_d[:, 2:-1, :]))
        vtmp = vtmp.at[:, npt:n - npt, :].set(v4[:, npt - 1:n - npt - 1, :])
    grid = cdgrid.base
    return pad_halo_vector(
        utmp, vtmp,
        grid.cos_angle, grid.sin_angle,
        grid.cos_angle_padded_h2, grid.sin_angle_padded_h2,
        interp_offsets=grid.halo_interp_offsets_h2,
        halo=2,
    )


def d2a2c_d_to_a_4d(u_d, v_d, cdgrid):
    """4D (all-levels-one-message) :func:`d2a2c_d_to_a` (#811).

    The D→A covariant averages are pure-local — they slice ``u_d``/``v_d`` on the
    horizontal axes and ride the trailing level axis via broadcasting — and the
    halo=2 VECTOR exchange is done ONCE for all levels with
    :func:`pad_halo_vector_4d` (rotation angles broadcast over levels).  This is
    what lets the moisture substep reconstruct the transport winds without a
    ``vmap(pad_halo_vector)`` (the wind-halo ``batch_axes`` failure under MPI
    face-scatter; #811).  BIT-IDENTICAL to per-level ``d2a2c_d_to_a`` on
    single-rank.

    u_d : (6, n, n+1, nlev); v_d : (6, n+1, n, nlev).  Returns
    ``(utmp_pad, vtmp_pad)`` each ``(6, n+4, n+4, nlev)``.
    """
    n = cdgrid.n
    npt = min(4, n // 2)
    utmp = 0.5 * (u_d[:, :, :-1] + u_d[:, :, 1:])   # (6, n, n, nlev)
    vtmp = 0.5 * (v_d[:, :-1, :] + v_d[:, 1:, :])
    if n > 2 * npt and npt > 0:
        u4 = (_A2 * (u_d[:, :, :-3] + u_d[:, :, 3:])
              + _A1 * (u_d[:, :, 1:-2] + u_d[:, :, 2:-1]))
        utmp = utmp.at[:, :, npt:n - npt].set(u4[:, :, npt - 1:n - npt - 1])
        v4 = (_A2 * (v_d[:, :-3, :] + v_d[:, 3:, :])
              + _A1 * (v_d[:, 1:-2, :] + v_d[:, 2:-1, :]))
        vtmp = vtmp.at[:, npt:n - npt, :].set(v4[:, npt - 1:n - npt - 1, :])
    grid = cdgrid.base
    return pad_halo_vector_4d(
        utmp, vtmp,
        grid.cos_angle, grid.sin_angle,
        grid.cos_angle_padded_h2, grid.sin_angle_padded_h2,
        interp_offsets=grid.halo_interp_offsets_h2,
        halo=2,
    )


def d2a2c_uc_4th_local(utmp_pad):
    """A→C x-dir 4th-order interior stencil on the h2-padded covariant
    utmp — leading-axis-agnostic.  Returns the full (·, n+1, n) 4th-order
    uc (n = utmp_pad.shape[1]-4).  The global d2a2c overlays this onto
    its 2nd-order base in the [npt+1:n-npt] face-interior band; under
    tiling, an INTERIOR tile is entirely 4th-order, so its uc IS this
    core on the tile's padded utmp (P4 phase-1b approach C).  No
    face-edge specials (C1/C2/C3, edge_interpolate4) — those stay in
    d2a2c_vect for face-boundary cells.
    """
    h = 2
    n = utmp_pad.shape[1] - 2 * h
    return (_A2 * (utmp_pad[:, h - 2:n + h - 1, h:-h]
                   + utmp_pad[:, h + 1:n + h + 2, h:-h])
            + _A1 * (utmp_pad[:, h - 1:n + h, h:-h]
                     + utmp_pad[:, h:n + h + 1, h:-h]))


def d2a2c_uc_c123_local(utmp_pad, at_high):
    """One-sided C1/C2/C3 uc value at the FACE-edge cell i=1 (W) or
    i=n-1 (E) — the d2a2c_vect edge special (sw_core.F90:3586/3594),
    leading-axis-agnostic on the h2-padded utmp (n = shape[1]-4).

    ``at_high=False`` -> i=1: C1*utmp_pad[h+2] + C2*utmp_pad[h+1] +
    C3*utmp_pad[h].  ``at_high=True`` -> i=n-1: C1*utmp_pad[n+h-3] +
    C2*utmp_pad[n+h-2] + C3*utmp_pad[n+h-1].  Returns the (·, n) row of
    uc.  Pure utmp_pad stencil — no halo; an edge tile overlays this on
    its uc at the local i=1 / i=nl-1 face when the tile touches the
    W/E face edge (P4 phase-1b edge specials, first piece).
    """
    h = 2
    n = utmp_pad.shape[1] - 2 * h
    if at_high:
        return (_C1 * utmp_pad[:, n + h - 3, h:-h]
                + _C2 * utmp_pad[:, n + h - 2, h:-h]
                + _C3 * utmp_pad[:, n + h - 1, h:-h])
    return (_C1 * utmp_pad[:, h + 2, h:-h]
            + _C2 * utmp_pad[:, h + 1, h:-h]
            + _C3 * utmp_pad[:, h, h:-h])


def d2a2c_uc_edge_interp_local(ua_pad, dx_pad, se_pad, sw_pad, at_high):
    """edge_interpolate4 + upwind sin_sg uc value at the FACE boundary
    i=0 (W) / i=n (E) — the d2a2c_vect edge special
    (sw_core.F90:3587/3589-3592), leading-axis-agnostic.

    Exact mirror of d2a2c_vect lines 779-794 for one ``i_bdy``:
      ua_pad : (·, n+4, n+4) tile contravariant ua (h2-padded both axes)
      dx_pad : (·, n+4, n)   tile dx, rows h2-padded / cols unpadded
      se_pad : (·, n+2, n+2) tile sin_sg E-component (cross-face h1 pad)
      sw_pad : (·, n+2, n+2) tile sin_sg W-component
    ``n = ua_pad.shape[1]-4``.  Returns the (·, n) uc boundary row
    ``uc_bdy = where(ut_bdy>0, ut_bdy*sin_left, ut_bdy*sin_right)``.
    The outer upwind sine cell is CROSS-FACE (se/sw must be globally
    padded then tile-sliced — P4 phase-1b approach C; codex
    sin_sg-halo verdict).
    """
    h = 2
    n = ua_pad.shape[1] - 2 * h
    i_bdy = n if at_high else 0
    i_p = i_bdy + h
    ua4 = jnp.stack([ua_pad[:, i_p - 1, h:-h], ua_pad[:, i_p, h:-h],
                     ua_pad[:, i_p + 1, h:-h], ua_pad[:, i_p + 2, h:-h]],
                    axis=-1)
    dxa4 = jnp.stack([dx_pad[:, i_p - 1, :], dx_pad[:, i_p, :],
                      dx_pad[:, i_p + 1, :], dx_pad[:, i_p + 2, :]],
                     axis=-1)
    ut_bdy = _edge_interpolate4(ua4, dxa4)
    sin_left = se_pad[:, i_bdy, 1:-1]
    sin_right = sw_pad[:, i_bdy + 1, 1:-1]
    return jnp.where(ut_bdy > 0, ut_bdy * sin_left, ut_bdy * sin_right)


def d2a2c_vc_4th_local(vtmp_pad):
    """A→C y-dir 4th-order interior stencil — symmetric to
    :func:`d2a2c_uc_4th_local`.  Returns (·, n, n+1)
    (n = vtmp_pad.shape[2]-4)."""
    h = 2
    n = vtmp_pad.shape[2] - 2 * h
    return (_A2 * (vtmp_pad[:, h:-h, h - 2:n + h - 1]
                   + vtmp_pad[:, h:-h, h + 1:n + h + 2])
            + _A1 * (vtmp_pad[:, h:-h, h - 1:n + h]
                     + vtmp_pad[:, h:-h, h:n + h + 1]))


def d2a2c_ut_vt_local(uc, vc, u_d, v_d, cosa_u, rsin_u, cosa_v, rsin_v):
    """A→C contravariant transport winds ut/vt — pure pointwise CORE.

    ``ut = (uc - v_d*cosa_u)*rsin_u`` (and symmetric ``vt``).  No halo —
    a tile computes its own staggered ut ``(nl+1, nl)`` / vt
    ``(nl, nl+1)`` from its uc/vc + LOCAL v_d/u_d + sliced metrics (P4
    phase-1b approach C; codex: ut/vt need only the local staggered
    D-winds, no staggered halo).  The production d2a2c_vect applies
    this base then overlays face-boundary + adjacent-strip overrides
    (edge specials) — those stay in d2a2c_vect for boundary cells.
    """
    ut = (uc - v_d * cosa_u) * rsin_u
    vt = (vc - u_d * cosa_v) * rsin_v
    return ut, vt


def d2a2c_vc_c123_local(vtmp_pad, at_high):
    """One-sided C1/C2/C3 vc value at the FACE-edge cell j=1 (S) or
    j=n-1 (N) — the j-axis transpose of :func:`d2a2c_uc_c123_local`
    (sw_core.F90 vc C1/C2/C3).  n = vtmp_pad.shape[2]-4.  Returns the
    (·, n) vc column."""
    h = 2
    n = vtmp_pad.shape[2] - 2 * h
    if at_high:
        return (_C1 * vtmp_pad[:, h:-h, n + h - 3]
                + _C2 * vtmp_pad[:, h:-h, n + h - 2]
                + _C3 * vtmp_pad[:, h:-h, n + h - 1])
    return (_C1 * vtmp_pad[:, h:-h, h + 2]
            + _C2 * vtmp_pad[:, h:-h, h + 1]
            + _C3 * vtmp_pad[:, h:-h, h])


def d2a2c_vc_edge_interp_local(va_pad, dy_pad, sn_pad, ss_pad, at_high):
    """edge_interpolate4 + upwind sin_sg vc value at the FACE boundary
    j=0 (S) / j=n (N) — the j-axis transpose of
    :func:`d2a2c_uc_edge_interp_local` (sw_core.F90 vc face boundary).

      va_pad : (·, n+4, n+4) tile contravariant va
      dy_pad : (·, n, n+4)   tile dy, cols h2-padded / rows unpadded
      sn_pad : (·, n+2, n+2) tile sin_sg N-component (cross-face h1 pad)
      ss_pad : (·, n+2, n+2) tile sin_sg S-component
    Returns the (·, n) vc boundary column.  Outer upwind sine is
    cross-face (sn/ss globally padded then tile-sliced — approach C).
    """
    h = 2
    n = va_pad.shape[2] - 2 * h
    j_bdy = n if at_high else 0
    j_p = j_bdy + h
    va4 = jnp.stack([va_pad[:, h:-h, j_p - 1], va_pad[:, h:-h, j_p],
                     va_pad[:, h:-h, j_p + 1], va_pad[:, h:-h, j_p + 2]],
                    axis=-1)
    dya4 = jnp.stack([dy_pad[:, :, j_p - 1], dy_pad[:, :, j_p],
                      dy_pad[:, :, j_p + 1], dy_pad[:, :, j_p + 2]],
                     axis=-1)
    vt_bdy = _edge_interpolate4(va4, dya4)
    sin_below = sn_pad[:, 1:-1, j_bdy]
    sin_above = ss_pad[:, 1:-1, j_bdy + 1]
    return jnp.where(vt_bdy > 0, vt_bdy * sin_below, vt_bdy * sin_above)


def d2a2c_ua_va_local(utmp, vtmp, cos_sg5, rsin2):
    """A-grid contravariant winds from covariant utmp/vtmp — pointwise.

    ``ua = (utmp - vtmp*cos_sg5)*rsin2`` (symmetric ``va``).  No halo,
    so a tile computes its own ``(nl, nl)`` ua/va from sliced interior
    utmp/vtmp + cos_sg5/rsin2 — the global d2a2c pads utmp/vtmp then
    trims, which is pointwise-identical to operating on the interior
    (P4 phase-1b approach C, first A→C output).
    """
    ua = (utmp - vtmp * cos_sg5) * rsin2
    va = (vtmp - utmp * cos_sg5) * rsin2
    return ua, va


def _d2a2c_uc_iinterior(utmp_pad_face, a, b, nl, n, npt):
    """uc (1, nl+1, nl) for an i-INTERIOR tile column: 2nd-order base + a
    clamped 4th-order overlay on the global band [npt+1, n-npt) (no W/E
    specials), asymmetric -1 i-window so ``d2a2c_uc_4th_local(w)`` row r ==
    production uc[a+r] with no shift.  Shared by d2a2c_interior_local, the
    S/N edge tiles and the j-edge sides of a corner tile (P4 phase-1b)."""
    h = 2
    w = utmp_pad_face[None, a - 1:a + nl + 3, b:b + nl + 4]
    uc = 0.5 * (w[:, 2:nl + 3, h:-h] + w[:, 3:nl + 4, h:-h])     # 2nd base
    uc4 = d2a2c_uc_4th_local(w)
    k_lo = max(0, (npt + 1) - a)
    k_hi = min(nl + 1, (n - npt) - a)
    if n > 2 * npt + 2 and k_lo < k_hi:
        uc = uc.at[:, k_lo:k_hi, :].set(uc4[:, k_lo:k_hi, :])
    return uc


def _d2a2c_vc_jinterior(vtmp_pad_face, a, b, nl, n, npt):
    """vc (1, nl, nl+1) for a j-INTERIOR tile column — the j-axis transpose
    of :func:`_d2a2c_uc_iinterior` (asymmetric -1 j-window, no S/N specials).
    Shared by d2a2c_interior_local, the W/E edge tiles and the i-edge sides
    of a corner tile."""
    h = 2
    wv = vtmp_pad_face[None, a:a + nl + 4, b - 1:b + nl + 3]
    vc = 0.5 * (wv[:, h:-h, 2:nl + 3] + wv[:, h:-h, 3:nl + 4])   # 2nd base
    vc4 = d2a2c_vc_4th_local(wv)
    j_lo = max(0, (npt + 1) - b)
    j_hi = min(nl + 1, (n - npt) - b)
    if n > 2 * npt + 2 and j_lo < j_hi:
        vc = vc.at[:, :, j_lo:j_hi].set(vc4[:, :, j_lo:j_hi])
    return vc


def _d2a2c_uc_iedge(utmp_pad_face, a, b, nl, n, npt,
                    ua_pad_face, dx_pad_face, se_pad_face, sw_pad_face,
                    at_high):
    """uc (1, nl+1, nl) for an i-EDGE tile column (W: at_high=False, a=0;
    E: at_high=True, a=n-nl) — the shared FV3 A→C x-direction edge stencil
    (2nd base + clamped 4th overlay + C1/C2/C3 + edge_interpolate4+upwind).
    Used by d2a2c_edge_w/e_local and by both i-edge corners.  Returns
    ``(uc, (i_spec, sin_left, sin_right))`` where the tuple is the
    boundary-face (i=0 / i=n) ut-override data (P4 phase-1b)."""
    h = 2
    if at_high:   # E: asymmetric -1 window, no overlay shift; specials hi
        w = utmp_pad_face[None, a - 1:a + nl + 3, b:b + nl + 4]
        uc = 0.5 * (w[:, 2:nl + 3, h:-h] + w[:, 3:nl + 4, h:-h])
        uc4 = d2a2c_uc_4th_local(w)
        k_lo = max(0, (npt + 1) - a)
        k_hi = min(nl + 1, (n - npt) - a)
        if n > 2 * npt + 2 and k_lo < k_hi:
            uc = uc.at[:, k_lo:k_hi, :].set(uc4[:, k_lo:k_hi, :])
        ws = utmp_pad_face[None, a:a + nl + 4, b:b + nl + 4]
        uc = uc.at[:, nl - 1, :].set(d2a2c_uc_c123_local(ws, True))
        ua_e = ua_pad_face[None, a:a + nl + 4, b:b + nl + 4]
        dx_e = dx_pad_face[None, a:a + nl + 4, b:b + nl]
        se_e = se_pad_face[None, a:a + nl + 2, b:b + nl + 2]
        sw_e = sw_pad_face[None, a:a + nl + 2, b:b + nl + 2]
        i_spec = nl
        sin_left, sin_right = se_e[:, nl, 1:-1], sw_e[:, nl + 1, 1:-1]
    else:         # W: symmetric window + 1 overlay shift; specials lo
        w = utmp_pad_face[None, 0:nl + 4, b:b + nl + 4]
        uc = 0.5 * (w[:, h - 1:nl + h, h:-h] + w[:, h:nl + h + 1, h:-h])
        uc4 = (_A2 * (w[:, 0:nl + 1, h:-h] + w[:, 3:nl + 4, h:-h])
               + _A1 * (w[:, 1:nl + 2, h:-h] + w[:, 2:nl + 3, h:-h]))
        i_lo = npt + 1
        k_hi = min(n - npt, nl + 1)
        if n > 2 * npt + 2 and i_lo < k_hi:
            uc = uc.at[:, i_lo:k_hi, :].set(uc4[:, i_lo - 1:k_hi - 1, :])
        uc = uc.at[:, 1, :].set(d2a2c_uc_c123_local(w, False))
        ua_e = ua_pad_face[None, 0:nl + 4, b:b + nl + 4]
        dx_e = dx_pad_face[None, 0:nl + 4, b:b + nl]
        se_e = se_pad_face[None, 0:nl + 2, b:b + nl + 2]
        sw_e = sw_pad_face[None, 0:nl + 2, b:b + nl + 2]
        i_spec = 0
        sin_left, sin_right = se_e[:, 0, 1:-1], sw_e[:, 1, 1:-1]
    uc = uc.at[:, i_spec, :].set(
        d2a2c_uc_edge_interp_local(ua_e, dx_e, se_e, sw_e, at_high))
    return uc, (i_spec, sin_left, sin_right)


def _d2a2c_vc_jedge(vtmp_pad_face, a, b, nl, n, npt,
                    va_pad_face, dy_pad_face, sn_pad_face, ss_pad_face,
                    at_high):
    """vc (1, nl, nl+1) for a j-EDGE tile column (S: at_high=False, b=0;
    N: at_high=True, b=n-nl) — the j-axis transpose of
    :func:`_d2a2c_uc_iedge`.  Returns ``(vc, (j_spec, sin_below,
    sin_above))`` (the j=0 / j=n vt-override data)."""
    h = 2
    if at_high:   # N: asymmetric -1 window, no shift; specials hi
        wv = vtmp_pad_face[None, a:a + nl + 4, b - 1:b + nl + 3]
        vc = 0.5 * (wv[:, h:-h, 2:nl + 3] + wv[:, h:-h, 3:nl + 4])
        vc4 = d2a2c_vc_4th_local(wv)
        m_lo = max(0, (npt + 1) - b)
        m_hi = min(nl + 1, (n - npt) - b)
        if n > 2 * npt + 2 and m_lo < m_hi:
            vc = vc.at[:, :, m_lo:m_hi].set(vc4[:, :, m_lo:m_hi])
        wsv = vtmp_pad_face[None, a:a + nl + 4, b:b + nl + 4]
        vc = vc.at[:, :, nl - 1].set(d2a2c_vc_c123_local(wsv, True))
        va_e = va_pad_face[None, a:a + nl + 4, b:b + nl + 4]
        dy_e = dy_pad_face[None, a:a + nl, b:b + nl + 4]
        sn_e = sn_pad_face[None, a:a + nl + 2, b:b + nl + 2]
        ss_e = ss_pad_face[None, a:a + nl + 2, b:b + nl + 2]
        j_spec = nl
        sin_below, sin_above = sn_e[:, 1:-1, nl], ss_e[:, 1:-1, nl + 1]
    else:         # S: symmetric window + 1 shift; specials lo
        wv = vtmp_pad_face[None, a:a + nl + 4, 0:nl + 4]
        vc = 0.5 * (wv[:, h:-h, h - 1:nl + h] + wv[:, h:-h, h:nl + h + 1])
        vc4 = (_A2 * (wv[:, h:-h, 0:nl + 1] + wv[:, h:-h, 3:nl + 4])
               + _A1 * (wv[:, h:-h, 1:nl + 2] + wv[:, h:-h, 2:nl + 3]))
        j_lo = npt + 1
        m_hi = min(n - npt, nl + 1)
        if n > 2 * npt + 2 and j_lo < m_hi:
            vc = vc.at[:, :, j_lo:m_hi].set(vc4[:, :, j_lo - 1:m_hi - 1])
        vc = vc.at[:, :, 1].set(d2a2c_vc_c123_local(wv, False))
        va_e = va_pad_face[None, a:a + nl + 4, 0:nl + 4]
        dy_e = dy_pad_face[None, a:a + nl, 0:nl + 4]
        sn_e = sn_pad_face[None, a:a + nl + 2, 0:nl + 2]
        ss_e = ss_pad_face[None, a:a + nl + 2, 0:nl + 2]
        j_spec = 0
        sin_below, sin_above = sn_e[:, 1:-1, 0], ss_e[:, 1:-1, 1]
    vc = vc.at[:, :, j_spec].set(
        d2a2c_vc_edge_interp_local(va_e, dy_e, sn_e, ss_e, at_high))
    return vc, (j_spec, sin_below, sin_above)


def d2a2c_interior_local(utmp_pad_face, vtmp_pad_face, ti, tj, nl,
                         u_d, v_d, cos_sg5, rsin2,
                         cosa_u, rsin_u, cosa_v, rsin_v):
    """Full A→C d2a2c for an INTERIOR tile (0<ti<kt-1, 0<tj<kt-1) —
    composes the approach-C pieces with the CORRECT per-output windows.

    ``utmp_pad_face`` / ``vtmp_pad_face`` are this face's FULL h2-padded
    covariant winds ``(n+4, n+4)`` (from d2a2c_d_to_a); the tile's
    windows are sliced HERE so the index logic lives in one place.  The
    other args are the tile's already-sliced staggered/cell metrics +
    local D-winds.

    Per-output windows (h=2; tile cells span global ``[ti*nl, (ti+1)*nl)``):
      * ua/va — interior utmp/vtmp ``(nl, nl)`` at padded
        ``[ti*nl+h : ti*nl+h+nl]`` (symmetric);
      * uc — production ``uc[I] = A2*(utmp_pad[I-1]+utmp_pad[I+2]) +
        A1*(utmp_pad[I]+utmp_pad[I+1])`` for u-faces ``I in
        [ti*nl, ti*nl+nl]``, which needs ``utmp_pad`` rows
        ``[ti*nl-1 : ti*nl+nl+3]`` — ASYMMETRIC -1 on the staggered i
        axis (the prior symmetric window gave ``uc_4th[ti*nl+j] =
        uc_prod[ti*nl+j+1]``, off by one); cols symmetric
        ``[tj*nl : tj*nl+nl+4]`` (d2a2c_uc_4th_local trims ``[h:-h]``);
      * vc — symmetric rows ``[ti*nl : ti*nl+nl+4]``, asymmetric -1
        cols ``[tj*nl-1 : tj*nl+nl+3]`` (the j-staggered analogue);
      * ut/vt — pointwise from uc/vc + the local D-winds/metrics.

    Valid for INTERIOR tiles (``ti*nl-1 >= 0``; no face-boundary cells, so
    no C1/C2/C3 or edge_interpolate4 specials).  uc/vc carry the 2nd-order
    base + a clamped 4th-order overlay on the global band ``[npt+1, n-npt)``;
    for the realistic ``nl >= npt+1`` the band covers the whole tile (pure
    4th), but the base/clamp also makes small tiles whose faces fall outside
    the band bit-identical to global d2a2c_vect (which uses 2nd base there).
    Returns ``(ua, va, uc, vc, ut, vt)`` (leading singleton axis kept).
    """
    h = 2
    n = utmp_pad_face.shape[0] - 2 * h
    npt = min(4, n // 2)
    a, b = ti * nl, tj * nl
    ua, va = d2a2c_ua_va_local(
        utmp_pad_face[None, a + h:a + h + nl, b + h:b + h + nl],
        vtmp_pad_face[None, a + h:a + h + nl, b + h:b + h + nl], cos_sg5, rsin2)
    uc = _d2a2c_uc_iinterior(utmp_pad_face, a, b, nl, n, npt)
    vc = _d2a2c_vc_jinterior(vtmp_pad_face, a, b, nl, n, npt)
    ut, vt = d2a2c_ut_vt_local(
        uc, vc, u_d, v_d, cosa_u, rsin_u, cosa_v, rsin_v)
    return ua, va, uc, vc, ut, vt


def d2a2c_edge_w_local(utmp_pad_face, vtmp_pad_face, tj, nl, n,
                       u_d, v_d, cos_sg5, rsin2,
                       cosa_u, rsin_u, cosa_v, rsin_v,
                       ua_pad_face, dx_pad_face, se_pad_face, sw_pad_face):
    """A→C d2a2c for a W-EDGE, j-interior tile (ti=0, 0<tj<kt-1) —
    everything EXCEPT the adjacent-strip vt[0] (which reads a neighbour
    ut j-halo, a stage-level coupling).

    Mirrors production d2a2c_vect restricted to the tile (a=ti*nl=0):
    uc = 2nd-order base + 4th-order overlay where the global face is in
    [npt+1:n-npt] + C1/C2/C3 at i=1 + edge_interpolate4 at i=0; vc =
    pure 4th-order (tj interior, no S/N); ut = base + the i=0 face
    boundary override (uc/sin_upwind); vt = pointwise base.  Outputs
    (ua, va, uc, vc, ut, vt); vt[:, 0, :] is the base (NOT the
    adjacent-strip) and is excluded from the parity gate.  P4 phase-1b.
    """
    h = 2
    npt = min(4, n // 2)
    b = tj * nl
    ua, va = d2a2c_ua_va_local(
        utmp_pad_face[None, h:h + nl, b + h:b + h + nl],
        vtmp_pad_face[None, h:h + nl, b + h:b + h + nl], cos_sg5, rsin2)
    uc, (i_spec, sin_left, sin_right) = _d2a2c_uc_iedge(
        utmp_pad_face, 0, b, nl, n, npt,
        ua_pad_face, dx_pad_face, se_pad_face, sw_pad_face, False)
    vc = _d2a2c_vc_jinterior(vtmp_pad_face, 0, b, nl, n, npt)
    # ut: base + the i=0 face-boundary override (uc/sin_upwind).
    ut = (uc - v_d * cosa_u) * rsin_u
    sin_up = jnp.where(uc[:, i_spec, :] > 0, sin_left, sin_right)
    ut = ut.at[:, i_spec, :].set(uc[:, i_spec, :] / jnp.maximum(sin_up, _EPS))
    # vt: pointwise base (vt[:,0,:] is base, NOT the adjacent strip).
    vt = (vc - u_d * cosa_v) * rsin_v
    return ua, va, uc, vc, ut, vt


def d2a2c_edge_e_local(utmp_pad_face, vtmp_pad_face, tj, nl, n,
                       u_d, v_d, cos_sg5, rsin2,
                       cosa_u, rsin_u, cosa_v, rsin_v,
                       ua_pad_face, dx_pad_face, se_pad_face, sw_pad_face):
    """A→C d2a2c for an E-EDGE, j-interior tile (ti=kt-1, 0<tj<kt-1) —
    everything EXCEPT the adjacent-strip vt[nl-1] (which reads a neighbour
    ut j-halo, a stage-level coupling).

    Mirror of :func:`d2a2c_edge_w_local` on the high-i face (a=ti*nl=n-nl):
    uc = 2nd-order base + 4th-order overlay on the global band [npt+1:n-npt)
    (for the E tile only its LOW faces) + C1/C2/C3 at i=n-1 +
    edge_interpolate4 at i=n; vc = pure 4th-order (tj interior, no S/N);
    ut = base + the i=n face boundary override (uc/sin_upwind); vt =
    pointwise base.  Outputs (ua, va, uc, vc, ut, vt); vt[:, nl-1, :] is the
    base (NOT the adjacent-strip) and is excluded from the parity gate.

    Unlike the W tile (symmetric window + a +1 overlay shift), the E tile's
    low faces are 4th-order interior cuts that need the ASYMMETRIC -1
    i-window ``w = utmp_pad[a-1:a+nl+3]`` — with it, ``d2a2c_uc_4th_local(w)``
    row r equals production ``uc[a+r]`` directly (no shift).  P4 phase-1b.
    """
    h = 2
    npt = min(4, n // 2)
    a, b = n - nl, tj * nl
    ua, va = d2a2c_ua_va_local(
        utmp_pad_face[None, a + h:a + h + nl, b + h:b + h + nl],
        vtmp_pad_face[None, a + h:a + h + nl, b + h:b + h + nl], cos_sg5, rsin2)
    uc, (i_spec, sin_left, sin_right) = _d2a2c_uc_iedge(
        utmp_pad_face, a, b, nl, n, npt,
        ua_pad_face, dx_pad_face, se_pad_face, sw_pad_face, True)
    vc = _d2a2c_vc_jinterior(vtmp_pad_face, a, b, nl, n, npt)
    # ut: base + the i=n face-boundary override (uc/sin_upwind).
    ut = (uc - v_d * cosa_u) * rsin_u
    sin_up = jnp.where(uc[:, i_spec, :] > 0, sin_left, sin_right)
    ut = ut.at[:, i_spec, :].set(uc[:, i_spec, :] / jnp.maximum(sin_up, _EPS))
    # vt: pointwise base (vt[:,nl-1,:] is base, NOT the adjacent strip).
    vt = (vc - u_d * cosa_v) * rsin_v
    return ua, va, uc, vc, ut, vt


def d2a2c_edge_s_local(utmp_pad_face, vtmp_pad_face, ti, nl, n,
                       u_d, v_d, cos_sg5, rsin2,
                       cosa_u, rsin_u, cosa_v, rsin_v,
                       va_pad_face, dy_pad_face, sn_pad_face, ss_pad_face):
    """A→C d2a2c for a S-EDGE, i-interior tile (tj=0, 0<ti<kt-1) — the
    j-axis transpose of :func:`d2a2c_edge_w_local` (b=tj*nl=0):

    vc = 2nd-order base + 4th-order overlay on the global j-band [npt+1:n-npt)
    + C1/C2/C3 at j=1 + edge_interpolate4 at j=0; uc = 2nd base + clamped 4th
    overlay (ti interior, no W/E specials); vt = base + the j=0 face boundary
    override (vc/sin_upwind); ut = pointwise base.  Outputs
    (ua, va, uc, vc, ut, vt); ut[:, :, 0] is the base (NOT the South
    adjacent-strip, which reads a neighbour vt i-halo at stage level) and is
    excluded from the parity gate.  P4 phase-1b.
    """
    h = 2
    npt = min(4, n // 2)
    a = ti * nl
    ua, va = d2a2c_ua_va_local(
        utmp_pad_face[None, a + h:a + h + nl, h:h + nl],
        vtmp_pad_face[None, a + h:a + h + nl, h:h + nl], cos_sg5, rsin2)
    uc = _d2a2c_uc_iinterior(utmp_pad_face, a, 0, nl, n, npt)
    vc, (j_spec, sin_below, sin_above) = _d2a2c_vc_jedge(
        vtmp_pad_face, a, 0, nl, n, npt,
        va_pad_face, dy_pad_face, sn_pad_face, ss_pad_face, False)
    # vt: base + the j=0 face-boundary override (vc/sin_upwind).
    vt = (vc - u_d * cosa_v) * rsin_v
    sin_up = jnp.where(vc[:, :, j_spec] > 0, sin_below, sin_above)
    vt = vt.at[:, :, j_spec].set(vc[:, :, j_spec] / jnp.maximum(sin_up, _EPS))
    # ut: pointwise base (ut[:,:,0] is base, NOT the South adjacent strip).
    ut = (uc - v_d * cosa_u) * rsin_u
    return ua, va, uc, vc, ut, vt


def d2a2c_edge_n_local(utmp_pad_face, vtmp_pad_face, ti, nl, n,
                       u_d, v_d, cos_sg5, rsin2,
                       cosa_u, rsin_u, cosa_v, rsin_v,
                       va_pad_face, dy_pad_face, sn_pad_face, ss_pad_face):
    """A→C d2a2c for a N-EDGE, i-interior tile (tj=kt-1, 0<ti<kt-1) — the
    j-axis transpose of :func:`d2a2c_edge_e_local` (b=tj*nl=n-nl):

    vc = 2nd base + clamped 4th overlay on the tile's LOW j-faces +
    C1/C2/C3 at j=n-1 + edge_interpolate4 at j=n; uc = 2nd base + clamped 4th
    overlay (ti interior); vt = base + the j=n face boundary override; ut =
    pointwise base.  Outputs (ua, va, uc, vc, ut, vt); ut[:, :, nl-1] is the
    base (NOT the North adjacent-strip) and is excluded from the parity gate.
    Like the E tile, the low j-faces are 4th-order interior cuts needing the
    asymmetric -1 j-window (no overlay shift).  P4 phase-1b.
    """
    h = 2
    npt = min(4, n // 2)
    a, b = ti * nl, n - nl
    ua, va = d2a2c_ua_va_local(
        utmp_pad_face[None, a + h:a + h + nl, b + h:b + h + nl],
        vtmp_pad_face[None, a + h:a + h + nl, b + h:b + h + nl], cos_sg5, rsin2)
    uc = _d2a2c_uc_iinterior(utmp_pad_face, a, b, nl, n, npt)
    vc, (j_spec, sin_below, sin_above) = _d2a2c_vc_jedge(
        vtmp_pad_face, a, b, nl, n, npt,
        va_pad_face, dy_pad_face, sn_pad_face, ss_pad_face, True)
    # vt: base + the j=n face-boundary override (vc/sin_upwind).
    vt = (vc - u_d * cosa_v) * rsin_v
    sin_up = jnp.where(vc[:, :, j_spec] > 0, sin_below, sin_above)
    vt = vt.at[:, :, j_spec].set(vc[:, :, j_spec] / jnp.maximum(sin_up, _EPS))
    # ut: pointwise base (ut[:,:,nl-1] is base, NOT the North adjacent strip).
    ut = (uc - v_d * cosa_u) * rsin_u
    return ua, va, uc, vc, ut, vt


def d2a2c_corner_local(utmp_pad_face, vtmp_pad_face, ti, tj, nl, n,
                       u_d, v_d, cos_sg5, rsin2,
                       cosa_u, rsin_u, cosa_v, rsin_v,
                       ua_pad_face, dx_pad_face, se_pad_face, sw_pad_face,
                       va_pad_face, dy_pad_face, sn_pad_face, ss_pad_face):
    """A→C d2a2c for a CORNER tile (ti,tj each in {0,kt-1}) — the union of an
    i-edge (W/E) uc column and a j-edge (S/N) vc column.  For kt=2 EVERY tile
    is a corner (6*kt²=24 devices), so this is the np=24 sub-face unlock.

    uc = the i-edge stencil (W if ti=0 else E); vc = the j-edge stencil (S if
    tj=0 else N); ut = base + the i-face boundary override; vt = base + the
    j-face boundary override.  BOTH adjacent strips are deferred (each reads a
    neighbour transport-wind halo at stage level): the vt i-strip (row i=0 /
    i=n-1) and the ut j-strip (col j=0 / j=n-1) are left as the pointwise base
    and excluded from the parity gate.  The i-face/j-face boundary overrides
    do NOT overlap either strip (i_spec,j_spec in {0,n}; strips at {0,n-1} for
    i/j in [2,n-2]).  P4 phase-1b.
    """
    h = 2
    npt = min(4, n // 2)
    a, b = ti * nl, tj * nl
    at_high_i = (a == n - nl)   # ti == kt-1 -> E side
    at_high_j = (b == n - nl)   # tj == kt-1 -> N side
    ua, va = d2a2c_ua_va_local(
        utmp_pad_face[None, a + h:a + h + nl, b + h:b + h + nl],
        vtmp_pad_face[None, a + h:a + h + nl, b + h:b + h + nl], cos_sg5, rsin2)
    uc, (i_spec, sin_left, sin_right) = _d2a2c_uc_iedge(
        utmp_pad_face, a, b, nl, n, npt,
        ua_pad_face, dx_pad_face, se_pad_face, sw_pad_face, at_high_i)
    vc, (j_spec, sin_below, sin_above) = _d2a2c_vc_jedge(
        vtmp_pad_face, a, b, nl, n, npt,
        va_pad_face, dy_pad_face, sn_pad_face, ss_pad_face, at_high_j)
    # ut: base + i-face override (the vt i-strip is deferred -> base).
    ut = (uc - v_d * cosa_u) * rsin_u
    sin_ui = jnp.where(uc[:, i_spec, :] > 0, sin_left, sin_right)
    ut = ut.at[:, i_spec, :].set(uc[:, i_spec, :] / jnp.maximum(sin_ui, _EPS))
    # vt: base + j-face override (the ut j-strip is deferred -> base).
    vt = (vc - u_d * cosa_v) * rsin_v
    sin_vj = jnp.where(vc[:, :, j_spec] > 0, sin_below, sin_above)
    vt = vt.at[:, :, j_spec].set(vc[:, :, j_spec] / jnp.maximum(sin_vj, _EPS))
    return ua, va, uc, vc, ut, vt


def d2a2c_uc_tile_unified(u_block, ua_block, dx_block, se_block, sw_block,
                          a, n, npt, is_lo_i, is_hi_i):
    """uc for ONE tile in the GSPMD/shard_map form — fixed-shape masks + a
    ``jnp.where`` on the boundary flags instead of per-tile static slices, so
    a SINGLE traced function serves every tile under a (6,kt,kt) mesh.

    ``u_block`` is the WIDE padded covariant-u block ``(1, nl+5, nl+4)`` with
    ``u_block[:, p, :] = utmp_pad[a-1+p, b:b+nl+4]`` (the one extra low cell
    feeds the no-shift 4th overlay for E/N/interior tiles; for a low-edge
    tile, a=0, that cell is garbage but its faces 0/1 are overwritten by the
    edge specials).  The symmetric block is ``u_block[:, 1:, :]``.
    ``ua_block``/``dx_block``/``se_block``/``sw_block`` are the standard
    symmetric tile blocks (``[a:a+nl+4]`` etc.) the edge_interpolate4 reads.
    ``a``, ``is_lo_i``, ``is_hi_i`` may be traced (``lax.axis_index`` under
    shard_map).  Returns ``(1, nl+1, nl)`` uc, equal to the specific
    W/E/interior kernel's uc for the matching tile (P4 phase-1b)."""
    h = 2
    nl = u_block.shape[1] - 5
    usym = u_block[:, 1:, :]            # (1, nl+4, nl+4) == [a:a+nl+4]
    # 2nd-order base.
    uc = 0.5 * (usym[:, h - 1:nl + h, h:-h] + usym[:, h:nl + h + 1, h:-h])
    # 4th overlay (no shift) from the wide block; uc4[k] == production uc[a+k].
    uc4 = (_A2 * (u_block[:, 0:nl + 1, h:-h] + u_block[:, 3:nl + 4, h:-h])
           + _A1 * (u_block[:, 1:nl + 2, h:-h] + u_block[:, 2:nl + 3, h:-h]))
    faces = a + jnp.arange(nl + 1)
    in_band = ((faces >= npt + 1) & (faces < n - npt)
               & (n > 2 * npt + 2))[None, :, None]
    uc = jnp.where(in_band, uc4, uc)
    # C1/C2/C3 one-sided value: at local i=1 when this is the low-i edge
    # (global i=1), at local i=nl-1 when the high-i edge (global i=n-1).
    c123_lo = d2a2c_uc_c123_local(usym, False)[:, None, :]
    c123_hi = d2a2c_uc_c123_local(usym, True)[:, None, :]
    kk = jnp.arange(nl + 1)[None, :, None]
    uc = jnp.where(is_lo_i & (kk == 1), c123_lo, uc)
    uc = jnp.where(is_hi_i & (kk == nl - 1), c123_hi, uc)
    # edge_interpolate4 + upwind: at local i=0 (low edge) / i=nl (high edge).
    ei_lo = d2a2c_uc_edge_interp_local(
        ua_block, dx_block, se_block, sw_block, False)[:, None, :]
    ei_hi = d2a2c_uc_edge_interp_local(
        ua_block, dx_block, se_block, sw_block, True)[:, None, :]
    uc = jnp.where(is_lo_i & (kk == 0), ei_lo, uc)
    uc = jnp.where(is_hi_i & (kk == nl), ei_hi, uc)
    return uc


def d2a2c_vc_tile_unified(v_block, va_block, dy_block, sn_block, ss_block,
                          b, n, npt, is_lo_j, is_hi_j):
    """vc for ONE tile in the GSPMD/shard_map form — the j-axis transpose of
    :func:`d2a2c_uc_tile_unified`.  ``v_block`` is the WIDE low-j-padded
    covariant-v block ``(1, nl+4, nl+5)`` with ``v_block[:, :, q] =
    vtmp_pad[..., b-1+q]`` (the extra low-j cell feeds the no-shift 4th
    overlay; for a low-j-edge tile, b=0, it is garbage but overwritten by the
    j=0/1 specials).  ``vsym = v_block[:, :, 1:]`` is the symmetric
    ``[b:b+nl+4]`` block.  ``b``/``is_lo_j``/``is_hi_j`` may be traced.
    Returns ``(1, nl, nl+1)`` vc, equal to the specific S/N/interior kernel's
    vc for the matching tile (P4 phase-1b)."""
    h = 2
    nl = v_block.shape[2] - 5
    vsym = v_block[:, :, 1:]            # (1, nl+4, nl+4) == [b:b+nl+4]
    vc = 0.5 * (vsym[:, h:-h, h - 1:nl + h] + vsym[:, h:-h, h:nl + h + 1])
    vc4 = (_A2 * (v_block[:, h:-h, 0:nl + 1] + v_block[:, h:-h, 3:nl + 4])
           + _A1 * (v_block[:, h:-h, 1:nl + 2] + v_block[:, h:-h, 2:nl + 3]))
    faces = b + jnp.arange(nl + 1)
    in_band = ((faces >= npt + 1) & (faces < n - npt)
               & (n > 2 * npt + 2))[None, None, :]
    vc = jnp.where(in_band, vc4, vc)
    c123_lo = d2a2c_vc_c123_local(vsym, False)[:, :, None]
    c123_hi = d2a2c_vc_c123_local(vsym, True)[:, :, None]
    mm = jnp.arange(nl + 1)[None, None, :]
    vc = jnp.where(is_lo_j & (mm == 1), c123_lo, vc)
    vc = jnp.where(is_hi_j & (mm == nl - 1), c123_hi, vc)
    ei_lo = d2a2c_vc_edge_interp_local(
        va_block, dy_block, sn_block, ss_block, False)[:, :, None]
    ei_hi = d2a2c_vc_edge_interp_local(
        va_block, dy_block, sn_block, ss_block, True)[:, :, None]
    vc = jnp.where(is_lo_j & (mm == 0), ei_lo, vc)
    vc = jnp.where(is_hi_j & (mm == nl), ei_hi, vc)
    return vc


def d2a2c_tile_unified(u_block, v_block, ua_block, dx_block, se_block, sw_block,
                       va_block, dy_block, sn_block, ss_block,
                       u_d, v_d, cos_sg5, rsin2, cosa_u, rsin_u, cosa_v, rsin_v,
                       a, b, n, npt, is_lo_i, is_hi_i, is_lo_j, is_hi_j):
    """Full per-tile A→C d2a2c in the GSPMD/shard_map form — the SINGLE traced
    body the (6,kt,kt) shard_map runs on every device.  Composes
    :func:`d2a2c_ua_va_local` + the flag-driven :func:`d2a2c_uc_tile_unified`
    / :func:`d2a2c_vc_tile_unified` + flag-driven ut/vt face overrides.  The
    two adjacent strips (vt at i=0/n-1, ut at j=0/n-1) are DEFERRED — the
    stage applies them via the same-face-neighbour ppermute / global
    post-gather — so this matches the specific per-tile kernels EXACTLY except
    at the deferred strip cells.  ``u_block``/``v_block`` are the wide
    low-padded covariant blocks; the symmetric blocks are ``u_block[:,1:,:]``
    / ``v_block[:,:,1:]``.  Position/flags may be traced.  P4 phase-1b."""
    h = 2
    nl = u_block.shape[1] - 5
    usym = u_block[:, 1:, :]
    vsym = v_block[:, :, 1:]
    ua, va = d2a2c_ua_va_local(
        usym[:, h:h + nl, h:h + nl], vsym[:, h:h + nl, h:h + nl],
        cos_sg5, rsin2)
    uc = d2a2c_uc_tile_unified(u_block, ua_block, dx_block, se_block, sw_block,
                               a, n, npt, is_lo_i, is_hi_i)
    vc = d2a2c_vc_tile_unified(v_block, va_block, dy_block, sn_block, ss_block,
                               b, n, npt, is_lo_j, is_hi_j)
    kk = jnp.arange(nl + 1)[None, :, None]
    mm = jnp.arange(nl + 1)[None, None, :]
    # ut: base + the i-face boundary override (uc/sin_upwind), flag-gated.
    ut = (uc - v_d * cosa_u) * rsin_u
    sin_lo = jnp.where(uc[:, 0, :] > 0,
                       se_block[:, 0, 1:-1], sw_block[:, 1, 1:-1])
    ut_lo = (uc[:, 0, :] / jnp.maximum(sin_lo, _EPS))[:, None, :]
    ut = jnp.where(is_lo_i & (kk == 0), ut_lo, ut)
    sin_hi = jnp.where(uc[:, nl, :] > 0,
                       se_block[:, nl, 1:-1], sw_block[:, nl + 1, 1:-1])
    ut_hi = (uc[:, nl, :] / jnp.maximum(sin_hi, _EPS))[:, None, :]
    ut = jnp.where(is_hi_i & (kk == nl), ut_hi, ut)
    # vt: base + the j-face boundary override, flag-gated.
    vt = (vc - u_d * cosa_v) * rsin_v
    sin_lo_j = jnp.where(vc[:, :, 0] > 0,
                         sn_block[:, 1:-1, 0], ss_block[:, 1:-1, 1])
    vt_lo = (vc[:, :, 0] / jnp.maximum(sin_lo_j, _EPS))[:, :, None]
    vt = jnp.where(is_lo_j & (mm == 0), vt_lo, vt)
    sin_hi_j = jnp.where(vc[:, :, nl] > 0,
                         sn_block[:, 1:-1, nl], ss_block[:, 1:-1, nl + 1])
    vt_hi = (vc[:, :, nl] / jnp.maximum(sin_hi_j, _EPS))[:, :, None]
    vt = jnp.where(is_hi_j & (mm == nl), vt_hi, vt)
    return ua, va, uc, vc, ut, vt


class _D2A2CFields(NamedTuple):
    """Global padded fields the A→C d2a2c step consumes (output of
    :func:`d2a2c_global_fields`)."""
    utmp_pad: jnp.ndarray      # (6, n+4, n+4) D→A covariant u
    vtmp_pad: jnp.ndarray      # (6, n+4, n+4) D→A covariant v
    cos_sg5: jnp.ndarray       # (6, n, n) cell-centre cos_sg
    rsin2: jnp.ndarray         # (6, n, n) cell-centre rsin2
    ua_pad: jnp.ndarray        # (6, n+4, n+4) A-grid contravariant u
    va_pad: jnp.ndarray        # (6, n+4, n+4) A-grid contravariant v
    dxc_pad_x: jnp.ndarray     # (6, n+4, n) dx, h2-padded on i
    dyc_pad_y: jnp.ndarray     # (6, n, n+4) dy, h2-padded on j
    se_pad_x: jnp.ndarray      # (6, n+2, n+2) sin_sg E, h1 halo
    sw_pad_x: jnp.ndarray      # (6, n+2, n+2) sin_sg W, h1 halo
    sn_pad_y: jnp.ndarray      # (6, n+2, n+2) sin_sg N, h1 halo
    ss_pad_y: jnp.ndarray      # (6, n+2, n+2) sin_sg S, h1 halo


def d2a2c_global_fields(u_d, v_d, cdgrid):
    """Compute the global padded fields the A→C step (and the tiled stage)
    consume: D→A covariant winds (:func:`d2a2c_d_to_a`), the A-grid
    contravariant ua/va, the staggered dx/dy, and the halo-padded sin_sg
    edge components.  Single source for both d2a2c_vect and the tiled
    per-tile kernels (P4 phase-1b approach C — the tiled stage runs the cheap
    D→A globally then shards these into the per-tile A→C)."""
    grid = cdgrid.base
    h = 2
    utmp_pad, vtmp_pad = d2a2c_d_to_a(u_d, v_d, cdgrid)
    cos_sg5 = cdgrid.cos_sg[:, :, :, 4]
    rsin2 = cdgrid.rsin2_cell
    cos_sg5_pad = jnp.pad(cos_sg5, [(0, 0), (h, h), (h, h)], mode='edge')
    rsin2_pad = jnp.pad(rsin2, [(0, 0), (h, h), (h, h)], mode='edge')
    ua_pad = (utmp_pad - vtmp_pad * cos_sg5_pad) * rsin2_pad
    va_pad = (vtmp_pad - utmp_pad * cos_sg5_pad) * rsin2_pad
    dxc_pad_x = jnp.pad(grid.dx, [(0, 0), (h, h), (0, 0)], mode='edge')
    dyc_pad_y = jnp.pad(grid.dy, [(0, 0), (0, 0), (h, h)], mode='edge')
    offsets = grid.halo_interp_offsets
    se_pad_x = pad_halo(cdgrid.sin_sg[:, :, :, 2], interp_offsets=offsets)
    sw_pad_x = pad_halo(cdgrid.sin_sg[:, :, :, 0], interp_offsets=offsets)
    sn_pad_y = pad_halo(cdgrid.sin_sg[:, :, :, 3], interp_offsets=offsets)
    ss_pad_y = pad_halo(cdgrid.sin_sg[:, :, :, 1], interp_offsets=offsets)
    return _D2A2CFields(utmp_pad, vtmp_pad, cos_sg5, rsin2, ua_pad, va_pad,
                        dxc_pad_x, dyc_pad_y, se_pad_x, sw_pad_x,
                        sn_pad_y, ss_pad_y)


def d2a2c_ua_va_halo(u_d, v_d, cdgrid):
    """FV3 D->A ``ua``/``va`` with exactly ONE A-grid halo ring.

    The corner-divergence routine needs A-grid winds that already carry a
    cross-panel ring.  Building them by a four-corner average of the D-grid
    corner winds and then scalar-padding is NOT FV3's operation: FV3 passes
    the output of ``d2a2c_vect`` (sw_core.F90:148-160), which is the
    covariant-to-contravariant D->A algebra, not an average.

    Inputs are FV3-COVARIANT D winds.  Returns ``(ua, va)`` of shape
    ``(6, n+2, n+2)`` -- physical cells at ``[1:-1, 1:-1]`` -- sliced from
    the h2 ring that :func:`d2a2c_global_fields` already computes, so no
    additional halo exchange is issued.
    """
    fields = d2a2c_global_fields(u_d, v_d, cdgrid)
    return fields.ua_pad[:, 1:-1, 1:-1], fields.va_pad[:, 1:-1, 1:-1]


def d2a2c_ua_va_halo_4d(u_d, v_d, cdgrid):
    """All-levels-one-message counterpart of :func:`d2a2c_ua_va_halo`.

    Returns ``(6, n+2, n+2, nlev)``.  Uses :func:`d2a2c_global_fields_4d`,
    whose single vector halo covers every level in one message.
    """
    fields = d2a2c_global_fields_4d(u_d, v_d, cdgrid)
    return (fields.ua_pad[:, 1:-1, 1:-1, :],
            fields.va_pad[:, 1:-1, 1:-1, :])


def d2a2c_global_fields_4d(u_d, v_d, cdgrid):
    """4D (all-levels-one-message) :func:`d2a2c_global_fields` (#811).

    The single VECTOR wind halo is done once via :func:`d2a2c_d_to_a_4d`; every
    other pad is on a grid CONSTANT (level-independent) so it stays 2D and
    broadcasts over the level axis.  The returned :class:`_D2A2CFields` has 4D
    wind fields (``utmp_pad``/``vtmp_pad``/``ua_pad``/``va_pad``, shape
    ``(6, n+4, n+4, nlev)``) and the SAME 2D grid-constant fields as the
    per-level version — so ``jax.vmap`` can map the wind fields (axis -1) and
    capture the constants (``None``) when running ``d2a2c_vect``'s A→C tail via
    its ``global_fields=`` fast path.
    """
    grid = cdgrid.base
    h = 2
    utmp_pad, vtmp_pad = d2a2c_d_to_a_4d(u_d, v_d, cdgrid)   # (6, n+4, n+4, nlev)
    cos_sg5 = cdgrid.cos_sg[:, :, :, 4]
    rsin2 = cdgrid.rsin2_cell
    cos_sg5_pad = jnp.pad(cos_sg5, [(0, 0), (h, h), (h, h)], mode='edge')
    rsin2_pad = jnp.pad(rsin2, [(0, 0), (h, h), (h, h)], mode='edge')
    # [..., None] broadcasts the 2D grid constant over the trailing level axis.
    ua_pad = (utmp_pad - vtmp_pad * cos_sg5_pad[..., None]) * rsin2_pad[..., None]
    va_pad = (vtmp_pad - utmp_pad * cos_sg5_pad[..., None]) * rsin2_pad[..., None]
    dxc_pad_x = jnp.pad(grid.dx, [(0, 0), (h, h), (0, 0)], mode='edge')
    dyc_pad_y = jnp.pad(grid.dy, [(0, 0), (0, 0), (h, h)], mode='edge')
    offsets = grid.halo_interp_offsets
    se_pad_x = pad_halo(cdgrid.sin_sg[:, :, :, 2], interp_offsets=offsets)
    sw_pad_x = pad_halo(cdgrid.sin_sg[:, :, :, 0], interp_offsets=offsets)
    sn_pad_y = pad_halo(cdgrid.sin_sg[:, :, :, 3], interp_offsets=offsets)
    ss_pad_y = pad_halo(cdgrid.sin_sg[:, :, :, 1], interp_offsets=offsets)
    return _D2A2CFields(utmp_pad, vtmp_pad, cos_sg5, rsin2, ua_pad, va_pad,
                        dxc_pad_x, dyc_pad_y, se_pad_x, sw_pad_x,
                        sn_pad_y, ss_pad_y)


def d2a2c_adjacent_strips(uc, vc, ut, vt, cosa_u, cosa_v, n):
    """Non-duogrid adjacent-strip recompute of the transport winds at the
    face edges (FV3 sw_core.F90:670-722): vt at i=0/n-1 and ut at j=0/n-1,
    each over the [2,n-2] interior strip, blending the cross-component
    transport wind from the two adjacent cells.  Shared by d2a2c_vect and the
    tiled stage's global post-gather pass.  vt strips run first; the ut strips
    read the updated vt but only at i-cells [1,n-2] (the West/East vt strips
    touch i-cell 0/n-1 only — no overlap, FV3 ordering preserved).  Returns
    (ut, vt).  P4 phase-1b.
    """
    if n >= 4:
        j_lo, j_hi = 2, n - 1  # j_face range [2, n-2]
        ut_w = (ut[:, 0, j_lo - 1:j_hi - 1] + ut[:, 1, j_lo - 1:j_hi - 1]
                + ut[:, 0, j_lo:j_hi] + ut[:, 1, j_lo:j_hi])
        vt = vt.at[:, 0, j_lo:j_hi].set(
            vc[:, 0, j_lo:j_hi] - 0.25 * cosa_v[:, 0, j_lo:j_hi] * ut_w)
        ut_e = (ut[:, n - 1, j_lo - 1:j_hi - 1] + ut[:, n, j_lo - 1:j_hi - 1]
                + ut[:, n - 1, j_lo:j_hi] + ut[:, n, j_lo:j_hi])
        vt = vt.at[:, n - 1, j_lo:j_hi].set(
            vc[:, n - 1, j_lo:j_hi] - 0.25 * cosa_v[:, n - 1, j_lo:j_hi] * ut_e)
        i_lo, i_hi = 2, n - 1  # i_face range [2, n-2]
        vt_s = (vt[:, i_lo - 1:i_hi - 1, 0] + vt[:, i_lo:i_hi, 0]
                + vt[:, i_lo - 1:i_hi - 1, 1] + vt[:, i_lo:i_hi, 1])
        ut = ut.at[:, i_lo:i_hi, 0].set(
            uc[:, i_lo:i_hi, 0] - 0.25 * cosa_u[:, i_lo:i_hi, 0] * vt_s)
        vt_n = (vt[:, i_lo - 1:i_hi - 1, n - 1] + vt[:, i_lo:i_hi, n - 1]
                + vt[:, i_lo - 1:i_hi - 1, n] + vt[:, i_lo:i_hi, n])
        ut = ut.at[:, i_lo:i_hi, n - 1].set(
            uc[:, i_lo:i_hi, n - 1] - 0.25 * cosa_u[:, i_lo:i_hi, n - 1] * vt_n)
    return ut, vt


def d2a2c_tile_strips(uc, vc, ut, vt, ut_lo, ut_hi, vt_lo, vt_hi,
                      cosa_u, cosa_v, a, b, n, nl,
                      is_lo_i, is_hi_i, is_lo_j, is_hi_j):
    """Per-tile form of :func:`d2a2c_adjacent_strips` (the tiled stage's
    step 2a — removes the global post-gather strip pass).

    Inputs are one tile's kernel outputs (``uc``/``ut`` ``(1, nl+1, nl)``,
    ``vc``/``vt`` ``(1, nl, nl+1)``) plus the four 1-cell same-face
    neighbour halos (each ``(1, nl+1)``):

    - ``ut_lo``/``ut_hi``: neighbour ut CELL columns ``b-1`` / ``b+nl``
      (tile ``tile_j∓1``'s local column ``nl-1`` / ``0``),
    - ``vt_lo``/``vt_hi``: neighbour vt CELL rows ``a-1`` / ``a+nl``
      (tile ``tile_i∓1``'s local row ``nl-1`` / ``0``).

    The strip masks (global range ``[2, n-2]``) provably exclude every
    position whose halo would be the wrapped/garbage value at a face
    boundary, so the stage can feed periodic-``ppermute`` halos
    unconditionally.  Writes cover the FULL local staggered range
    (including the duplicated shared faces), so neighbouring tiles'
    duplicated copies remain bit-identical — same global cells, same
    fp ops — and downstream consumers never depend on tile ownership.
    Production-order equivalence: the in-mask vt-strip reads
    (post-face-override ut at cell cols ``[1, n-2]``) and ut-strip reads
    (vt at cell rows ``[1, n-2]``) are disjoint from all strip writes
    (i/j cells ``{0, n-1}``), so applying both from the PRE-strip fields
    matches :func:`d2a2c_adjacent_strips` exactly.  ``a``/``b`` and the
    side flags may be traced (shard_map ``axis_index``).  Returns
    ``(ut, vt)``.  P4 phase-1b.
    """
    # Extended cell-axis views: [lo halo | local | hi halo].
    ut_ext = jnp.concatenate(
        [ut_lo[:, :, None], ut, ut_hi[:, :, None]], axis=2)  # (1,nl+1,nl+2)
    vt_ext = jnp.concatenate(
        [vt_lo[:, None, :], vt, vt_hi[:, None, :]], axis=1)  # (1,nl+2,nl+1)
    mm = jnp.arange(nl + 1)[None, :]
    j_in = (b + mm >= 2) & (b + mm <= n - 2)   # vt-strip staggered cols
    i_in = (a + mm >= 2) & (a + mm <= n - 2)   # ut-strip staggered rows

    # vt strips first (FV3 order; reads PRE-strip ut).
    # West (cell row 0 of an is_lo_i tile): vt[0, j] = vc[0, j]
    #   - 0.25*cosa_v[0, j]*(ut[0:2, j-1] + ut[0:2, j]).
    sum_w = (ut_ext[:, 0, :-1] + ut_ext[:, 1, :-1]
             + ut_ext[:, 0, 1:] + ut_ext[:, 1, 1:])
    val_w = vc[:, 0, :] - 0.25 * cosa_v[:, 0, :] * sum_w
    vt = vt.at[:, 0, :].set(jnp.where(is_lo_i & j_in, val_w, vt[:, 0, :]))
    # East (cell row nl-1 == global n-1 of an is_hi_i tile).
    sum_e = (ut_ext[:, nl - 1, :-1] + ut_ext[:, nl, :-1]
             + ut_ext[:, nl - 1, 1:] + ut_ext[:, nl, 1:])
    val_e = vc[:, nl - 1, :] - 0.25 * cosa_v[:, nl - 1, :] * sum_e
    vt = vt.at[:, nl - 1, :].set(
        jnp.where(is_hi_i & j_in, val_e, vt[:, nl - 1, :]))

    # ut strips (read vt at interior cell rows only — disjoint from the
    # vt-strip writes above, so vt_ext built from the PRE-strip vt is
    # production-exact at every in-mask position).
    # South (cell col 0 of an is_lo_j tile).
    sum_s = (vt_ext[:, :-1, 0] + vt_ext[:, 1:, 0]
             + vt_ext[:, :-1, 1] + vt_ext[:, 1:, 1])
    val_s = uc[:, :, 0] - 0.25 * cosa_u[:, :, 0] * sum_s
    ut = ut.at[:, :, 0].set(jnp.where(is_lo_j & i_in, val_s, ut[:, :, 0]))
    # North (cell col nl-1 == global n-1 of an is_hi_j tile).
    sum_n = (vt_ext[:, :-1, nl - 1] + vt_ext[:, 1:, nl - 1]
             + vt_ext[:, :-1, nl] + vt_ext[:, 1:, nl])
    val_n = uc[:, :, nl - 1] - 0.25 * cosa_u[:, :, nl - 1] * sum_n
    ut = ut.at[:, :, nl - 1].set(
        jnp.where(is_hi_j & i_in, val_n, ut[:, :, nl - 1]))
    return ut, vt


def d2a2c_vect(u_d, v_d, cdgrid, global_fields=None):
    """FV3 D-grid → A-grid → C-grid (covariant). FV3 sw_core.F90 d2a2c_vect.

    Dispatches to _d2a2c_vect_duogrid when dg.ng>=2 (FV3 dg%is_initialized branch).
    Returns ua/va (A-cov), uc/vc (C-cov), ut/vt (C-contravariant transport).

    ``global_fields`` (#811): a precomputed :class:`_D2A2CFields` (the sole
    cross-face wind halo).  When provided the internal ``d2a2c_global_fields``
    call is SKIPPED and the pure-local A→C tail runs on the supplied fields —
    the fast path :func:`d2a2c_vect_4d` uses to reconstruct the transport winds
    for all levels from ONE ``pad_halo_vector_4d`` (no ``vmap(pad_halo)``).
    Mirrors the ``padded=`` kwarg idiom on the ``operators_3d`` stencils.
    """
    # Duogrid path: 4th-order everywhere, skip edge/corner specials (FV3 sw_core.F90:3419)
    dg = cdgrid.base.duogrid
    if dg is not None and dg.ng >= 2:
        return _d2a2c_vect_duogrid(u_d, v_d, cdgrid)

    n = cdgrid.n
    npt = min(4, n // 2)

    # iter-108 faithfulness gap (Priority 3): Fortran's cube-vertex corner
    # overrides for utmp/vtmp and ua/va (sw_core.F90:3527-3545 and 3620-3640 —
    # sign-flipped copies of the OTHER component from the adjacent face) are
    # NOT PORTED to this non-duogrid path.  Python instead uses pad_halo_vector
    # + fill_corners_h1/fill_corners_h2 (2-point edge-halo AVERAGE) — a
    # DIFFERENT convention that gives DIFFERENT values at cube-vertex cells
    # (O(1) on random input, O(dx^2) on smooth fields).  The numerical impact
    # on the non-duogrid FB path has NOT been quantified.  (Duogrid path via
    # _d2a2c_vect_duogrid is unaffected — Fortran also skips these via
    # dg%is_initialized.)

    # Steps 1+2: D→A covariant utmp/vtmp + halo=2 vector exchange.
    # Extracted to d2a2c_d_to_a (P4 phase-1b approach C): the tiled
    # stage runs THIS step in the global view (cheap averages + one
    # vector halo) and shards utmp_pad/vtmp_pad into the per-tile A→C.
    h = 2
    # Steps 1-3 global fields (D→A covariant winds + halo, A-grid
    # contravariant ua/va, staggered dx/dy, halo-padded sin_sg) — the single
    # source shared with the tiled per-tile stage (d2a2c_global_fields).
    F = (global_fields if global_fields is not None
         else d2a2c_global_fields(u_d, v_d, cdgrid))
    utmp_pad, vtmp_pad = F.utmp_pad, F.vtmp_pad  # each (6, n+4, n+4)

    # iter-938: Fortran cube-corner sign-flip overrides (sw_core.F90:3527-3545, 3620-3639) available
    # as _apply_fortran_d2a2c_corner_overrides but output-dead without edge_interpolate4 j-slice extension.

    ua_pad, va_pad = F.ua_pad, F.va_pad
    ua = ua_pad[:, h:-h, h:-h]  # (6, n, n)
    va = va_pad[:, h:-h, h:-h]

    # Step 4a: A→C x-dir. u-face i ↔ padded indices i+h-1, i+h
    uc = 0.5 * (utmp_pad[:, h-1:n+h, h:-h]
                + utmp_pad[:, h:n+h+1, h:-h])  # (6, n+1, n)

    if n > 2 * npt + 2:
        uc_4th = d2a2c_uc_4th_local(utmp_pad)  # (6, n+1, n)
        i_lo = npt + 1
        i_hi = n - npt
        uc = uc.at[:, i_lo:i_hi, :].set(uc_4th[:, i_lo - 1:i_hi - 1, :])

    # One-sided c1/c2/c3 stencil at i=1, n-1 (FV3 sw_core.F90:3586,3594)
    if n > 3:
        uc = uc.at[:, 1, :].set(
            _C1 * utmp_pad[:, h + 2, h:-h] + _C2 * utmp_pad[:, h + 1, h:-h]
            + _C3 * utmp_pad[:, h, h:-h])
        uc = uc.at[:, n - 1, :].set(
            _C1 * utmp_pad[:, n + h - 3, h:-h] + _C2 * utmp_pad[:, n + h - 2, h:-h]
            + _C3 * utmp_pad[:, n + h - 1, h:-h])

    # Face boundary (i=0, i=n): edge_interpolate4 on ua (FV3:3587,3603); halo=2 straddles boundary.
    # Upwind sin_sg from halo cell: ut>0 → sin_sg(i-1,j,3); ut<=0 → sin_sg(i,j,1)
    dxc_pad_x = F.dxc_pad_x
    se_pad_x, sw_pad_x = F.se_pad_x, F.sw_pad_x  # (6, n+2, n+2)
    for i_bdy in ([0, n] if n >= 2 else []):
        i_p = i_bdy + h  # padded offset: cell i → padded index i+h
        # ua_pad stencil: 4 cells centred on u-face i_bdy
        ua4 = jnp.stack([ua_pad[:, i_p - 1, h:-h], ua_pad[:, i_p, h:-h],
                         ua_pad[:, i_p + 1, h:-h], ua_pad[:, i_p + 2, h:-h]],
                        axis=-1)
        dxa4 = jnp.stack([dxc_pad_x[:, i_p - 1, :], dxc_pad_x[:, i_p, :],
                          dxc_pad_x[:, i_p + 1, :], dxc_pad_x[:, i_p + 2, :]],
                         axis=-1)
        ut_bdy = _edge_interpolate4(ua4, dxa4)

        # FV3:3589-3592 upwind sin_sg from halo cell
        sin_left = se_pad_x[:, i_bdy, 1:-1]
        sin_right = sw_pad_x[:, i_bdy + 1, 1:-1]  # W-edge of cell to RIGHT of face i_bdy
        uc_bdy = jnp.where(ut_bdy > 0, ut_bdy * sin_left, ut_bdy * sin_right)
        uc = uc.at[:, i_bdy, :].set(uc_bdy)

    # Contravariant ut (rsin_u = 1/sin²)
    ut = (uc - v_d * cdgrid.cosa_u) * cdgrid.rsin_u

    # Face boundary: ut = uc/sin_upwind to recover edge_interpolate4 result (FV3:3587,3603)
    for i_bdy in ([0, n] if n >= 2 else []):
        sin_left = se_pad_x[:, i_bdy, 1:-1]
        sin_right = sw_pad_x[:, i_bdy + 1, 1:-1]
        sin_upwind = jnp.where(uc[:, i_bdy, :] > 0, sin_left, sin_right)
        ut = ut.at[:, i_bdy, :].set(
            uc[:, i_bdy, :] / jnp.maximum(sin_upwind, _EPS))

    # Step 4b: A→C y-dir. v-face j ↔ padded j+h-1, j+h
    vc = 0.5 * (vtmp_pad[:, h:-h, h-1:n+h] + vtmp_pad[:, h:-h, h:n+h+1])  # (6, n, n+1)

    if n > 2 * npt + 2:
        vc_4th = d2a2c_vc_4th_local(vtmp_pad)  # (6, n, n+1)
        j_lo = npt + 1
        j_hi = n - npt
        vc = vc.at[:, :, j_lo:j_hi].set(vc_4th[:, :, j_lo - 1:j_hi - 1])

    # One-sided c1/c2/c3 at j=1, n-1
    if n > 3:
        vc = vc.at[:, :, 1].set(
            _C1 * vtmp_pad[:, h:-h, h + 2] + _C2 * vtmp_pad[:, h:-h, h + 1]
            + _C3 * vtmp_pad[:, h:-h, h])
        vc = vc.at[:, :, n - 1].set(
            _C1 * vtmp_pad[:, h:-h, n + h - 3] + _C2 * vtmp_pad[:, h:-h, n + h - 2]
            + _C3 * vtmp_pad[:, h:-h, n + h - 1])

    # Face boundary y-dir: edge_interpolate4 on va (symmetric to x-dir)
    dyc_pad_y = F.dyc_pad_y
    sn_pad_y, ss_pad_y = F.sn_pad_y, F.ss_pad_y
    for j_bdy in ([0, n] if n >= 2 else []):
        j_p = j_bdy + h
        va4 = jnp.stack([va_pad[:, h:-h, j_p - 1], va_pad[:, h:-h, j_p],
                         va_pad[:, h:-h, j_p + 1], va_pad[:, h:-h, j_p + 2]],
                        axis=-1)
        dya4 = jnp.stack([dyc_pad_y[:, :, j_p - 1], dyc_pad_y[:, :, j_p],
                          dyc_pad_y[:, :, j_p + 1], dyc_pad_y[:, :, j_p + 2]],
                         axis=-1)
        vt_bdy = _edge_interpolate4(va4, dya4)

        # Upwind sin_sg from halo cell
        sin_below = sn_pad_y[:, 1:-1, j_bdy]
        sin_above = ss_pad_y[:, 1:-1, j_bdy + 1]
        vc_bdy = jnp.where(vt_bdy > 0, vt_bdy * sin_below, vt_bdy * sin_above)
        vc = vc.at[:, :, j_bdy].set(vc_bdy)

    vt = (vc - u_d * cdgrid.cosa_v) * cdgrid.rsin_v

    # Override at face boundary: vt = vc/sin_upwind
    for j_bdy in [0, n]:
        sin_below = sn_pad_y[:, 1:-1, j_bdy]
        sin_above = ss_pad_y[:, 1:-1, j_bdy + 1]
        sin_upwind = jnp.where(vc[:, :, j_bdy] > 0, sin_below, sin_above)
        vt = vt.at[:, :, j_bdy].set(
            vc[:, :, j_bdy] / jnp.maximum(sin_upwind, _EPS))

    # Non-duogrid adjacent-strip recompute of ut/vt at the face edges
    # (FV3 sw_core.F90:670-722) — extracted so the tiled stage reuses it.
    ut, vt = d2a2c_adjacent_strips(
        uc, vc, ut, vt, cdgrid.cosa_u, cdgrid.cosa_v, n)

    return ua, va, uc, vc, ut, vt


def d2a2c_vect_4d(u_d, v_d, cdgrid):
    """4D (all-levels-one-message) :func:`d2a2c_vect` for the flux-form moisture
    substep (#811).

    Does the ONE cross-face wind halo (+ grid-constant halos) globally via
    :func:`d2a2c_global_fields_4d`, then ``vmap``s the PURE-LOCAL A→C tail of
    ``d2a2c_vect`` over levels (through its ``global_fields=`` fast path, so the
    A→C numerics are shared bit-for-bit — no duplication).  This replaces the
    per-level ``vmap(d2a2c_vect)`` whose internal ``pad_halo_vector`` was a
    vmapped cross-face ``sendrecv`` (``batch_axes`` failure under MPI
    face-scatter).  BIT-IDENTICAL to ``jax.vmap(d2a2c_vect)`` on single-rank.

    NON-duogrid only (the duogrid ``_d2a2c_vect_duogrid`` path is not 4D-ified;
    the moisture substep runs on the non-duogrid grid).

    u_d : (6, n, n+1, nlev); v_d : (6, n+1, n, nlev).  Returns
    ``(ua, va, uc, vc, ut, vt)``, each with a trailing level axis.
    """
    dg = cdgrid.base.duogrid
    if dg is not None and dg.ng >= 2:
        raise NotImplementedError(
            "d2a2c_vect_4d does not support duogrid grids (the 4th-order-"
            "everywhere duogrid A→C path is not 4D-ified); the flux-form "
            "moisture substep runs on the non-duogrid grid (#811).")

    F = d2a2c_global_fields_4d(u_d, v_d, cdgrid)
    # Map the 4D wind fields over the trailing level axis; capture the 2D grid
    # constants (None).  Mirrors _D2A2CFields' field order.
    f_axes = _D2A2CFields(
        utmp_pad=-1, vtmp_pad=-1, cos_sg5=None, rsin2=None,
        ua_pad=-1, va_pad=-1, dxc_pad_x=None, dyc_pad_y=None,
        se_pad_x=None, sw_pad_x=None, sn_pad_y=None, ss_pad_y=None)
    return jax.vmap(
        lambda fk, udk, vdk: d2a2c_vect(udk, vdk, cdgrid, global_fields=fk),
        in_axes=(f_axes, -1, -1), out_axes=-1)(F, u_d, v_d)


# ==============================================================================
# Shared FV3 c_sw helpers (used by _c_sw)
# ==============================================================================


def sina_u_v_from_sin_sg(cdgrid):
    """Return `sina_u` (6, n+1, n) and `sina_v` (6, n, n+1) constructed
    from FV3 sub-grid `sin_sg` per ``fv_grid_utils.F90:505-518``.

    Interior faces:
      sina_u(i,j) = 0.5*(sin_sg(i-1,j,3) + sin_sg(i,j,1))
      sina_v(i,j) = 0.5*(sin_sg(i,j-1,4) + sin_sg(i,j,2))

    Panel-edge faces: use the single-side sin_sg at the outermost cell,
    matching the sina_u/sina_v construction in `cubed_sphere_cdgrid.py`
    and `d_sw5_corner_divergence`.

    Using this formulation instead of ``sqrt(1 - cosa_u**2)`` is the
    Fortran-faithful convention — the two are only identical when
    ``cosa**2 + sina**2 = 1`` exactly, which is NOT the case for
    halo-averaged ``cosa_u = 0.5*(cos_sg(E) + cos_sg(W))``.
    """
    sg = cdgrid.sin_sg  # (6, n, n, 9): 0=W, 1=S, 2=E, 3=N, 4=center, ...
    sin_E = sg[:, :, :, 2]
    sin_W = sg[:, :, :, 0]
    sin_N = sg[:, :, :, 3]
    sin_S = sg[:, :, :, 1]

    sina_u_int = 0.5 * (sin_E[:, :-1, :] + sin_W[:, 1:, :])  # (6, n-1, n)
    sina_u = jnp.concatenate(
        [sin_W[:, :1, :], sina_u_int, sin_E[:, -1:, :]], axis=1,
    )  # (6, n+1, n)

    sina_v_int = 0.5 * (sin_N[:, :, :-1] + sin_S[:, :, 1:])  # (6, n, n-1)
    sina_v = jnp.concatenate(
        [sin_S[:, :, :1], sina_v_int, sin_N[:, :, -1:]], axis=2,
    )  # (6, n, n+1)

    return sina_u, sina_v


def _ke_upwind(uc, vc, ua, va, u_d, v_d, cdgrid, use_duogrid):
    """FV3 c_sw KE upwind selection (sw_core.F90:303-365).

    Returns ke_u, ke_v — the upwind-selected covariant velocities for KE.
    Non-duogrid path applies sin_sg/cos_sg conversion at face boundaries.
    """
    n = cdgrid.n
    ke_u = jnp.where(ua > 0, uc[:, :-1, :], uc[:, 1:, :])
    ke_v = jnp.where(va > 0, vc[:, :, :-1], vc[:, :, 1:])

    if not use_duogrid:
        sg = cdgrid.sin_sg
        cg = cdgrid.cos_sg
        # West edge (cell 0, ua > 0): uc*sin_sg(W) + v*cos_sg(W)
        ke_bdy_l = uc[:, 0, :] * sg[:, 0, :, 0] + v_d[:, 0, :] * cg[:, 0, :, 0]
        ke_u = ke_u.at[:, 0, :].set(
            jnp.where(ua[:, 0, :] > 0, ke_bdy_l, ke_u[:, 0, :]))
        # East edge (cell n-1, ua <= 0): uc*sin_sg(E) + v*cos_sg(E)
        ke_bdy_r = uc[:, n, :] * sg[:, n-1, :, 2] + v_d[:, n, :] * cg[:, n-1, :, 2]
        ke_u = ke_u.at[:, n-1, :].set(
            jnp.where(ua[:, n-1, :] > 0, ke_u[:, n-1, :], ke_bdy_r))
        # South edge (cell j=0, va > 0): vc*sin_sg(S) + u*cos_sg(S)
        ke_bdy_b = vc[:, :, 0] * sg[:, :, 0, 1] + u_d[:, :, 0] * cg[:, :, 0, 1]
        ke_v = ke_v.at[:, :, 0].set(
            jnp.where(va[:, :, 0] > 0, ke_bdy_b, ke_v[:, :, 0]))
        # North edge (cell j=n-1, va <= 0): vc*sin_sg(N) + u*cos_sg(N)
        ke_bdy_t = vc[:, :, n] * sg[:, :, n-1, 3] + u_d[:, :, n] * cg[:, :, n-1, 3]
        ke_v = ke_v.at[:, :, n-1].set(
            jnp.where(va[:, :, n-1] > 0, ke_v[:, :, n-1], ke_bdy_t))

    return ke_u, ke_v


def _del6_vt_flux(nord, damp, q, cdgrid, use_duogrid=False):
    """FV3 del6_vt_flux: del-n vorticity damping (sw_core.F90:2008-2121).

    nord 0=del-2, 1=del-4, 2=del-6. damp = (damp_v * da_min_c)^(nord+1).
    Used in d_sw6 when damp_v > 1e-5. Returns (fx2, fy2) raw diffusive fluxes.
    """
    n = cdgrid.n
    grid = cdgrid.base
    sg = cdgrid.sin_sg
    dy = cdgrid.dy_edge_x
    dx = cdgrid.dx_edge_y
    rdxc = cdgrid.rdxc
    rdyc = cdgrid.rdyc
    rarea = 1.0 / grid.area

    # Route halos through duogrid remap when active (FV3 bounded_domain path)
    dg = grid.duogrid if use_duogrid else None
    _offs = None if use_duogrid else grid.halo_interp_offsets

    # iter-937b: defer damp factor to end (float32 overflow guard; LINEAR in d2)
    d2 = q

    # Laplacian diffusive fluxes (USE_SG path, sw_core.F90:2064-2082)
    d2_pad = pad_halo(d2, interp_offsets=_offs, duogrid=dg)
    sin_E = sg[:, :, :, 2]
    sin_W = sg[:, :, :, 0]
    sin_N = sg[:, :, :, 3]
    sin_S = sg[:, :, :, 1]
    se_pad = pad_halo(sin_E, interp_offsets=_offs, duogrid=dg)
    sw_pad = pad_halo(sin_W, interp_offsets=_offs, duogrid=dg)
    sn_pad = pad_halo(sin_N, interp_offsets=_offs, duogrid=dg)
    ss_pad = pad_halo(sin_S, interp_offsets=_offs, duogrid=dg)

    sin_uv_x = 0.5 * (se_pad[:, :n+1, 1:-1] + sw_pad[:, 1:n+2, 1:-1])
    sin_uv_y = 0.5 * (sn_pad[:, 1:-1, :n+1] + ss_pad[:, 1:-1, 1:n+2])

    fx2 = sin_uv_x * dy * (d2_pad[:, :-1, 1:-1] - d2_pad[:, 1:, 1:-1]) * rdxc
    fy2 = sin_uv_y * dx * (d2_pad[:, 1:-1, :-1] - d2_pad[:, 1:-1, 1:]) * rdyc

    # Higher-order iteration (sw_core.F90:2084-2119)
    for _it in range(nord):
        d2 = (fx2[:, :-1, :] - fx2[:, 1:, :] + fy2[:, :, :-1] - fy2[:, :, 1:]) * rarea
        d2_pad = pad_halo(d2, interp_offsets=_offs, duogrid=dg)
        fx2 = sin_uv_x * dy * (d2_pad[:, 1:, 1:-1] - d2_pad[:, :-1, 1:-1]) * rdxc
        fy2 = sin_uv_y * dx * (d2_pad[:, 1:-1, 1:] - d2_pad[:, 1:-1, :-1]) * rdyc

    # iter-937b: apply deferred damp factor
    fx2 = damp * fx2
    fy2 = damp * fy2

    return fx2, fy2


def _divergence_corner_duo(u_d, v_d, ua, va, cdgrid, *,
                           dxc=None, dyc=None, rarea_c=None):
    """FV3 divergence_corner_duo (sw_core.F90:2345-2447). Corner divergence for nord>0 hyperviscosity.

    Cross-velocity correction via cos_sg/sin_sg. Face-boundary zeroing + 0.25 attenuation.
    ``dxc``/``dyc``/``rarea_c`` overrides: the faithful-ring D5 bundle
    (bounded-gridstruct geometry) — default None keeps cdgrid's fields
    byte-identical for every existing caller.
    """
    n = cdgrid.n
    sg = cdgrid.sin_sg
    cg = cdgrid.cos_sg
    dxc = cdgrid.dxc if dxc is None else dxc   # (6, n+1, n)
    dyc = cdgrid.dyc if dyc is None else dyc   # (6, n, n+1)
    rarea_c = cdgrid.rarea_c if rarea_c is None else rarea_c

    # iter-657/949: mode='edge' here is a numerical no-op (face-boundary zeroing kills the diff)
    ua_pad = jnp.pad(ua, [(0, 0), (1, 1), (0, 0)], mode='edge')  # (6, n+2, n)
    va_pad = jnp.pad(va, [(0, 0), (0, 0), (1, 1)], mode='edge')  # (6, n, n+2)

    # uf: u-flux at v-face (FV3:2413-2418). uf = (u - 0.25*va_avg*cos_sum)*dyc*0.5*sin_sum
    va_below = va_pad[:, :, :-1]
    va_above = va_pad[:, :, 1:]

    cos_N = cg[:, :, :, 3]
    cos_S = cg[:, :, :, 1]  # (6, n, n)
    sin_N = sg[:, :, :, 3]  # (6, n, n)
    sin_S = sg[:, :, :, 1]  # (6, n, n)
    cos_N_pad = jnp.pad(cos_N, [(0, 0), (0, 0), (1, 1)], mode='edge')
    cos_S_pad = jnp.pad(cos_S, [(0, 0), (0, 0), (1, 1)], mode='edge')
    sin_N_pad = jnp.pad(sin_N, [(0, 0), (0, 0), (1, 1)], mode='edge')
    sin_S_pad = jnp.pad(sin_S, [(0, 0), (0, 0), (1, 1)], mode='edge')

    cos_sum_u = cos_N_pad[:, :, :-1] + cos_S_pad[:, :, 1:]  # (6, n, n+1)
    sin_sum_u = sin_N_pad[:, :, :-1] + sin_S_pad[:, :, 1:]
    uf = (u_d - 0.25 * (va_below + va_above) * cos_sum_u) * dyc * 0.5 * sin_sum_u

    # vf: v-flux at u-face (FV3:2420-2425). Symmetric to uf.
    ua_left = ua_pad[:, :-1, :]
    ua_right = ua_pad[:, 1:, :]

    cos_E = cg[:, :, :, 2]
    cos_W = cg[:, :, :, 0]  # (6, n, n) W-edge
    sin_E = sg[:, :, :, 2]
    sin_W = sg[:, :, :, 0]
    cos_E_pad = jnp.pad(cos_E, [(0, 0), (1, 1), (0, 0)], mode='edge')
    cos_W_pad = jnp.pad(cos_W, [(0, 0), (1, 1), (0, 0)], mode='edge')
    sin_E_pad = jnp.pad(sin_E, [(0, 0), (1, 1), (0, 0)], mode='edge')
    sin_W_pad = jnp.pad(sin_W, [(0, 0), (1, 1), (0, 0)], mode='edge')

    cos_sum_v = cos_E_pad[:, :-1, :] + cos_W_pad[:, 1:, :]  # (6, n+1, n)
    sin_sum_v = sin_E_pad[:, :-1, :] + sin_W_pad[:, 1:, :]
    vf = (v_d - 0.25 * (ua_left + ua_right) * cos_sum_v) * dxc * 0.5 * sin_sum_v

    # divg_d at corners (FV3:2427-2442). divg_d = (vf[j-1]-vf[j] + uf[i-1]-uf[i])*rarea_c
    vf_pad = jnp.pad(vf, [(0, 0), (0, 0), (1, 1)], mode='edge')  # (6, n+1, n+2)
    uf_pad = jnp.pad(uf, [(0, 0), (1, 1), (0, 0)], mode='edge')  # (6, n+2, n+1)

    divg_d = (vf_pad[:, :, :-1] - vf_pad[:, :, 1:]
              + uf_pad[:, :-1, :] - uf_pad[:, 1:, :]) * rarea_c

    # Face-boundary zeroing (FV3:2431-2434)
    divg_d = divg_d.at[:, 0, :].set(0.0)
    divg_d = divg_d.at[:, n, :].set(0.0)
    divg_d = divg_d.at[:, :, 0].set(0.0)
    divg_d = divg_d.at[:, :, n].set(0.0)

    # 0.25x attenuation at face-adjacent cells (FV3:2437-2440)
    divg_d = divg_d.at[:, 1, :].multiply(0.25)
    divg_d = divg_d.at[:, n - 1, :].multiply(0.25)
    divg_d = divg_d.at[:, :, 1].multiply(0.25)
    divg_d = divg_d.at[:, :, n - 1].multiply(0.25)

    return divg_d


def _pad_corner_scalar_cross_face(field, n):
    """1-ring cross-face halo pad for a B-grid corner scalar.

    (6, n+1, n+1) -> (6, n+3, n+3).  The halo row one beyond each panel edge
    is filled with the corner row ONE INSIDE the neighbouring face's shared
    edge (scalar value copy, index-reversed per CONNECTIVITY).

    Fortran oracle (2026-07-10 port): for duogrid with nord>0, dyn_core.F90
    :651-652 runs a dedicated B-grid ghost exchange of divgd between c_sw
    and d_sw — `ext_scalar(divgd, dg, bd, domain, 1, 1)` →
    fv_duogrid.F90::ext_scalar_3d(istag=jstag=1): mpp_update_domains at
    NORTH+EAST corner position followed by cube_rmp onto the duogrid
    extension points.  The ghost ring therefore holds the NEIGHBOUR's
    divgd — including its divergence_corner_duo panel-edge zeroing and
    0.25x attenuation — interpolated to the extension positions.  This
    helper APPROXIMATES the cube_rmp interpolation by a nearest-row value
    copy (measured: max tangential position error ~1 dx at cube vertices,
    ~0.1 dx at mid-edge; codex 2026-07-10: near strip ends the nominal
    remap targets fall between neighbour indices, so with the zero/0.25
    boundary profile the copied endpoint reads 0 where cube_rmp would give
    ~0.0625q — the tests lock THIS approximation, not oracle semantics).
    A ghost ring built from the RAW (un-attenuated) divergence instead was
    measured to destabilise the colliding-modon C48 FB run (NaN at day 8
    vs stable decay through day 20+ with the attenuated ghost).

    Pad corners (vertex-diagonal points) are never read by the d_sw5
    gradient stencil; they keep the edge-copy base pad.
    """
    padded = jnp.pad(field, [(0, 0), (1, 1), (1, 1)], mode='edge')
    for face in range(6):
        for edge in (WEST, EAST, SOUTH, NORTH):
            nbr_f, nbr_e, rev = CONNECTIVITY[face][edge]
            # Neighbour corner row one inside its shared edge.
            if nbr_e == WEST:
                strip = field[nbr_f, 1, :]
            elif nbr_e == EAST:
                strip = field[nbr_f, n - 1, :]
            elif nbr_e == SOUTH:
                strip = field[nbr_f, :, 1]
            else:  # NORTH
                strip = field[nbr_f, :, n - 1]
            if rev:
                strip = strip[::-1]
            if edge == WEST:
                padded = padded.at[face, 0, 1:n + 2].set(strip)
            elif edge == EAST:
                padded = padded.at[face, n + 2, 1:n + 2].set(strip)
            elif edge == SOUTH:
                padded = padded.at[face, 1:n + 2, 0].set(strip)
            else:  # NORTH
                padded = padded.at[face, 1:n + 2, n + 2].set(strip)
    return padded


def _apply_legacy_d_sw4_corner_ke_fix(
        ke, ut, vt, u_d, v_d, dt,
        bounded_domain: bool):
    """iter-869: 4 cube-vertex KE overrides (FV3 sw_core.F90:1442-1465; legacy d_sw4 non-duogrid).

    Gated by Fortran .not.bounded_domain. RHS reads halo cells (mode='edge' fallback;
    cross-face D-grid edge halo deferred to iter-870+).
    """
    if bounded_domain:
        return ke

    dt6 = dt / 6.0

    # Pad with mode='edge' for halo cells (cross-face upgrade in iter-870+)
    ut_pad = jnp.pad(ut, [(0, 0), (0, 0), (1, 1)], mode='edge')   # (6, n+1, n+2)
    vt_pad = jnp.pad(vt, [(0, 0), (1, 1), (0, 0)], mode='edge')
    u_pad = jnp.pad(u_d, [(0, 0), (1, 1), (0, 0)], mode='edge')
    jnp.pad(v_d, [(0, 0), (0, 0), (1, 1)], mode='edge')

    ke.shape[1] - 1

    # SW corner
    sw_value = dt6 * (
        (ut[:, 0, 0] + ut_pad[:, 0, 0]) * u_d[:, 0, 0]
        + (vt[:, 0, 0] + vt_pad[:, 0, 0]) * v_d[:, 0, 0]
        + (ut[:, 0, 0] + vt[:, 0, 0]) * u_pad[:, 0, 0]
    )

    # SE corner
    se_value = dt6 * (
        (ut[:, -1, 0] + ut_pad[:, -1, 0]) * u_d[:, -1, 0]
        + (vt_pad[:, -1, 0] + vt[:, -1, 0]) * v_d[:, -1, 0]
        + (ut[:, -1, 0] - vt[:, -1, 0]) * u_pad[:, -1, 0]
    )

    # NE corner
    ne_value = dt6 * (
        (ut[:, -1, -1] + ut[:, -1, -2]) * u_d[:, -1, -1]
        + (vt[:, -1, -1] + vt[:, -2, -1]) * v_d[:, -1, -1]
        + (ut[:, -1, -2] + vt[:, -2, -1]) * u_pad[:, -1, -1]
    )

    # NW corner
    nw_value = dt6 * (
        (ut[:, 0, -1] + ut[:, 0, -2]) * u_d[:, 0, -1]
        + (vt[:, 0, -1] + vt_pad[:, 0, -1]) * v_d[:, 0, -1]
        + (ut[:, 0, -2] - vt[:, 0, -1]) * u_pad[:, 0, -1]
    )

    ke_fixed = ke.at[:, 0, 0].set(sw_value)
    ke_fixed = ke_fixed.at[:, -1, 0].set(se_value)
    ke_fixed = ke_fixed.at[:, -1, -1].set(ne_value)
    ke_fixed = ke_fixed.at[:, 0, -1].set(nw_value)
    return ke_fixed


def _apply_legacy_d_sw5_corner_corrections(field_at_corners, edge_halo_field):
    """iter-862: cube-vertex corner adjustments (FV3 sw_core.F90:1709-1715, 1773-1776 d_sw5 legacy).

    SW/SE -= edge_halo; NE/NW += edge_halo. Caller gates on non-duogrid + iter-862 opt-in.
    """
    f = field_at_corners
    f = f.at[:, 0, 0].add(-edge_halo_field[:, 0, 0])
    f = f.at[:, -1, 0].add(-edge_halo_field[:, -1, 0])
    f = f.at[:, -1, -1].add(edge_halo_field[:, -1, -1])
    f = f.at[:, 0, -1].add(edge_halo_field[:, 0, -1])
    return f


def d_sw5_corner_divergence(u_d, v_d, ua, va, cdgrid, dt,
                             d2_bg=0.0, dddmp=0.0, d4_bg=0.16, nord=1,
                             apply_legacy_corner_corrections=False,
                             cross_face_halo=False):
    """FV3 d_sw5 corner divergence damping (sw_core.F90:1641-1821).

    nord=0: del-2 Smag adaptive. nord>0: iterated Laplacian.
    apply_legacy_corner_corrections (iter-862): Fortran-structural corner corrections at non-duogrid;
    RHS halo data incomplete (mode='edge' fallback), gated default-OFF until cross-face halo helper lands.
    Returns ke_damping increment for ke_corner.

    Iterated-Laplacian halo (Fortran oracle: sw_core.F90:1737-1785): the
    cross-face ghost ring was PORTED 2026-07-10 as an OPT-IN
    (``cross_face_halo=True``, duogrid only).  It mirrors Fortran's
    dedicated B-grid exchange for duogrid+nord>0 (dyn_core.F90:651-652:
    `ext_scalar(divgd, dg, bd, domain, 1, 1)` = mpp corner halo update of
    the neighbour's ATTENUATED divgd + cube_rmp onto extension points) via
    `_pad_corner_scalar_cross_face` — NOT the raw local
    divergence_corner_duo ghost values (a raw ghost ring destabilises the
    C48 colliding-modon FB run at day 8).  MEASURED 2026-07-10: even the
    attenuated (ext_scalar-faithful) ghost pumps a seam mode in the FB
    chain on long horizons (modon NaN at day ~60-65), while the default
    zero ghost ring (mode='edge' pad of the boundary-zeroed divg_d) runs
    the same case 120 days clean with healthy vortex decay — hence the
    default stays False (truth tier over oracle tier).  The in-loop
    fill_corners is gated `.not. duogrid`, so no corner rotation is
    involved.  Only the EXPERIMENTAL FB chain (fv3_fb_sw_step)
    calls this; production (fv3_sw_tendencies) does not.  Update the
    test_d_sw5_iterated_laplacian_halo_gap_documentation_marker test
    together with this note.
    """
    # Dispatch hardening (2026-07-10 review): the cross-face ghost ring is
    # built from duogrid tables — silently no-opping on a non-duogrid grid
    # would run different physics than requested.  Static config → fn-entry
    # raise (repo dispatch doctrine).
    if cross_face_halo not in (False, True, "faithful"):
        raise ValueError(
            "d_sw5_corner_divergence: cross_face_halo must be False "
            "(zero-ring), True (nearest-row attenuated ghost) or "
            f"'faithful' (certified k2e ring map); got {cross_face_halo!r}")
    if cross_face_halo and cdgrid.base.duogrid is None:
        raise ValueError(
            "d_sw5_corner_divergence: cross_face_halo=True requires a "
            "duogrid grid (create_cubed_sphere(..., use_duogrid=True)).")
    if (cross_face_halo == "faithful"
            and getattr(cdgrid.base, "gnomonic_form", "") != "ed"):
        # codex bgring-r1 P0-2: the certified ring map's k2e/corner
        # tables are ED-lattice-specific (equiangular tables differ by
        # up to 0.96 at C12) — a non-ED grid would silently run wrong
        # weights.
        raise ValueError(
            "d_sw5_corner_divergence: cross_face_halo='faithful' is "
            "certified for the ED gnomonic duogrid only "
            "(create_fv3_native_cubed_sphere); got gnomonic_form="
            f"{getattr(cdgrid.base, 'gnomonic_form', None)!r}")

    n = cdgrid.n
    cosa_u = cdgrid.cosa_u
    cosa_v = cdgrid.cosa_v
    # iter-87: shared helper sin_sg sub-grid form (Fortran-faithful vs sqrt(1-cosa²))
    sina_u, sina_v = sina_u_v_from_sin_sg(cdgrid)

    dxc = cdgrid.dxc          # (6, n+1, n)
    dyc = cdgrid.dyc          # (6, n, n+1)
    rarea_c = cdgrid.rarea_c  # (6, n+1, n+1)
    da_min_c = jnp.min(1.0 / rarea_c)  # minimum corner area
    d5_bundle = None
    if cross_face_halo == "faithful":
        # codex converge-r1 rank-1: a real ghost ring activates the D5
        # metric-halo coefficients that are inert under the zero ring —
        # ring + metrics must be consistent TOGETHER.  The bundle is
        # the BOUNDED gridstruct's D5 geometry (real native halo
        # strips) in create layout, built once per n (trace-time
        # numpy, jnp constants under jit).
        from legoesm.grids.duogrid_bgrid_ring import build_d5_metric_bundle

        d5_bundle = build_d5_metric_bundle(n)
        dxc = jnp.asarray(d5_bundle["dxc"], dtype=dxc.dtype)
        dyc = jnp.asarray(d5_bundle["dyc"], dtype=dyc.dtype)
        rarea_c = jnp.asarray(d5_bundle["rarea_c"], dtype=rarea_c.dtype)
        da_min_c = jnp.asarray(d5_bundle["da_min_c"],
                               dtype=jnp.result_type(rarea_c))

    # iter-656: ua/va padding inside nord==0 branch only (not module scope)
    if nord == 0:
        # del-2 divergence damping (FV3:1644-1724). iter-655: pad_halo cross-face for nord=0
        dg = cdgrid.base.duogrid
        _offs = None if dg is not None else cdgrid.base.halo_interp_offsets
        ua_full = pad_halo(ua, halo=1, interp_offsets=_offs, duogrid=dg)
        va_full = pad_halo(va, halo=1, interp_offsets=_offs, duogrid=dg)
        ua_pad = ua_full[:, :, 1:-1]  # (6, n+2, n)
        va_pad = va_full[:, 1:-1, :]  # (6, n, n+2)

        # ptc, vort at face midpoints (FV3:1644-1658)
        va_below = va_pad[:, :, :-1]
        va_above = va_pad[:, :, 1:]
        ptc = (u_d - 0.5 * (va_below + va_above) * cosa_v) * dyc * sina_v

        ua_left = ua_pad[:, :-1, :]
        ua_right = ua_pad[:, 1:, :]
        vort = (v_d - 0.5 * (ua_left + ua_right) * cosa_u) * dxc * sina_u

        # delpc at corners. iter-655: vort/ptc on edge midpoints; mode='edge' same-face fallback
        # (proper edge-midpoint cross-face halo helper deferred)
        vort_pad = jnp.pad(vort, [(0, 0), (0, 0), (1, 1)], mode='edge')
        ptc_pad = jnp.pad(ptc, [(0, 0), (1, 1), (0, 0)], mode='edge')

        delpc = (vort_pad[:, :, :-1] - vort_pad[:, :, 1:]
                 + ptc_pad[:, :-1, :] - ptc_pad[:, 1:, :])

        # iter-862: cube-vertex corner corrections (FV3:1709-1715, gated by .not.duogrid).
        # iter-958: corrections are no-ops on duogrid (boundary-zeroed by _divergence_corner_duo).
        # Default OFF: RHS data uses mode='edge' (cross-face DGRID halo deferred to iter-863+).
        if (apply_legacy_corner_corrections
                and cdgrid.base.duogrid is None):
            delpc = _apply_legacy_d_sw5_corner_corrections(delpc, vort_pad)

        delpc = rarea_c * delpc

        # Smag adaptive damp (FV3:1720-1721)
        damp = da_min_c * jnp.maximum(
            d2_bg, jnp.minimum(0.20, dddmp * jnp.abs(delpc * dt)))
        ke_damping = damp * delpc

    else:
        # Higher-order div damping (FV3:1725-1821). _divergence_corner_duo → nord iterations →
        # del-2 + del-(2*nord+2) composite
        # cross_face_halo default False = zero ghost ring (the panel-edge
        # rows of divg_d are zeroed, so the mode='edge' pad below yields a
        # zero ring).  Measured (2026-07-10, C48 colliding modons, FB):
        # zero-ring nord=1 d4_bg=0.16 dddmp=0.2 runs 120 days clean;
        # cross_face_halo=True (Fortran ext_scalar-faithful attenuated
        # ghost) destabilises at day ~60-65 in the FB chain.  Stability
        # (truth tier) outranks oracle-matching -> default stays False.
        # (non-duogrid + cross_face_halo=True raises at fn entry.)
        use_cross_face_halo = cross_face_halo
        if d5_bundle is not None:
            divg_d = _divergence_corner_duo(
                u_d, v_d, ua, va, cdgrid,
                dxc=dxc, dyc=dyc, rarea_c=rarea_c)
        else:
            divg_d = _divergence_corner_duo(u_d, v_d, ua, va, cdgrid)
        delpc = divg_d

        # dd8 (FV3:1811)
        dd8 = (da_min_c * d4_bg) ** (nord + 1)

        # Smag del-2 part (FV3:1790-1805)
        if dddmp > 1e-5:
            rarea = 1.0 / cdgrid.base.area
            dx_u = cdgrid.dx_edge_y
            dy_v = cdgrid.dy_edge_x
            wk = rarea * (u_d[:, :, :-1] * dx_u[:, :, :-1]
                          - u_d[:, :, 1:] * dx_u[:, :, 1:]
                          - v_d[:, :-1, :] * dy_v[:, :-1, :]
                          + v_d[:, 1:, :] * dy_v[:, 1:, :])
            # iter-972: a2b_ord4 4th-order (FV3:1795 a2b_ord4 call)
            wk_corner = interp_center_to_corner_a2b_ord4(wk, cdgrid)
            # FV3_3D iter 183: double-where for grad-safe sqrt at rest state
            _smag_arg = delpc ** 2 + wk_corner ** 2
            _safe_smag_arg = jnp.where(_smag_arg > 0.0, _smag_arg, 1.0)
            _smag_root = jnp.where(
                _smag_arg > 0.0, jnp.sqrt(_safe_smag_arg), 0.0,
            )
            smag_vort = jnp.abs(dt) * _smag_root
            damp2 = da_min_c * jnp.maximum(
                d2_bg, jnp.minimum(0.20, dddmp * smag_vort))
        else:
            damp2 = da_min_c * d2_bg

        # Iterated Laplacian using divg_u/divg_v metrics (FV3:1737-1787, fv_grid_utils.F90:709-735)
        # divg_u = sina_v * dyc / dx; divg_v = sina_u * dxc / dy
        dx = cdgrid.dx_edge_y
        dy = cdgrid.dy_edge_x   # (6, n+1, n)
        divg_u_met = sina_v * dyc / jnp.maximum(dx, _EPS)  # (6, n, n+1)
        divg_v_met = sina_u * dxc / jnp.maximum(dy, _EPS)  # (6, n+1, n)
        # Metric halo: mode='edge' copies the half-offset staggered metric
        # nearest the edge into the ghost face.  This is consistent with the
        # nearest-row ghost surrogate below (codex 2026-07-10: the true
        # extended-grid metric differs by ~3-4% at C48, ~15% at C12 — NOT an
        # exact mirror).  On the default zero-ring path the outer scalar
        # difference is identically zero, so the ghost metric is inert there.
        divg_u_pad = jnp.pad(divg_u_met, [(0, 0), (1, 1), (0, 0)],
                             mode='edge')  # (6, n+2, n+1)
        divg_v_pad = jnp.pad(divg_v_met, [(0, 0), (0, 0), (1, 1)],
                             mode='edge')  # (6, n+1, n+2)
        if d5_bundle is not None:
            # faithful lane: the BOUNDED gridstruct's real divg_u/divg_v
            # incl the native halo rows (replaces both the edge-pad AND
            # the cdgrid-interior approximation — codex converge-r1:
            # these coefficients multiply the ring-activated gradients
            # directly, so they must be geometry-consistent with it)
            divg_u_pad = jnp.asarray(d5_bundle["divg_u_pad"],
                                     dtype=divg_u_pad.dtype)
            divg_v_pad = jnp.asarray(d5_bundle["divg_v_pad"],
                                     dtype=divg_v_pad.dtype)

        # codex 2026-07-10 (HIGH): the opt-in one-ring re-copy is NOT
        # equivalent to Fortran's shrinking wider-halo in-place evolution for
        # nord>=2 (interpolation and the Laplacian do not commute, and early
        # nt>0 iterations consume corner-region halo data a one-ring pad
        # never represents).  Restrict the opt-in to nord==1.
        if use_cross_face_halo and nord > 1:
            raise ValueError(
                "d_sw5_corner_divergence: cross_face_halo=True supports "
                "nord=1 only (one-ring ghost re-copy is not faithful to the "
                "Fortran wider-halo evolution for nord>=2); use the default "
                "zero-ring for nord>=2.")

        for _it in range(nord):
            if use_cross_face_halo == "faithful":
                # 2026-07-20: the FAITHFUL exchange (dyn_core.F90:652
                # ext_scalar B-grid ghost = mpp copy + k2e cube_rmp
                # Lagrange ring + corner Lagrange) as a static linear
                # map extracted by impulse-probing the certified numpy
                # ext_scalar_sixface(·,"B") — weights ARE the certified
                # code's output (duogrid_bgrid_ring).  Map is
                # grid-static: built once per n (disk-cached), applied
                # as a jit-safe gather/segment-sum.
                from legoesm.grids.duogrid_bgrid_ring import (
                    apply_bgrid_ring1,
                    build_bgrid_ring1_map,
                )

                _grid_nord = int(getattr(
                    getattr(cdgrid.base, "duogrid", None), "k2e_nord",
                    2))
                ring_map = build_bgrid_ring1_map(
                    n, k2e_nord=_grid_nord)   # cached, trace-time;
                # order follows the GRID (codex r9: a default-2 map on
                # an explicit nord-4 grid is the mixed-order hazard)
                divg_d_pad = apply_bgrid_ring1(divg_d, ring_map, n)
            elif use_cross_face_halo:
                # 2026-07-10 opt-in port (dyn_core.F90:652 ext_scalar B-grid
                # ghost exchange + sw_core.F90:1737-1787 duogrid nord loop):
                # the ghost ring holds the neighbour's ATTENUATED divg_d via
                # a nearest-row copy (cube_rmp tangential remap NOT applied;
                # see _pad_corner_scalar_cross_face).  The in-loop
                # fill_corners is `.not. duogrid` → no corner fills.
                # NOTE: `cross_face_halo`/`nord` must be static Python values
                # under jit (Python branching).
                divg_d_pad = _pad_corner_scalar_cross_face(divg_d, n)
            else:
                # Default zero-ring: divg_d panel-edge rows are zeroed by
                # _divergence_corner_duo, so this edge-pad yields a ZERO
                # ghost ring on the first iteration; for nord>=2 later
                # iterations it imposes a zero-normal-gradient ghost of the
                # updated (generally nonzero) boundary rows.  Non-duogrid:
                # iter-132 O(1) approx vs Fortran MPI fill_corners halo
                # (unported).
                divg_d_pad = jnp.pad(divg_d, [(0, 0), (1, 1), (1, 1)],
                                     mode='edge')

            # x/y gradient → corner convergence (FV3:1748-1769)
            vc_lap = ((divg_d_pad[:, 1:n+3, 1:n+2]
                       - divg_d_pad[:, 0:n+2, 1:n+2]) * divg_u_pad)
            uc_lap = ((divg_d_pad[:, 1:n+2, 1:n+3]
                       - divg_d_pad[:, 1:n+2, 0:n+2]) * divg_v_pad)
            divg_d = (uc_lap[:, :, :-1] - uc_lap[:, :, 1:]
                      + vc_lap[:, :-1, :] - vc_lap[:, 1:, :])

            # iter-862 (FV3:1773-1776, non-duogrid only): cube-vertex corner corrections inside n-loop
            if (apply_legacy_corner_corrections
                    and cdgrid.base.duogrid is None):
                divg_d = _apply_legacy_d_sw5_corner_corrections(
                    divg_d, uc_lap)

            divg_d = divg_d * rarea_c

        # Composite damping del-2 + del-(2*nord+2) (FV3:1814-1820)
        ke_damping = damp2 * delpc + dd8 * divg_d

    return ke_damping


def _corner_vorticity(uc, vc, cdgrid, use_duogrid, u_d=None, v_d=None):
    """FV3 c_sw corner vorticity from C-grid circulation (sw_core.F90:378-408).

    ``u_d``/``v_d`` (FB chain only, COVARIANT convention): when provided on a
    duogrid with ng>=3, the panel-edge fx/fy halo rows are rebuilt from the
    D winds via ``_pad_halo_uc_vc_via_d2a2c`` (4th-order covariant cross-face
    halo) instead of the cc-average + ``pad_halo_vector`` + re-stagger
    reconstruction — 2026-07-10 audit: on W2/C36 this cuts the corner
    vort_abs seam error 15.7e-5 -> 0.7e-5 s^-1 (the reconstruction was the
    dominant remaining seam term after the covariant-entry conversion).
    """
    n = cdgrid.n
    fx_circ = uc * cdgrid.dxc    # (6, n+1, n)
    fy_circ = vc * cdgrid.dyc    # (6, n, n+1)

    dg = cdgrid.base.duogrid
    if (use_duogrid and u_d is not None and v_d is not None
            and dg is not None and dg.ng >= 3):
        uc_jh, vc_ih = _pad_halo_uc_vc_via_d2a2c(u_d, v_d, cdgrid)
        # Metric halo: edge-copy (iter94: proven low-impact; see note below)
        fx_pad = jnp.concatenate([
            (uc_jh[:, :, 0] * cdgrid.dxc[:, :, 0])[:, :, None],
            fx_circ,
            (uc_jh[:, :, n + 1] * cdgrid.dxc[:, :, -1])[:, :, None],
        ], axis=2)  # (6, n+1, n+2)
        fy_pad = jnp.concatenate([
            (vc_ih[:, 0, :] * cdgrid.dyc[:, 0, :])[:, None, :],
            fy_circ,
            (vc_ih[:, n + 1, :] * cdgrid.dyc[:, -1, :])[:, None, :],
        ], axis=1)  # (6, n+2, n+1)
    # Boundary halo: non-duogrid uses linear extrap (FV3:396-400); duogrid uses cross-face uc/vc
    # iter-836: mode='edge' on fx/fy loses cross-face rotation (15.6% error at cube vertex on W2)
    elif use_duogrid and n >= 2:
        # iter-836b: halo-only fix; PRESERVE interior fx_circ/fy_circ exactly (4th-order A→C in _d2a2c_vect_duogrid)
        uc_cc = 0.5 * (uc[:, :-1, :] + uc[:, 1:, :])
        vc_cc = 0.5 * (vc[:, :, :-1] + vc[:, :, 1:])
        grid = cdgrid.base
        dg = grid.duogrid
        # iter-837: uc/vc are FV3 COVARIANT — pass cos_theta/sin_theta for pad_halo_vector covariant branch
        uc_cc_pad, vc_cc_pad = pad_halo_vector(
            uc_cc, vc_cc,
            grid.cos_angle, grid.sin_angle,
            grid.cos_angle_padded, grid.sin_angle_padded,
            interp_offsets=None, duogrid=dg, halo=1,
            cos_theta=cdgrid.cosa_cell,
            sin_theta=cdgrid.sina_cell,
        )
        # Extract HALO rows only and reconstruct face-staggered values
        uc_halo_j_below = 0.5 * (uc_cc_pad[:, :-1, 0] + uc_cc_pad[:, 1:, 0])
        uc_halo_j_above = 0.5 * (uc_cc_pad[:, :-1, n + 1]
                                  + uc_cc_pad[:, 1:, n + 1])
        vc_halo_i_left = 0.5 * (vc_cc_pad[:, 0, :-1] + vc_cc_pad[:, 0, 1:])
        vc_halo_i_right = 0.5 * (vc_cc_pad[:, n + 1, :-1]
                                  + vc_cc_pad[:, n + 1, 1:])
        # Metric halo: edge-copy the boundary dxc/dyc into the halo row.
        # iter93 oracle finding: FV3 in DUOGRID mode skips the dxc/dyc
        # edge-extrapolation + mpp_update the non-duogrid path uses
        # (fv_grid_tools.F90:899-916 + :1107 are gated `.not. duogrid`) and
        # instead carries the exact CROSS-FACE metric halo from its extended grid,
        # so this edge-copy is formally a faithfulness gap.  BUT iter94 RULED IT
        # OUT as a meaningful FB-residual term: replacing edge-copy with O(dx²)
        # linear extrapolation (2*edge - first-interior) of dxc/dyc changed the FB
        # W2 C36 day-1 max|u_d| by <0.2% (48.60 → 48.53, still day-2 NaN).  ⇒ the
        # FB residual is NOT in the corner-vorticity METRIC halo (consistent with
        # iter82-83 "structural corner coupling, not metric"); the remaining
        # candidate inside `_corner_vorticity` is the uc/vc halo RECONSTRUCTION
        # (the 2-pt center-avg + re-stagger above), not the metric.  Kept as
        # edge-copy — the validated baseline; the exact cross-face metric is a
        # known-LOW-priority TODO, proven not to move the residual.
        dxc_halo_j_below = cdgrid.dxc[:, :, 0]    # (6, n+1)
        dxc_halo_j_above = cdgrid.dxc[:, :, -1]   # (6, n+1)
        dyc_halo_i_left = cdgrid.dyc[:, 0, :]     # (6, n+1)
        dyc_halo_i_right = cdgrid.dyc[:, -1, :]   # (6, n+1)

        fx_halo_j_below = uc_halo_j_below * dxc_halo_j_below   # (6, n+1)
        fx_halo_j_above = uc_halo_j_above * dxc_halo_j_above   # (6, n+1)
        fy_halo_i_left = vc_halo_i_left * dyc_halo_i_left      # (6, n+1)
        fy_halo_i_right = vc_halo_i_right * dyc_halo_i_right   # (6, n+1)

        # Interior fx_pad = fx_circ; halo rows at j=-1 and j=n ONLY are
        # replaced with the rotated values.  Interior values unchanged.
        fx_pad = jnp.concatenate([
            fx_halo_j_below[:, :, jnp.newaxis],   # (6, n+1, 1) j=-1
            fx_circ,                               # (6, n+1, n) interior
            fx_halo_j_above[:, :, jnp.newaxis],   # (6, n+1, 1) j=n
        ], axis=2)  # (6, n+1, n+2)
        fy_pad = jnp.concatenate([
            fy_halo_i_left[:, jnp.newaxis, :],    # (6, 1, n+1) i=-1
            fy_circ,                               # (6, n, n+1) interior
            fy_halo_i_right[:, jnp.newaxis, :],   # (6, 1, n+1) i=n
        ], axis=1)  # (6, n+2, n+1)
    else:
        # Non-duogrid path: edge padding + linear extrapolation at the 4
        # panel-edge boundaries (sw_core.F90:396-400).
        fx_pad = jnp.pad(fx_circ, [(0, 0), (0, 0), (1, 1)], mode='edge')
        fy_pad = jnp.pad(fy_circ, [(0, 0), (1, 1), (0, 0)], mode='edge')
        if n > 2:
            fx_pad = fx_pad.at[:, :, 0].set(2 * fx_circ[:, :, 0] - fx_circ[:, :, 1])
            fx_pad = fx_pad.at[:, :, n + 1].set(2 * fx_circ[:, :, n - 1] - fx_circ[:, :, n - 2])
            fy_pad = fy_pad.at[:, 0, :].set(2 * fy_circ[:, 0, :] - fy_circ[:, 1, :])
            fy_pad = fy_pad.at[:, n + 1, :].set(2 * fy_circ[:, n - 1, :] - fy_circ[:, n - 2, :])

    # Direct corner vorticity: vort(i,j) = fx(i,j-1) - fx(i,j) - fy(i-1,j) + fy(i,j)
    vort = (fx_pad[:, :, :-1] - fx_pad[:, :, 1:]
            - fy_pad[:, :-1, :] + fy_pad[:, 1:, :])

    # Corner corrections for non-duogrid (FV3 sw_core.F90:396-400)
    if not use_duogrid:
        vort = vort.at[:, 0, 0].add(fy_pad[:, 0, 0])
        vort = vort.at[:, n, 0].add(-fy_pad[:, n + 1, 0])
        vort = vort.at[:, n, n].add(-fy_pad[:, n + 1, n])
        vort = vort.at[:, 0, n].add(fy_pad[:, 0, n])

    rarea_c = 1.0 / cdgrid.area_corner
    return cdgrid.f_corner + rarea_c * vort


def _vorticity_flux(v_d, u_d, uc, vc, vort_abs, cdgrid, use_duogrid):
    """FV3 c_sw vorticity transport flux (sw_core.F90:416-480).

    Returns fy1, vort_x (x-face) and fx1, vort_y (y-face).
    Uses 1/sin (NOT 1/sin²) per FV3 comment at sw_core.F90:417.

    `sina_u` / `sina_v` come from the sin_sg sub-grid as
    ``0.5*(sin_sg(i-1,j,3) + sin_sg(i,j,1))`` (matching
    `fv_grid_utils.F90:505-518`) rather than ``sqrt(1 - cosa**2)``.
    The two are not identical because `cosa_u` is a halo-averaged
    value of `cos_sg` and the trigonometric identity does not hold
    on averaged quantities.
    """
    n = cdgrid.n
    sina_u, sina_v = sina_u_v_from_sin_sg(cdgrid)

    fy1 = (v_d - uc * cdgrid.cosa_u) / jnp.maximum(sina_u, _EPS)
    if not use_duogrid:
        fy1 = fy1.at[:, 0, :].set(v_d[:, 0, :])
        fy1 = fy1.at[:, n, :].set(v_d[:, n, :])
    vort_x = jnp.where(fy1 > 0, vort_abs[:, :, :-1], vort_abs[:, :, 1:])

    fx1 = (u_d - vc * cdgrid.cosa_v) / jnp.maximum(sina_v, _EPS)
    if not use_duogrid:
        fx1 = fx1.at[:, :, 0].set(u_d[:, :, 0])
        fx1 = fx1.at[:, :, n].set(u_d[:, :, n])
    vort_y = jnp.where(fx1 > 0, vort_abs[:, :-1, :], vort_abs[:, 1:, :])

    return fy1, vort_x, fx1, vort_y


# ==============================================================================
# c_sw: C-grid half-step
# ==============================================================================

def _c_sw(h, u_d, v_d, h_s, cdgrid, dt, g):
    """FV3 c_sw: C-grid half of forward-backward. Mass via 1st-order upwind; (uc,vc) via vort + KE grad."""
    n = cdgrid.n
    dt2 = 0.5 * dt
    dg = cdgrid.base.duogrid
    use_duogrid = dg is not None and dg.ng >= 2

    # 1. d2a2c_vect
    ua, va, uc, vc, ut, vt = d2a2c_vect(u_d, v_d, cdgrid)

    # 2. Scale transport: ut/vt * dt2 * edge_length * sin_sg_upwind (FV3 fv_grid_utils.F90:570)
    dy = cdgrid.dy_edge_x   # (6, n+1, n)
    dx = cdgrid.dx_edge_y   # (6, n, n+1)
    sg = cdgrid.sin_sg
    grid = cdgrid.base
    _offs = None if use_duogrid else grid.halo_interp_offsets
    _dg = dg if use_duogrid else None

    # x-direction
    sin_east = sg[:, :, :, 2]
    sin_west = sg[:, :, :, 0]
    se_pad = pad_halo(sin_east, interp_offsets=_offs, duogrid=_dg)
    sw_pad = pad_halo(sin_west, interp_offsets=_offs, duogrid=_dg)
    sin_upwind_x = jnp.where(ut > 0, se_pad[:, :n+1, 1:-1],
                                      sw_pad[:, 1:n+2, 1:-1])
    ut_scaled = dt2 * ut * dy * sin_upwind_x

    # y-direction
    sin_north = sg[:, :, :, 3]
    sin_south = sg[:, :, :, 1]
    sn_pad = pad_halo(sin_north, interp_offsets=_offs, duogrid=_dg)
    ss_pad = pad_halo(sin_south, interp_offsets=_offs, duogrid=_dg)
    sin_upwind_y = jnp.where(vt > 0, sn_pad[:, 1:-1, :n+1],
                                      ss_pad[:, 1:-1, 1:n+2])
    vt_scaled = dt2 * vt * dx * sin_upwind_y

    # 3. First-order upwind mass transport
    h_pad = pad_halo_auto(h, cdgrid)
    h_left = h_pad[:, :-1, 1:-1]   # (6, n+1, n)
    h_right = h_pad[:, 1:, 1:-1]
    fx = jnp.where(ut_scaled > 0, h_left, h_right) * ut_scaled

    h_bot = h_pad[:, 1:-1, :-1]    # (6, n, n+1)
    h_top = h_pad[:, 1:-1, 1:]
    fy = jnp.where(vt_scaled > 0, h_bot, h_top) * vt_scaled

    # Duogrid flux synchronization (FV3 dyn_core.F90:853-900)
    if use_duogrid:
        fx, fy = synchronize_cgrid_fluxes(fx, fy, n)

    rarea = 1.0 / cdgrid.base.area
    h_star = h + (fx[:, :-1, :] - fx[:, 1:, :] + fy[:, :, :-1] - fy[:, :, 1:]) * rarea

    # 4. KE at cc (FV3:303-372). c_sw uses KE only (no g*h)
    ke_u, ke_v = _ke_upwind(uc, vc, ua, va, u_d, v_d, cdgrid, use_duogrid)
    ke_total = dt2 * 0.5 * (ua * ke_u + va * ke_v)

    # 5. Corner vorticity (FV3:378-408). FB chain: pass the (covariant)
    # D winds so the seam halo uses the d2a2c covariant cross-face path.
    vort_abs = _corner_vorticity(uc, vc, cdgrid, use_duogrid,
                                 u_d=u_d, v_d=v_d)

    # 6. Vorticity flux at C-faces (FV3:416-480)
    fy1, vort_x, fx1, vort_y = _vorticity_flux(
        v_d, u_d, uc, vc, vort_abs, cdgrid, use_duogrid)
    fy1 = dt2 * fy1
    fx1 = dt2 * fx1

    # 7. KE gradient at C-faces
    ke_pad = pad_halo_auto(ke_total, cdgrid)
    dke_x = cdgrid.rdxc * (ke_pad[:, :-1, 1:-1] - ke_pad[:, 1:, 1:-1])
    dke_y = cdgrid.rdyc * (ke_pad[:, 1:-1, :-1] - ke_pad[:, 1:-1, 1:])

    # 8. Update C-grid covariant velocities
    uc_new = uc + fy1 * vort_x + dke_x
    vc_new = vc - fx1 * vort_y + dke_y

    return h_star, uc_new, vc_new, ua, va


# ==============================================================================
# Complete forward-backward step
# ==============================================================================

def require_duogrid_fb(cdgrid, entry_name):
    """Fn-entry guard: the FB chain is DUOGRID-ONLY (codex 2026-07-10 F1).

    On non-duogrid grids the covariant-wind FB chain mixes conventions at
    panel seams: the D→A halo in ``d2a2c_d_to_a`` falls back to an
    orthogonal-rotation cross-face copy (no cos_theta/sin_theta covariant
    rotation) and ``_u_orth_at_v_points`` falls back to a same-face
    edge-pad — both silently wrong at seams.  All FB validation (W2,
    colliding modons) ran duogrid; non-duogrid FB was already
    known-degraded.  Dispatch-hardening doctrine: raise loudly instead of
    running silently-wrong seam numerics.
    """
    dg = cdgrid.base.duogrid
    if dg is None or dg.ng < 2:
        raise ValueError(
            f"{entry_name} requires a duogrid cubed-sphere grid "
            f"(create_cubed_sphere(..., use_duogrid=True) with ng >= 2); "
            f"got duogrid="
            f"{'None' if dg is None else f'ng={dg.ng}'}. The FB "
            f"covariant-wind chain mixes conventions at panel seams on "
            f"non-duogrid grids. For non-duogrid grids use the production "
            f"FV3EdgeShallowWaterModel (fv3_sw_tendencies + RK3), "
            f"which is calibrated for the model's own orthogonal "
            f"wind convention end-to-end.")


def fv3_forward_backward_step(h, u_d, v_d, h_s, cdgrid, dt, g=constants.g,
                               div_damp=0.0, hyperdiff_coeff=0.0,
                               apply_legacy_d_sw4_corner_ke_fix=False,
                               apply_legacy_d_sw5_corner_corrections=False,
                               apply_fortran_xppm_boundary=False):
    """EXPERIMENTAL FV3 forward-backward step (unstable by ~50 steps; use fv3_sw_tendencies+RK3 for production).

    Phase 1: c_sw (C-grid half). Phase 2: p_grad_c. Phase 3: _d_sw_native (FV3 PPM + KE/vort transport).
    div_damp here is LEGACY/UNUSED (FB chain uses d_sw5 d2_bg/dddmp/d4_bg/nord instead).
    DUOGRID-ONLY: raises ValueError on non-duogrid grids (see require_duogrid_fb).
    d_sw5 ``cross_face_halo`` deliberately NOT plumbed here — see the
    fv3_fb_sw_step docstring NOTE (destabilizes the 120d modon; research-only).
    """
    require_duogrid_fb(cdgrid, "fv3_forward_backward_step")

    # Phase 0: model-orthogonal → covariant v (see fv3_fb_sw_step)
    v_d = fb_v_d_to_covariant(u_d, v_d, cdgrid)

    # Phase 1: c_sw
    h_star, uc_new, vc_new, ua, va = _c_sw(
        h, u_d, v_d, h_s, cdgrid, dt, g)

    # Phase 2: backward pressure gradient
    dt2 = 0.5 * dt
    dp_x, dp_y = _p_grad_c(h_star, h_s, cdgrid, dt2, g)
    uc_new = uc_new + dp_x
    vc_new = vc_new + dp_y

    # Phase 3: d_sw_native (iter-871c forwards iter-869b/iter-871b opt-in flags)
    h_new, u_d_new, v_d_new = _d_sw_native(
        h, u_d, v_d, h_s, uc_new, vc_new, ua, va, cdgrid, dt, g,
        div_damp=div_damp,
        apply_legacy_d_sw4_corner_ke_fix=apply_legacy_d_sw4_corner_ke_fix,
        apply_legacy_d_sw5_corner_corrections=(
            apply_legacy_d_sw5_corner_corrections),
        apply_fortran_xppm_boundary=apply_fortran_xppm_boundary)

    # covariant → model-orthogonal v (inverse of Phase 0)
    v_d_new = fb_v_d_to_orthogonal(u_d_new, v_d_new, cdgrid)

    return h_new, u_d_new, v_d_new


# ==============================================================================
# NEW: FV3-faithful forward-backward WITHOUT Arakawa-Lamb gradient
# ==============================================================================

def _p_grad_c(h_star, h_s, cdgrid, dt2, g):
    """Backward p-gradient g*grad(h_star+h_s) at C-faces (gravity-wave stability via implicit coupling)."""
    p = g * (h_star + h_s)
    p_pad = pad_halo_auto(p, cdgrid)
    # 2-point gradient (same sign as c_sw KE gradient)
    dp_x = dt2 * cdgrid.rdxc * (p_pad[:, :-1, 1:-1] - p_pad[:, 1:, 1:-1])
    dp_y = dt2 * cdgrid.rdyc * (p_pad[:, 1:-1, :-1] - p_pad[:, 1:-1, 1:])
    return dp_x, dp_y


def ppm_transport_1d(field, courant, rdelta, axis, external_halo: int = 0,
                      apply_d_sw3_boundary_fix: bool = False,
                      boundary_fix_dx_field=None, rd_prepadded: bool = False,
                      boundary_fix_edges=None):
    """PPM hord=9 staggered-field transport (FV3 ytp_v/xtp_u, sw_core.F90:2897-3353, 2540-2894 jord=9).

    Used for B-grid KE transport in d_sw3. N cells → N+1 interface fluxes.
    external_halo (iter-945): when > 0, field already has cross-face halo on sweep axis.

    rd_prepadded (cube tiled np>6 stage, task #3 U1): when True, ``rdelta``
    is ALREADY the depth-1 edge-padded array of shape ``(6, N+2, M)`` along
    the sweep axis (so the internal ``jnp.pad(rd, ((0,0),(1,1),(0,0)),
    'edge')`` is SKIPPED).  A sub-face TILE supplies its own ``rd_pad`` with
    a REAL depth-1 neighbour-tile halo at interior cuts (the global edge-pad
    is wrong there — the upwind CFL cell lives in the neighbour tile).
    Default False is BIT-IDENTICAL to the prior behaviour.

    boundary_fix_edges (2026-07-10 tiled follow-up): the d_sw3 one-sided
    overrides + cube-vertex bl=br=0 zeroing are only valid at GLOBAL
    cube-face boundaries.  Static 4-tuple of Python bools
    ``(sweep_lo, sweep_hi, cross_lo, cross_hi)`` gating, respectively, the
    south/north sweep-axis override blocks and the M-axis-endpoint vertex
    zeroing at M index 0 / -1.  ``None`` (the serial full-face default)
    applies all four — bit-identical to the previous always-on behaviour.
    A sub-face TILE passes the flags for the global edges its window
    touches (see ``legoesm.parallel.tiled_transport``).
    """
    # Transpose so sweep axis is axis 1 for uniform indexing
    if axis == 1:
        v = field    # (6, N + 2*ext, M)
        c = courant  # (6, N+1, M)
        rd = rdelta  # (6, N, M)
    else:
        v = jnp.swapaxes(field, 1, 2)     # (6, N + 2*ext, M)
        c = jnp.swapaxes(courant, 1, 2)   # (6, N+1, M)
        rd = jnp.swapaxes(rdelta, 1, 2)   # (6, N, M)

    nn = v.shape[1] - 2 * external_halo  # interior cells along sweep axis

    # Fail loudly on a mis-sized prepadded rdelta (codex U1 MEDIUM): the
    # flux slices rd_pad[:, :nn+1] / [:, 1:nn+2], so a too-LONG rd would be
    # silently truncated.  Shapes are static at trace time → cheap check.
    if rd_prepadded and rd.shape[1] != nn + 2:
        raise ValueError(
            f"ppm_transport_1d: rd_prepadded expects rdelta depth-1 "
            f"padded to length nn+2={nn + 2} on the sweep axis, got "
            f"{rd.shape[1]}.")

    # Bring field to total halo h3=4 (external preserved; rest mode='edge')
    h3 = 4
    if external_halo == h3:
        vp = v
    elif external_halo < h3:
        gap = h3 - external_halo
        vp = jnp.pad(v, [(0, 0), (gap, gap), (0, 0)], mode='edge')
    else:
        trim = external_halo - h3
        vp = v[:, trim:v.shape[1] - trim, :]
    # vp shape: (6, N + 2*h3, M) — same as pre-iter-945 internal layout

    # --- Monotone slopes (FV3 sw_core.F90:3165-3171) ---
    # dm[j] at each padded cell. We compute for padded cells 1..N+4 (need j±1).
    xt = 0.25 * (vp[:, 2:, :] - vp[:, :-2, :])  # (6, N+4, M), at padded cells 1..N+4
    vm = vp[:, 1:-1, :]  # (6, N+4, M)
    vhi = jnp.maximum(jnp.maximum(vp[:, :-2, :], vm), vp[:, 2:, :])
    vlo = jnp.minimum(jnp.minimum(vp[:, :-2, :], vm), vp[:, 2:, :])
    dm = jnp.sign(xt) * jnp.minimum(
        jnp.abs(xt), jnp.minimum(vhi - vm, vm - vlo))
    # dm[k] = slope at padded cell k+1 (k=0..N+3)

    # --- Cell differences (FV3 sw_core.F90:3173-3177) ---
    dq = vp[:, 1:, :] - vp[:, :-1, :]  # (6, N+5, M), dq[k] = v[k+1]-v[k] at padded k

    # --- Edge values (FV3 sw_core.F90:3180-3183) ---
    # al at interface between padded cells k and k+1:
    #   al = 0.5*(v[k]+v[k+1]) + r3*(dm_at_k - dm_at_k+1)
    # dm_at_k = dm[k-1] (dm starts at padded cell 1, so dm index = padded cell - 1)
    r3 = 1.0 / 3.0
    al = (0.5 * (vp[:, 1:-2, :] + vp[:, 2:-1, :])
          + r3 * (dm[:, :-1, :] - dm[:, 1:, :]))
    # al[k] = interface between padded cells k+1 and k+2, k=0..N+2
    # al shape: (6, N+3, M)

    # --- PPM reconstruction jord=9 from ytp_v (sw_core.F90:3194-3204) ---
    # IMPORTANT: ytp_v/xtp_u (B-grid wind transport) uses pmp/lac limiter
    # for jord=9, NOT pert_ppm(iv=0).  The transported fields (D-grid winds)
    # are SIGNED, so the positive-definite constraint would incorrectly zero
    # the reconstruction for negative wind cells.  Only tp_core.F90 xppm/yppm
    # (mass/vorticity transport via fv_tp_2d) uses pert_ppm(iv=0) for iord=9.
    #
    # Need bl/br for cells -1..N (for flux at interfaces 0..N).
    nc = nn + 2  # cells -1..N

    # Indices in padded coordinates for cells -1..N:
    # Cell j (original 0-based) sits at padded index j+h3.
    # Cell -1 → padded h3-1=3; cell N → padded h3+N=nn+4.
    # al[k] = edge between padded cells k+1 and k+2.
    # Left edge of cell j: al[j+h3-2]. Right edge: al[j+h3-1].
    al_l = al[:, 1:1+nc, :]    # al_left for cells -1..N
    al_r = al[:, 2:2+nc, :]    # al_right for cells -1..N
    v_c = vp[:, h3-1:h3-1+nc, :]   # v at cells -1..N

    # pmp/lac coefficients (sw_core.F90:3197-3202):
    #   pmp_1 = -2*dq[j],  lac_1 = pmp_1 + 1.5*dq[j+1]  (bl direction)
    #   pmp_2 = 2*dq[j-1], lac_2 = pmp_2 - 1.5*dq[j-2]  (br direction)
    # where dq[j] = v[j+1]-v[j] (single difference).
    # dq_at_cell(j) = dq[j+h3] in padded indexing (dq[k]=vp[k+1]-vp[k]).
    # For cells -1..N: dq starts at index h3-1.
    p_off = h3 - 1  # dq at cell j is at dq index j + p_off
    pmp_1 = -2.0 * dq[:, p_off:p_off+nc, :]           # -2*dq[j] for j=-1..N
    lac_1 = pmp_1 + 1.5 * dq[:, p_off+1:p_off+1+nc, :]  # + 1.5*dq[j+1]
    pmp_2 = 2.0 * dq[:, p_off-1:p_off-1+nc, :]        # 2*dq[j-1]
    lac_2 = pmp_2 - 1.5 * dq[:, p_off-2:p_off-2+nc, :]  # - 1.5*dq[j-2]

    z = jnp.zeros_like(pmp_1)
    bl = jnp.minimum(
        jnp.maximum(jnp.maximum(z, pmp_1), lac_1),
        jnp.maximum(al_l - v_c,
                     jnp.minimum(jnp.minimum(z, pmp_1), lac_1)))
    br = jnp.minimum(
        jnp.maximum(jnp.maximum(z, pmp_2), lac_2),
        jnp.maximum(al_r - v_c,
                     jnp.minimum(jnp.minimum(z, pmp_2), lac_2)))
    # bl, br: (6, nc, M) for cells -1..N (index 0..nc-1)

    # Iter-967: d_sw3 cube-edge boundary fix (sw_core.F90:3239-3316
    # ytp_v branch; sw_core.F90:2819-2863 xtp_u branch — same formula
    # mirrored across the sweep axis).
    #
    # CRITICAL Fortran detail: d_sw3 calls ytp_v / xtp_u with
    # ``bounded_domain=.false.`` HARDCODED at lines 1315-1316 and
    # 1373-1374, regardless of the global bounded_domain flag.  The
    # boundary fix in xtp_u at line 2819 fires when
    # ``(.not. bounded_domain .or. .not. dg%is_initialized)`` — with
    # the HARDCODED .false., the condition becomes
    # ``(.not. .false. .or. ...) = .true.``, so the fix ALWAYS FIRES
    # for d_sw3 wind transport on cube-face boundaries (regardless
    # of duogrid status).
    #
    # Constants from sw_core.F90:38: s11=11/14, s14=4/7, s15=3/14.
    # Index map: Fortran cell j ↔ Python k = j (when bl/br is indexed
    # 0..nc-1 over Python cells -1..N, where Python cell j = Fortran
    # cell j+1 → bl/br index k_python = Fortran j_fortran).
    #
    # Boundary overrides at js=1 (south boundary):
    #   br(2) = al(3) - v(2)
    #   xt = s15*v(1) + s11*v(2) - s14*dm(2)
    #   br(1) = xt - v(1);  bl(2) = xt - v(2)
    #   bl(0) = s14*dm(-1) - s11*dq(-1)
    #   xt = (length-weighted xt of v(0)/v(-1) and v(1)/v(2) extrap)
    #   bl(1) = xt - v(1);  br(0) = xt - v(0)
    #   pert_ppm(v(2), bl(2), br(2), iv=-1) → standard PPM constraint
    if apply_d_sw3_boundary_fix:
        # Global-edge gating (see docstring): static Python bools, so the
        # serial default (None → all True) traces the identical graph.
        sweep_lo, sweep_hi, cross_lo, cross_hi = (
            (True, True, True, True) if boundary_fix_edges is None
            else boundary_fix_edges)
        s11_c = 11.0 / 14.0
        s14_c = 4.0 / 7.0
        s15_c = 3.0 / 14.0
        # Index map: vp[h3+j_F-1], dm/dq[h3+j_F-2], al[h3+j_F-3]. bl/br at j_F (Python k)

        # SOUTH (Fortran j_F ∈ {-1,0,1,2,3})
        v_jm1 = vp[:, h3 - 2, :]
        v_j0 = vp[:, h3 - 1, :]
        v_j1 = vp[:, h3, :]
        v_j2 = vp[:, h3 + 1, :]
        dm_jm1 = dm[:, h3 - 3, :]
        dm_j2 = dm[:, h3, :]
        dq_jm1 = dq[:, h3 - 3, :]
        al_j3 = al[:, h3, :]

        # NORTH (Fortran j_F ∈ {N-1,N,N+1,N+2})
        v_npy_m2 = vp[:, h3 + nn - 2, :]
        v_npy_m1 = vp[:, h3 + nn - 1, :]
        v_npy = vp[:, h3 + nn, :]
        v_npy_p1 = vp[:, h3 + nn + 1, :]
        dm_npy_m2 = dm[:, h3 + nn - 3, :]
        dm_npy_p1 = dm[:, h3 + nn, :]
        dq_npy = dq[:, h3 + nn - 1, :]
        al_npy_m2 = al[:, h3 + nn - 4, :]

        # Optional length-weighted xt via dx (interior shape, padded to h3=4)
        if boundary_fix_dx_field is not None:
            if axis == 2:
                dxf_int = jnp.swapaxes(boundary_fix_dx_field, 1, 2)
            else:
                dxf_int = boundary_fix_dx_field
            # Pad sweep axis to total halo h3 to match vp.
            dxf = jnp.pad(dxf_int, [(0, 0), (h3, h3), (0, 0)],
                           mode='edge')
            dx_m2 = dxf[:, h3 - 2, :]
            dx_m1 = dxf[:, h3 - 1, :]
            dx_1 = dxf[:, h3, :]
            dx_2 = dxf[:, h3 + 1, :]
            dx_npy_m2 = dxf[:, h3 + nn - 2, :]
            dx_npy_m1 = dxf[:, h3 + nn - 1, :]
            dx_npy = dxf[:, h3 + nn, :]
            dx_npy_p1 = dxf[:, h3 + nn + 1, :]
        else:
            dx_m2 = dx_m1 = dx_1 = dx_2 = None
            dx_npy_m2 = dx_npy_m1 = dx_npy = dx_npy_p1 = None

        # SOUTH boundary fix (overrides bl/br at k=0,1,2) — global lo edge only
        if sweep_lo:
            br = br.at[:, 2, :].set(al_j3 - v_j2)
            # xt = s15*v(1) + s11*v(2) - s14*dm(2)
            xt_s = s15_c * v_j1 + s11_c * v_j2 - s14_c * dm_j2
            br = br.at[:, 1, :].set(xt_s - v_j1)
            bl = bl.at[:, 2, :].set(xt_s - v_j2)
            # bl(0) = s14*dm(-1) - s11*dq(-1)
            bl = bl.at[:, 0, :].set(s14_c * dm_jm1 - s11_c * dq_jm1)
            # ELSE branch (length-weighted xt for bl(1), br(0)):
            if dx_m1 is not None:
                x0L = 0.5 * (
                    ((2.0 * dx_m1 + dx_m2) * v_j0 - dx_m1 * v_jm1)
                    / jnp.maximum(dx_m1 + dx_m2, _EPS)
                )
                x0R = 0.5 * (
                    ((2.0 * dx_1 + dx_2) * v_j1 - dx_1 * v_j2)
                    / jnp.maximum(dx_1 + dx_2, _EPS)
                )
                xt_s2 = x0L + x0R
            else:
                xt_s2 = 0.5 * ((1.5 * v_j0 - 0.5 * v_jm1)
                                + (1.5 * v_j1 - 0.5 * v_j2))
            bl = bl.at[:, 1, :].set(xt_s2 - v_j1)
            br = br.at[:, 0, :].set(xt_s2 - v_j0)

        # NORTH boundary fix (overrides bl/br at k=N-1,N,N+1) — global hi edge
        k_nm2 = nn - 1
        k_nm1 = nn
        k_n = nn + 1

        if sweep_hi:
            # bl(npy-2) = al(npy-2) - v(npy-2)
            bl = bl.at[:, k_nm2, :].set(al_npy_m2 - v_npy_m2)
            # xt = s15*v(npy-1) + s11*v(npy-2) + s14*dm(npy-2)
            xt_n = s15_c * v_npy_m1 + s11_c * v_npy_m2 + s14_c * dm_npy_m2
            br = br.at[:, k_nm2, :].set(xt_n - v_npy_m2)
            bl = bl.at[:, k_nm1, :].set(xt_n - v_npy_m1)
            # br(npy) = s11*dq(npy) - s14*dm(npy+1)
            br = br.at[:, k_n, :].set(s11_c * dq_npy - s14_c * dm_npy_p1)
            # ELSE branch (length-weighted xt for br(npy-1), bl(npy)):
            if dx_npy_m1 is not None:
                x0L_n = 0.5 * (
                    ((2.0 * dx_npy_m1 + dx_npy_m2) * v_npy_m1
                      - dx_npy_m1 * v_npy_m2)
                    / jnp.maximum(dx_npy_m1 + dx_npy_m2, _EPS)
                )
                x0R_n = 0.5 * (
                    ((2.0 * dx_npy + dx_npy_p1) * v_npy
                      - dx_npy * v_npy_p1)
                    / jnp.maximum(dx_npy + dx_npy_p1, _EPS)
                )
                xt_n2 = x0L_n + x0R_n
            else:
                xt_n2 = 0.5 * ((1.5 * v_npy_m1 - 0.5 * v_npy_m2)
                                + (1.5 * v_npy - 0.5 * v_npy_p1))
            br = br.at[:, k_nm1, :].set(xt_n2 - v_npy_m1)
            bl = bl.at[:, k_n, :].set(xt_n2 - v_npy)

        # Cube-VERTEX zeroing (sw_core.F90 ytp_v:3263-3274/3302-3313,
        # xtp_u:2824-2829/2847-2852): at the two perpendicular panel-edge
        # rows (M index 0 and -1), the boundary cells go piecewise-constant
        # (bl=br=0) — FV3's reflection control at the 3-face cube vertex.
        # Overwrites the two-sided x0L+x0R blend at those 4 points per
        # edge-end, exactly as the Fortran if(j==1 .or. j==npy) branch does.
        # Gated to the sweep blocks that fired AND the M endpoints that are
        # GLOBAL perpendicular edges (a tile-interior M endpoint is a
        # neighbour-tile cut, not a cube vertex).
        for _kk in (([0, 1] if sweep_lo else [])
                    + ([k_nm1, k_n] if sweep_hi else [])):
            if cross_lo:
                bl = bl.at[:, _kk, 0].set(0.0)
                br = br.at[:, _kk, 0].set(0.0)
            if cross_hi:
                bl = bl.at[:, _kk, -1].set(0.0)
                br = br.at[:, _kk, -1].set(0.0)

        # pert_ppm(iv=1) at j=2 and j=npy-2
        if sweep_lo:
            bl_2 = bl[:, 2, :]
            br_2 = br[:, 2, :]
            bl_2_new, br_2_new = pert_ppm(bl_2, br_2)
            bl = bl.at[:, 2, :].set(bl_2_new)
            br = br.at[:, 2, :].set(br_2_new)
        if sweep_hi:
            bl_nm2 = bl[:, k_nm2, :]
            br_nm2 = br[:, k_nm2, :]
            bl_nm2_new, br_nm2_new = pert_ppm(bl_nm2, br_nm2)
            bl = bl.at[:, k_nm2, :].set(bl_nm2_new)
            br = br.at[:, k_nm2, :].set(br_nm2_new)

    # Flux evaluation (FV3 sw_core.F90:3339-3349). cfl = c*rdy_upwind
    # rd_prepadded (task #3 U1): a sub-face tile supplies rd ALREADY depth-1
    # edge-padded (real neighbour-tile halo at interior cuts); skip the pad.
    rd_pad = rd if rd_prepadded else jnp.pad(
        rd, [(0, 0), (1, 1), (0, 0)], mode='edge')
    rdy_pos = rd_pad[:, :nn+1, :]
    rdy_neg = rd_pad[:, 1:nn+2, :]

    v_pos = vp[:, h3-1:h3-1+nn+1, :]
    v_neg = vp[:, h3:h3+nn+1, :]
    bl_pos = bl[:, :nn+1, :]
    br_pos = br[:, :nn+1, :]
    bl_neg = bl[:, 1:nn+2, :]
    br_neg = br[:, 1:nn+2, :]

    cfl_pos = jnp.abs(c) * rdy_pos
    cfl_neg = jnp.abs(c) * rdy_neg

    flux_pos = v_pos + (1.0 - cfl_pos) * (br_pos - cfl_pos * (bl_pos + br_pos))
    flux_neg = v_neg + (1.0 - cfl_neg) * (bl_neg - cfl_neg * (bl_neg + br_neg))

    flux = jnp.where(c > 0, flux_pos, flux_neg)

    # Transpose back
    if axis == 2:
        flux = jnp.swapaxes(flux, 1, 2)

    return flux


def bgrid_corner_courant_local(uc_pad, vc_pad, cosa, rsina, dt5):
    """B-grid contravariant corner Courant numbers ``(vb, ub)`` — the
    pointwise Step 1/3 of :func:`_bgrid_ke_transport`, factored so a sub-face
    tile can compute its corner-block ``(vb, ub)`` from its slice of the
    GLOBALLY cross-face-halo'd ``uc_pad``/``vc_pad`` + corner metrics
    (approach-C cube tiling, task #3 — mirrors ``d2a2c_ua_va_local``: the
    cross-face halo runs in the global view, this core is pure pointwise).

    Shape-generic (global or per-tile):
      ``uc_pad`` ``(.., A, B+1)`` j-padded uc; ``vc_pad`` ``(.., A+1, B)``
      i-padded vc; ``cosa``/``rsina`` corner metrics ``(.., A, B)``; returns
      ``vb, ub`` ``(.., A, B)``.  The two adjacent-sums collapse the padded
      axis: ``vc_sum`` over i, ``uc_sum`` over j.
    """
    vc_sum = vc_pad[:, :-1, :] + vc_pad[:, 1:, :]
    uc_sum = uc_pad[:, :, :-1] + uc_pad[:, :, 1:]
    vb = dt5 * (vc_sum - uc_sum * cosa) * rsina
    ub = dt5 * (uc_sum - vc_sum * cosa) * rsina
    return vb, ub


def _bgrid_ke_transport(u_d, v_d, uc, vc, cdgrid, dt):
    """FV3 d_sw3: B-grid KE at corners via 1D PPM transport with contravariant Courant numbers.

    Matches sw_core.F90:1201-1388 (duogrid/bounded_domain branch).
    """
    n = cdgrid.n
    dt5 = 0.5 * dt
    cosa = cdgrid.cosa_corner    # (6, n+1, n+1)
    rsina = cdgrid.rsin2_corner  # (6, n+1, n+1) = 1/sin²

    dg = cdgrid.base.duogrid
    use_duogrid = dg is not None and dg.ng >= 2

    # Step 1: B-grid contravariant v-velocity Courant number.
    # vb = dt/2 * (vc_sum - uc_sum*cosa)*rsina at corners.
    # iter-947 → 2026-07-10: NEW-corrected cross-face halo via _pad_halo_uc_vc_new_via_neighbor_delta (ng>=3)
    if use_duogrid and dg.ng >= 3:
        uc_pad, vc_pad = _pad_halo_uc_vc_new_via_neighbor_delta(
            uc, vc, u_d, v_d, cdgrid)
    else:
        vc_pad = jnp.pad(vc, [(0, 0), (1, 1), (0, 0)], mode='edge')
        uc_pad = jnp.pad(uc, [(0, 0), (0, 0), (1, 1)], mode='edge')
    # Corner Courant (vb, ub) — pointwise, factored to bgrid_corner_courant_local
    # so the sub-face tiled stage reuses the EXACT core (task #3 cube tiling).
    # Both are pure functions of (uc_sum, vc_sum, cosa, rsina); computing ub
    # here (before the ytp_v sweep) instead of at the old Step-3 site is
    # bit-identical — no data dependency on the sweep.
    vb, ub = bgrid_corner_courant_local(uc_pad, vc_pad, cosa, rsina, dt5)  # (6,n+1,n+1)

    # iter-945: cross-face halo for (u_d, v_d) PPM sweep via _pad_halo_dgrid_for_ppm
    # (FV3 mpp_update_domains DGRID_NE analogue). iter-950 NEGATIVE: h_dg=3 regresses v_ll_Linf.
    if use_duogrid:
        h_dg = 2
        u_d_ihalo, v_d_jhalo = _pad_halo_dgrid_for_ppm(
            u_d, v_d, cdgrid, halo=h_dg)
    else:
        h_dg = 0
        u_d_ihalo = u_d
        v_d_jhalo = v_d

    # Step 2: PPM ytp_v hord=9 (FV3:1315).  d_sw3 hardcodes
    # bounded_domain=.false. (sw_core.F90:1315-1316), so the one-sided
    # edge overrides ALWAYS fire in Fortran even with duogrid halos —
    # halo AND boundary fix together, plus the cube-vertex bl=br=0
    # zeroing.  (iter-967 tested the fix INSTEAD of the halo and
    # without the vertex zeroing — confounded negative.)
    rdy = 1.0 / jnp.maximum(cdgrid.dy_edge_x, _EPS)  # (6, n+1, n)
    transported_y = ppm_transport_1d(
        v_d_jhalo, vb, rdy, axis=2, external_halo=h_dg,
        apply_d_sw3_boundary_fix=True,
        boundary_fix_dx_field=cdgrid.dy_edge_x)

    # --- Step 3: B-grid contravariant u-velocity (Courant number) ---
    # ub computed above with vb via bgrid_corner_courant_local (bit-identical).

    # --- Step 4: transport u_d in x-direction using ub (PPM hord=9) ---
    # Same d_sw3 hardcoded-.false. one-sided edge overrides as Step 2
    # (xtp_u branch, sw_core.F90:2819-2863 incl. vertex zeroing).
    rdx = 1.0 / jnp.maximum(cdgrid.dx_edge_y, _EPS)  # (6, n, n+1)
    transported_x = ppm_transport_1d(
        u_d_ihalo, ub, rdx, axis=1, external_halo=h_dg,
        apply_d_sw3_boundary_fix=True,
        boundary_fix_dx_field=cdgrid.dx_edge_y)

    # Step 5: BGRID_NE component sync (FV3 dyn_core.F90:968-1019). Fortran fires inside if(duogrid) block.
    # iter-102: route vector avg through geographic frame (avoids per-seam rotation tables).
    # iter-944b: gate on duogrid (Fortran-faithful).
    ubbtemp = transported_y
    vbbtemp = vb
    ubb = ub
    vbb = transported_x
    if use_duogrid:
        # iter3: exact non-orthogonal corner c2l z-matrix (z11=cos_angle_corner,
        # z12=sin_angle_corner, z21/z22 = j-tangent rows) — fixes the O(1)
        # vertex corruption of the prior orthogonal rotation.
        ubb, vbbtemp = synchronize_bgrid_ne_corner_geo(
            ubb, vbbtemp,
            cdgrid.cos_angle_corner, cdgrid.sin_angle_corner,
            cdgrid.z21_corner, cdgrid.z22_corner, n)

    # Step 6: KE at corners = 0.5*(ubbtemp*vbbtemp + ubb*vbb) (FV3 dyn_core.F90:1013-1020 Lin-Rood)
    ke_corner = 0.5 * (ubbtemp * vbbtemp + ubb * vbb)

    return ke_corner


def _d_sw_native(h, u_d, v_d, h_s, uc, vc, ua, va, cdgrid, dt, g,
                 div_damp=0.0, d2_bg=0.0, dddmp=0.0, d4_bg=0.16, nord=1,
                 damp_v=0.0, nord_v=0,
                 apply_legacy_d_sw4_corner_ke_fix=False,
                 apply_legacy_d_sw5_corner_corrections=False,
                 apply_fortran_xppm_boundary=False,
                 cross_face_halo=False):
    """FV3 d_sw1..d_sw6 D-grid full-step (dyn_core.F90).

    d_sw1: transport velocity + PPM mass transport. d_sw3: B-grid KE transport.
    d_sw5: corner div damping + vorticity transport. d_sw6: wind replacement + vorticity damping.
    d_sw5 damping uses d2_bg/dddmp/d4_bg/nord. damp_v/nord_v for d_sw6 vorticity damping (FV3 vtdm4).
    div_damp LEGACY/UNUSED in FB chain.
    """

    # Step 1: contravariant transport velocity (FV3 d_sw1).
    # iter-947 → 2026-07-10: forward OLD u_d/v_d for duogrid NEW-corrected halo via _pad_halo_uc_vc_new_via_neighbor_delta
    ut, vt = _d_sw1_recompute_ut_vt(
        uc, vc, cdgrid, dt, u_d_old=u_d, v_d_old=v_d)
    # iter-944b: REVERTED iter-944 CGRID_NE (ut, vt) sync — Fortran only syncs MASS flux

    # Step 2: PPM mass transport (FV3 d_sw1, sw_core.F90:886-887 unconditional nord=nord_v, damp_c=damp_v).
    # Outer threshold lives in fv_tp_2d (damp_c > 1e-4); no Python outer guard.
    # Fortran.  Codex adversarial review flagged this as a non-
    # faithful accretion.  Iter-728 removes the guard: always
    # forward ``nord=nord_v, damp_c=damp_v`` and let the internal
    # tp_core.F90:217 gate handle damp_v=0.  Fortran-exact; no
    # behavioural change for any damp_v because both the Python and
    # the Fortran internal gates are ``> 1e-4`` (so 0 <= damp_v
    # <= 1e-4 is a no-op both ways, and damp_v > 1e-4 invokes the
    # del-n smoother both ways).
    #
    # NOTE on the Fortran param naming (dyn_core.F90:762-770):
    #   damp_t = damp_v = damp_vt(k) = flagstruct%vtdm4
    #   nord_t = nord_v(k)
    # So mass damping (sw_core.F90:886-887) and vorticity damping
    # (sw_core.F90:1948) share one coefficient pair.  ``q_con``
    # transport at sw_core.F90:942-943 uses ``damp_t`` / ``nord_t``
    # — same numeric value per the dyn_core assignment, but the
    # code-path names are distinct.  We forward damp_v/nord_v here
    # because the delp call at sw_core.F90:886-887 names them
    # literally.
    # iter-888b: forward xppm_boundary for tp_core.F90:614-628 / 632-647 s11/s14/s15 boundary formula
    h_new = transport_step(h, ut, vt, dt, cdgrid,
                           nord=nord_v, damp_c=damp_v,
                           apply_fortran_xppm_boundary=(
                               apply_fortran_xppm_boundary))

    # Step 3: cc vorticity from D-grid circulation (CCW: bottom - top + right - left)
    dx_u = cdgrid.dx_edge_y  # (6, n, n+1)
    dy_v = cdgrid.dy_edge_x  # (6, n+1, n)

    vt_circ = u_d * dx_u
    ut_circ = v_d * dy_v

    rarea = 1.0 / cdgrid.base.area
    zeta = rarea * (vt_circ[:, :, :-1] - vt_circ[:, :, 1:]
                    + ut_circ[:, 1:, :] - ut_circ[:, :-1, :])
    zeta_abs = zeta + cdgrid.base.f

    # Step 4: B-grid KE transport (FV3 d_sw3, sw_core.F90:1201-1388)
    ke_corner = _bgrid_ke_transport(u_d, v_d, uc, vc, cdgrid, dt)

    # iter-869: optional d_sw4 cube-vertex KE fix (FV3:1442-1465, non-bounded_domain)
    if apply_legacy_d_sw4_corner_ke_fix:
        ke_corner = _apply_legacy_d_sw4_corner_ke_fix(
            ke_corner, ut, vt, u_d, v_d, dt,
            bounded_domain=cdgrid.base.bounded_domain)

    # Step 5: corner div damping into KE (FV3 d_sw5, sw_core.F90:1641-1821). nord=1, d4_bg=0.16 defaults.
    use_d_sw5_damping = (d2_bg > 1e-10 or dddmp > 1e-10 or d4_bg > 1e-10)
    if use_d_sw5_damping:
        # iter-871b: forward iter-862 corner-corrections flag through FB wrapper
        ke_damping = d_sw5_corner_divergence(
            u_d, v_d, ua, va, cdgrid, dt,
            d2_bg=d2_bg, dddmp=dddmp, d4_bg=d4_bg, nord=nord,
            apply_legacy_corner_corrections=(
                apply_legacy_d_sw5_corner_corrections),
            cross_face_halo=cross_face_halo)
        ke_corner = ke_corner + ke_damping

    # iter-944b: REVERTED iter-942 ke_corner sync (FV3 reference has it commented out)

    # Step 6: KE gradient at D-grid edges (FV3 d_sw6, sw_core.F90:1935-1944)
    ke_diff_u_scaled = ke_corner[:, :-1, :] - ke_corner[:, 1:, :]  # (6, n, n+1)
    ke_diff_v_scaled = ke_corner[:, :, :-1] - ke_corner[:, :, 1:]  # (6, n+1, n)

    # Step 7: vorticity transport to D-edges (FV3 d_sw5 fv_tp_2d).
    # iter-864: apply_cgrid_flux_sync=False matches Fortran's commented-out sync block (dyn_core.F90:1124-1207)
    crx, cry, xfx, yfx, ra_x, ra_y = compute_transport_quantities(
        ut, vt, dt, cdgrid)
    fx_vort, fy_vort = fv_tp_2d(
        zeta_abs, crx, cry, xfx, yfx, ra_x, ra_y, cdgrid,
        apply_cgrid_flux_sync=False,
        apply_fortran_xppm_boundary=apply_fortran_xppm_boundary)
    # iter-944b: REVERTED iter-944 vorticity flux sync (FV3 reference has it commented out)

    # Step 8: D-grid wind update (FV3 d_sw6, sw_core.F90:1935-1944). Incremental: u*dx += ke_diff + fy_vort
    rdx_u = 1.0 / jnp.maximum(dx_u, _EPS)  # (6, n, n+1)
    rdy_v = 1.0 / jnp.maximum(dy_v, _EPS)  # (6, n+1, n)

    u_d_new = u_d + (ke_diff_u_scaled + fy_vort) * rdx_u
    v_d_new = v_d + (ke_diff_v_scaled - fx_vort) * rdy_v

    # Step 9: vorticity damping (FV3 d_sw6:1948-2000). damp4 = (damp_v*da_min_c)^(nord_v+1)
    if damp_v > 1e-5:
        da_min_c = jnp.min(cdgrid.area_corner)
        damp4 = (damp_v * da_min_c) ** (nord_v + 1)
        dg = cdgrid.base.duogrid
        _use_dg = dg is not None and dg.ng >= 2
        fx2, fy2 = _del6_vt_flux(nord_v, damp4, zeta, cdgrid,
                                  use_duogrid=_use_dg)
        u_d_new = u_d_new + fy2 * rdx_u
        v_d_new = v_d_new - fx2 * rdy_v

    return h_new, u_d_new, v_d_new


def _u_orth_at_v_points(u_d, v_d_orth, cdgrid):
    """Model-orthogonal u (V·x̂) at v_d points (x-faces): 4-pt average of
    u_d with a depth-1 cross-face i-halo (duogrid; orthogonal basis) or an
    edge-pad fallback (non-duogrid).  (u_d, v_d_orth) MUST be the model's
    orthogonal pair — the halo basis is "orthogonal"."""
    n = cdgrid.n
    dg = cdgrid.base.duogrid
    if dg is not None and dg.ng >= 2:
        h = 2
        u_pad, _ = _pad_halo_dgrid_for_ppm(
            u_d, v_d_orth, cdgrid, halo=h, basis="orthogonal")
        # u_pad: (6, n+2h, n+1); padded i-index p ↔ cell p-h
    else:
        h = 1
        u_pad = jnp.pad(u_d, [(0, 0), (1, 1), (0, 0)], mode='edge')
    return 0.25 * (u_pad[:, h - 1:h + n, :-1] + u_pad[:, h:h + n + 1, :-1]
                   + u_pad[:, h - 1:h + n, 1:] + u_pad[:, h:h + n + 1, 1:])


def fb_v_d_to_covariant(u_d, v_d, cdgrid):
    """Model-orthogonal v_d → true FV3 covariant v (2026-07-10 FB fix).

    Convention (measured, cubed_sphere_cdgrid.py angle tables): the model's
    D winds are the ORTHOGONAL pair u_d = V·x̂ (x̂ = unit i-tangent,
    angle_edge_x/angle_edge_y) and v_d = V·rot90(x̂); FV3's Fortran chain
    assumes COVARIANT winds u = V·x̂ (identical) and v = V·ŷ (ŷ = unit
    j-line tangent).  With ŷ = cosa·x̂ + sina·rot90(x̂) (grid verified
    right-handed; cosa_u/sina_u match the x̂↔ĵ-edge-tangent angle to 3e-4
    everywhere incl. panel seams):

        v_cov = cosa_u·(V·x̂) + sina_u·v_d,   V·x̂ at v-points = 4-pt avg
                                              of u_d (cross-face i-halo).

    O(cosa_u) reaches 0.5 at cube vertices — feeding v_d directly into the
    covariant formulas (ut/vt rsin, KE c·C products, circulation, B-grid
    Courant, one_grad_p) was the FB panel-edge instability root cause.
    """
    sina_u, _ = sina_u_v_from_sin_sg(cdgrid)
    ubar = _u_orth_at_v_points(u_d, v_d, cdgrid)
    return cdgrid.cosa_u * ubar + sina_u * v_d


def fb_v_d_to_orthogonal(u_d, v_cov, cdgrid):
    """Approximate inverse of :func:`fb_v_d_to_covariant` for the FB exit —
    a DELIBERATE seam-filtering approximation, NOT an exact coordinate
    inverse (codex 2026-07-10 F4 reframe).

    Two fixed-point passes: the cross-face u halo needs the orthogonal
    pair, so pass 1 inverts with an edge-pad ū (exact in the interior),
    pass 2 rebuilds ū with the proper orthogonal halo.  Interior: exact
    inverse (same ū).  Seams: the 2-pass truncation leaves a residual that
    is 2nd order (halo-u sensitivity × pass-1 seam error) — a
    dt-INDEPENDENT ~7e-4 m/s per-step kick at seam rows (W2 C36).  That
    residual is kept ON PURPOSE: it acts as a weak seam filter that damps
    the seam mode the covariant chain pumps.

    2026-07-10 MEASURED (do NOT "fix" by adding a 3rd pass / exact
    inverse; commit 59a89ae12 reverted exactly that, 3da499f27): one more
    pass contracts the roundtrip seam residual ~10x (7e-4 → 7e-5 m/s) but
    WORSENS the W2 C36 2-day FB drift — max|dv| 6.2 → 9.6 m/s, max|u|
    41.6 → 42.3 (scripts/tmp/_fb_edge_repro.py, single-variable
    pass-count probe; 6h unchanged at 38.9).  Truth tier (measured
    long-run drift) outranks the roundtrip-identity metric — same
    doctrine as the d_sw5 cross_face_halo default-OFF.

    jax.grad implication: because the 2-pass inverse is not the exact
    inverse of :func:`fb_v_d_to_covariant`, the FB roundtrip is not an
    identity map at seams — gradients through an FB step differentiate
    the APPROXIMATE (filtered) map, including its dt-independent seam
    residual, not an idealised exact-roundtrip step.  The map is smooth
    (jnp.maximum floor only guards sina_u≈0, which does not occur on the
    cubed sphere), so grads stay finite/well-defined; but loss terms that
    probe seam-row winds at ~1e-3 m/s precision will see the residual and
    its gradient.  Changing the pass count changes BOTH the primal and
    the gradient — keep primal/adjoint consistent (2 passes).
    """
    sina_u, _ = sina_u_v_from_sin_sg(cdgrid)
    rs = 1.0 / jnp.maximum(sina_u, _EPS)
    n = cdgrid.n
    u_pad0 = jnp.pad(u_d, [(0, 0), (1, 1), (0, 0)], mode='edge')
    ubar0 = 0.25 * (u_pad0[:, :n + 1, :-1] + u_pad0[:, 1:n + 2, :-1]
                    + u_pad0[:, :n + 1, 1:] + u_pad0[:, 1:n + 2, 1:])
    v = (v_cov - cdgrid.cosa_u * ubar0) * rs
    dg = cdgrid.base.duogrid
    if dg is None or dg.ng < 2:
        return v
    ubar = _u_orth_at_v_points(u_d, v, cdgrid)
    return (v_cov - cdgrid.cosa_u * ubar) * rs


def fv3_fb_sw_step(h, u_d, v_d, h_s, cdgrid, dt, g=constants.g,
                   div_damp=0.0, d2_bg=0.0, dddmp=0.0, d4_bg=0.16, nord=1,
                   damp_v=0.0, nord_v=0,
                   apply_legacy_d_sw4_corner_ke_fix=False,
                   apply_legacy_d_sw5_corner_corrections=False,
                   apply_fortran_xppm_boundary=False,
                   cross_face_halo=False):
    """EXPERIMENTAL FV3 forward-backward SW step (unstable at C16; use fv3_sw_tendencies+RK3 for prod).

    Phase 1: c_sw (dt/2). Phase 2: p_grad_c (dt/2). Phase 3: _d_sw_native d_sw1-6 chain.
    div_damp LEGACY/UNUSED (FB uses d_sw5 d2_bg/dddmp/d4_bg/nord); damp_v / nord_v for vorticity damping.
    DUOGRID-ONLY: raises ValueError on non-duogrid grids (see require_duogrid_fb).

    NOTE (codex 2026-07-10 F5): the d_sw5 ``cross_face_halo`` option
    (Fortran's attenuated cross-face divergence ghost, dyn_core.F90:651)
    is deliberately NOT plumbed to this entry point or any model config:
    the faithful attenuated ghost measurably DESTABILIZES the 120-day
    colliding-modon run, while the default zero-ring ghost runs clean —
    truth tier over oracle tier.  ``cross_face_halo=True`` is
    research-only via a direct ``d_sw5_corner_divergence`` call (nord=1
    only; raises on nord>=2).
    """
    require_duogrid_fb(cdgrid, "fv3_fb_sw_step")
    dt2 = 0.5 * dt

    # Phase 0 (2026-07-10 wind-convention fix): the prognostic winds are the
    # model's ORTHOGONAL pair; the FV3 chain below is Fortran-verbatim
    # COVARIANT.  Convert v at entry, run covariant, convert back at exit
    # (u is identical in both conventions — V·x̂).  See fb_v_d_to_covariant.
    v_d = fb_v_d_to_covariant(u_d, v_d, cdgrid)

    # Phase 1: c_sw
    h_star, uc_new, vc_new, ua, va = _c_sw(
        h, u_d, v_d, h_s, cdgrid, dt, g)

    # Phase 2: p_grad_c
    dp_x, dp_y = _p_grad_c(h_star, h_s, cdgrid, dt2, g)
    uc_new = uc_new + dp_x
    vc_new = vc_new + dp_y

    # Phase 3: d_sw_native (iter-871c forwards iter-869b/iter-871b opt-in flags)
    h_new, u_d_new, v_d_new = _d_sw_native(
        h, u_d, v_d, h_s, uc_new, vc_new, ua, va, cdgrid, dt, g,
        div_damp=div_damp, d2_bg=d2_bg, dddmp=dddmp, d4_bg=d4_bg, nord=nord,
        damp_v=damp_v, nord_v=nord_v,
        apply_legacy_d_sw4_corner_ke_fix=apply_legacy_d_sw4_corner_ke_fix,
        apply_legacy_d_sw5_corner_corrections=(
            apply_legacy_d_sw5_corner_corrections),
        apply_fortran_xppm_boundary=apply_fortran_xppm_boundary,
        cross_face_halo=cross_face_halo)

    # Phase 4: one_grad_p — D-grid BACKWARD pressure-gradient update on the
    # prognostic winds (FV3 dyn_core.F90:2347 one_grad_p / 1529 grad1_p_update).
    # iter77 ROOT-CAUSE FIX: pre-iter77 the FB step applied the pressure gradient
    # ONLY at the C-grid (_p_grad_c on uc/vc, Phase 2), so the prognostic D-grid
    # winds u_d/v_d never felt the PGF — the vector-invariant momentum eqn was
    # missing its -∇Φ term.  For a steady geostrophic state (W2) the winds then
    # had NO restoring force balancing Coriolis, seeding a dt-independent growing
    # mode that NaN'd ~3 h regardless of dt or dissipation.
    #
    # Vector-invariant form: du = -dt·∂Φ/∂x with the geopotential Φ = g·(h+h_s)
    # at the B-grid CORNERS (a2b_ord4, FV3's a2b(gz)), BACKWARD-centred on the
    # post-mass-update height h_new.  Same corner-difference staggering + dt-LINEAR
    # scaling as the d_sw KE gradient (ke_corner ∝ dt, verified), and the same
    # sign convention (Φ[i]-Φ[i+1] mirrors ke_corner[i]-ke_corner[i+1]).
    gz_b = interp_center_to_corner_a2b_ord4(
        g * (h_new + h_s), cdgrid)  # (6, n+1, n+1) geopotential at corners
    rdx_u = 1.0 / jnp.maximum(cdgrid.dx_edge_y, _EPS)  # (6, n, n+1) — u_d edge
    rdy_v = 1.0 / jnp.maximum(cdgrid.dy_edge_x, _EPS)  # (6, n+1, n) — v_d edge
    u_d_new = u_d_new + dt * rdx_u * (gz_b[:, :-1, :] - gz_b[:, 1:, :])
    v_d_new = v_d_new + dt * rdy_v * (gz_b[:, :, :-1] - gz_b[:, :, 1:])

    # Phase 5: covariant → model-orthogonal v (inverse of Phase 0).
    v_d_new = fb_v_d_to_orthogonal(u_d_new, v_d_new, cdgrid)

    return h_new, u_d_new, v_d_new
